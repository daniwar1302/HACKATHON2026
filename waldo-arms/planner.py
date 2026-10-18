"""Pick-and-place over the arm_server protocol: one planner for the real G1 and for waldo-sim.

Movement lessons from the sim, built in:
  * straight-line hand paths (~3 cm steps, each solved from the previous pose): the arm doesn't swing through the
    desk or the torso the way one big joint-space move does;
  * a lift-clear route out of the hanging pose (hands start below the desk top, under its front edge);
  * level-jaw tilted grasps: the hand comes in ~60 deg below horizontal, heading perpendicular to the jaw direction
    so both fingers stay level (straight down only reaches 32 cm out at desk height; 60 deg reaches 40 cm);
  * the wrist turns gradually on the way down instead of one big swing;
  * a contact stage for the final descent and lift (finger pads may touch the desk);
  * fingertip grasps for thin items, rim pinches for rings wider than the 95 mm Dex1 opening;
  * robustness: candidate grasps (both hands, jaw +-15/30 deg, pitches 45-80 deg) screened by IK first, executed best
    first, the grip verified after lifting, and on any failure the hand recovers (opens, retreats) and tries the next.

    p = Planner(arm, desk_top_z=-0.06)       # arm(cmd, **kw) -> reply dict, raising PlanError on ok=false
    r = p.pick_item(xyz, jaw_deg=30, height_m=0.012)   # -> {"side", "attempts", ...}; PlanError if every candidate fails
    p.place(r["side"], box_min, box_max)
    p.home(r["side"])
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass

import numpy as np


class PlanError(RuntimeError):
    pass


class Arm:
    """Minimal arm_server client: arm(cmd, **kw) -> reply dict, raising PlanError when ok is false."""

    def __init__(self, addr: str, timeout_s: float = 120):
        import zmq
        self.sock = zmq.Context.instance().socket(zmq.REQ)
        self.sock.setsockopt(zmq.RCVTIMEO, int(timeout_s * 1000))
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.connect(addr)

    def __call__(self, cmd, **kw):
        self.sock.send(json.dumps(dict(cmd=cmd, **kw)).encode())
        r = json.loads(self.sock.recv())
        if not r.get("ok", False):
            raise PlanError(r.get("error") or f"{cmd} failed")
        return r


def heading_for(jaw_deg: float, side: str = "left") -> float:
    """Approach heading (deg, desk plane): perpendicular to the jaw direction so both fingers stay level, pointing
    away from the robot (within +-100 deg of +x), coming in from that hand's own side when both qualify."""
    h = (jaw_deg - 90 + 180) % 360 - 180
    cands = [c for c in (h, (h + 360) % 360 - 180) if abs(c) <= 100] or [h + 180 if abs(h) > 90 else h]
    sgn = -1 if side == "left" else 1
    return max(cands, key=lambda c: (sgn * math.sin(math.radians(c)) >= -1e-9, -abs(c)))


def tilted_R(pitch_deg: float, jaw_deg: float, side: str = "left") -> np.ndarray:
    """Grasp frame: hand x along heading_for(jaw, side), tilted pitch_deg below horizontal; jaws close along the
    horizontal direction jaw_deg (exactly level)."""
    p, h = math.radians(pitch_deg), math.radians(heading_for(jaw_deg, side))
    x = np.array([math.cos(p) * math.cos(h), math.cos(p) * math.sin(h), -math.sin(p)])
    y = np.array([math.cos(math.radians(jaw_deg)), math.sin(math.radians(jaw_deg)), 0.0])
    y = y - x * (y @ x)
    y /= np.linalg.norm(y)
    return np.column_stack([x, y, np.cross(x, y)])


def _log(R):
    c = max(-1.0, min(1.0, (np.trace(R) - 1) / 2))
    a = math.acos(c)
    if a < 1e-9:
        return np.zeros(3)
    return a * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * math.sin(a))


def _exp(v):
    a = np.linalg.norm(v)
    if a < 1e-9:
        return np.eye(3)
    k = v / a
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * K @ K


def slerp(R0, R1, t):
    return R0 @ _exp(t * _log(R0.T @ R1))


@dataclass
class Grasp:
    side: str
    jaw: float
    pitch: float
    R: np.ndarray
    pre: np.ndarray
    grip: np.ndarray
    back: np.ndarray
    cost: float


class Planner:
    STEP, SPEED = 0.03, 0.15                 # m per straight-line segment, m/s along the path
    JAW_OFFSETS = (0, 15, -15, 30, -30)      # a parallel gripper still holds a long item 30 deg off square
    PITCHES = (60, 50, 70, 45, 80)
    TRIES_PER_SIDE = 3

    def __init__(self, arm, desk_top_z: float = -0.06, log=None, twin=True, scene_kind: str = "sort"):
        self.arm, self.desk, self.log = arm, desk_top_z, log or (lambda *a: None)
        self.R = {"left": None, "right": None}
        self.twin = None
        if twin:
            try:                               # collision-free motion planning on a MuJoCo digital twin
                from motion import Twin
                self.twin = Twin(desk_top_z, kind=scene_kind)
            except Exception as e:             # no mujoco / assets on this machine: straight-line moves only
                self.log(f"motion planning unavailable ({e}); using straight-line moves")

    def q14(self):
        return np.array(self.arm("state")["arm"]["q"], float)

    def move_free(self, side, target, R=None):
        """Plan a collision-free joint path to the hand at `target` (orientation R, or level jaws) and run it."""
        if self.twin is None:
            self.lin(side, target, R)
            return
        from motion import timed
        path = self.twin.plan(side, self.q14(), target, R)
        if path is None:
            raise PlanError(f"no collision-free path to {np.round(target, 3).tolist()}")
        q, times = timed(path)
        self.arm("trajectory", q=q, times=times, side=side)
        self.R[side] = R

    # ----- primitives -----
    def hand(self, side):
        st = self.arm("state")
        return np.array(st["fk" if side == "left" else "fk_right"]["grasp_xyz"])

    def _move(self, side, xyz, R, duration, contact=False):
        kw = {"R": R.tolist()} if R is not None else {}
        if contact:
            kw["contact"] = True
        self.arm("move_ee", side=side, xyz=[float(v) for v in xyz], duration=float(duration), **kw)
        self.R[side] = R

    def lin(self, side, to, R=None, step=None, speed=None, contact=False):
        """Straight-line hand move. With the twin: solved locally and sent as ONE smooth trajectory (no stop at each
        step). Without it: a series of small move_ee steps."""
        step, speed = step or self.STEP, speed or self.SPEED
        p0, to = self.hand(side), np.asarray(to, float)
        dist = float(np.linalg.norm(to - p0))
        n = max(1, math.ceil(dist / step))
        pts = [p0 + (to - p0) * k / n for k in range(1, n + 1)]
        if self.twin is not None:
            from motion import cartesian_path, timed
            path = cartesian_path(self.twin, side, self.q14(), pts, R)
            if path is not None:
                q, times = timed(path, max_joint_speed=0.6 if contact else 0.8)
                kw = {"contact": True} if contact else {}
                self.arm("trajectory", q=q, times=times, side=side, **kw)
                self.R[side] = R
                return
        for p in pts:
            self._move(side, p, R, max(0.2, dist / n / speed), contact)

    def lin_turn(self, side, to, R1, step=0.02):
        """Straight-line move that rotates the hand to R1 a little per step (never one big wrist swing)."""
        f = self.arm("fk", side=side)
        p0, R0, to = np.array(f["grasp_xyz"]), np.array(f["grasp_R"]), np.asarray(to, float)
        n = max(3, math.ceil(np.linalg.norm(to - p0) / step), math.ceil(np.linalg.norm(_log(R0.T @ R1)) / math.radians(10)))
        for k in range(1, n + 1):
            self._move(side, p0 + (to - p0) * k / n, slerp(R0, R1, k / n), 0.3)

    @staticmethod
    def stage(name, fn):
        try:
            return fn()
        except PlanError as e:
            raise PlanError(f"{name}: {e}") from None

    def lift_points(self, side):
        sy = 1 if side == "left" else -1
        return [np.array([0.10, 0.26 * sy, -0.10]), np.array([0.14, 0.26 * sy, self.desk + 0.14])]

    def ensure_clear(self, side):
        """From wherever the hand is, get it up over the desk with level jaws at a safe height."""
        if self.twin is not None:
            p = self.hand(side)
            if p[0] > 0.15 and p[2] < self.desk + 0.10:     # low over the desk: lift straight up a little first
                self.stage("retreat up", lambda: self.lin(side, [p[0], p[1], self.desk + 0.10], self.R[side], contact=True))
            return
        p = self.hand(side)
        safe_z = self.desk + 0.16
        if p[0] > 0.15 and p[2] < safe_z:                  # over the desk: straight up first, keeping orientation
            self.stage("retreat up", lambda: self.lin(side, [p[0], p[1], safe_z], self.R[side]))
        elif p[0] <= 0.15 and p[2] < self.desk + 0.05:     # hanging / below the desk edge: the lift route
            lp = self.lift_points(side)
            self.stage("lift out of hang", lambda: self._move(side, lp[0], None, 0.8))   # starts outside the workspace box
            self.stage("lift above desk", lambda: self.lin(side, lp[1], None))
        if self.R[side] is not None:
            self.stage("level wrist", lambda: self.lin_turn(side, self.hand(side), self._level_R(side)))

    def _level_R(self, side):
        """Level-jaw orientation the IK picks by itself at the current hand position (what move_ee without R does)."""
        q = self.arm("ik", side=side, xyz=[float(v) for v in self.hand(side)])
        return np.array(self.arm("fk", side=side, q7=q["q7"])["grasp_R"])

    def recover(self, side):
        """After a failed attempt: open, back off up and out, level the wrist. Best effort; reports if stuck."""
        try:
            self.arm("open", side=side)
        except PlanError:
            pass
        try:
            self.ensure_clear(side)
            return True
        except PlanError as e:
            self.log(f"recover {side}: {e}")
            try:   # smaller steps straight up, keeping whatever orientation we have
                p = self.hand(side)
                self.lin(side, [p[0], p[1], self.desk + 0.16], self.R[side], step=0.01)
                return True
            except PlanError:
                return False

    def ik_ok(self, side, pt, R):
        try:
            self.arm("ik", side=side, xyz=[float(v) for v in pt], R=R.tolist())
            return True
        except PlanError:
            return False

    # ----- grasp candidates -----
    def candidates(self, side, xyz, jaw_deg, height_m, thin=False, round_item=False):
        """Feasible grasps for one hand, cheapest first. IK-screened at the pre-grasp, the grip and the lift point."""
        x, y, z = (float(v) for v in xyz)
        jaws = range(0, 180, 30) if round_item else [jaw_deg + o for o in self.JAW_OFFSETS]
        grip_z = z - (0.002 if height_m < 0.025 else 0.0)        # centre height, 2 mm lower for thin items
        out = []
        for i, jaw in enumerate(jaws):
            hd = math.radians(heading_for(jaw, side))
            back = np.array([math.cos(hd), math.sin(hd), 0.0])
            for j, pitch in enumerate(self.PITCHES):
                R = tilted_R(pitch, jaw, side)
                grip = np.array([x, y, grip_z])
                if thin:
                    grip = grip - 0.022 * R[:, 0]                  # fingertip pads at the item, jaw centre above
                pre = grip - 0.035 * back + np.array([0, 0, 0.07])
                lift = grip - 0.02 * back + np.array([0, 0, 0.13])
                if all(self.ik_ok(side, p, R) for p in (pre, grip)):
                    out.append(Grasp(side, jaw, pitch, R, pre, grip, back, cost=(0 if round_item else i) * 1.0 + j * 0.6))
                    break   # best pitch for this jaw found
        return sorted(out, key=lambda g: g.cost)

    def execute(self, g: Grasp, lift=0.13):
        side = g.side
        self.arm("enable")
        self.ensure_clear(side)
        if self.twin is not None:
            self.stage("to pre-grasp", lambda: self.move_free(side, g.pre, g.R))
        else:
            self.stage("over target", lambda: self.lin(side, [g.pre[0], g.pre[1], self.desk + 0.14], None))
            self.stage("approach (turning)", lambda: self.lin_turn(side, g.pre, g.R))
        self.stage("descend", lambda: self.lin(side, g.grip, g.R, step=0.015, speed=0.10, contact=True))
        self.arm("grasp", side=side)
        self.stage("lift", lambda: self.lin(side, g.grip - 0.02 * g.back + np.array([0, 0, lift]), g.R, contact=True))
        st = self.arm("gripper_state", side=side)
        held = st.get("holding")
        if held is None:          # a gripper service without contact sensing: closed-on-something if the gap stayed open
            held = st.get("gap_mm", 0) > 9
        if not held:
            raise PlanError(f"grasp: nothing held after lifting (gap {st.get('gap_mm')} mm)")
        return st

    def pick_item(self, xyz, jaw_deg=0.0, height_m=0.02, prefer_side=None, thin=None, round_item=False, rim_radius_m=None):
        """Pick an item centred at xyz (pelvis frame). jaw_deg = direction the jaws close (across the item).
        rim_radius_m: pinch the wall of a ring (tape roll) at that radius instead of grabbing across it."""
        xyz = np.asarray(xyz, float)
        thin = height_m < 0.006 if thin is None else thin
        sides = [prefer_side] if prefer_side in ("left", "right") else (["left", "right"] if xyz[1] > 0 else ["right", "left"])
        if prefer_side in ("left", "right"):
            sides.append("right" if prefer_side == "left" else "left")
        attempts = []
        for side in sides:
            target, jaw, rnd = xyz, jaw_deg, round_item
            if rim_radius_m:
                # pinch the wall on the side facing this hand's shoulder; jaws close radially across the wall
                sh = np.array([0.0, 0.15 if side == "left" else -0.15])
                d = sh - xyz[:2]
                d /= np.linalg.norm(d)
                target = np.array([xyz[0] + d[0] * rim_radius_m, xyz[1] + d[1] * rim_radius_m, xyz[2]])
                jaw, rnd = math.degrees(math.atan2(d[1], d[0])), False
            cands = self.candidates(side, target, jaw, height_m, thin, rnd)
            if not cands:
                attempts.append(f"{side}: no reachable level-jaw grasp")
                continue
            for g in cands[:self.TRIES_PER_SIDE]:
                try:
                    st = self.execute(g)
                    return {"side": side, "jaw_deg": round(g.jaw, 1), "pitch_deg": g.pitch, "gripper": st,
                            "attempts": attempts + [f"{side} jaw {g.jaw:.0f} pitch {g.pitch}: ok"]}
                except PlanError as e:
                    attempts.append(f"{side} jaw {g.jaw:.0f} pitch {g.pitch}: {e}")
                    self.log(attempts[-1])
                    if not self.recover(side):
                        attempts.append(f"{side}: stuck after recovery")
                        break
        raise PlanError("; ".join(attempts[-4:]) or "no attempts")

    # ----- pressing things: switches, buttons, keys -----
    TIP = 0.030      # closed Dex1 fingertips sit ~3 cm beyond the jaw centre along the hand's x axis

    @staticmethod
    def push_R(push):
        """Hand frame for poking along `push` with the fingertips: hand x = push, jaws level (y horizontal)."""
        x = np.asarray(push, float) / np.linalg.norm(push)
        y = np.cross([0.0, 0.0, 1.0], x)
        y = np.array([0.0, 1.0, 0.0]) if np.linalg.norm(y) < 1e-6 else y / np.linalg.norm(y)
        return np.column_stack([x, y, np.cross(x, y)])

    def press(self, point, push, depth=0.006, side=None, hold_s=0.4, standoff=0.06):
        """Press a switch / button / key at `point` (pelvis frame) by pushing the closed fingertips along `push`
        `depth` metres past the surface, hold, and back off. Returns the side used."""
        point, push = np.asarray(point, float), np.asarray(push, float) / np.linalg.norm(push)
        sides = [side] if side else (["left", "right"] if point[1] > 0 else ["right", "left"])
        last = None
        for sd in sides:
            R = self.push_R(push)
            centre = point + push * depth - self.TIP * push
            pre = centre - push * (depth + standoff)
            try:
                self.arm("enable")
                self.arm("grasp", side=sd)                       # close the jaws: poke with the fingertips
                self.ensure_clear(sd)
                self.stage("to press point", lambda: self.move_free(sd, pre, R))
                self.stage("press", lambda: self.lin(sd, centre, R, step=0.01, speed=0.08, contact=True))
                self.arm("gripper_state", side=sd)
                if hold_s and "sim_wait" in getattr(self, "_sim_cmds", ("sim_wait",)):
                    try:
                        self.arm("sim_wait", seconds=hold_s)
                    except PlanError:
                        pass
                self.stage("release", lambda: self.lin(sd, pre, R, contact=True))
                return sd
            except PlanError as e:
                last = f"{sd}: {e}"
                self.recover(sd)
        raise PlanError(last or "press failed")

    def place(self, side, box_min, box_max, drop_dz=0.07):
        """Carry the held item over the container's centre and open. Keeps the grasp orientation when reachable."""
        cx, cy = (box_min[0] + box_max[0]) / 2, (box_min[1] + box_max[1]) / 2
        p = self.hand(side)
        target = np.array([cx - 0.03, cy, max(box_max[2] + drop_dz, p[2])])
        R = self.R[side]
        if R is not None and not all(self.ik_ok(side, pt, R) for pt in ([p[0], p[1], target[2]], target)):
            R = None
        if self.twin is not None:
            self.stage("carry", lambda: self.move_free(side, target, R))
        else:
            self.stage("raise to carry height", lambda: self.lin(side, [p[0], p[1], target[2]], R))
            self.stage("carry", lambda: self.lin(side, target, R))
        self.arm("open", side=side)

    def home(self, side):
        """Rest the arm low by the side (like a person's idle arm): out of the camera's view and the other hand's way."""
        self.ensure_clear(side)
        if self.twin is not None:
            from motion import REST, timed
            q_now = self.q14()
            goal = q_now.copy()
            goal[slice(0, 7) if side == "left" else slice(7, 14)] = REST[side]
            path = self.twin.rrt_connect(side, q_now, goal)
            if path:
                q, times = timed(path)
                self.stage("rest", lambda: self.arm("trajectory", q=q, times=times, side=side))
                self.R[side] = None
                return
        self.stage("to side", lambda: self.lin(side, self.lift_points(side)[1], None))

"""Pick-and-place planner over the arm_server protocol, so the same code drives waldo-sim and the real G1.

What it adds on top of single move_ee calls (each lesson came from the sim):
  * straight-line hand paths: moves are split into ~3 cm steps, each solved from the previous pose, so the hand
    follows the intended line and the arm doesn't swing through the desk or the torso;
  * a lift-clear route out of the hanging pose (hands start below the desk top, under its front edge);
  * the wrist turns above the target, at a clear height, never next to obstacles;
  * a tilted top-down approach (60 deg from horizontal reaches every jaw angle out to x = 0.40 m at desk height;
    straight down only reaches 0.32 m), trying the 180-degree-equivalent jaw angle and nearby pitches if IK fails;
  * a contact stage for the final descent: finger pads may touch the desk (needed for flat items).

    arm = Arm("tcp://127.0.0.1:15810")            # or the real arm_server through the tunnel
    p = Planner(arm, desk_top_z=-0.06)
    p.pick("left", xyz, jaw_deg)                    # returns gripper state; raises PlanError with the failing stage
    p.place("left", box_min, box_max)
    p.home("left")
"""
from __future__ import annotations

import json
import math

import numpy as np
import zmq


class PlanError(RuntimeError):
    pass


class Arm:
    """Minimal arm_server client: arm(cmd, **kw) -> reply dict, raising PlanError when ok is false."""

    def __init__(self, addr: str, timeout_s: float = 120):
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
    """Approach heading (deg, in the desk plane) for a jaw direction: perpendicular to it, so both fingers stay level,
    pointing away from the robot (within +-100 deg of +x), and, when both directions qualify, coming in from that
    hand's own side (the left hand moves toward -y, the right toward +y) so it never reaches across the body."""
    h = (jaw_deg - 90 + 180) % 360 - 180
    cands = [c for c in (h, (h + 360) % 360 - 180) if abs(c) <= 100] or [h + 180 if abs(h) > 90 else h]
    sgn = -1 if side == "left" else 1
    return max(cands, key=lambda c: (sgn * math.sin(math.radians(c)) >= -1e-9, -abs(c)))


def tilted_R(pitch_deg: float, jaw_deg: float, side: str = "left") -> np.ndarray:
    """Grasp frame: hand x approaches along heading_for(jaw, side) tilted pitch_deg below horizontal; jaws close along
    the horizontal direction jaw_deg (exactly level, both fingertips at the same height)."""
    p, h = math.radians(pitch_deg), math.radians(heading_for(jaw_deg, side))
    x = np.array([math.cos(p) * math.cos(h), math.cos(p) * math.sin(h), -math.sin(p)])
    y = np.array([math.cos(math.radians(jaw_deg)), math.sin(math.radians(jaw_deg)), 0.0])
    y = y - x * (y @ x)           # already perpendicular by construction; this just removes rounding
    y /= np.linalg.norm(y)
    return np.column_stack([x, y, np.cross(x, y)])


def _log(R):   # rotation matrix -> rotation vector
    c = max(-1.0, min(1.0, (np.trace(R) - 1) / 2))
    a = math.acos(c)
    if a < 1e-9:
        return np.zeros(3)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * math.sin(a))
    return a * w


def _exp(v):   # rotation vector -> rotation matrix (Rodrigues)
    a = np.linalg.norm(v)
    if a < 1e-9:
        return np.eye(3)
    k = v / a
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * K @ K


def slerp(R0, R1, t):
    return R0 @ _exp(t * _log(R0.T @ R1))


class Planner:
    STEP, SPEED = 0.03, 0.15          # m per straight-line segment, m/s along the path

    def __init__(self, arm, desk_top_z: float = -0.06):
        self.arm, self.desk = arm, desk_top_z
        self.R = {"left": None, "right": None}      # orientation last commanded per hand (None = level jaws)

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
        step, speed = step or self.STEP, speed or self.SPEED
        p0, to = self.hand(side), np.asarray(to, float)
        dist = float(np.linalg.norm(to - p0))
        n = max(1, math.ceil(dist / step))
        for k in range(1, n + 1):
            self._move(side, p0 + (to - p0) * k / n, R, max(0.2, dist / n / speed), contact)

    def lin_turn(self, side, to, R1, step=0.02):
        """Straight-line move that also rotates the hand from its current orientation to R1, a little per step,
        so the wrist never makes one big swing."""
        f = self.arm("fk", side=side)
        p0, R0, to = np.array(f["grasp_xyz"]), np.array(f["grasp_R"]), np.asarray(to, float)
        ang = np.linalg.norm(_log(R0.T @ R1))
        n = max(3, math.ceil(np.linalg.norm(to - p0) / step), math.ceil(ang / math.radians(10)))
        for k in range(1, n + 1):
            self._move(side, p0 + (to - p0) * k / n, slerp(R0, R1, k / n), 0.3)

    def stage(self, name, fn):
        try:
            return fn()
        except PlanError as e:
            raise PlanError(f"{name}: {e}") from None

    def lift_points(self, side):
        sy = 1 if side == "left" else -1
        return [np.array([0.10, 0.26 * sy, -0.10]), np.array([0.14, 0.26 * sy, self.desk + 0.14])]   # verified clear route

    def ensure_clear(self, side):
        """From wherever the hand is, get it up over the desk with level jaws at a safe height."""
        p = self.hand(side)
        safe_z = self.desk + 0.16
        if p[0] > 0.15 and p[2] < safe_z:                 # over the desk: straight up first, keeping orientation
            self.stage("retreat up", lambda: self.lin(side, [p[0], p[1], safe_z], self.R[side]))
        elif p[0] <= 0.15 and p[2] < self.desk + 0.05:      # hanging / below the desk edge: the lift route
            lp = self.lift_points(side)
            self.stage("lift out of hang", lambda: self._move(side, lp[0], None, 0.8))   # starts outside the workspace box
            self.stage("lift above desk", lambda: self.lin(side, lp[1], None))
        if self.R[side] is not None:
            self.stage("level wrist", lambda: self._move(side, self.hand(side), None, 0.6))

    def feasible_R(self, side, points, jaw_deg):
        """First grasp orientation whose IK reaches all `points`: pitch 60 first, then 50/70/45/80; the jaw angle and
        its 180-degree twin, then up to 15 deg off square (a parallel gripper still holds a pen 15 deg off)."""
        for pitch in (60, 50, 70, 45, 80):
            for jaw in (jaw_deg, jaw_deg + 15, jaw_deg - 15):   # jaw and jaw+180 are the same grasp now
                R = tilted_R(pitch, jaw, side)
                try:
                    for pt in points:
                        self.arm("ik", side=side, xyz=[float(v) for v in pt], R=R.tolist())
                    return R
                except PlanError:
                    continue
        raise PlanError(f"no reachable grasp orientation (jaw {jaw_deg:.0f} deg)")

    # ----- tasks -----
    def pick(self, side, xyz, jaw_deg, grip_dz=0.004, lift=0.13, thin=False):
        """thin=True (sheets, cards): aim the fingertip pads rather than the jaw centre at the item, since tilted
        fingertips reach ~3 cm below the jaw centre and would otherwise press the desk and close over the top."""
        x, y, z = (float(v) for v in xyz)
        grip = np.array([x, y, z + grip_dz])
        self.arm("enable")
        self.ensure_clear(side)
        def plan():
            for jaw in (jaw_deg, jaw_deg + 15, jaw_deg - 15):
                hd = math.radians(heading_for(jaw, side))
                back = np.array([math.cos(hd), math.sin(hd), 0.0])     # approach comes in along the heading
                try:
                    R = self.feasible_R(side, [grip + np.array([0, 0, 0.07]), grip], jaw)
                    g = grip - 0.022 * R[:, 0] if thin else grip       # fingertip pads at the item, jaw centre above
                    pre = g - 0.035 * back + np.array([0, 0, 0.07])
                    self.feasible_R(side, [pre, g], jaw)
                    return R, pre, back, g
                except PlanError:
                    continue
            raise PlanError(f"no reachable level-jaw grasp (jaw {jaw_deg:.0f} deg)")
        R, pre, back, grip = self.stage("plan grasp", plan)
        high = np.array([pre[0], pre[1], self.desk + 0.14])
        self.stage("over target", lambda: self.lin(side, high, None))
        self.stage("approach (turning)", lambda: self.lin_turn(side, pre, R))
        self.stage("descend", lambda: self.lin(side, grip, R, step=0.015, speed=0.06, contact=True))
        self.arm("grasp", side=side)
        self.stage("lift", lambda: self.lin(side, grip - 0.02 * back + np.array([0, 0, lift]), R, contact=True))
        g = self.arm("gripper_state", side=side)
        if not g.get("holding"):
            raise PlanError(f"grasp: nothing held after lifting (gap {g.get('gap_mm')} mm)")
        return g

    def place(self, side, box_min, box_max, drop_dz=0.08):
        cx, cy = (box_min[0] + box_max[0]) / 2, (box_min[1] + box_max[1]) / 2
        p = self.hand(side)
        target = np.array([cx - 0.03, cy, max(box_max[2] + drop_dz, p[2])])
        R = self.R[side]
        try:   # keep the grasp orientation while carrying if it reaches; otherwise carry with level jaws
            for pt in ([p[0], p[1], target[2]], target):
                self.arm("ik", side=side, xyz=[float(v) for v in pt], **({"R": R.tolist()} if R is not None else {}))
        except PlanError:
            R = None
        self.stage("raise to carry height", lambda: self.lin(side, [p[0], p[1], target[2]], R))
        self.stage("carry", lambda: self.lin(side, target, R))
        self.arm("open", side=side)

    def home(self, side):
        """Park above the desk's side, level jaws: out of the camera's way and of the other hand's (no need to hang)."""
        self.ensure_clear(side)
        self.stage("to side", lambda: self.lin(side, self.lift_points(side)[1], None))

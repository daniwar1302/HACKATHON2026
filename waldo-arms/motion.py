"""Collision-free arm motion planning on a MuJoCo digital twin (RRT-Connect + shortcutting), for the real G1 and the sim.

The twin is the same scene waldo-sim builds (Unitree's G1 + Dex1 URDF, desk, containers). The planner syncs it to the
robot's current joint angles, finds a goal configuration with IK, searches joint space for a collision-free path,
smooths it, and returns waypoints for arm_server's `trajectory` command (which re-checks every sample itself).

Collision rules match the sim: arm vs the robot's own body / other arm tolerates <5 mm (convex hulls are bulkier than
the links), arm vs desk / containers ~1 mm; finger pads may touch the desk in a contact stage; movable items ignored.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "waldo-sim"))
import scene  # noqa: E402

ARM = scene.ARM
SIDES = scene.SIDES

# Natural working posture (shoulder pitch, roll, yaw, elbow, wrist roll, pitch, yaw): elbow down and close to the body,
# forearm forward, wrist neutral. IK is pulled toward it in the null space so arms never "chicken-wing".
NATURAL = {"left": np.array([-0.45, 0.15, 0.0, 1.25, 0.0, 0.0, 0.0]),
           "right": np.array([-0.45, -0.15, 0.0, 1.25, 0.0, 0.0, 0.0])}
# Resting pose for an idle arm: by the hip, elbow bent, hand 9 cm clear of the desk's front edge (shoulder pitch 0.10
# parked it 7 mm under the edge: every path out of rest then grazed the desk).
REST = {"left": np.array([0.30, 0.20, 0.0, 0.90, 0.0, 0.0, 0.0]),
        "right": np.array([0.30, -0.20, 0.0, 0.90, 0.0, 0.0, 0.0])}


class Twin:
    def __init__(self, desk_top_z: float = -0.06, seed: int = 0, kind: str = "sort"):
        # kind: sort (desk + containers + items), devices / lab (sim stations), bare (desk only: a real cell whose
        # devices the twin doesn't know; the arm_server's checks and a slow, watched first press cover the rest)
        stations = kind in ("devices", "lab", "cell", "rack", "bare")
        self.m = (scene.build(scene.Desk(top_z=desk_top_z), items={}, containers={}, devices=False if kind == "bare" else kind,
                              twin=True)
                  if stations else scene.build(scene.Desk(top_z=desk_top_z)))
        self.d = mujoco.MjData(self.m)
        m = self.m
        self.qadr = {s: [m.joint(f"{s}_{j}_joint").qposadr[0] for j in ARM] for s in SIDES}
        self.dadr = {s: [m.joint(f"{s}_{j}_joint").dofadr[0] for j in ARM] for s in SIDES}
        self.lim = {s: np.array([m.joint(f"{s}_{j}_joint").range for j in ARM]) for s in SIDES}
        self.site = {s: m.site(f"{s}_grasp").id for s in SIDES}
        self.arm_bodies = {s: {b for b in range(m.nbody) if m.body(b).name.startswith(s) and any(
            k in m.body(b).name for k in ("shoulder", "elbow", "wrist", "dex1"))} for s in SIDES}
        self.fingers = {s: {m.body(f"{s}_dex1_finger_link_{k}").id for k in (1, 2)} for s in SIDES}
        self.containers = set() if stations else {m.body(f"{n}_model").id for n in scene.CONTAINERS}
        # devices are obstacles to plan around (they count like containers), items are ignored
        self.touchable = set()
        if stations:
            names = {"lab": scene.LAB, "devices": scene.DEVICES, "cell": scene.CELL, "rack": scene.RACK, "bare": {}}[kind]
            self.containers = {b for b in range(m.nbody) if any(m.body(b).name.startswith(n) for n in names)}
            self.touchable = set(self.containers)        # in a contact stage the gripper may touch devices (it presses them)
        self.items = set() if stations else {scene.item_body(m, n) for n in scene.ITEMS}
        for n in ([] if stations else scene.ITEMS):   # items: no collisions, parked apart
            b = scene.item_body(m, n)
            qa = m.jnt_qposadr[m.body_jntadr[b]]
            self.d.qpos[qa:qa + 3] = [-2.0, 0.3 * list(scene.ITEMS).index(n), -2.0]
            for g in range(m.ngeom):            # (all parked items overlapping overflowed MuJoCo's contact arena,
                if m.geom_bodyid[g] == b:       #  which can silently drop real arm contacts)
                    m.geom_contype[g] = m.geom_conaffinity[g] = 0
        # The real/sim grippers sit open (24.5 mm out per finger); the twin's fingers must match or its "clear"
        # paths put the fingertips into the desk. Open is also the conservative (widest) shape.
        self.finger_q = [m.joint(f"{s}_dex1_finger_joint_{k}").qposadr[0] for s in SIDES for k in (1, 2)]
        self.d.qpos[self.finger_q] = scene.FINGER[1]
        self.PAD = 0.006
        for g in range(m.ngeom):                 # static obstacles report contacts from PAD away
            b = m.geom_bodyid[g]
            if b == 0 or b in self.containers:
                m.geom_margin[g] = self.PAD
        self.rng = np.random.default_rng(seed)
        # keep-out boxes move at runtime; the world body's compile-time BVH wouldn't see them (0.03 ms per check)
        m.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_MIDPHASE
        mujoco.mj_forward(m, self.d)

    def set_keepouts(self, boxes):
        """Registered keep-out zones [(lo xyz, hi xyz)] become solid obstacles (the gripper may still touch them in a
        contact stage, like the desk: picking from a fixture, lowering into a bin). Replaces the previous set."""
        m = self.m
        if len(boxes) > scene.KEEPOUTS:
            raise ValueError(f"at most {scene.KEEPOUTS} keep-out zones")
        for i in range(scene.KEEPOUTS):
            g = m.geom(f"keepout{i}").id
            if i < len(boxes):
                lo, hi = np.asarray(boxes[i][0], float), np.asarray(boxes[i][1], float)
                size = np.maximum((hi - lo) / 2, 1e-3)
                m.geom_pos[g], m.geom_size[g] = (lo + hi) / 2, size
                m.geom_contype[g] = m.geom_conaffinity[g] = 1
                m.geom_margin[g] = self.PAD
            else:
                m.geom_pos[g], m.geom_size[g] = [-5, -5, -5 - i], [1e-3] * 3
                m.geom_contype[g] = m.geom_conaffinity[g] = 0
            m.geom_rbound[g] = float(np.linalg.norm(m.geom_size[g]))
            m.geom_aabb[g] = np.concatenate([[0, 0, 0], m.geom_size[g]])
        mujoco.mj_forward(m, self.d)

    # ----- state -----
    def set_q14(self, q14):
        self.d.qpos[self.qadr["left"]], self.d.qpos[self.qadr["right"]] = q14[:7], q14[7:]

    def q14(self):
        return np.concatenate([self.d.qpos[self.qadr["left"]], self.d.qpos[self.qadr["right"]]])

    # ----- collision -----
    def collides(self, q14, contact_ok=()):
        self.set_q14(q14)
        mujoco.mj_forward(self.m, self.d)
        m, d = self.m, self.d
        for c in d.contact[:d.ncon]:
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            for s in SIDES:
                arm = self.arm_bodies[s]
                if b1 in arm and b2 not in arm:
                    mine, hit = b1, b2
                elif b2 in arm and b1 not in arm:
                    mine, hit = b2, b1
                else:
                    continue
                if hit in self.items:
                    continue
                if s in contact_ok and "dex1" in m.body(mine).name and hit == 0:
                    continue
                if s in contact_ok and "dex1" in m.body(mine).name and hit in self.touchable:
                    continue
                own = hit not in self.containers and hit != 0
                # plan with padding, execute without: the desk/containers/devices count as hit within PAD of the arm
                # (contacts are detected that early because those geoms carry a margin), so tracking error has room
                if (-c.dist >= 0.005) if own else (c.dist <= self.PAD):
                    return True
        return False

    def edge_free(self, qa, qb, contact_ok=(), step=math.radians(2.0)):
        n = max(1, int(np.abs(qb - qa).max() / step))
        return all(not self.collides(qa + (qb - qa) * k / n, contact_ok) for k in range(1, n + 1))

    # ----- kinematics -----
    def fk(self, side, q14):
        self.set_q14(q14)
        mujoco.mj_kinematics(self.m, self.d)
        return self.d.site_xpos[self.site[side]].copy(), self.d.site_xmat[self.site[side]].reshape(3, 3).copy()

    def ik(self, side, q14_seed, target, R=None, iters=300):
        """Damped least squares on the jaw centre (level jaws if R is None). Returns (q14, pos_err, rot_err)."""
        m, d = self.m, self.d
        q14 = np.array(q14_seed, float)
        sl = slice(0, 7) if side == "left" else slice(7, 14)
        q = q14[sl].copy()
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        tgt = np.asarray(target, float)
        for _ in range(iters):
            q14[sl] = q
            self.set_q14(q14)
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            p, Rc = d.site_xpos[self.site[side]], d.site_xmat[self.site[side]].reshape(3, 3)
            mujoco.mj_jacSite(m, d, jp, jr, self.site[side])
            Jp, Jr = jp[:, self.dadr[side]], jr[:, self.dadr[side]]
            ep = tgt - p
            if R is not None:
                er = 0.5 * sum(np.cross(Rc[:, i], R[:, i]) for i in range(3))
                J, e = np.vstack([Jp, 0.3 * Jr]), np.concatenate([ep, 0.3 * er])
            else:
                y = Rc[:, 1]
                J, e = np.vstack([Jp, 0.3 * (np.cross(y, [0, 0, 1.0]) @ Jr)[None, :]]), np.concatenate([ep, [-0.3 * y[2]]])
            if np.linalg.norm(ep) < 3e-4 and np.linalg.norm(e[3:]) < 2e-3:
                break
            Jinv = J.T @ np.linalg.inv(J @ J.T + 1e-4 * np.eye(len(e)))
            dq = Jinv @ e + (np.eye(7) - Jinv @ J) @ (0.08 * (NATURAL[side] - q))   # natural posture, null space
            q = np.clip(q + np.clip(dq, -0.15, 0.15), self.lim[side][:, 0] + 0.01, self.lim[side][:, 1] - 0.01)
        q14[sl] = q
        p, Rc = self.fk(side, q14)
        oe = float(np.linalg.norm(0.5 * sum(np.cross(Rc[:, i], R[:, i]) for i in range(3)))) if R is not None else abs(float(Rc[2, 1]))
        return q14, float(np.linalg.norm(tgt - p)), oe

    def ik_goals(self, side, q14_now, target, R=None, n=4, contact_ok=()):
        """Collision-free IK solutions, most natural first: seeded from the current pose, the natural posture and a
        few elbow-down variants (random seeds only if none of those work). Ranked by distance from the natural
        posture plus distance from where the arm is, so moves stay short and human-like."""
        sl = slice(0, 7) if side == "left" else slice(7, 14)
        sgn = 1 if side == "left" else -1
        templates = [q14_now[sl], NATURAL[side], NATURAL[side] + [0.3, 0.1 * sgn, 0, -0.3, 0, 0, 0],
                     NATURAL[side] + [-0.3, 0.1 * sgn, 0, 0.2, 0, -0.4, 0], NATURAL[side] + [0, 0, 0.4 * sgn, 0, 0, 0, 0]]
        def solve(seeds):
            out = []
            for t in seeds:
                s0 = q14_now.copy()
                s0[sl] = t
                q, pe, oe = self.ik(side, s0, target, R)
                if pe < 4e-3 and oe < 0.08 and not self.collides(q, contact_ok):
                    if all(np.abs(q[sl] - g[sl]).max() > 0.15 for g in out):
                        out.append(q)
            return out
        goals = solve(templates)
        if not goals:
            goals = solve([self.rng.uniform(self.lim[side][:, 0], self.lim[side][:, 1]) for _ in range(12)])
        cost = lambda g: np.linalg.norm(g[sl] - NATURAL[side]) + 0.5 * np.linalg.norm(g[sl] - q14_now[sl])
        return sorted(goals, key=cost)[:n]

    # ----- planning -----
    def rrt_connect(self, side, q_start, q_goal, contact_ok=(), step=0.15, iters=4000):
        """Joint-space RRT-Connect for one arm (the other arm fixed). Returns a list of q14 waypoints or None."""
        sl = slice(0, 7) if side == "left" else slice(7, 14)
        lo, hi = self.lim[side][:, 0] + 0.01, self.lim[side][:, 1] - 0.01
        if self.edge_free(q_start, q_goal, contact_ok):
            return [q_start, q_goal]
        A, B = [q_start.copy()], [q_goal.copy()]
        pa, pb = [-1], [-1]

        def extend(T, P, q_target):
            i = int(np.argmin([np.linalg.norm(t[sl] - q_target[sl]) for t in T]))
            d = q_target[sl] - T[i][sl]
            dist = np.linalg.norm(d)
            q_new = T[i].copy()
            q_new[sl] = T[i][sl] + (d if dist <= step else d / dist * step)
            if self.edge_free(T[i], q_new, contact_ok):
                T.append(q_new)
                P.append(i)
                return len(T) - 1, dist <= step
            return None, False

        for it in range(iters):
            q_rand = q_start.copy()
            q_rand[sl] = self.rng.uniform(lo, hi) if self.rng.random() > 0.1 else B[0][sl]
            ia, _ = extend(A, pa, q_rand)
            if ia is None:
                A, B, pa, pb = B, A, pb, pa
                continue
            while True:                       # connect B toward the new node
                ib, reached = extend(B, pb, A[ia])
                if ib is None or reached:
                    break
            if ib is not None and reached:
                path_a, k = [], ia
                while k != -1:
                    path_a.append(A[k]); k = pa[k]
                path_b, k = [], ib
                while k != -1:
                    path_b.append(B[k]); k = pb[k]
                path = path_a[::-1] + path_b
                if np.linalg.norm(path[0][sl] - q_start[sl]) > 1e-9:
                    path = path[::-1]
                return self.shortcut(path, contact_ok)
            A, B, pa, pb = B, A, pb, pa
        return None

    def shortcut(self, path, contact_ok=(), rounds=60):
        path = list(path)
        for _ in range(rounds):
            if len(path) <= 2:
                break
            i, j = sorted(self.rng.choice(len(path), 2, replace=False))
            if j - i > 1 and self.edge_free(path[i], path[j], contact_ok):
                path = path[:i + 1] + path[j:]
        return path

    def plan(self, side, q14_now, target, R=None, contact_ok=()):
        """Collision-free joint path from q14_now to the hand at target (orientation R or level jaws)."""
        if self.collides(q14_now, contact_ok):
            q14_now = q14_now.copy()       # starting in (tolerated) contact: plan anyway from where we are
        for goal in self.ik_goals(side, q14_now, target, R, contact_ok=contact_ok):
            path = self.rrt_connect(side, q14_now, goal, contact_ok)
            if path:
                return path
        return None


def timed(path, max_joint_speed=1.2, dt=0.02, min_T=0.5):
    """Waypoints -> one smooth trajectory: dense samples along the path with a single min-jerk speed profile over its
    whole length (accelerate once, flow through every waypoint without stopping, decelerate once).
    Returns (q list, times) for arm_server's `trajectory` command."""
    P = np.asarray(path, float)
    seg = np.abs(np.diff(P, axis=0)).max(axis=1)              # joint-space length per segment (max-norm)
    L = float(seg.sum())
    if L < 1e-9:
        return [list(map(float, P[0])), list(map(float, P[-1]))], [0.0, min_T]
    T = max(min_T, 1.875 * L / max_joint_speed)               # min-jerk peak speed = 1.875 x average
    cum = np.concatenate([[0.0], np.cumsum(seg)]) / L
    n = max(2, int(math.ceil(T / dt)) + 1)
    qs, ts = [], []
    for k in range(n):
        tau = k / (n - 1)
        u = 10 * tau ** 3 - 15 * tau ** 4 + 6 * tau ** 5          # min-jerk progress along the path
        i = min(int(np.searchsorted(cum, u, side="right")) - 1, len(P) - 2)
        f = 0.0 if cum[i + 1] == cum[i] else (u - cum[i]) / (cum[i + 1] - cum[i])
        qs.append(list(map(float, P[i] + (P[i + 1] - P[i]) * f)))
        ts.append(tau * T)
    return qs, ts


def cartesian_path(twin, side, q14_now, points, R=None):
    """Joint waypoints that carry the hand through Cartesian `points` (each solved from the previous): a straight-line
    hand move as one trajectory. Returns None if a point can't be reached accurately."""
    q, out = np.array(q14_now, float), [np.array(q14_now, float)]
    for p in points:
        q, pe, oe = twin.ik(side, q, p, R)
        if pe > 4e-3 or oe > 0.08:
            return None
        out.append(q.copy())
    return out

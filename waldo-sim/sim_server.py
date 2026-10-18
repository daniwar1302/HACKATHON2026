#!/usr/bin/env python3
"""waldo-sim: the desk-sorting G1 in MuJoCo, speaking the real robot's protocols so waldo-arms, agents, VLMs and
routines run against it unchanged.

Serves, on its own ports (no clash with tunnels to the real robot):
  arm_server protocol   ZMQ REP  :15810   state enable release freeze ik check move_ee park set_table open grasp
                                          gripper ... + sim_reset {seed, n_items}, sim_objects, sim_point {item, noise_px}
  teleimager config     ZMQ REP  :16000   head camera intrinsics + depth scale
  head depth            ZMQ PUB  :15556   raw z16 (mm), aligned to color
  camera viewer         HTTP     :18766   /frame/head_camera.jpg  /frame/waldo_webcam.jpg  /api/cameras

Physics, not shortcuts: Unitree's G1+Dex1 URDF; arms track commands through stiff position actuators with gravity
compensation; each Dex1 finger squeezes with at most 20 N, so a grasp holds or slips by contact and friction.
IK targets the jaw centre (15.2 cm ahead of the wrist) with either level jaws (the real arm_server's default) or a
top-down approach at a chosen jaw angle (`approach: "down", jaw_deg`), which pens and markers need.

  MUJOCO_GL=glfw uv run --with mujoco --with pyzmq --with pillow --with numpy waldo-sim/sim_server.py [--speed 1]
"""
from __future__ import annotations

import argparse
import io
import json
import math
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import mujoco
import numpy as np
import zmq
from PIL import Image

import scene
from scene import ARM, FINGER, SIDES

ARM_PORT, CFG_PORT, DEPTH_PORT, HTTP_PORT = 15810, 16000, 15556, 18766
W, H = scene.W, scene.H
WS = {"x": (0.10, 0.60), "y": (-0.55, 0.55), "z": (-0.35, 0.45)}
READY = {"left": [0.28, 0.20, 0.05], "right": [0.28, -0.20, 0.05]}
HANG = {"shoulder_pitch": 0.28, "shoulder_roll": 0.22, "elbow": 0.98}
SPAWN = {"x": (0.26, 0.38), "y": (-0.20, 0.20)}     # visible from the head camera AND reachable at desk height
RADIUS = {"marker_black": 0.064, "marker_blue": 0.064, "pen": 0.07, "pencil": 0.088, "eraser": 0.028,
          "sticky_notes": 0.054, "sticker_sheet": 0.057, "tape": 0.056, "scissors": 0.108, "crayons": 0.07}


def motor_q(finger):   # finger slide (m) -> Dex1 motor angle reported by the real service (0 closed .. 5.4 open)
    return round((finger - FINGER[0]) / (FINGER[1] - FINGER[0]) * 5.4, 3)


def down_R(jaw_deg: float) -> np.ndarray:
    """Grasp frame for a top-down approach: hand x points down, jaws close along (cos a, sin a, 0)."""
    a = math.radians(jaw_deg)
    x, y = np.array([0.0, 0.0, -1.0]), np.array([math.cos(a), math.sin(a), 0.0])
    return np.column_stack([x, y, np.cross(x, y)])


class Sim:
    def __init__(self, speed: float, desk: scene.Desk):
        self.desk = desk
        self.m = scene.build(desk)
        self.d = mujoco.MjData(self.m)
        self.speed = speed
        m = self.m
        self.qadr = {s: [m.joint(f"{s}_{j}_joint").qposadr[0] for j in ARM] for s in SIDES}
        self.dadr = {s: [m.joint(f"{s}_{j}_joint").dofadr[0] for j in ARM] for s in SIDES}
        self.act = {s: [m.actuator(f"{s}_{j}_joint").id for j in ARM] for s in SIDES}
        self.lim = {s: np.array([m.joint(f"{s}_{j}_joint").range for j in ARM]) for s in SIDES}
        self.fing_act = {s: [m.actuator(f"{s}_dex1_finger_joint_{k}").id for k in (1, 2)] for s in SIDES}
        self.fing_q = {s: [m.joint(f"{s}_dex1_finger_joint_{k}").qposadr[0] for k in (1, 2)] for s in SIDES}
        self.fing_body = {s: {m.body(f"{s}_dex1_finger_link_{k}").id for k in (1, 2)} for s in SIDES}
        self.site = {s: m.site(f"{s}_grasp").id for s in SIDES}
        self.arm_bodies = {s: {b for b in range(m.nbody) if m.body(b).name.startswith(s) and any(
            k in m.body(b).name for k in ("shoulder", "elbow", "wrist", "dex1"))} for s in SIDES}
        self.items = {n: scene.item_body(m, n) for n in scene.ITEMS}
        self.item_qadr = {n: m.jnt_qposadr[m.body_jntadr[b]] for n, b in self.items.items()}
        self.item_dadr = {n: m.jnt_dofadr[m.body_jntadr[b]] for n, b in self.items.items()}
        self.containers = {n: m.body(f"{n}_model").id for n in scene.CONTAINERS}
        self.static_geoms = {g for g in range(m.ngeom) if m.geom_bodyid[g] == 0 or m.geom_bodyid[g] in self.containers.values()}
        mujoco.mj_forward(m, self.d)
        self.box = {n: self._aabb(b) for n, b in self.containers.items()}
        self.extent = {n: self._body_extent(b) for n, b in self.items.items()}
        self.rest_z = {}
        self.enabled = self.frozen = self.moving = False
        self.freeze_reason, self.table = "", None
        self.reset(0)

    def _aabb(self, body):
        pts = []
        for g in range(self.m.ngeom):
            if self.m.geom_bodyid[g] == body and self.m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and self.m.geom_group[g] == 3:
                mid = self.m.geom_dataid[g]
                v = self.m.mesh_vert[self.m.mesh_vertadr[mid]:self.m.mesh_vertadr[mid] + self.m.mesh_vertnum[mid]]
                pts.append((self.d.geom_xmat[g].reshape(3, 3) @ v.T).T + self.d.geom_xpos[g])
        p = np.vstack(pts)
        return p.min(0), p.max(0)

    def _body_extent(self, body):
        """Item size along its own x, y, z axes (m), from its collision shapes."""
        m, d = self.m, self.d
        Rb, pb, pts = d.xmat[body].reshape(3, 3), d.xpos[body], []
        for g in range(m.ngeom):
            if m.geom_bodyid[g] != body or not (m.geom_contype[g] or m.geom_conaffinity[g]):
                continue
            Rg, pg, sz, t = d.geom_xmat[g].reshape(3, 3), d.geom_xpos[g], m.geom_size[g], m.geom_type[g]
            if t == mujoco.mjtGeom.mjGEOM_MESH:
                mid = m.geom_dataid[g]
                local = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
            elif t == mujoco.mjtGeom.mjGEOM_BOX:
                local = np.array([[a, b, c] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)]) * sz
            else:   # capsule / cylinder: axis z, radius sz[0], half-length sz[1]
                r, h = sz[0], sz[1] + (sz[0] if t == mujoco.mjtGeom.mjGEOM_CAPSULE else 0)
                local = np.array([[a * r, b * r, c * h] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)])
            pts.append(((Rg @ local.T).T + pg - pb) @ Rb)   # into the body frame
        p = np.vstack(pts)
        return p.max(0) - p.min(0)

    # ----- world -----
    def reset(self, seed: int, n_items: int | None = None):
        m, d = self.m, self.d
        mujoco.mj_resetData(m, d)
        rng = np.random.default_rng(seed)
        for s in SIDES:
            for j, a in zip(ARM, self.qadr[s]):
                d.qpos[a] = HANG.get(j, 0.0) * (-1 if (s == "right" and j == "shoulder_roll") else 1)
            for a in self.fing_q[s]:
                d.qpos[a] = FINGER[1]
        names = list(scene.ITEMS)
        chosen = list(rng.choice(names, size=min(n_items or len(names), len(names)), replace=False))
        placed = []
        for n in names:
            qa = self.item_qadr[n]
            if n not in chosen:                    # parked out of the way, behind the robot
                d.qpos[qa:qa + 3] = [-0.8, 0.15 * names.index(n) - 0.7, scene.FLOOR_Z + 0.05]
                continue
            for _ in range(500):
                xy = np.array([rng.uniform(*SPAWN["x"]), rng.uniform(*SPAWN["y"])])
                clear_of_containers = all(not (lo[0] - 0.07 - RADIUS[n] < xy[0] < hi[0] + 0.07 + RADIUS[n] and
                                               lo[1] - 0.07 - RADIUS[n] < xy[1] < hi[1] + 0.07 + RADIUS[n])
                                          for lo, hi in self.box.values())
                if clear_of_containers and all(np.linalg.norm(xy - p) > RADIUS[n] + RADIUS[o] + 0.01 for o, p in placed):
                    break
            placed.append((n, xy))
            yaw = rng.uniform(-math.pi, math.pi)
            d.qpos[qa:qa + 2] = xy
            q0, qz, out = d.qpos[qa + 3:qa + 7].copy(), np.array([math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]), np.zeros(4)
            mujoco.mju_mulQuat(out, qz, q0)
            d.qpos[qa + 3:qa + 7] = out
            d.qpos[qa + 2] = self.desk.top_z + 0.03   # dropped from 3 cm, settles below
        self.chosen = chosen
        mujoco.mj_forward(m, d)
        self._hold_ctrl()
        for s in SIDES:
            for a in self.fing_act[s]:
                d.ctrl[a] = FINGER[1]
        self.enabled = self.frozen = self.moving = False
        self.table = None
        self.step(600)                             # 1.2 s to settle

    def _hold_ctrl(self):
        for i in range(self.m.nu):
            self.d.ctrl[i] = self.d.qpos[self.m.jnt_qposadr[self.m.actuator_trnid[i, 0]]]

    def step(self, n=1):
        for _ in range(n):
            mujoco.mj_step(self.m, self.d)

    # ----- arms -----
    def arm_q(self, s):
        return self.d.qpos[self.qadr[s]].copy()

    def q14(self):
        return np.concatenate([self.arm_q("left"), self.arm_q("right")])

    def grasp_pose(self, s, q7=None):
        save = None
        if q7 is not None:
            save = self.d.qpos[self.qadr[s]].copy()
            self.d.qpos[self.qadr[s]] = q7
        mujoco.mj_kinematics(self.m, self.d)
        p, R = self.d.site_xpos[self.site[s]].copy(), self.d.site_xmat[self.site[s]].reshape(3, 3).copy()
        if save is not None:
            self.d.qpos[self.qadr[s]] = save
            mujoco.mj_kinematics(self.m, self.d)
        return p, R

    def ik(self, s, target, R_target=None, iters=300):
        """Damped least squares on the jaw centre. Orientation: R_target if given, else level jaws (hand y horizontal,
        like the real arm_server's jaws_horizontal default). Tries the current pose and two natural seeds."""
        m, d = self.m, self.d
        save = d.qpos.copy()
        tgt = np.asarray(target, float)
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        seeds = [self.arm_q(s), np.array([-0.3, 0.3, 0, 1.2, 0, 0, 0]), np.array([-0.8, 0.2, 0, 1.4, 0, -0.6, 0])]
        if s == "right":
            seeds[1][1] *= -1
            seeds[2][1] *= -1
        best = None
        for q in seeds:
            q = np.clip(q.copy(), self.lim[s][:, 0] + 0.01, self.lim[s][:, 1] - 0.01)
            for _ in range(iters):
                d.qpos[self.qadr[s]] = q
                mujoco.mj_kinematics(m, d)
                mujoco.mj_comPos(m, d)
                p, R = d.site_xpos[self.site[s]], d.site_xmat[self.site[s]].reshape(3, 3)
                mujoco.mj_jacSite(m, d, jp, jr, self.site[s])
                Jp, Jr = jp[:, self.dadr[s]], jr[:, self.dadr[s]]
                ep = tgt - p
                if R_target is not None:
                    er = 0.5 * sum(np.cross(R[:, i], R_target[:, i]) for i in range(3))
                    J, e = np.vstack([Jp, 0.3 * Jr]), np.concatenate([ep, 0.3 * er])
                else:
                    y = R[:, 1]
                    J = np.vstack([Jp, 0.3 * (np.cross(y, [0, 0, 1.0]) @ Jr)[None, :]])
                    e = np.concatenate([ep, [-0.3 * y[2]]])
                if np.linalg.norm(ep) < 3e-4 and np.linalg.norm(e[3:]) < 2e-3:
                    break
                dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(len(e)), e)
                q = np.clip(q + np.clip(dq, -0.15, 0.15), self.lim[s][:, 0] + 0.01, self.lim[s][:, 1] - 0.01)
            d.qpos[self.qadr[s]] = q
            mujoco.mj_kinematics(m, d)
            pe = float(np.linalg.norm(tgt - d.site_xpos[self.site[s]]))
            R = d.site_xmat[self.site[s]].reshape(3, 3)
            oe = float(np.linalg.norm(0.5 * sum(np.cross(R[:, i], R_target[:, i]) for i in range(3)))) if R_target is not None else abs(float(R[2, 1]))
            score = pe + 0.05 * oe
            if best is None or score < best[3]:
                best = (q.copy(), pe, oe, score)
            if pe < 2e-3 and oe < 0.05:      # continuity first (like the real arm_server): keep the first good answer
                break
        d.qpos[:] = save
        mujoco.mj_forward(m, d)
        return best[0], best[1], best[2]

    def collisions(self, q14, contact_ok=()):
        """Arm contacts with the body, the other arm, the desk or a container at joint angles q14 (items excluded).
        contact_ok: sides whose finger pads may touch the desk (a grasp's final descent onto a flat item)."""
        m, d = self.m, self.d
        save = d.qpos.copy()
        d.qpos[self.qadr["left"]], d.qpos[self.qadr["right"]] = q14[:7], q14[7:]
        mujoco.mj_forward(m, d)
        bad = []
        item_bodies = set(self.items.values())
        for c in d.contact[:d.ncon]:
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            for s in SIDES:
                arm, other = self.arm_bodies[s], self.arm_bodies["right" if s == "left" else "left"]
                hit, hg = (b2, c.geom2) if b1 in arm else (b1, c.geom1) if b2 in arm else (None, None)
                if hit is None or hit in arm or hit in item_bodies:
                    continue
                mine = b1 if b1 in arm else b2
                if s in contact_ok and mine in self.fing_body[s] and hit == 0:
                    continue
                what = ("the desk" if hg in self.static_geoms and hit == 0 else
                        next((n for n, b in self.containers.items() if b == hit), None) or
                        ("the other arm" if hit in other else m.body(hit).name))
                bad.append(f"{m.body(mine).name} / {what}")
        d.qpos[:] = save
        mujoco.mj_forward(m, d)
        return bad

    def sweep(self, q_from, q_to, step_deg=3.0, contact_ok=()):
        start = set(self.collisions(q_from, contact_ok))   # contacts that already exist don't block moving away
        n = max(2, int(np.abs(np.asarray(q_to) - q_from).max() / math.radians(step_deg)) + 1)
        for k in range(1, n + 1):
            new = [b for b in self.collisions(q_from + (np.asarray(q_to) - q_from) * k / n, contact_ok) if b not in start]
            if new:
                return {"ok": False, "reason": f"collision {new[0]} at sample {k}/{n}", "samples": k}
        return {"ok": True, "reason": "", "samples": n}

    def move(self, q14_goal, duration):
        q0, q1 = self.q14(), np.asarray(q14_goal, float)
        steps = max(1, int(duration / self.m.opt.timestep))
        self.moving, t_wall = True, time.monotonic()
        for k in range(1, steps + 1):
            a = k / steps
            a = a * a * (3 - 2 * a)
            q = q0 + (q1 - q0) * a
            for s, sl in (("left", slice(0, 7)), ("right", slice(7, 14))):
                self.d.ctrl[self.act[s]] = q[sl]
            self.step()
            if self.speed > 0:
                lag = k * self.m.opt.timestep / self.speed - (time.monotonic() - t_wall)
                if lag > 0:
                    time.sleep(lag)
        self.step(100)                              # 0.2 s settle
        self.moving = False

    # ----- grippers -----
    def holding(self, s):
        """The item both finger pads are touching, if any."""
        touch = {1: set(), 2: set()}
        fl = sorted(self.fing_body[s])
        for c in self.d.contact[:self.d.ncon]:
            b1, b2 = self.m.geom_bodyid[c.geom1], self.m.geom_bodyid[c.geom2]
            for k, fb in zip((1, 2), fl):
                if fb in (b1, b2):
                    touch[k].add(b2 if b1 == fb else b1)
        both = touch[1] & touch[2] & set(self.items.values())
        return next((n for n, b in self.items.items() if b in both), None)

    def set_fingers(self, s, q, seconds):
        for a in self.fing_act[s]:
            self.d.ctrl[a] = q
        for _ in range(int(seconds / self.m.opt.timestep)):
            self.step()

    def gstat(self, s):
        q = float(np.mean(self.d.qpos[self.fing_q[s]]))
        h = self.holding(s)
        return {"q": motor_q(q), "gap_mm": round(2 * (0.023 + q) * 1000, 1), "phase": "holding" if h else "idle",
                "holding": bool(h), "object": h}

    # ----- scoring -----
    def objects(self):
        out = {}
        for n in self.chosen:
            b = self.items[n]
            p, R = self.d.xpos[b], self.d.xmat[b].reshape(3, 3)
            inside = next((c for c, (lo, hi) in self.box.items()
                           if lo[0] <= p[0] <= hi[0] and lo[1] <= p[1] <= hi[1] and p[2] <= hi[2] + 0.01), None)
            ex = self.extent[n]
            yaw = math.degrees(math.atan2(R[1, 0], R[0, 0]))
            jaw = yaw + (90 if ex[0] > ex[1] else 0)            # jaws close across the narrow horizontal axis
            out[n] = {"xyz": [round(float(v), 4) for v in p], "yaw_deg": round(yaw, 1),
                      "grip_jaw_deg": round((jaw + 90) % 180 - 90, 1), "width_mm": round(float(min(ex[0], ex[1])) * 1000, 1),
                      "height_mm": round(float(ex[2]) * 1000, 1),
                      "category": scene.ITEMS[n][4], "in_container": inside,
                      "held_by": next((s for s in SIDES if self.holding(s) == n), None),
                      "lifted_m": round(float(p[2] - self.desk.top_z), 4)}
        return out

    def T_pelvis_torso(self):
        mujoco.mj_kinematics(self.m, self.d)
        t = self.m.body("torso_link").id
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = self.d.xmat[t].reshape(3, 3), self.d.xpos[t]   # world frame == pelvis frame
        return T


def handle(sim: Sim, r: dict, cam_project) -> dict:
    cmd, side = r.get("cmd"), r.get("side", "left")
    if cmd == "state":
        return {"arm": {"fsm": [500, 0], "enabled": sim.enabled, "frozen": sim.frozen, "freeze_reason": sim.freeze_reason,
                        "moving": sim.moving, "weight": 1.0 if sim.enabled else 0.0, "q": sim.q14().round(4).tolist()},
                "fk": {"grasp_xyz": sim.grasp_pose("left")[0].round(4).tolist()},
                "fk_right": {"grasp_xyz": sim.grasp_pose("right")[0].round(4).tolist()},
                "gripper": sim.gstat("left"), "gripper_right": sim.gstat("right"), "table": sim.table,
                "T_pelvis_torso": sim.T_pelvis_torso().round(6).tolist()}
    if cmd == "fk":                                # same reply shape as the real arm_server
        q7 = np.asarray(r["q7"], float) if "q7" in r else None
        pz, Rz = sim.grasp_pose(side, q7)
        return {"grasp_xyz": pz.round(4).tolist(), "grasp_R": Rz.round(4).tolist()}
    if cmd == "prepare_enable":
        return {"token": "sim", "expires_s": 15.0}
    if cmd == "enable":
        sim.enabled, sim.frozen = True, False
        return {"status": {"enabled": True}}
    if cmd == "release":
        sim.enabled = False
        return {}
    if cmd == "freeze":
        sim.frozen, sim.freeze_reason = True, r.get("reason", "client")
        return {}
    if cmd == "unfreeze":
        sim.frozen = False
        return {}
    if cmd == "set_table":
        sim.table = None if r.get("z") is None else {"z": float(r["z"]), "x": r.get("x"), "y": r.get("y")}
        return {"table": sim.table or {}}
    if cmd in ("ik", "move_ee"):
        xyz = np.asarray(r["xyz"], float)
        for k, v in zip("xyz", xyz):
            if not WS[k][0] <= v <= WS[k][1]:
                raise ValueError(f"target {k}={v:.3f} outside workspace {WS[k]}")
        R = np.asarray(r["R"], float) if "R" in r else (down_R(float(r.get("jaw_deg", 0))) if r.get("approach") == "down" else None)
        q7, pe, oe = sim.ik(side, xyz, R)
        out = {"q7": q7.round(5).tolist(), "q7_deg": np.degrees(q7).round(1).tolist(), "pos_err": round(pe, 4),
               "jaw_tilt": round(oe, 3)}
        if pe > r.get("max_pos_err", 0.01) or oe > r.get("max_jaw_tilt", 0.15):
            return {**out, "ok": False, "error": "IK did not converge within tolerance"}
        if cmd == "ik" or r.get("dry"):
            return out
        if not sim.enabled:
            raise RuntimeError("not enabled")
        if sim.frozen:
            raise RuntimeError(f"frozen: {sim.freeze_reason}")
        goal = sim.q14()
        goal[(slice(0, 7) if side == "left" else slice(7, 14))] = q7
        chk = sim.sweep(sim.q14(), goal, contact_ok=(side,) if r.get("contact") else ())
        if not chk["ok"] and not r.get("force"):
            raise ValueError(f"path check: {chk['reason']} (send force=true to override)")
        sim.move(goal, float(r.get("duration", 3.0)))
        p, _ = sim.grasp_pose(side)
        return {**out, "duration": r.get("duration", 3.0), "check": chk, "reached_err": round(float(np.linalg.norm(p - xyz)), 4)}
    if cmd == "check":
        q = np.asarray(r["q"], float) if "q" in r else None
        if q is None:
            q = sim.q14()
            q[(slice(0, 7) if side == "left" else slice(7, 14))] = r["q7"]
        return {"check": sim.sweep(sim.q14(), q), "q_start": sim.q14().round(5).tolist()}
    if cmd == "park":
        for s in SIDES:
            if r.get("side", "both") in ("both", s):
                p, _ = sim.grasp_pose(s)
                for xyz in ([max(p[0], 0.13), p[1], p[2] + 0.15], READY[s]):
                    handle(sim, {"cmd": "move_ee", "side": s, "xyz": xyz, "duration": 2.5, "max_pos_err": 0.03}, cam_project)
        return {}
    if cmd == "grasp":
        sim.set_fingers(side, FINGER[0], 0.6)
        return sim.gstat(side)
    if cmd == "open":
        sim.set_fingers(side, FINGER[1], 0.5)
        return sim.gstat(side)
    if cmd == "gripper":                            # motor angle 0..5.4 like the real service
        sim.set_fingers(side, FINGER[0] + float(r["q"]) / 5.4 * (FINGER[1] - FINGER[0]), 0.5)
        return sim.gstat(side)
    if cmd in ("gripper_stop", "gripper_state"):
        return sim.gstat(side)
    if cmd == "sim_reset":
        sim.reset(int(r.get("seed", 0)), r.get("n_items"))
        return {"objects": sim.objects()}
    if cmd == "sim_objects":
        return {"objects": sim.objects(), "desk_top_z": sim.desk.top_z,
                "containers": {n: {"category": c[3], "box_min": sim.box[n][0].round(3).tolist(), "box_max": sim.box[n][1].round(3).tolist()}
                               for n, c in scene.CONTAINERS.items()}}
    if cmd == "sim_point":
        # Oracle pointer: the item's top-centre projected into the head image, with VLM-like pixel noise.
        n = r["item"]
        b = sim.items[n]
        top = sim.d.xpos[b].copy()
        top[2] += 0.004
        u, v = cam_project(top)
        rng = np.random.default_rng(r.get("seed"))
        nz = float(r.get("noise_px", 0))
        return {"item": n, "u": round(u + rng.normal(0, nz), 1), "v": round(v + rng.normal(0, nz), 1), "noise_px": nz}
    raise ValueError(f"unknown cmd {cmd}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--speed", type=float, default=1.0, help="sim-time / wall-time while moving; 0 = as fast as possible")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--desk-top", type=float, default=scene.Desk.top_z, help="desk top z in the pelvis frame (m)")
    a = ap.parse_args()
    sim = Sim(a.speed, scene.Desk(top_z=a.desk_top))
    sim.reset(a.seed)
    m = sim.m
    head, fy = m.cam("head_camera").id, scene.FY
    K = {"fx": fy, "fy": fy, "ppx": W / 2, "ppy": H / 2}
    cfg = {"head_camera": {"image_shape": [H, W], "depth_scale": 0.001, "enable_depth": True, "type": "sim", "color_intrinsics": K}}

    def cam_project(p_world):
        mujoco.mj_kinematics(m, sim.d)
        mujoco.mj_camlight(m, sim.d)
        pos, R = sim.d.cam_xpos[head], sim.d.cam_xmat[head].reshape(3, 3)
        pc = R.T @ (np.asarray(p_world) - pos)          # MuJoCo camera: x right, y up, looking along -z
        return K["ppx"] + fy * pc[0] / -pc[2], K["ppy"] - fy * pc[1] / -pc[2]

    rgb_r, dep_r = mujoco.Renderer(m, H, W), mujoco.Renderer(m, H, W)
    dep_r.enable_depth_rendering()
    room = mujoco.MjvCamera()
    room.lookat[:] = [0.3, 0, -0.1]
    room.distance, room.azimuth, room.elevation = 1.4, 210, -28

    def render(which, depth=False):
        r = dep_r if depth else rgb_r
        r.update_scene(sim.d, camera=head if which == "head" else room)
        return r.render()

    ctx = zmq.Context.instance()
    arm_sock, cfg_sock, pub = ctx.socket(zmq.REP), ctx.socket(zmq.REP), ctx.socket(zmq.PUB)
    arm_sock.bind(f"tcp://127.0.0.1:{ARM_PORT}")
    cfg_sock.bind(f"tcp://127.0.0.1:{CFG_PORT}")
    pub.bind(f"tcp://127.0.0.1:{DEPTH_PORT}")
    jobs: queue.Queue = queue.Queue()

    class Http(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/api/cameras":
                body, ctype = json.dumps({"connected": True, "cameras": [{"id": "head_camera", "state": "live"}, {"id": "waldo_webcam", "state": "live"}]}).encode(), "application/json"
            elif self.path in ("/frame/head_camera.jpg", "/frame/waldo_webcam.jpg"):
                done, box = threading.Event(), {}
                jobs.put(("head" if "head" in self.path else "room", box, done))
                done.wait(5)
                body, ctype = box.get("jpg", b""), "image/jpeg"
            else:
                self.send_response(404)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("content-type", ctype)
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    threading.Thread(target=ThreadingHTTPServer(("127.0.0.1", HTTP_PORT), Http).serve_forever, daemon=True).start()
    print(f"waldo-sim (desk) up: arm :{ARM_PORT}  config :{CFG_PORT}  depth :{DEPTH_PORT}  http :{HTTP_PORT}  "
          f"(speed {a.speed}x, desk top z {sim.desk.top_z})", flush=True)
    poller = zmq.Poller()
    poller.register(arm_sock, zmq.POLLIN)
    poller.register(cfg_sock, zmq.POLLIN)
    last_depth = 0.0
    while True:
        socks = dict(poller.poll(5))
        if cfg_sock in socks:
            cfg_sock.recv()
            cfg_sock.send_json(cfg)
        if arm_sock in socks:
            try:
                out = handle(sim, json.loads(arm_sock.recv()), cam_project)
                out.setdefault("ok", True)
            except Exception as e:
                out = {"ok": False, "error": str(e)}
            arm_sock.send(json.dumps(out).encode())
        while not jobs.empty():
            which, box, done = jobs.get()
            buf = io.BytesIO()
            Image.fromarray(render(which)).save(buf, "JPEG", quality=88)
            box["jpg"] = buf.getvalue()
            done.set()
        if time.monotonic() - last_depth > 1 / 15:
            last_depth = time.monotonic()
            pub.send((np.clip(render("head", depth=True), 0, 65.0) * 1000).astype(np.uint16).tobytes())
        sim.step()


if __name__ == "__main__":
    main()

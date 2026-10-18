#!/usr/bin/env python3
"""waldo-sim: a MuJoCo G1 that speaks the real robot's protocols, so waldo-arms (and agents, VLMs, routines) run
against it unchanged. For rehearsing tasks and optimizing them before touching the robot.

Serves, on its own ports (no clash with tunnels to the real robot):
  arm_server protocol   ZMQ REP  :15810   state enable release freeze ik check move_ee park set_table open grasp ...
                                          + sim_reset {seed}  sim_objects  (ground truth for scoring)
  teleimager config     ZMQ REP  :16000   head camera intrinsics + depth scale
  head depth            ZMQ PUB  :15556   raw z16 (mm), aligned to color
  camera viewer         HTTP     :18766   /frame/head_camera.jpg  /frame/waldo_webcam.jpg  /api/cameras

Model: MuJoCo Menagerie unitree_g1 (via robot_descriptions), pelvis fixed in the air like the gantry, arms kinematic
(commanded joint angles are applied exactly; objects have physics). Head camera placed at the URDF d435 mount so
point_to_xyz's math matches. Grasping is simplified: 'grasp' attaches the nearest object within GRASP_RADIUS of the
jaw centre and carries it until 'open'. Pose accuracy, IK limits and path collisions are real; grip physics is not.

  MUJOCO_GL=glfw uv run --with mujoco --with robot_descriptions --with pyzmq --with pillow --with numpy sim_server.py
  then point waldo-arms at it:  G1_ARM_ADDR=tcp://localhost:15810 G1_CAM_BASE=http://localhost:18766 \\
                                G1_CAM_HOST=localhost G1_DEPTH_PORT=15556 G1_CAMCFG_PORT=16000
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
import queue
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import mujoco
import numpy as np
import zmq
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "waldo-arms"))
import vision  # noqa: E402  (same URDF camera mount the real pipeline uses)

ARM_PORT, CFG_PORT, DEPTH_PORT, HTTP_PORT = 15810, 16000, 15556, 18766
W, H = 640, 480
PELVIS_HEIGHT = 0.80                  # world z of the pelvis (gantry height; legs hang)
GRASP_X = 0.13                        # jaw centre ahead of the wrist_yaw link (Dex1 approx.)
GRASP_RADIUS = 0.05
TABLE = {"x": (0.18, 0.50), "y": (-0.35, 0.35), "z_top": -0.12}   # pelvis frame. Hands only reach z ~ -0.1 out to x ~ 0.35
WS = {"x": (0.10, 0.60), "y": (-0.55, 0.55), "z": (-0.35, 0.45)}
SIDES = ("left", "right")
ARM = ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw"]
READY = {"left": [0.28, 0.20, 0.05], "right": [0.28, -0.20, 0.05]}
CUBES = {"red_cube": (0.9, 0.15, 0.15), "green_cube": (0.15, 0.75, 0.2), "blue_cube": (0.2, 0.3, 0.9)}
CUBE_HALF = 0.02


def build(menagerie_xml: str) -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(menagerie_xml)
    pelvis = spec.body("pelvis")
    for j in list(pelvis.joints):            # no floating base: the gantry holds the pelvis
        if j.type == mujoco.mjtJoint.mjJNT_FREE:
            spec.delete(j)
    pelvis.pos = [0, 0, PELVIS_HEIGHT]
    torso = spec.body("torso_link")
    T = vision.T_TORSO_CAM                   # optical frame in torso coords (x right, y down, z forward)
    xo, yo = T[:3, 0], T[:3, 1]              # MuJoCo cameras look along -z with y up: x_c = x_o, y_c = -y_o
    fovy = math.degrees(2 * math.atan(H / 2 / 604.4344))
    torso.add_camera(name="head_camera", pos=T[:3, 3].tolist(), xyaxes=[*xo.tolist(), *(-yo).tolist()], fovy=fovy)
    for s in SIDES:
        spec.body(f"{s}_wrist_yaw_link").add_site(name=f"{s}_grasp", pos=[GRASP_X, 0, 0], size=[0.012, 0, 0],
                                                  rgba=[1, 1, 0, 0.6])
    wb = spec.worldbody
    wb.add_light(pos=[0.4, 0, 2.5], dir=[0, 0, -1], diffuse=[0.8, 0.8, 0.8])
    wb.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[3, 3, 0.1], rgba=[0.35, 0.3, 0.25, 1])
    cx, cy = sum(TABLE["x"]) / 2, sum(TABLE["y"]) / 2
    hx, hy = (TABLE["x"][1] - TABLE["x"][0]) / 2, (TABLE["y"][1] - TABLE["y"][0]) / 2
    top = PELVIS_HEIGHT + TABLE["z_top"]
    wb.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[cx, cy, top / 2], size=[hx, hy, top / 2],
                rgba=[0.85, 0.85, 0.82, 1])
    for name, rgb in CUBES.items():
        b = wb.add_body(name=name, pos=[cx, cy, top + CUBE_HALF])
        b.add_freejoint()
        b.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[CUBE_HALF] * 3, rgba=[*rgb, 1], mass=0.05, friction=[1.0, 0.01, 0.001])
    wb.add_camera(name="waldo_webcam", pos=[0.9, -1.4, 1.2], xyaxes=[0.85, 0.53, 0, -0.2, 0.32, 0.93], fovy=55)
    return spec.compile()


class Sim:
    def __init__(self, model: mujoco.MjModel, speed: float):
        self.m, self.d, self.speed = model, mujoco.MjData(model), speed
        self.qadr = {s: [self.m.joint(f"{s}_{j}_joint").qposadr[0] for j in ARM] for s in SIDES}
        self.dadr = {s: [self.m.joint(f"{s}_{j}_joint").dofadr[0] for j in ARM] for s in SIDES}
        self.lim = {s: np.array([self.m.joint(f"{s}_{j}_joint").range for j in ARM]) for s in SIDES}
        self.site = {s: self.m.site(f"{s}_grasp").id for s in SIDES}
        self.pelvis, self.torso = self.m.body("pelvis").id, self.m.body("torso_link").id
        self.cube_q = {n: self.m.joint(self.m.body(n).jntadr[0]).qposadr[0] for n in CUBES}
        self.cube_v = {n: self.m.joint(self.m.body(n).jntadr[0]).dofadr[0] for n in CUBES}
        self.arm_bodies = {s: {self.m.body(f"{s}_{b}_link").id for b in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")} for s in SIDES}
        self.table_geom = self.m.geom("table").id
        self.robot_qadr = [self.m.joint(i).qposadr[0] for i in range(self.m.njnt) if self.m.joint(i).type != mujoco.mjtJoint.mjJNT_FREE]
        self.cmd = None
        self.enabled = self.frozen = self.moving = False
        self.freeze_reason, self.table = "", None
        self.held: dict[str, tuple[str, np.ndarray] | None] = {s: None for s in SIDES}
        self.grip_q = {s: 5.0 for s in SIDES}
        self.reset(0)

    # ----- world -----
    def reset(self, seed: int):
        mujoco.mj_resetData(self.m, self.d)
        rng = np.random.default_rng(seed)
        q = self.d.qpos
        hang = {"shoulder_pitch": 0.28, "shoulder_roll": 0.22, "elbow": 0.98}   # measured hanging pose
        for s in SIDES:
            for j, a in zip(ARM, self.qadr[s]):
                q[a] = hang.get(j, 0.0) * (-1 if (s == "right" and j == "shoulder_roll") else 1)
        self.cmd = q[self.robot_qadr].copy()
        taken = []
        for n in CUBES:
            for _ in range(100):
                xy = np.array([rng.uniform(0.24, 0.34), rng.uniform(-0.22, 0.22)])   # within reach at table height
                if all(np.linalg.norm(xy - t) > 0.08 for t in taken):
                    break
            taken.append(xy)
            p = self.world(np.array([xy[0], xy[1], TABLE["z_top"] + CUBE_HALF]))
            q[self.cube_q[n]:self.cube_q[n] + 3] = p
            q[self.cube_q[n] + 3:self.cube_q[n] + 7] = [1, 0, 0, 0]
        self.held = {s: None for s in SIDES}
        self.grip_q = {s: 5.0 for s in SIDES}
        self.enabled = self.frozen = self.moving = False
        mujoco.mj_forward(self.m, self.d)
        for _ in range(200):                  # let the cubes settle on the table
            self.step()

    def world(self, p_pelvis):
        mujoco.mj_kinematics(self.m, self.d)
        return self.d.xpos[self.pelvis] + self.d.xmat[self.pelvis].reshape(3, 3) @ p_pelvis

    def to_pelvis(self, p_world):
        return self.d.xmat[self.pelvis].reshape(3, 3).T @ (np.asarray(p_world) - self.d.xpos[self.pelvis])

    def step(self):
        self.d.qpos[self.robot_qadr] = self.cmd          # arms are kinematic: exactly the commanded angles
        self.d.qvel[:] = np.where(np.isin(np.arange(self.m.nv), self._robot_dofs()), 0.0, self.d.qvel)
        for s in SIDES:                                  # a held cube rides with the jaws
            if self.held[s]:
                name, off = self.held[s]
                mujoco.mj_kinematics(self.m, self.d)
                R = self.d.site_xmat[self.site[s]].reshape(3, 3)
                self.d.qpos[self.cube_q[name]:self.cube_q[name] + 3] = self.d.site_xpos[self.site[s]] + R @ off
                self.d.qvel[self.cube_v[name]:self.cube_v[name] + 6] = 0
        mujoco.mj_step(self.m, self.d)

    def _robot_dofs(self):
        if not hasattr(self, "_rd"):
            self._rd = [self.m.joint(i).dofadr[0] for i in range(self.m.njnt) if self.m.joint(i).type != mujoco.mjtJoint.mjJNT_FREE]
        return self._rd

    # ----- arms -----
    def arm_q(self, s):
        return self.d.qpos[self.qadr[s]].copy()

    def grasp_xyz(self, s, q7=None):
        if q7 is not None:
            save = self.d.qpos[self.qadr[s]].copy()
            self.d.qpos[self.qadr[s]] = q7
        mujoco.mj_kinematics(self.m, self.d)
        p = self.to_pelvis(self.d.site_xpos[self.site[s]])
        if q7 is not None:
            self.d.qpos[self.qadr[s]] = save
            mujoco.mj_kinematics(self.m, self.d)
        return p

    def ik(self, s, target, q_init=None, iters=200):
        """Damped least squares on the jaw-centre position, joint limits enforced, mild pull to the start pose."""
        save = self.d.qpos.copy()
        q = (self.arm_q(s) if q_init is None else np.array(q_init, float)).copy()
        q0 = q.copy()
        tw = self.world(np.asarray(target, float))
        jac = np.zeros((3, self.m.nv))
        for _ in range(iters):
            self.d.qpos[self.qadr[s]] = q
            mujoco.mj_kinematics(self.m, self.d)
            mujoco.mj_comPos(self.m, self.d)
            err = tw - self.d.site_xpos[self.site[s]]
            if np.linalg.norm(err) < 5e-4:
                break
            mujoco.mj_jacSite(self.m, self.d, jac, None, self.site[s])
            J = jac[:, self.dadr[s]]
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-3 * np.eye(3), err) + 0.01 * (q0 - q)
            q = np.clip(q + np.clip(dq, -0.2, 0.2), self.lim[s][:, 0] + 0.02, self.lim[s][:, 1] - 0.02)
        self.d.qpos[self.qadr[s]] = q
        mujoco.mj_kinematics(self.m, self.d)
        pe = float(np.linalg.norm(tw - self.d.site_xpos[self.site[s]]))
        self.d.qpos[:] = save
        mujoco.mj_forward(self.m, self.d)
        return q, pe

    def collisions(self, q14):
        """Contacts between an arm and the body, the other arm or the table at joint angles q14."""
        save = self.d.qpos.copy()
        self.d.qpos[self.qadr["left"]], self.d.qpos[self.qadr["right"]] = q14[:7], q14[7:]
        mujoco.mj_forward(self.m, self.d)
        bad = []
        for c in self.d.contact[:self.d.ncon]:
            b1, b2 = self.m.geom_bodyid[c.geom1], self.m.geom_bodyid[c.geom2]
            for s in SIDES:
                arm = self.arm_bodies[s]
                hit = b2 if b1 in arm else b1 if b2 in arm else None
                if hit is None or hit in arm or self.m.body(hit).name in CUBES:
                    continue
                if c.geom1 == self.table_geom or c.geom2 == self.table_geom:
                    bad.append(f"{s} arm / table")
                elif hit in self.arm_bodies["right" if s == "left" else "left"]:
                    bad.append("left arm / right arm")
                elif self.m.body(hit).name in ("torso_link", "pelvis", "waist_yaw_link", "waist_roll_link", "head_link") or "hip" in self.m.body(hit).name:
                    bad.append(f"{s} arm / {self.m.body(hit).name}")
        self.d.qpos[:] = save
        mujoco.mj_forward(self.m, self.d)
        return bad

    def sweep(self, q_from, q_to, step_deg=3.0):
        n = max(2, int(np.abs(np.asarray(q_to) - q_from).max() / math.radians(step_deg)) + 1)
        for k in range(n + 1):
            bad = self.collisions(q_from + (np.asarray(q_to) - q_from) * k / n)
            if bad:
                return {"ok": False, "reason": f"self-collision {bad[0]} at sample {k}/{n}", "samples": k}
        return {"ok": True, "reason": "", "samples": n}

    def q14(self):
        return np.concatenate([self.arm_q("left"), self.arm_q("right")])

    def move(self, q14_goal, duration):
        """Smoothstep from the current pose to q14_goal over `duration` seconds of sim time."""
        q0, q1 = self.q14(), np.asarray(q14_goal, float)
        steps = max(1, int(duration / self.m.opt.timestep))
        self.moving = True
        t_wall = time.monotonic()
        for k in range(1, steps + 1):
            a = k / steps
            a = a * a * (3 - 2 * a)
            q = q0 + (q1 - q0) * a
            self.cmd[[self.robot_qadr.index(i) for i in self.qadr["left"]]] = q[:7]
            self.cmd[[self.robot_qadr.index(i) for i in self.qadr["right"]]] = q[7:]
            self.step()
            if self.speed > 0:   # pace to real time / speed so a viewer looks right
                lag = k * self.m.opt.timestep / self.speed - (time.monotonic() - t_wall)
                if lag > 0:
                    time.sleep(lag)
        self.moving = False

    # ----- grippers -----
    def grasp(self, s):
        self.grip_q[s] = 0.0
        mujoco.mj_kinematics(self.m, self.d)
        tip = self.d.site_xpos[self.site[s]]
        best = min(CUBES, key=lambda n: np.linalg.norm(self.d.qpos[self.cube_q[n]:self.cube_q[n] + 3] - tip))
        dist = float(np.linalg.norm(self.d.qpos[self.cube_q[best]:self.cube_q[best] + 3] - tip))
        if dist <= GRASP_RADIUS and not any(h and h[0] == best for h in self.held.values()):
            R = self.d.site_xmat[self.site[s]].reshape(3, 3)
            self.held[s] = (best, R.T @ (self.d.qpos[self.cube_q[best]:self.cube_q[best] + 3] - tip))
            return {"q": 0.6, "phase": "holding", "holding": True, "object": best}
        return {"q": 0.0, "phase": "closed", "holding": False, "nearest": best, "nearest_m": round(dist, 3)}

    def open(self, s):
        self.grip_q[s], self.held[s] = 5.0, None
        return {"q": 5.0, "phase": "moving", "holding": False}

    def objects(self):
        out = {}
        for n in CUBES:
            p = self.to_pelvis(self.d.qpos[self.cube_q[n]:self.cube_q[n] + 3])
            out[n] = {"xyz": [round(float(v), 4) for v in p], "held_by": next((s for s in SIDES if self.held[s] and self.held[s][0] == n), None),
                      "lifted_m": round(float(p[2] - (TABLE["z_top"] + CUBE_HALF)), 4)}
        return out

    def T_pelvis_torso(self):
        mujoco.mj_kinematics(self.m, self.d)
        Rp, Rt = self.d.xmat[self.pelvis].reshape(3, 3), self.d.xmat[self.torso].reshape(3, 3)
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = Rp.T @ Rt, Rp.T @ (self.d.xpos[self.torso] - self.d.xpos[self.pelvis])
        return T


def handle(sim: Sim, r: dict) -> dict:
    cmd, side = r.get("cmd"), r.get("side", "left")
    gstat = lambda s: {"q": sim.grip_q[s], "phase": "holding" if sim.held[s] else "idle", "holding": bool(sim.held[s])}
    if cmd == "state":
        return {"arm": {"fsm": [500, 0], "enabled": sim.enabled, "frozen": sim.frozen, "freeze_reason": sim.freeze_reason,
                        "moving": sim.moving, "weight": 1.0 if sim.enabled else 0.0, "q": sim.q14().round(4).tolist()},
                "fk": {"grasp_xyz": sim.grasp_xyz("left").round(4).tolist()}, "fk_right": {"grasp_xyz": sim.grasp_xyz("right").round(4).tolist()},
                "gripper": gstat("left"), "gripper_right": gstat("right"), "table": sim.table,
                "T_pelvis_torso": sim.T_pelvis_torso().round(6).tolist()}
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
        q7, pe = sim.ik(side, xyz)
        out = {"q7": q7.round(5).tolist(), "q7_deg": np.degrees(q7).round(1).tolist(), "pos_err": round(pe, 4), "jaw_tilt": 0.0}
        if pe > r.get("max_pos_err", 0.01):
            return {**out, "ok": False, "error": "IK did not converge within tolerance"}
        if cmd == "ik" or r.get("dry"):
            return out
        if not sim.enabled:
            raise RuntimeError("not enabled")
        if sim.frozen:
            raise RuntimeError(f"frozen: {sim.freeze_reason}")
        goal = sim.q14()
        goal[:7 if side == "left" else 14][-7:] = q7
        chk = sim.sweep(sim.q14(), goal)
        if not chk["ok"] and not r.get("force"):
            raise ValueError(f"path check: {chk['reason']} (send force=true to override)")
        sim.move(goal, float(r.get("duration", 3.0)))
        return {**out, "duration": r.get("duration", 3.0), "check": chk}
    if cmd == "check":
        q = np.asarray(r["q"], float) if "q" in r else None
        if q is None:
            q = sim.q14()
            q[:7 if side == "left" else 14][-7:] = r["q7"]
        return {"check": sim.sweep(sim.q14(), q), "q_start": sim.q14().round(5).tolist()}
    if cmd == "park":
        for s in SIDES:
            if r.get("side", "both") in ("both", s):
                for xyz in ([max(sim.grasp_xyz(s)[0], 0.13), sim.grasp_xyz(s)[1], sim.grasp_xyz(s)[2] + 0.15], READY[s]):
                    handle(sim, {"cmd": "move_ee", "side": s, "xyz": xyz, "duration": 2.5, "max_pos_err": 0.03})
        return {}
    if cmd == "grasp":
        return sim.grasp(side)
    if cmd == "open":
        return sim.open(side)
    if cmd in ("gripper_stop", "gripper_state"):
        return gstat(side)
    if cmd == "sim_reset":
        sim.reset(int(r.get("seed", 0)))
        return {"objects": sim.objects()}
    if cmd == "sim_objects":
        return {"objects": sim.objects(), "table_z_top": TABLE["z_top"]}
    raise ValueError(f"unknown cmd {cmd}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--speed", type=float, default=1.0, help="sim-time / wall-time while moving; 0 = as fast as possible")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    from robot_descriptions import g1_mj_description
    model = build(g1_mj_description.MJCF_PATH)
    sim = Sim(model, a.speed)
    sim.reset(a.seed)
    head_cam, room_cam = model.cam("head_camera").id, model.cam("waldo_webcam").id
    fy = (H / 2) / math.tan(math.radians(model.cam_fovy[head_cam]) / 2)
    cfg = {"head_camera": {"image_shape": [H, W], "depth_scale": 0.001, "enable_depth": True, "type": "sim",
                           "color_intrinsics": {"fx": fy, "fy": fy, "ppx": W / 2, "ppy": H / 2}}}
    rgb_r, dep_r = mujoco.Renderer(model, H, W), mujoco.Renderer(model, H, W)
    dep_r.enable_depth_rendering()

    def render(cam, depth=False):
        r = dep_r if depth else rgb_r
        r.update_scene(sim.d, camera=cam)
        return r.render()

    ctx = zmq.Context.instance()
    arm_sock, cfg_sock, pub = ctx.socket(zmq.REP), ctx.socket(zmq.REP), ctx.socket(zmq.PUB)
    arm_sock.bind(f"tcp://127.0.0.1:{ARM_PORT}")
    cfg_sock.bind(f"tcp://127.0.0.1:{CFG_PORT}")
    pub.bind(f"tcp://127.0.0.1:{DEPTH_PORT}")
    jobs: queue.Queue = queue.Queue()   # HTTP threads ask the main thread to render (GL context lives here)

    class Http(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/api/cameras":
                body, ctype = json.dumps({"connected": True, "cameras": [{"id": "head_camera", "state": "live"}, {"id": "waldo_webcam", "state": "live"}]}).encode(), "application/json"
            elif self.path in ("/frame/head_camera.jpg", "/frame/waldo_webcam.jpg"):
                done, box = threading.Event(), {}
                jobs.put((head_cam if "head" in self.path else room_cam, box, done))
                done.wait(5)
                body, ctype = box.get("jpg", b""), "image/jpeg"
            else:
                self.send_response(404); self.end_headers(); return
            self.send_response(200); self.send_header("content-type", ctype); self.send_header("content-length", str(len(body))); self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    threading.Thread(target=ThreadingHTTPServer(("127.0.0.1", HTTP_PORT), Http).serve_forever, daemon=True).start()
    print(f"waldo-sim up: arm :{ARM_PORT}  config :{CFG_PORT}  depth :{DEPTH_PORT}  http :{HTTP_PORT}  (speed {a.speed}x)", flush=True)
    poller = zmq.Poller()
    poller.register(arm_sock, zmq.POLLIN)
    poller.register(cfg_sock, zmq.POLLIN)
    last_depth = 0.0
    while True:
        socks = dict(poller.poll(5))
        if cfg_sock in socks:
            cfg_sock.recv(); cfg_sock.send_json(cfg)
        if arm_sock in socks:
            try:
                req = json.loads(arm_sock.recv())
                out = handle(sim, req)
                out.setdefault("ok", True)
            except Exception as e:
                out = {"ok": False, "error": str(e)}
            arm_sock.send(json.dumps(out).encode())
        while not jobs.empty():
            cam, box, done = jobs.get()
            buf = io.BytesIO()
            Image.fromarray(render(cam)).save(buf, "JPEG", quality=88)
            box["jpg"] = buf.getvalue(); done.set()
        if time.monotonic() - last_depth > 1 / 15:     # 15 Hz depth, newest-frame-only subscribers
            last_depth = time.monotonic()
            z = render(head_cam, depth=True)
            pub.send((np.clip(z, 0, 65.0) * 1000).astype(np.uint16).tobytes())
        sim.step()


if __name__ == "__main__":
    main()

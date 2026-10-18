#!/usr/bin/env python3
"""waldo-arms: MCP server that lets an agent move the G1's hands and Dex1 grippers through Michael's arm_server.

arm_server (Jetson, ZMQ :5810) does the IK, self-collision and table path checks, workspace guard and watchdogs.
This server only exposes its commands as agent tools, adds a table floor, and serves camera frames.

Runs on Waldo (talks to 192.168.123.164:5810 directly) or on a laptop with tunnels:
  ssh -N -L 5810:192.168.123.164:5810 -L 8766:10.2.1.180:8766 waldo@100.120.182.66
  G1_ARM_ADDR=tcp://localhost:5810 G1_CAM_BASE=http://localhost:8766 G1_CAM_HOST=localhost \
    uv run --with "mcp>=2.2" --with pyzmq --with numpy --with pillow server.py   (also tunnel 55556 + 60000)
"""
import io
import json
import os
import re
import threading
import time
import urllib.request
from pathlib import Path

import numpy as np

import zmq
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError  # message reaches the agent; plain exceptions don't
from mcp.server.mcpserver.utilities.types import Image

import vision

ADDR = os.environ.get("G1_ARM_ADDR", "tcp://192.168.123.164:5810")
CAM = os.environ.get("G1_CAM_BASE", "http://10.2.1.180:8766")
WS = {"x": (0.10, 0.60), "y": (-0.55, 0.55), "z": (-0.35, 0.45)}  # arm_server's pelvis-frame workspace (m)
CAM_HOST = os.environ.get("G1_CAM_HOST", "192.168.123.164")   # the Jetson's teleimager (depth + camera config)
DEPTH_PORT, CAMCFG_PORT = int(os.environ.get("G1_DEPTH_PORT", "55556")), int(os.environ.get("G1_CAMCFG_PORT", "60000"))
# Head camera config as teleimager reported it on 2026-10-17 (used only if the live request fails).
HEAD_FALLBACK = {"image_shape": [480, 640], "depth_scale": 0.001,
                 "color_intrinsics": {"fx": 604.9346, "fy": 604.4344, "ppx": 326.5778, "ppy": 247.7904}}
TABLE_CLEARANCE = 0.03  # hand targets stay this far above a declared table top
table_z = None
ROUTINES = Path(os.environ.get("G1_ROUTINES", Path(__file__).resolve().parent / "routines"))
recording: list[dict] = []   # every successful action this session, as routine steps
_replaying = False           # don't re-record steps while a routine runs

mcp = MCPServer("waldo-arms", instructions=(
    "Moves the Unitree G1's hands and Dex1 grippers (robot hangs in a gantry; no walking). "
    "Frame: pelvis, metres, x forward, y left, z up. Hands hang near [0.04, +-0.25, -0.21]; ready pose "
    "[0.28, +-0.20, 0.05]; reachable box x 0.10-0.60, y +-0.55, z -0.35-0.45. "
    "Workflow: status -> enable -> look / check_reach -> move_hand (small steps) -> gripper -> release when done. "
    "Seeing -> acting: look (annotated head image, 640x480) -> pick the target pixel (or have a VLM point) -> "
    "point_to_xyz([[u, v]]) -> move_hand to that xyz (approach from above: same x/y, z + 0.08 first). "
    "Use look after moves to verify. freeze holds in place; release hands the arms back to Unitree's controller. "
    "The robot must be in FSM 500/501/801 (status shows it); you cannot change that here. "
    "CODIFYING: every successful move_hand / gripper / set_table / park is recorded as a JSON step. When a "
    "sequence worked, call save_routine(name, description) to keep it (or pass your own steps); later "
    "run_routine(name, dry_run=true) checks every step without moving, then run_routine(name, dry_run=false) "
    "replays it. Step format: {op: move_hand, side, xyz:[x,y,z], seconds} | {op: gripper, side, action} | "
    "{op: wait, seconds} | {op: park} | {op: set_table, z}. "
    "FAST PATH (skills, one call each, planned + collision-checked + verified): pick(u, v) the object under a point "
    "-> place(side, container=name) (list_containers; register_container once per station) or place(side, u, v) "
    "under a point -> home(side). Fixed devices: list_controls, "
    "run_task(name) (e.g. power_cycle_router, enter_bootloader, start_cycle, arm_stop, stop_machine). "
    "SENSE, ~50 ms, no VLM: read_light(points), read_display(box) for 7-segment codes, read_gauge(name) for dials "
    "(register_gauge once). Poll them to verify every action. For limits that need a fast reaction (overpressure -> "
    "STOP), arm a guard(gauge, above, then_task) BEFORE starting (it runs in the background), then guard_status. Every physical action "
    "leaves a receipt (receipts, show_receipt: before/after frames)."))


class _Arm:
    def __init__(self):
        self.ctx = zmq.Context.instance()
        self.sock = None
        self.lock = threading.RLock()        # guards fire from a background thread; one REQ socket, one caller at a time

    def __call__(self, cmd, timeout_s: float = 30, **kw):
        with self.lock:
            return self._call(cmd, timeout_s, **kw)

    def _call(self, cmd, timeout_s, **kw):
        if self.sock is None:
            self.sock = self.ctx.socket(zmq.REQ)
            self.sock.setsockopt(zmq.LINGER, 0)
            self.sock.connect(ADDR)
        self.sock.setsockopt(zmq.RCVTIMEO, int(timeout_s * 1000))
        self.sock.send(json.dumps(dict(cmd=cmd, **kw)).encode())
        try:
            r = json.loads(self.sock.recv())
        except zmq.Again:
            self.sock.close()
            self.sock = None  # a REQ socket is unusable after a timeout
            raise ToolError(f"arm_server did not answer {cmd!r} at {ADDR}: is it running?")
        if not r.get("ok", False):
            raise ToolError(r.get("error") or f"{cmd} failed")
        return r


arm = _Arm()


def _side(side):
    if side not in ("left", "right"):
        raise ToolError("side must be 'left' or 'right'")
    return side


def _target(x, y, z):
    xyz = [float(x), float(y), float(z)]
    for k, v in zip("xyz", xyz):
        lo, hi = WS[k]
        if not lo <= v <= hi:
            raise ToolError(f"{k}={v:.3f} m is outside the reachable box {WS[k]}")
    if table_z is not None and xyz[2] < table_z + TABLE_CLEARANCE:
        raise ToolError(f"z={xyz[2]:.3f} m is below the table floor ({table_z:.3f} + {TABLE_CLEARANCE} m clearance)")
    return xyz


@mcp.tool()
def status() -> dict:
    """Arm controller state: FSM, enabled/frozen, both hands' positions (pelvis frame, m), grippers, table."""
    r = arm("state", timeout_s=5)
    a = r["arm"]
    grip = lambda g: None if not g else {k: g.get(k) for k in ("q", "phase", "tau", "holding") if k in g}
    fsm_id = a["fsm"][0] if isinstance(a["fsm"], (list, tuple)) else a["fsm"]     # arm_server reports [id, mode]
    return {"fsm": a["fsm"], "arm_capable_fsm": fsm_id in (500, 501, 801), "enabled": a["enabled"],
            "frozen": a["frozen"], "freeze_reason": a["freeze_reason"], "moving": a["moving"],
            "left_hand_xyz": (r.get("fk") or {}).get("grasp_xyz"), "right_hand_xyz": (r.get("fk_right") or {}).get("grasp_xyz"),
            "left_gripper": grip(r.get("gripper")), "right_gripper": grip(r.get("gripper_right")),
            "table": r.get("table"), "table_floor_z": table_z}


@mcp.tool()
def enable() -> str:
    """Take the arms from Unitree's controller (2 s ramp from the measured pose, no snap). Needed before moving."""
    tok = arm("prepare_enable")["token"]
    arm("enable", prepared_token=tok, ramp_s=2.0)
    return "enabled"


@mcp.tool()
def release() -> str:
    """Hand the arms back to Unitree's controller (2 s fade). Do this when the task is done."""
    arm("release", ramp_s=2.0)
    return "released"


@mcp.tool()
def freeze(reason: str = "agent") -> str:
    """Hold both arms exactly where they are now. Use when something looks wrong."""
    arm("freeze", reason=reason)
    return "frozen (release to hand back, or unfreeze=true on the next move)"


@mcp.tool()
def check_reach(side: str, x: float, y: float, z: float) -> dict:
    """Can this hand reach (x, y, z) with level jaws, and is the path from the current pose collision-free? Never moves."""
    xyz = _target(x, y, z)
    try:
        ik = arm("ik", side=_side(side), xyz=xyz)
    except ToolError as e:  # "not reachable" is an answer here, not a failure
        return {"reachable": False, "reason": str(e)}
    q = ik.get("q7")
    path = arm("check", side=side, q7=q)["check"] if q else {"ok": False, "reason": "no IK solution"}
    return {"reachable": bool(q) and ik.get("ok", True), "pos_err_m": ik.get("pos_err"),
            "path_clear": path.get("ok"), "path_problem": path.get("reason") or None}


@mcp.tool()
def move_hand(side: str, x: float, y: float, z: float, seconds: float = 3.0) -> dict:
    """Move one hand's grasp point (between the Dex1 jaws) to (x, y, z), jaws level. Blocks until it arrives.
    The server refuses unreachable targets and paths that collide with the body or the table.
    Prefer steps of <= 15 cm and 2-4 s; check with look afterwards."""
    if not 1.0 <= seconds <= 10.0:
        raise ToolError("seconds must be 1-10")
    r = arm("move_ee", side=_side(side), xyz=_target(x, y, z), duration=seconds, timeout_s=seconds + 20)
    if not _replaying:
        recording.append({"op": "move_hand", "side": side, "xyz": [round(float(v), 4) for v in (x, y, z)], "seconds": float(seconds)})
    now = (status().get(f"{side}_hand_xyz"))
    return {"arrived_xyz": now, "ik_pos_err_m": r.get("pos_err"), "jaw_tilt": r.get("jaw_tilt")}


@mcp.tool()
def gripper(side: str, action: str) -> dict:
    """Dex1 gripper: action 'open', 'close' (grasp with a bounded squeeze; keeps holding force), or 'stop'."""
    cmd = {"open": "open", "close": "grasp", "stop": "gripper_stop"}.get(action)
    if cmd is None:
        raise ToolError("action must be open, close or stop")
    g = arm(cmd, side=_side(side))
    if not _replaying:
        recording.append({"op": "gripper", "side": side, "action": action})
    return {k: g.get(k) for k in ("q", "phase", "tau", "holding") if k in g}


@mcp.tool()
def set_table(z: float | None, x_min: float = 0.10, x_max: float = 0.60, y_min: float = -0.55, y_max: float = 0.55) -> str:
    """Declare a table top at height z (pelvis frame, m) so paths keep the hands above it and targets stay
    3 cm over it. z=None clears it."""
    global table_z
    if z is None:
        arm("set_table", z=None)
        table_z = None
        return "table cleared"
    arm("set_table", z=float(z), x=[x_min, x_max], y=[y_min, y_max])
    table_z = float(z)
    if not _replaying:
        recording.append({"op": "set_table", "z": table_z, "x": [x_min, x_max], "y": [y_min, y_max]})
    return f"table at z={table_z:.3f} m; hand targets must stay at or above {table_z + TABLE_CLEARANCE:.3f} m"


@mcp.tool()
def park() -> str:
    """Lift both hands 15 cm, then move them to the ready pose [0.28, +-0.20, 0.05]. Needs enable."""
    arm("park", side="both", timeout_s=40)
    if not _replaying:
        recording.append({"op": "park"})
    return "parked at the ready pose"


# ---------- codifying: recorded steps -> named routines ----------
OPS = {"move_hand", "gripper", "wait", "park", "set_table"}


def _valid_step(st: dict) -> dict:
    """Check one routine step and return it normalized. Raises ToolError with the reason."""
    if not isinstance(st, dict) or st.get("op") not in OPS:
        raise ToolError(f"bad step {st!r}: op must be one of {sorted(OPS)}")
    op = st["op"]
    if op == "move_hand":
        xyz = st.get("xyz")
        if not (isinstance(xyz, (list, tuple)) and len(xyz) == 3):
            raise ToolError(f"move_hand step needs xyz [x, y, z]: {st!r}")
        sec = float(st.get("seconds", 3.0))
        if not 1.0 <= sec <= 10.0:
            raise ToolError(f"move_hand seconds must be 1-10: {st!r}")
        return {"op": op, "side": _side(st.get("side")), "xyz": _target(*xyz), "seconds": sec}
    if op == "gripper":
        if st.get("action") not in ("open", "close", "stop"):
            raise ToolError(f"gripper step needs action open/close/stop: {st!r}")
        return {"op": op, "side": _side(st.get("side")), "action": st["action"]}
    if op == "wait":
        sec = float(st.get("seconds", 1.0))
        if not 0.0 <= sec <= 30.0:
            raise ToolError(f"wait seconds must be 0-30: {st!r}")
        return {"op": op, "seconds": sec}
    if op == "set_table":
        return {"op": op, "z": None if st.get("z") is None else float(st["z"]),
                "x": list(st.get("x", [0.10, 0.60])), "y": list(st.get("y", [-0.55, 0.55]))}
    return {"op": op}


def _path(name: str) -> Path:
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", name or ""):
        raise ToolError("routine name: lowercase letters, digits, - and _, up to 64 chars (e.g. 'wave-hello')")
    return ROUTINES / f"{name}.json"


@mcp.tool()
def recorded_steps() -> list[dict]:
    """The successful actions recorded this session, in order, as routine steps (what save_routine would keep)."""
    return recording


@mcp.tool()
def clear_recording() -> str:
    """Start a fresh recording, e.g. right before demonstrating the sequence you want to save."""
    recording.clear()
    return "recording cleared"


@mcp.tool()
def save_routine(name: str, description: str, steps: list[dict] | None = None, overwrite: bool = False) -> dict:
    """Save a named routine. Without `steps`, saves recorded_steps(); with `steps`, saves those (you can
    write or edit a sequence directly). Every step is validated (bounds, table floor, op format) first."""
    path = _path(name)
    if path.exists() and not overwrite:
        raise ToolError(f"routine {name!r} exists; pass overwrite=true to replace it")
    raw = recording if steps is None else steps
    if not raw:
        raise ToolError("nothing to save: no recorded steps and no steps given")
    clean = [_valid_step(st) for st in raw]
    ROUTINES.mkdir(parents=True, exist_ok=True)
    doc = {"name": name, "description": description, "saved": time.strftime("%Y-%m-%d %H:%M:%S"),
           "frame": "pelvis, metres: x forward, y left, z up", "steps": clean}
    path.write_text(json.dumps(doc, indent=2) + "\n")
    return {"saved": str(path), "steps": len(clean)}


@mcp.tool()
def list_routines() -> list[dict]:
    """All saved routines: name, description, number of steps."""
    out = []
    for f in sorted(ROUTINES.glob("*.json")) if ROUTINES.exists() else []:
        try:
            d = json.loads(f.read_text())
            out.append({"name": d.get("name", f.stem), "description": d.get("description", ""), "steps": len(d.get("steps", []))})
        except (OSError, ValueError):
            out.append({"name": f.stem, "description": "(unreadable file)", "steps": 0})
    return out


@mcp.tool()
def show_routine(name: str) -> dict:
    """The full routine file: description and every step."""
    path = _path(name)
    if not path.exists():
        raise ToolError(f"no routine {name!r}; see list_routines")
    return json.loads(path.read_text())


@mcp.tool()
def run_routine(name: str, dry_run: bool = True) -> dict:
    """Replay a saved routine. dry_run=true (default) validates every step and checks each hand target is
    reachable, without moving. dry_run=false needs the arms enabled and runs the steps in order through the
    same checks as the individual tools; it stops at the first refused step and reports which one."""
    global _replaying
    steps = [_valid_step(st) for st in show_routine(name)["steps"]]
    if dry_run:
        report = []
        for i, st in enumerate(steps):
            if st["op"] == "move_hand":
                r = check_reach(st["side"], *st["xyz"])
                report.append({"step": i, "op": "move_hand", "side": st["side"], "xyz": st["xyz"], "reachable": r.get("reachable")})
            else:
                report.append({"step": i, **st})
        bad = [r for r in report if r.get("reachable") is False]
        return {"routine": name, "dry_run": True, "ok": not bad, "steps": report,
                "note": "path collision is checked again live, from wherever the arm actually is"}
    if not status()["enabled"]:
        raise ToolError("arms are not enabled: call enable first")
    pre = run_routine(name, dry_run=True)  # refuse up front rather than stopping halfway through
    if not pre["ok"]:
        bad = [r["step"] for r in pre["steps"] if r.get("reachable") is False]
        raise ToolError(f"routine {name!r} not started: step(s) {bad} unreachable (see run_routine dry_run=true)")
    done = []
    _replaying = True
    try:
        for i, st in enumerate(steps):
            try:
                if st["op"] == "move_hand":
                    move_hand(st["side"], *st["xyz"], seconds=st["seconds"])
                elif st["op"] == "gripper":
                    gripper(st["side"], st["action"])
                elif st["op"] == "wait":
                    time.sleep(st["seconds"])
                elif st["op"] == "park":
                    park()
                elif st["op"] == "set_table":
                    set_table(st["z"], st["x"][0], st["x"][1], st["y"][0], st["y"][1])
            except ToolError as e:
                raise ToolError(f"routine {name!r} stopped at step {i} ({st['op']}) after {len(done)} done: {e}")
            done.append(i)
    finally:
        _replaying = False
    return {"routine": name, "ran_steps": len(done), "ok": True}


# ---------- vision: pixels <-> pelvis-frame targets ----------
_cfg, _depth_cache = None, (0.0, None)


def _head_config() -> dict:
    """Head camera intrinsics + depth scale, asked once from teleimager's config responder (port 60000)."""
    global _cfg
    if _cfg is None:
        sock = zmq.Context.instance().socket(zmq.REQ)
        sock.setsockopt(zmq.LINGER, 0)
        sock.setsockopt(zmq.RCVTIMEO, 1500)
        try:
            sock.connect(f"tcp://{CAM_HOST}:{CAMCFG_PORT}")
            sock.send(b"cfg")
            head = json.loads(sock.recv())["head_camera"]
            _cfg = {"image_shape": head["image_shape"], "depth_scale": float(head.get("depth_scale", 0.001)),
                    "color_intrinsics": head["color_intrinsics"], "source": "live"}
        except (zmq.Again, KeyError, ValueError):
            _cfg = dict(HEAD_FALLBACK, source="fallback")
        finally:
            sock.close()
    return _cfg


def _depth_m() -> np.ndarray:
    """Latest aligned head depth frame in metres. One short-lived SUB (CONFLATE: newest frame only), cached 150 ms,
    so a burst of point queries costs one frame and nothing streams in the background."""
    global _depth_cache
    t, frame = _depth_cache
    if frame is not None and time.monotonic() - t < 0.15:
        return frame
    cfg = _head_config()
    h, w = cfg["image_shape"]
    sock = zmq.Context.instance().socket(zmq.SUB)
    sock.setsockopt(zmq.LINGER, 0)
    sock.setsockopt(zmq.CONFLATE, 1)
    sock.setsockopt(zmq.RCVTIMEO, 1500)
    sock.setsockopt_string(zmq.SUBSCRIBE, "")
    try:
        sock.connect(f"tcp://{CAM_HOST}:{DEPTH_PORT}")
        raw = sock.recv()
    except zmq.Again:
        raise ToolError(f"no depth frame from {CAM_HOST}:{DEPTH_PORT} in 1.5 s (teleimager down, or tunnel missing)")
    finally:
        sock.close()
    if len(raw) == h * w * 2:                      # raw z16, aligned to color
        z = np.frombuffer(raw, np.uint16).reshape(h, w)
    elif raw[:8] == b"\x89PNG\r\n\x1a\n":          # 16-bit PNG, if the server compresses
        from PIL import Image as PILImage
        z = np.asarray(PILImage.open(io.BytesIO(raw)))
    else:
        raise ToolError(f"unexpected depth message: {len(raw)} bytes (expected {h}x{w}x2 raw z16 or PNG)")
    frame = z.astype(np.float32) * cfg["depth_scale"]
    _depth_cache = (time.monotonic(), frame)
    return frame


def _pelvis_from_cam() -> tuple[np.ndarray, str]:
    """Camera pose in the pelvis frame, with the live waist angles from arm_server when it's up."""
    try:
        T_pt = arm("state", timeout_s=3).get("T_pelvis_torso")
        if T_pt:
            return vision.pelvis_from_cam(np.array(T_pt)), "live waist"
    except ToolError:
        pass
    return vision.pelvis_from_cam(np.eye(4)), "waist assumed straight (arm_server unreachable)"


@mcp.tool()
def point_to_xyz(points: list[list[float]], scale: str = "pixels") -> dict:
    """Turn points on the head-camera image into hand targets: [[u, v], ...] -> pelvis-frame xyz (m) for move_hand.
    scale: 'pixels' (640x480 head image, as look returns it), 'norm1000' (0-1000, Gemini Robotics-ER style) or
    'norm1' (0-1). All points share one depth frame. confidence = fraction of valid depth around the point
    (low: dark/shiny/edge; 0 means it searched wider to find any). Point at the part you want the jaws to reach."""
    t0 = time.monotonic()
    if not points or len(points) > 32:
        raise ToolError("give 1-32 points as [[u, v], ...]")
    cfg = _head_config()
    h, w = cfg["image_shape"]
    K = cfg["color_intrinsics"]
    depth = _depth_m()
    T, waist = _pelvis_from_cam()
    out = []
    for pt in points:
        try:
            u, v = vision.to_pixels(float(pt[0]), float(pt[1]), scale, w, h)
        except (ValueError, TypeError, IndexError):
            raise ToolError(f"bad point {pt!r}; scale must be pixels, norm1000 or norm1")
        z, conf = vision.depth_at(depth, u, v)
        if z is None:
            out.append({"u": round(u, 1), "v": round(v, 1), "xyz": None, "why": "no valid depth here (outside image, too close/far, or a hole)"})
            continue
        xyz = vision.pixel_to_pelvis(u, v, z, K, T)
        inside = all(WS[k][0] <= c <= WS[k][1] for k, c in zip("xyz", xyz))
        out.append({"u": round(u, 1), "v": round(v, 1), "xyz": [round(float(c), 4) for c in xyz],
                    "depth_m": round(z, 3), "confidence": round(conf, 2), "in_reach_box": inside})
    return {"points": out, "camera_config": cfg["source"], "mount": "URDF d435_joint", "waist": waist,
            "ms": round((time.monotonic() - t0) * 1000)}


def _annotate(jpeg: bytes) -> bytes:
    """Pixel grid every 80 px with coordinates, and both hands' computed positions projected onto the image.
    If the L/R dots sit on the real grippers, the camera calibration is right."""
    from PIL import Image as PILImage, ImageDraw
    im = PILImage.open(io.BytesIO(jpeg)).convert("RGB")
    d = ImageDraw.Draw(im)
    W, H = im.size
    for x in range(80, W, 80):
        d.line([(x, 0), (x, H)], fill=(255, 255, 0), width=1)
        d.text((x + 2, 2), str(x), fill=(255, 255, 0))
    for y in range(80, H, 80):
        d.line([(0, y), (W, y)], fill=(255, 255, 0), width=1)
        d.text((2, y + 2), str(y), fill=(255, 255, 0))
    try:
        r = arm("state", timeout_s=3)
        T = vision.pelvis_from_cam(np.array(r["T_pelvis_torso"]))
        K = _head_config()["color_intrinsics"]
        for label, key, col in (("L", "fk", (0, 255, 0)), ("R", "fk_right", (255, 0, 255))):
            xyz = (r.get(key) or {}).get("grasp_xyz")
            px = vision.pelvis_to_pixel(xyz, K, T) if xyz else None
            if px and 0 <= px[0] < W and 0 <= px[1] < H:
                d.ellipse([px[0] - 7, px[1] - 7, px[0] + 7, px[1] + 7], outline=col, width=3)
                tx = px[0] + 9 if px[0] < W - 60 else px[0] - 55   # keep the label inside the image
                d.text((tx, px[1] - 6), f"{label} hand", fill=col)
    except (ToolError, KeyError, TypeError):
        d.text((8, H - 16), "hand overlay unavailable (arm_server down)", fill=(255, 80, 80))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=85)
    return buf.getvalue()


@mcp.tool()
def look(camera: str = "head_camera", annotate: bool = True) -> Image:
    """One JPEG frame: 'head_camera' (the robot's eyes, 640x480, looking down at the workspace) or 'waldo_webcam'
    (room view). For the head camera, annotate=true adds a pixel grid (every 80 px) and marks where the robot thinks
    each hand is (L/R): use the grid to name points for point_to_xyz, and the marks to check a move landed."""
    if camera not in ("head_camera", "waldo_webcam"):
        raise ToolError("camera must be head_camera or waldo_webcam")
    try:
        with urllib.request.urlopen(f"{CAM}/frame/{camera}.jpg", timeout=5) as r:
            data = r.read()
    except OSError as e:
        raise ToolError(f"camera unreachable at {CAM} ({e}). On a laptop, open the 8766 tunnel to waldo; "
                        "if the head feed is stalled, restart g1-camera-viewer on waldo")
    return Image(data=_annotate(data) if annotate and camera == "head_camera" else data, format="jpeg")


# ---------- skills: one call per intent, for agents and onboard VLAs that only point ----------
STATIONS = Path(os.environ.get("G1_STATIONS", Path(__file__).resolve().parent / "stations.json"))
TAUGHT_DIR = Path(os.environ.get("G1_TAUGHT_DIR", Path(__file__).resolve().parent.parent / "waldo-sim"))
GAUGES = Path(os.environ.get("G1_GAUGES", Path(__file__).resolve().parent / "gauges.json"))
RECEIPTS = Path(os.environ.get("G1_RECEIPTS", Path(__file__).resolve().parent / "receipts"))
_plan = None


_sim = None
_recording = None        # {"name", "log"} while record_transfer is on: what the skills send, for save_transfer


def _is_sim() -> bool:
    """waldo-sim answers sim_* commands; the real arm_server doesn't."""
    global _sim
    if _sim is None:
        try:
            arm("sim_objects", timeout_s=3)
            _sim = True
        except ToolError:
            _sim = False
    return _sim


def _parm(cmd, **kw):
    """arm_server for planner.py / teach.py: their retries expect PlanError. On the real robot `trajectory` is
    non-blocking, so wait for it here, polling `state` (which also keeps arm_server's 2 s heartbeat alive), and fail
    if the arm froze. Sim-only bits become real equivalents: sim_wait sleeps, `contact` is dropped."""
    from planner import PlanError
    sim = _is_sim()
    if cmd == "sim_wait" and not sim:
        time.sleep(float(kw.get("seconds", 0)))
        return {"ok": True}
    if not sim:
        kw.pop("contact", None)
    rec = _recording is not None and cmd in ("trajectory", "move_ee", "grasp", "open", "gripper", "park")
    if rec and cmd in ("move_ee", "park"):       # Cartesian goals re-solve IK on replay; record the joints instead
        q_before = arm("state", timeout_s=3)["arm"]["q"]
    elif rec:
        _recording["log"].append([cmd, json.loads(json.dumps(kw, default=float))])
    try:
        r = arm(cmd, timeout_s=120, **kw)
        if cmd == "trajectory" and not sim:
            deadline = time.monotonic() + float(kw["times"][-1]) + 5.0
            time.sleep(0.1)
            while True:
                st = arm("state", timeout_s=3)["arm"]
                if st.get("frozen"):
                    raise PlanError(f"arm froze during the trajectory: {st.get('freeze_reason')}")
                if not st.get("moving"):
                    break
                if time.monotonic() > deadline:
                    arm("freeze", reason="trajectory overran")
                    raise PlanError("trajectory overran its time; arm frozen")
                time.sleep(0.2)
        if rec and cmd in ("move_ee", "park"):
            q_after = arm("state", timeout_s=3)["arm"]["q"]
            _recording["log"].append(["trajectory", {"q": [q_before, q_after], "times": [0.0, float(kw.get("duration", 3.0))],
                                                     "side": kw.get("side", "left"), **({"contact": True} if kw.get("contact") else {})}])
        return r
    except ToolError as e:
        raise PlanError(str(e))


def _planner():
    global _plan
    from planner import Planner
    desk = table_z if table_z is not None else -0.06
    if _plan is None or _plan.desk != desk:
        # the twin plans around what it knows: G1_TWIN=bare (desk only, a real cell) or a sim scene (sort, cell, lab)
        _plan = Planner(_parm, desk, scene_kind=os.environ.get("G1_TWIN", "bare"))
        _push_keepouts()
    return _plan


CONTENTS_M = 0.06


def _push_keepouts():
    """Every registered container and keep-out zone is an obstacle in the planner's twin."""
    if _plan is not None and _plan.twin is not None:
        # containers hold things that stick up past the rim: route 6 cm above it (keep-outs are taken as given)
        _plan.twin.set_keepouts([(v["box_min"], v["box_max"][:2] + [v["box_max"][2] + (0 if v.get("keepout") else CONTENTS_M)])
                                 for v in _stations().values()])


def _object_at(u, v, scale, box=None):
    """Object geometry from a point, or from a box [u1, v1, u2, v2] (VLM bbox_2d; more robust in clutter)."""
    cfg = _head_config()
    h, w = cfg["image_shape"]
    try:
        if box is not None:
            (b1, b2), (b3, b4) = (vision.to_pixels(float(box[0]), float(box[1]), scale, w, h),
                                  vision.to_pixels(float(box[2]), float(box[3]), scale, w, h))
            box = [b1, b2, b3, b4]
            u, v = (b1 + b3) / 2, (b2 + b4) / 2
        elif u is None or v is None:
            raise ToolError("give a point (u, v) or a box [u1, v1, u2, v2]")
        else:
            u, v = vision.to_pixels(float(u), float(v), scale, w, h)
    except (ValueError, TypeError, IndexError):
        raise ToolError("scale must be pixels, norm1000 or norm1; box is [u1, v1, u2, v2]")
    T, _ = _pelvis_from_cam()
    depth = _depth_m()
    g = vision.segment(depth, u, v, cfg["color_intrinsics"], T, table_z if table_z is not None else -0.06, box=box)
    z, _ = vision.depth_at(depth, u, v)
    if g is not None and box is None and z is not None:
        g["point"] = vision.pixel_to_pelvis(u, v, z, cfg["color_intrinsics"], T).tolist()
    if g is None:
        raise ToolError(f"nothing stands above the desk at pixel ({u:.0f}, {v:.0f}); point at the object itself "
                        "(set_table first if the desk height is not -0.06)")
    return g


def _skill(fn):
    from planner import PlanError
    t0 = time.monotonic()
    try:
        out = fn()
    except PlanError as e:
        raise ToolError(f"{e} (arm left clear; try another point or side)")
    out["ms"] = round((time.monotonic() - t0) * 1000)
    return out


@mcp.tool()
def pick(u: float | None = None, v: float | None = None, box: list[float] | None = None, scale: str = "pixels",
         side: str | None = None, then_place_in: str | None = None) -> dict:
    """Pick up the object under one point on the head image, or inside a box [u1, v1, u2, v2] (a VLM's bbox_2d:
    better when objects touch). then_place_in: the container it's going to next (picks the hand that reaches both). Depth gives its footprint, height and the axis to close
    the jaws across, so you only point (pixel from look, or a VLM's norm1000 point). Tries both hands, several
    grasp angles and approach pitches, with collision-free planned motion; verifies something is held.
    Returns side (use it for place), the object geometry and the grasp used. Max graspable width ~90 mm."""
    g = _object_at(u, v, scale, box)
    if side is None and then_place_in:              # choose the hand on the destination's side: it can reach both ends
        dest = _stations().get(then_place_in)
        if dest is None:
            raise ToolError(f"unknown container {then_place_in!r}; registered: {sorted(_stations())}")
        side = "left" if (dest["box_min"][1] + dest["box_max"][1]) / 2 > 0 else "right"
    rim = None
    if g["ring"] or g["width"] > 0.09:
        if not g["ring"]:
            raise ToolError(f"object is {g['width'] * 1000:.0f} mm across its narrow side; the Dex1 opens to ~90 mm")
        rim = max(0.0, min(g["width"], g["length"]) / 2 - 0.006)     # pinch the wall of a ring (tape roll)
    p = _planner()
    r = _receipt("pick", {"u": u, "v": v, "box": box, "side": side},
                 lambda: _skill(lambda: p.pick_item(np.array(g["centre"]), g["jaw_deg"], height_m=g["height"],
                                                    prefer_side=_side(side) if side else None, rim_radius_m=rim,
                                                    round_item=not rim and g["length"] - g["width"] < 0.008,
                                                    width_m=g["width"])))
    return {"side": r["side"], "held": True, "gripper_gap_mm": r["gripper"].get("gap_mm"),
            "grasp": {"jaw_deg": r["jaw_deg"], "pitch_deg": r["pitch_deg"], "tries": len(r["attempts"])},
            "object": {k: (round(g[k], 3) if isinstance(g[k], float) else g[k]) for k in ("height", "width", "length", "ring")},
            "ms": r["ms"]}


def _stations() -> dict:
    return json.loads(STATIONS.read_text()) if STATIONS.exists() else {}


@mcp.tool()
def register_container(name: str, u: float | None = None, v: float | None = None, box: list[float] | None = None,
                       scale: str = "pixels", rim_z: float | None = None, drop_xy: list[float] | None = None) -> dict:
    """Remember a fixed container (cup, tray, bin, caddy) once, so place(container=name) works without looking:
    point at (or box) it on the head image; its footprint and rim come from depth. Point at the open spot to drop
    into (an empty compartment) or pass drop_xy [x, y] (m). rim_z overrides a rim the camera can't fully see
    (a tall back wall at the image edge). Containers are stations: register them when the cell is set up."""
    g = _object_at(u, v, scale, box)
    st = _stations()
    st[name] = {"box_min": [round(c, 3) for c in g["box_min"]],
                "box_max": [round(c, 3) for c in g["box_max"][:2]] + [round(rim_z if rim_z is not None else g["top_z"], 3)],
                "drop_xy": [round(c, 3) for c in (drop_xy or g.get("point") or g["centre"])[:2]]}
    STATIONS.write_text(json.dumps(st, indent=1))
    _push_keepouts()
    return {name: st[name]}


@mcp.tool()
def register_keepout(name: str, u: float | None = None, v: float | None = None, box: list[float] | None = None,
                     scale: str = "pixels", box_min: list[float] | None = None, box_max: list[float] | None = None,
                     margin_m: float = 0.01) -> dict:
    """Tell the planner about something it must route around: a fixture with parts standing in it, a machine, a
    panel, a light tower. Point at it / box it on the head image (footprint and height from depth), or give
    box_min / box_max (m, pelvis frame). Free moves avoid it; the gripper may still touch it while grasping from it.
    Registered containers count automatically. Register every fixed thing in the cell before running."""
    if box_min is not None and box_max is not None:
        lo, hi = [float(c) for c in box_min], [float(c) for c in box_max]
    else:
        g = _object_at(u, v, scale, box)
        lo, hi = g["box_min"], g["box_max"]
    st = _stations()
    st[name] = {"box_min": [round(c - margin_m, 3) for c in lo[:2]] + [round(lo[2], 3)],
                "box_max": [round(c + margin_m, 3) for c in hi[:2]] + [round(hi[2] + margin_m, 3)], "keepout": True}
    STATIONS.write_text(json.dumps(st, indent=1))
    _push_keepouts()
    return {name: st[name]}


@mcp.tool()
def list_containers() -> dict:
    """Registered containers (name -> footprint box, rim height, drop spot), for place(container=name)."""
    return _stations()


@mcp.tool()
def place(side: str, container: str | None = None, u: float | None = None, v: float | None = None,
          box: list[float] | None = None, scale: str = "pixels") -> dict:
    """Put what `side` holds into a container: by name (register_container / list_containers: fastest, no camera),
    or under a point on the head image (point at the open spot to drop into, away from dividers), or a box.
    The item is lifted clear of every rim, carried level, lowered in and released (fingers slide off before the arm
    moves). Then call home(side)."""
    _side(side)
    if container is not None:
        st = _stations()
        if container not in st:
            raise ToolError(f"unknown container {container!r}; registered: {sorted(st)}")
        g = st[container]
        at = g["drop_xy"]
    else:
        g = _object_at(u, v, scale, box)
        at = g.get("point")
    p = _planner()
    _receipt("place", {"side": side, "container": container, "u": u, "v": v, "box": box},
             lambda: _skill(lambda: p.place(side, g["box_min"], g["box_max"], at=at) or {}))
    return {"side": side, "released": True, "container": container or "pointed",
            "drop_xy": [round(c, 3) for c in (at or [])[:2]], "rim_z": round(g["box_max"][2], 3)}


@mcp.tool()
def home(side: str = "both") -> str:
    """Rest one or both arms low by the sides on a collision-free path (out of the camera's view and the other
    hand's way). Do this after place, and before handing control back."""
    sides = ["left", "right"] if side == "both" else [_side(side)]
    p = _planner()
    for s in sides:
        _receipt("home", {"side": s}, lambda: _skill(lambda: p.home(s) or {}))
    return f"{', '.join(sides)} at rest"


def _teach():
    import sys
    sim_dir = str(Path(__file__).resolve().parent.parent / "waldo-sim")     # teach.py lives with the sim
    if sim_dir not in sys.path:
        sys.path.insert(0, sim_dir)
    import teach
    return teach


def _taught() -> dict:
    """Every station's taught controls (taught.json = IT desk, taught_lab.json = dev board + machine)."""
    out = {}
    for f in ([TAUGHT_DIR] if TAUGHT_DIR.is_file() else sorted(TAUGHT_DIR.glob("taught*.json"))):   # one cell's file, or all
        out.update(json.loads(f.read_text()))
    if not out:
        raise ToolError(f"no taught controls in {TAUGHT_DIR}; teach them first (waldo-sim/teach.py)")
    return out


@mcp.tool()
def list_controls() -> dict:
    """Fixed-station controls (switches, buttons, keys) and multi-step tasks taught for this cell. Taught motions
    are verified joint trajectories: the fastest, most reliable way to operate a known device. A task is available
    when all its controls are taught."""
    teach, taught = _teach(), _taught()
    return {"controls": {n: t["side"] for n, t in taught.items()},
            "tasks": {n: [s[0] if len(s) < 3 else f"{s[0]}:{s[2]}" for s in steps] for n, steps in teach.ALL_TASKS.items()
                      if all(s[0] in taught for s in steps)},
            "phases": "tap | press (and hold) | release | hover (fingertip parked above, ready) | poke (fast press from hover) | rest"}


def _receipt(tool: str, args: dict, fn):
    """Proof of a physical action: head-camera frames before and after, the arguments and the result, appended to
    receipts/receipts.jsonl. Agents (and their humans) can show what the robot actually did."""
    RECEIPTS.mkdir(parents=True, exist_ok=True)
    rid = time.strftime("%Y%m%d-%H%M%S") + f"-{int(time.time() * 1000) % 1000:03d}-{tool}"
    def snap(tag):
        try:
            with urllib.request.urlopen(f"{CAM}/frame/head_camera.jpg", timeout=3) as r:
                (RECEIPTS / f"{rid}_{tag}.jpg").write_bytes(r.read())
            return f"{rid}_{tag}.jpg"
        except OSError:
            return None
    rec = {"id": rid, "tool": tool, "args": args, "before": snap("before")}
    t0 = time.monotonic()
    try:
        out = fn()
        rec.update(ok=True, result=out)
        return out
    except ToolError as e:
        rec.update(ok=False, error=str(e))
        raise
    finally:
        rec["ms"] = round((time.monotonic() - t0) * 1000)
        rec["after"] = snap("after")
        with open(RECEIPTS / "receipts.jsonl", "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")


@mcp.tool()
def receipts(last: int = 5) -> list[dict]:
    """The most recent physical actions (pick, place, home, run_task): what was asked, what happened, how long it
    took, and before/after head-camera frames (ids for show_receipt)."""
    f = RECEIPTS / "receipts.jsonl"
    if not f.exists():
        return []
    return [json.loads(line) for line in f.read_text().splitlines()[-max(1, min(last, 50)):]]


@mcp.tool()
def show_receipt(receipt_id: str) -> Image:
    """Before | after head-camera frames of one action, side by side (receipt ids from receipts)."""
    from PIL import Image as PILImage
    ims = []
    for tag in ("before", "after"):
        p = RECEIPTS / f"{receipt_id}_{tag}.jpg"
        if not p.exists():
            raise ToolError(f"no {tag} frame for receipt {receipt_id!r}")
        ims.append(PILImage.open(p).convert("RGB"))
    w, h = ims[0].size
    out = PILImage.new("RGB", (w * 2, h))
    out.paste(ims[0], (0, 0))
    out.paste(ims[1], (w, 0))
    buf = io.BytesIO()
    out.save(buf, "JPEG", quality=80)
    return Image(data=buf.getvalue(), format="jpeg")


_pending: dict = {}      # name -> taught control from teach_control, waiting for save_control


@mcp.tool()
def teach_control(name: str, u: float, v: float, push: str = "down", depth_mm: float = 5.0,
                  indicator: list[float] | None = None, side: str | None = None, scale: str = "pixels") -> dict:
    """Teach a new button / switch / key on the real cell by pointing at it: (u, v) = the spot to press on the head
    image; push = down (buttons, keys, rocker ends), forward (a switch facing the robot) or [x, y, z]. Plans an
    approach + press against a desk-only twin, runs it once through arm_server's own checks (watch it the first
    time; G1_JOINT_SPEED slows it), and reports whether `indicator` (a light's [u, v]) changed. If it did what you
    wanted, save_control(name) keeps it for run_task. Start from home: the press is replayed from where it began."""
    teach = _teach()
    cfg = _head_config()
    h, w = cfg["image_shape"]
    uu, vv = vision.to_pixels(float(u), float(v), scale, w, h)
    z, conf = vision.depth_at(_depth_m(), uu, vv)
    if z is None:
        raise ToolError(f"no depth at ({uu:.0f}, {vv:.0f}); point at the control's surface")
    T, _ = _pelvis_from_cam()
    point = vision.pixel_to_pelvis(uu, vv, z, cfg["color_intrinsics"], T)
    dirn = {"down": [0, 0, -1], "forward": [1, 0, 0]}.get(push, push)
    try:
        dirn = [float(c) for c in dirn]
    except (TypeError, ValueError):
        raise ToolError("push is down, forward or [x, y, z]")
    from motion import Twin
    from planner import PlanError
    p = _planner()
    twin = Twin(p.desk, kind="bare")
    twin.PAD = 0.004
    arm("enable")
    q0 = np.array(arm("state")["arm"]["q"], float)
    light = lambda: _light_state(indicator) if indicator else None
    before, tried, why = light(), 0, []
    for sd, d, approach, press in teach.candidates(p, twin, q0, point, dirn, depth_mm / 1000):
        if side and sd != side:
            continue
        tried += 1
        moved = []
        def go():
            arm("grasp", side=sd)
            teach.send(_parm, sd, approach)
            moved.append("approach")
            teach.send(_parm, sd, press, contact=True)
            _parm("sim_wait", seconds=0.3)
            teach.send(_parm, sd, press[::-1], contact=True)
            teach.send(_parm, sd, approach[::-1])
            moved.clear()
            return {}
        try:
            _receipt("teach_control", {"name": name, "u": u, "v": v, "push": push}, lambda: _skill(go))
        except ToolError as e:
            why.append(str(e)[:100])
            if moved:                 # rejected mid-way: back out the way we came, then stop for a human look
                try:
                    teach.send(_parm, sd, approach[::-1])
                except PlanError:
                    pass
                raise ToolError(f"press rejected after the approach ({why[-1]}); arm backed out. Check the camera, "
                                "then teach_control again with another side, push or depth")
            if tried >= 4:
                break
            continue
        after = light()
        _pending[name] = {"side": sd, "push": np.round(d, 4).tolist(),
                          "approach": [list(map(float, q)) for q in approach], "press": [list(map(float, q)) for q in press]}
        out = {"name": name, "side": sd, "point_xyz": [round(float(c), 4) for c in point], "tried": tried,
               "next": f"if it worked, save_control('{name}'); else teach_control again (other side / push / depth)"}
        if indicator:
            out.update(indicator_before=before, indicator_after=after,
                       changed=(before["on"], before.get("colour")) != (after["on"], after.get("colour")))
        return out
    raise ToolError(f"no press that the arm accepts ({tried} tried): {'; '.join(why[-2:]) or 'no reachable approach'}")


def _light_state(pt):
    l = read_light([pt])["lights"][0]
    return {"on": l["on"], "colour": l.get("colour")}


@mcp.tool()
def save_control(name: str) -> dict:
    """Keep the last teach_control(name) as a taught control (taught_real.json), so run_task(name) replays it."""
    if name not in _pending:
        raise ToolError(f"nothing pending for {name!r}; run teach_control first")
    f = TAUGHT_DIR if TAUGHT_DIR.is_file() else TAUGHT_DIR / "taught_real.json"
    taught = json.loads(f.read_text()) if f.exists() else {}
    taught[name] = _pending.pop(name)
    f.write_text(json.dumps(taught))
    return {"saved": name, "file": str(f), "controls": sorted(taught)}


@mcp.tool()
def record_transfer(name: str) -> dict:
    """Start recording a fixed transfer (e.g. infeed pocket -> machine nest) to teach it by doing it once: call home,
    then pick / place / home as usual while a human watches; check the result (look, read_light, machine starts),
    then save_transfer(name) to keep it, or save_transfer(name, discard=True). run_task(name) then replays exactly
    those motions: the reliable way to run fixtured parts. Re-record if pick needed more than one try."""
    global _recording
    _recording = {"name": name, "log": [], "t0": time.time()}
    return {"recording": name, "next": "home -> pick -> place -> home, check it worked, then save_transfer"}


@mcp.tool()
def save_transfer(name: str, discard: bool = False) -> dict:
    """Stop recording; keep the recorded motions as taught transfer `name` (replay with run_task), or discard them."""
    global _recording
    rec, _recording = _recording, None
    if rec is None or rec["name"] != name:
        raise ToolError(f"not recording {name!r}; call record_transfer first")
    if discard:
        return {"discarded": name, "commands": len(rec["log"])}
    if not any(c == "trajectory" for c, _ in rec["log"]):
        raise ToolError("nothing moved while recording")
    f = TAUGHT_DIR if TAUGHT_DIR.is_file() else TAUGHT_DIR / "taught_real.json"
    taught = json.loads(f.read_text()) if f.exists() else {}
    side = next((kw.get("side") for _, kw in rec["log"] if kw.get("side")), "left")
    taught[name] = {"log": rec["log"], "side": side}
    f.write_text(json.dumps(taught))
    return {"saved": name, "commands": len(rec["log"]), "seconds": round(time.time() - rec["t0"]), "file": str(f)}


@mcp.tool()
def run_task(name: str, phase: str = "tap") -> dict:
    """Run a taught task (list_controls: power_cycle_router, approve_2fa, light_on, reset_board, enter_bootloader,
    start_cycle, arm_stop, stop_machine, ...) or one taught control with a phase (tap, press, release, hover, poke,
    rest). Waits and two-hand holds are part of the task (enter_bootloader holds BOOT while the other hand taps
    RESET). Verify after with read_light / read_display / read_gauge."""
    teach, taught = _teach(), _taught()
    steps = teach.ALL_TASKS.get(name) or ([(name, 0.0, phase)] if name in taught else None)
    if steps is None:
        raise ToolError(f"unknown task or control {name!r}; known: {sorted(teach.ALL_TASKS) + sorted(taught)}")
    if phase not in teach.PHASES:
        raise ToolError(f"phase must be one of {list(teach.PHASES)}")
    missing = [s[0] for s in steps if s[0] not in taught]
    if missing:
        raise ToolError(f"not taught yet: {missing}")
    def go():
        arm("enable")
        if name in teach.ALL_TASKS:
            teach.run(_parm, name, taught)
        else:
            teach.play(_parm, name, taught, phase)
        return {}
    out = _receipt("run_task", {"name": name, "phase": phase}, lambda: _skill(go))
    return {"task": name, "done": [s[0] for s in steps], "ms": out["ms"]}


def _frame() -> np.ndarray:
    from PIL import Image as PILImage
    try:
        with urllib.request.urlopen(f"{CAM}/frame/head_camera.jpg", timeout=5) as r:
            return np.asarray(PILImage.open(io.BytesIO(r.read())).convert("RGB"), np.float32) / 255
    except OSError as e:
        raise ToolError(f"camera unreachable at {CAM} ({e})")


def _crop(im, box, scale):
    H, W, _ = im.shape
    try:
        u1, v1 = vision.to_pixels(float(box[0]), float(box[1]), scale, W, H)
        u2, v2 = vision.to_pixels(float(box[2]), float(box[3]), scale, W, H)
    except (ValueError, TypeError, IndexError):
        raise ToolError("box is [u1, v1, u2, v2]; scale pixels, norm1000 or norm1")
    u1, u2, v1, v2 = int(max(0, min(u1, u2))), int(min(W, max(u1, u2))), int(max(0, min(v1, v2))), int(min(H, max(v1, v2)))
    if u2 - u1 < 4 or v2 - v1 < 4:
        raise ToolError("box too small or outside the image")
    return im[v1:v2, u1:u2]


SEGMENTS = {"a": (0.5, 0.1), "b": (0.85, 0.3), "c": (0.85, 0.7), "d": (0.5, 0.9), "e": (0.15, 0.7), "f": (0.15, 0.3), "g": (0.5, 0.5)}
SEG_CHARS = {"abcdef": "0", "bc": "1", "abdeg": "2", "abcdg": "3", "bcfg": "4", "acdfg": "5", "acdefg": "6", "abc": "7",
             "abcdefg": "8", "abcdfg": "9", "g": "-", "cdefg": "b", "cdeg": "o", "defg": "t", "": " ", "adefg": "E",
             "aefg": "F", "bcefg": "H", "def": "L", "abefg": "P", "bcdef": "U", "eg": "r", "bcdeg": "d", "ceg": "n",
             "abcef": "n", "cde": "u"}
# (on a 7-segment display 'g' and '9' are the same glyph, as are 'S' and '5', 'O' and '0': read words with that in mind)


def seven_segment(crop: np.ndarray, digits: int) -> tuple[str, list[float]]:
    """Decode a 7-segment display: split into digit cells, test each segment's spot for lit pixels."""
    v = crop.max(2)
    lit = v > max(0.4, 0.55 * float(v.max()))
    h, w = lit.shape
    text, conf = "", []
    for k in range(digits):
        cell = lit[:, int(k * w / digits):int((k + 1) * w / digits)]
        ch, cw = cell.shape
        on, margins = "", []
        for sname, (x, y) in SEGMENTS.items():
            win = cell[int(max(0, (y - 0.08) * ch)):int((y + 0.08) * ch) + 1, int(max(0, (x - 0.1) * cw)):int((x + 0.1) * cw) + 1]
            f = float(win.mean()) if win.size else 0.0
            margins.append(abs(f - 0.3))
            if f > 0.3:
                on += sname
        text += SEG_CHARS.get(on, "?")
        conf.append(round(min(1.0, 2 * min(margins)), 2))
    return text, conf


@mcp.tool()
def read_display(box: list[float], digits: int = 4, scale: str = "pixels") -> dict:
    """Read a 7-segment / LED digit display (dev-board code, timer, meter) inside a tight box [u1, v1, u2, v2] around
    the digits on the head image: box just the lit digits, a few px of slack is fine (it decodes 81 slightly shifted
    boxes and votes; a box much bigger than the digits misreads). ~70 ms, no VLM. 7-segment glyphs are ambiguous by
    design: '9' may be 'g' ("9ood" = "good"), '5' may be 'S', '0' may be 'O'. agreement = share of clean decodes
    that agree (>0.6 is solid); below that, check with look. For free text (labels, screens), use look and read it yourself."""
    im = _frame()
    H, W, _ = im.shape
    try:
        (u1, v1), (u2, v2) = (vision.to_pixels(float(box[0]), float(box[1]), scale, W, H),
                              vision.to_pixels(float(box[2]), float(box[3]), scale, W, H))
    except (ValueError, TypeError, IndexError):
        raise ToolError("box is [u1, v1, u2, v2]; scale pixels, norm1000 or norm1")
    # decode jittered boxes and vote: a few px either way clips or shifts segments (fitting the box to the lit
    # pixels was tried: loose boxes pull in bright desk edges, and glowing segments have near-white cores)
    votes: dict = {}
    for d1 in (-3, 0, 3):
        for d2 in (-3, 0, 3):
            for d3 in (-3, 0, 3):
                for d4 in (-3, 0, 3):
                    try:
                        text, conf = seven_segment(_crop(im, [u1 + d1, v1 + d2, u2 + d3, v2 + d4], "pixels"), int(digits))
                    except ToolError:
                        continue
                    if "?" not in text:
                        votes.setdefault(text, []).append(min(conf))
    if not votes:
        text, conf = seven_segment(_crop(im, [u1, v1, u2, v2], "pixels"), int(digits))
        return {"text": text, "agreement": 0.0, "note": "no clean decode: box the digits (just the lit segments) and retry"}
    best = max(votes, key=lambda t: (len(votes[t]), max(votes[t])))
    total = sum(len(v) for v in votes.values())
    return {"text": best, "agreement": round(len(votes[best]) / total, 2),
            "alternatives": {t: len(v) for t, v in sorted(votes.items(), key=lambda kv: -len(kv[1]))[1:3]}}


def _gauges() -> dict:
    return json.loads(GAUGES.read_text()) if GAUGES.exists() else {}


_guards: dict = {}


@mcp.tool()
def guard(gauge: str, above: float, then_task: str, timeout_s: float = 60.0, poll_s: float = 0.2) -> dict:
    """Interlock, armed in the background: watches a registered gauge at ~5 Hz and runs `then_task` (e.g.
    stop_machine; run arm_stop first so it's a short poke) the moment the reading goes above `above`. Returns at once
    with a guard id, so arm it BEFORE starting the machine; then start, and check guard_status(id, wait_s=...).
    Tool calls run one at a time; a guard is how a limit stays watched while you do other things. It ends when it
    fires, after timeout_s, or once the reading has risen and fallen back to ~0 (cycle over)."""
    if gauge not in _gauges():
        raise ToolError(f"unknown gauge {gauge!r}; registered: {sorted(_gauges())}")
    gid = f"g{len(_guards) + 1}"
    st = _guards[gid] = {"gauge": gauge, "above": above, "then": then_task, "fired": False, "done": False, "peak": None,
                         "trace": [], "t0": time.monotonic()}

    def watch():
        rose = False
        try:
            while time.monotonic() - st["t0"] < timeout_s:
                val = read_gauge(gauge)["value"]
                st["peak"] = val if st["peak"] is None else max(st["peak"], val)
                st["trace"] = (st["trace"] + [round(val, 2)])[-20:]
                if val > above:
                    st.update(fired=True, at=round(val, 2), after_s=round(time.monotonic() - st["t0"], 1))
                    run_task(then_task)
                    break
                rose = rose or val > 0.2 * above
                if rose and val < 0.05 * above:
                    break
                time.sleep(poll_s)
        except Exception as e:                 # a guard that can't see must say so, loudly
            st["error"] = str(e)[:200]
        st["done"] = True

    threading.Thread(target=watch, daemon=True).start()
    return {"guard": gid, "armed": True, "watching": f"{gauge} > {above} -> {then_task}"}


@mcp.tool()
def guard_status(guard_id: str, wait_s: float = 0.0) -> dict:
    """A guard's state: fired (and at what reading), peak, recent trace, done. wait_s: wait up to this long for it to
    finish (lets time pass, e.g. while a machine cycle runs)."""
    st = _guards.get(guard_id)
    if st is None:
        raise ToolError(f"unknown guard {guard_id!r}")
    t_end = time.monotonic() + wait_s
    while not st["done"] and time.monotonic() < t_end:
        if _is_sim():
            _parm("sim_wait", seconds=0.2)          # a sim advances its clock on request; let the guard read each step
        time.sleep(0.25)
    return {k: v for k, v in st.items() if k != "t0"}


@mcp.tool()
def register_gauge(name: str, box: list[float], min_value: float, max_value: float, min_deg: float = -135.0,
                   max_deg: float = 135.0, unit: str = "", alarm_above: float | None = None, scale: str = "pixels") -> dict:
    """Remember an analog dial gauge once: its box on the head image and its scale (needle angle clockwise from
    straight up at min_value and at max_value; most gauges sweep -135..+135). Then read_gauge(name)."""
    g = _gauges()
    g[name] = {"box": [float(c) for c in box], "scale": scale, "min_value": min_value, "max_value": max_value,
               "min_deg": min_deg, "max_deg": max_deg, "unit": unit, "alarm_above": alarm_above}
    GAUGES.write_text(json.dumps(g, indent=1))
    return {name: g[name]}


def needle_angle(crop: np.ndarray) -> tuple[float, int]:
    """Needle direction in a dial crop, degrees clockwise from up: the red needle's pixels around the dial centre
    (the centroid of the white face). Returns (angle, needle pixel count)."""
    r, g, b = crop[..., 0], crop[..., 1], crop[..., 2]
    needle = (r > 0.45) & (r - g > 0.3) & (r - b > 0.3)
    face = (crop.min(2) > 0.75)
    if needle.sum() < 5 or face.sum() < 20:
        raise ToolError("no dial face / red needle found in the gauge box")
    fy, fx = np.nonzero(face)
    cy, cx = fy.mean(), fx.mean()
    ny, nx = np.nonzero(needle)
    dx, dy = nx - cx, ny - cy
    wgt = np.hypot(dx, dy)
    ang = np.degrees(np.arctan2((dx * wgt).sum(), (-dy * wgt).sum()))
    return float(ang), int(needle.sum())


@mcp.tool()
def read_gauge(name: str | None = None, box: list[float] | None = None, scale: str = "pixels") -> dict:
    """Read an analog gauge (registered by name, or a box with the default 0-10 / -135..+135 scale): needle angle ->
    value, with the alarm flag if registered. ~50 ms, no VLM: poll it while a machine runs."""
    if name is not None:
        g = _gauges().get(name)
        if g is None:
            raise ToolError(f"unknown gauge {name!r}; registered: {sorted(_gauges())}")
    elif box is not None:
        g = {"box": box, "scale": scale, "min_value": 0.0, "max_value": 10.0, "min_deg": -135.0, "max_deg": 135.0, "unit": "", "alarm_above": None}
    else:
        raise ToolError("give a registered gauge name or a box")
    ang, n = needle_angle(_crop(_frame(), g["box"], g["scale"]))
    val = g["min_value"] + (ang - g["min_deg"]) / (g["max_deg"] - g["min_deg"]) * (g["max_value"] - g["min_value"])
    out = {"value": round(val, 2), "unit": g["unit"], "needle_deg": round(ang, 1), "needle_pixels": n}
    if g.get("alarm_above") is not None:
        out["alarm"] = bool(val > g["alarm_above"])
    return out


@mcp.tool()
def read_light(points: list[list[float]], scale: str = "pixels") -> dict:
    """Read indicator lights / lamps / screens at points on the head image: on/off, colour and brightness from a small
    patch (no VLM needed: ~50 ms). Use it to verify a task (router LED back to green, lamp lit, machine running)."""
    import colorsys
    from PIL import Image as PILImage
    try:
        with urllib.request.urlopen(f"{CAM}/frame/head_camera.jpg", timeout=5) as r:
            im = np.asarray(PILImage.open(io.BytesIO(r.read())).convert("RGB"), np.float32) / 255
    except OSError as e:
        raise ToolError(f"camera unreachable at {CAM} ({e})")
    H, W, _ = im.shape
    out = []
    for pt in points[:32]:
        u, v = vision.to_pixels(float(pt[0]), float(pt[1]), scale, W, H)
        ui, vi = int(round(u)), int(round(v))
        patch = im[max(0, vi - 4):vi + 5, max(0, ui - 4):ui + 5].reshape(-1, 3)
        if not len(patch):
            out.append({"u": ui, "v": vi, "why": "outside the image"})
            continue
        bright = patch[patch.max(1) >= np.percentile(patch.max(1), 70)].mean(0)     # the lit part of the patch
        hh, ss, vv = colorsys.rgb_to_hsv(*bright)
        deg = hh * 360
        hue = next(n for lim, n in ((15, "red"), (40, "orange"), (70, "yellow"), (165, "green"), (200, "cyan"),
                                    (260, "blue"), (290, "purple"), (340, "magenta"), (361, "red")) if deg < lim)
        on = vv > 0.55 and (ss > 0.35 or vv > 0.9)
        out.append({"u": ui, "v": vi, "on": bool(on), "colour": hue if ss > 0.25 else "white",
                    "brightness": round(float(vv), 2), "saturation": round(float(ss), 2)})
    return {"lights": out}


if __name__ == "__main__":
    mcp.run()

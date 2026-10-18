#!/usr/bin/env python3
"""waldo-arms: MCP server that lets an agent move the G1's hands and Dex1 grippers through Michael's arm_server.

arm_server (Jetson, ZMQ :5810) does the IK, self-collision and table path checks, workspace guard and watchdogs.
This server only exposes its commands as agent tools, adds a table floor, and serves camera frames.

Runs on Waldo (talks to 192.168.123.164:5810 directly) or on a laptop with tunnels:
  ssh -N -L 5810:192.168.123.164:5810 -L 8766:10.2.1.180:8766 waldo@100.120.182.66
  G1_ARM_ADDR=tcp://localhost:5810 G1_CAM_BASE=http://localhost:8766 uv run --with "mcp>=2.2" --with pyzmq server.py
"""
import json
import os
import re
import time
import urllib.request
from pathlib import Path

import zmq
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError  # message reaches the agent; plain exceptions don't
from mcp.server.mcpserver.utilities.types import Image

ADDR = os.environ.get("G1_ARM_ADDR", "tcp://192.168.123.164:5810")
CAM = os.environ.get("G1_CAM_BASE", "http://10.2.1.180:8766")
WS = {"x": (0.10, 0.60), "y": (-0.55, 0.55), "z": (-0.35, 0.45)}  # arm_server's pelvis-frame workspace (m)
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
    "Use look after moves to verify. freeze holds in place; release hands the arms back to Unitree's controller. "
    "The robot must be in FSM 500/501/801 (status shows it); you cannot change that here. "
    "CODIFYING: every successful move_hand / gripper / set_table / park is recorded as a JSON step. When a "
    "sequence worked, call save_routine(name, description) to keep it (or pass your own steps); later "
    "run_routine(name, dry_run=true) checks every step without moving, then run_routine(name, dry_run=false) "
    "replays it. Step format: {op: move_hand, side, xyz:[x,y,z], seconds} | {op: gripper, side, action} | "
    "{op: wait, seconds} | {op: park} | {op: set_table, z}."))


class _Arm:
    def __init__(self):
        self.ctx = zmq.Context.instance()
        self.sock = None

    def __call__(self, cmd, timeout_s: float = 30, **kw):
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
    return {"fsm": a["fsm"], "arm_capable_fsm": a["fsm"] in (500, 501, 801), "enabled": a["enabled"],
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


@mcp.tool()
def look(camera: str = "head_camera") -> Image:
    """One JPEG frame: 'head_camera' (the robot's eyes, looking down at the workspace) or 'waldo_webcam' (room view)."""
    if camera not in ("head_camera", "waldo_webcam"):
        raise ToolError("camera must be head_camera or waldo_webcam")
    try:
        with urllib.request.urlopen(f"{CAM}/frame/{camera}.jpg", timeout=5) as r:
            return Image(data=r.read(), format="jpeg")
    except OSError as e:
        raise ToolError(f"camera unreachable at {CAM} ({e}). On a laptop, open the 8766 tunnel to waldo; "
                        "if the head feed is stalled, restart g1-camera-viewer on waldo")


if __name__ == "__main__":
    mcp.run()

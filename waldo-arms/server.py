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
import urllib.request

import zmq
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError  # message reaches the agent; plain exceptions don't
from mcp.server.mcpserver.utilities.types import Image

ADDR = os.environ.get("G1_ARM_ADDR", "tcp://192.168.123.164:5810")
CAM = os.environ.get("G1_CAM_BASE", "http://10.2.1.180:8766")
WS = {"x": (0.10, 0.60), "y": (-0.55, 0.55), "z": (-0.35, 0.45)}  # arm_server's pelvis-frame workspace (m)
TABLE_CLEARANCE = 0.03  # hand targets stay this far above a declared table top
table_z = None

mcp = MCPServer("waldo-arms", instructions=(
    "Moves the Unitree G1's hands and Dex1 grippers (robot hangs in a gantry; no walking). "
    "Frame: pelvis, metres, x forward, y left, z up. Hands hang near [0.04, +-0.25, -0.21]; ready pose "
    "[0.28, +-0.20, 0.05]; reachable box x 0.10-0.60, y +-0.55, z -0.35-0.45. "
    "Workflow: status -> enable -> look / check_reach -> move_hand (small steps) -> gripper -> release when done. "
    "Use look after moves to verify. freeze holds in place; release hands the arms back to Unitree's controller. "
    "The robot must be in FSM 500/501/801 (status shows it); you cannot change that here."))


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
    now = (status().get(f"{side}_hand_xyz"))
    return {"arrived_xyz": now, "ik_pos_err_m": r.get("pos_err"), "jaw_tilt": r.get("jaw_tilt")}


@mcp.tool()
def gripper(side: str, action: str) -> dict:
    """Dex1 gripper: action 'open', 'close' (grasp with a bounded squeeze; keeps holding force), or 'stop'."""
    cmd = {"open": "open", "close": "grasp", "stop": "gripper_stop"}.get(action)
    if cmd is None:
        raise ToolError("action must be open, close or stop")
    g = arm(cmd, side=_side(side))
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
    return f"table at z={table_z:.3f} m; hand targets must stay at or above {table_z + TABLE_CLEARANCE:.3f} m"


@mcp.tool()
def park() -> str:
    """Lift both hands 15 cm, then move them to the ready pose [0.28, +-0.20, 0.05]. Needs enable."""
    arm("park", side="both", timeout_s=40)
    return "parked at the ready pose"


@mcp.tool()
def look(camera: str = "head_camera") -> Image:
    """One JPEG frame: 'head_camera' (the robot's eyes, looking down at the workspace) or 'waldo_webcam' (room view)."""
    if camera not in ("head_camera", "waldo_webcam"):
        raise ToolError("camera must be head_camera or waldo_webcam")
    with urllib.request.urlopen(f"{CAM}/frame/{camera}.jpg", timeout=5) as r:
        return Image(data=r.read(), format="jpeg")


if __name__ == "__main__":
    mcp.run()

"""MCP server that lets Claude operate Waldo's Unitree G1.

Runs on Waldo over stdio (Claude Code starts it through SSH). Talks to the
robot directly with unitree_sdk2py over DDS on the robot interface, and to
the camera viewer on Waldo for frames.

Site rule: no walking. There is no tool that sends a non-zero velocity.
"""
from __future__ import annotations

import os
import time
import urllib.request

from mcp.server.fastmcp import FastMCP, Image

IFACE = os.environ.get("WALDO_IFACE", "enp5s0")
CAMERA_URL = os.environ.get("WALDO_CAMERA_URL", "http://10.2.1.180:8766")
CAMERAS = ("head_camera", "waldo_webcam")

FSM_ZERO_TORQUE, FSM_DAMP, FSM_STAND_LOCK, FSM_BALANCE = 0, 1, 4, 500
BALANCED = (200, 500)
FSM_NAMES = {
    0: "zero_torque (limp)",
    1: "damp (soft, limp)",
    2: "squat",
    3: "sit",
    4: "stand_locked",
    200: "balance (old firmware id)",
    500: "balance",
    702: "lie_to_stand",
    706: "squat_to_stand",
}

GESTURES = {
    "release arm": 99,
    "two-hand kiss": 11,
    "left kiss": 12,
    "right kiss": 13,
    "hands up": 15,
    "clap": 17,
    "high five": 18,
    "hug": 19,
    "heart": 20,
    "right heart": 21,
    "reject": 22,
    "right hand up": 23,
    "x-ray": 24,
    "face wave": 25,
    "high wave": 26,
    "shake hand": 27,
}

mcp = FastMCP(
    "waldo",
    instructions=(
        "You operate Waldo, a Unitree G1 humanoid. Walking is not allowed. "
        "Call get_state before acting, look before and after gestures, and call stop "
        "whenever unsure. A person holds the hand controller (L1+A damps the robot)."
    ),
)

_loco = None
_arm = None


def _clients():
    """Connect to DDS on first use so the server starts even if the robot is off."""
    global _loco, _arm
    if _loco is None:
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize
        from unitree_sdk2py.g1.arm.g1_arm_action_client import G1ArmActionClient
        from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient

        ChannelFactoryInitialize(0, IFACE)
        loco = LocoClient()
        loco.SetTimeout(3.0)
        loco.Init()
        arm = G1ArmActionClient()
        arm.SetTimeout(10.0)
        arm.Init()
        _loco, _arm = loco, arm
    return _loco, _arm


def _fsm() -> int:
    loco, _ = _clients()
    code, value = loco.GetFsmId()
    if code != 0 or value is None:
        raise RuntimeError(f"could not read FSM state (sdk code {code})")
    return int(value)


def _transition(target: int, timeout: float) -> int:
    """Request an FSM state and wait until it reads back stable for 1 s."""
    loco, _ = _clients()
    code = loco.SetFsmId(target)
    if code != 0:
        raise RuntimeError(f"FSM {target} request rejected (sdk code {code})")
    deadline = time.monotonic() + timeout
    stable, current = 0, None
    while time.monotonic() < deadline:
        current = _fsm()
        stable = stable + 1 if current == target else 0
        if stable >= 10:
            return current
        time.sleep(0.1)
    raise RuntimeError(f"FSM {target} did not settle; last state {current}")


@mcp.tool()
def get_state() -> dict:
    """Current locomotion state of the robot. Call this first."""
    fsm = _fsm()
    return {
        "fsm_id": fsm,
        "fsm_name": FSM_NAMES.get(fsm, "unknown"),
        "balanced": fsm in BALANCED,
        "can_gesture": fsm in BALANCED,
        "walking_allowed": False,
    }


@mcp.tool()
def look(camera: str = "head_camera") -> Image:
    """One JPEG frame from a camera: 'head_camera' (robot's view) or 'waldo_webcam' (room)."""
    if camera not in CAMERAS:
        raise ValueError(f"camera must be one of {CAMERAS}")
    with urllib.request.urlopen(f"{CAMERA_URL}/frame/{camera}.jpg", timeout=5) as resp:
        return Image(data=resp.read(), format="jpeg")


@mcp.tool()
def stand_up(supported: bool) -> dict:
    """Bring the robot from limp to actively balancing (zero torque -> damp -> locked stand -> balance).

    supported: true only when the operator confirmed the robot is on its gantry
    or has both feet on the floor. Does nothing if already balancing.
    """
    if supported is not True:
        return {"ok": False, "error": "operator must confirm gantry or feet on floor (supported=true)"}
    start = _fsm()
    if start in BALANCED:
        return {"ok": True, "note": "already balancing", "fsm_id": start}
    if start not in (FSM_ZERO_TORQUE, FSM_DAMP, FSM_STAND_LOCK):
        return {"ok": False, "error": f"unexpected starting state {start}; nothing sent"}
    steps = []
    if start == FSM_ZERO_TORQUE:
        steps.append(_transition(FSM_DAMP, 5))
    if start != FSM_STAND_LOCK:
        steps.append(_transition(FSM_STAND_LOCK, 15))
    steps.append(_transition(FSM_BALANCE, 15))
    loco, _ = _clients()
    loco.SetVelocity(0.0, 0.0, 0.0, 1.0)
    return {"ok": True, "from": start, "steps": steps, "fsm_id": _fsm()}


@mcp.tool()
def gesture(name: str) -> dict:
    """Play a built-in upper-body gesture while balancing.

    name: one of release arm, two-hand kiss, left kiss, right kiss, hands up, clap,
    high five, hug, heart, right heart, reject, right hand up, x-ray, face wave,
    high wave, shake hand. Use 'release arm' to return arms to neutral.
    """
    action_id = GESTURES.get(name)
    if action_id is None:
        return {"ok": False, "error": f"unknown gesture {name!r}", "available": sorted(GESTURES)}
    fsm = _fsm()
    if fsm not in BALANCED:
        return {"ok": False, "error": "robot is not balancing; call stand_up first", "fsm_id": fsm}
    _, arm = _clients()
    code = arm.ExecuteAction(action_id)
    if code == 3104:
        return {"ok": False, "error": "response timed out; the gesture may already be running, do not repeat blindly"}
    return {"ok": code == 0, "gesture": name, "sdk_code": code}


@mcp.tool()
def stop() -> dict:
    """Stop all motion and hold a balanced stand. Always safe; use whenever unsure."""
    loco, _ = _clients()
    code = loco.StopMove()
    return {"ok": code == 0, "sdk_code": code}


@mcp.tool()
def damp() -> dict:
    """EMERGENCY ONLY. Motors go soft and limp; a standing robot that is not on the gantry will fall."""
    loco, _ = _clients()
    code = loco.Damp()
    return {"ok": code == 0, "sdk_code": code}


if __name__ == "__main__":
    mcp.run()

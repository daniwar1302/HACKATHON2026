#!/usr/bin/env python3
"""MCP server for the Unitree G1, using only the SDK's built-in high-level calls.

Runs on YOUR laptop. Each tool SSHes to waldo and calls unitree_sdk2py there:
nothing is installed on waldo, and no custom joint trajectories are sent, only
Unitree's own commands (FSM modes, gestures, WaveHand, TTS, LED).

  claude mcp add g1 -- uv run --with "mcp>=2.2" ~/HACKATHON2026/waldo-console/g1_mcp.py
  G1_DRY_RUN=1 ...   # every tool returns the SDK calls it WOULD make, robot untouched

Env: G1_HOST (operator@lab-pc.example), G1_KEY (~/.ssh/id_ed25519_lab),
     G1_PY (waldo's unitree conda python), G1_IFACE (enp5s0).
"""
import json
import os
import shlex
import subprocess
from typing import Literal

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError  # message reaches the agent; plain exceptions don't

HOST = os.environ.get("G1_HOST", "operator@lab-pc.example")
KEY = os.path.expanduser(os.environ.get("G1_KEY", "~/.ssh/id_ed25519_lab"))
PY = os.environ.get("G1_PY", "~/miniconda3/envs/unitree/bin/python")
IFACE = os.environ.get("G1_IFACE", "enp5s0")
DRY = os.environ.get("G1_DRY_RUN") == "1"

FSM = {"zero_torque": 0, "damp": 1, "stand": 4}
FSM_NAMES = {0: "zero_torque", 1: "damp", 2: "squat", 3: "sit", 4: "stand", 500: "start", 501: "start",
             702: "lie2stand", 706: "squat2stand", 801: "balance_stand"}
# Verified 2026-10-17 in fsm 4: WaveHand returns code 0 but the arm does NOT move; built-in gestures
# are rejected outright (code 7404). Both need the balance stand.
# Michael's robot notes: 801 = "standing, balancing (arm gestures and rt/arm_sdk work)", entered with the REMOTE.
WAVE_FSM = {500, 501, 801}
GESTURE_FSM = {500, 501, 801}

# Fixed remote program: runs a JSON list of [client, method, args, sleep_after] calls.
# Arguments travel as JSON data in argv, never spliced into code.
REMOTE = r"""
import json, sys, time
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient
from unitree_sdk2py.g1.arm.g1_arm_action_client import G1ArmActionClient
from unitree_sdk2py.g1.audio.g1_audio_client import AudioClient
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
ChannelFactoryInitialize(0, sys.argv[1])
kinds = {"loco": LocoClient, "arm": G1ArmActionClient, "audio": AudioClient, "switch": MotionSwitcherClient}
made, out = {}, []
for client, method, args, pause in json.loads(sys.argv[2]):
    if client not in made:
        made[client] = kinds[client](); made[client].SetTimeout(10.0); made[client].Init()
    out.append(getattr(made[client], method)(*args))
    time.sleep(pause)
print("G1JSON" + json.dumps(out, default=str))
"""

# Every SDK method the server may call. Anything else is rejected before SSH.
ALLOWED = {
    ("loco", "GetFsmId"), ("loco", "SetFsmId"), ("loco", "SetTaskId"), ("loco", "ShakeHand"),
    ("loco", "HighStand"), ("loco", "LowStand"),
    ("arm", "GetActionList"), ("arm", "ExecuteAction"),
    ("audio", "TtsMaker"), ("audio", "GetVolume"), ("audio", "SetVolume"), ("audio", "LedControl"),
    ("switch", "CheckMode"),
}

mcp = MCPServer("g1", instructions=(
    "Controls a Unitree G1 humanoid hanging in a harness, via Unitree's own SDK commands. "
    "Call status first. wave_hand and built-in gestures need fsm 801 (balance stand), which a human "
    "enters with the robot's remote; set_mode('stand') (fsm 4) is not enough. "
    "stop() drops the robot to damp: use it whenever anything looks wrong."))


LOCK, LOCK_WAIT_S, LOCK_BUSY = "/tmp/g1-mcp.lock", 60, 75


def sdk(calls, queue=True):
    """Run [(client, method, args, sleep_after), ...] on waldo; return each call's result.

    Commands queue on a flock on waldo, so every MCP session and the web UI take turns.
    Pure reads, and stop (queue=False), skip the queue so they never wait behind a gesture.
    ponytail: the lock covers one sdk() call, so a tool's state check and its command are two
    turns; another session can slip a mode change between them. Hold one lock per tool if that bites.
    """
    for c, m, _, _ in calls:
        if (c, m) not in ALLOWED:
            raise ToolError(f"{c}.{m} is not on the allow-list")
    if DRY:
        return [f"DRY RUN: {c}.{m}{tuple(a)}" + (f" then wait {p}s" if p else "") for c, m, a, p in calls]
    reads_only = all(m.startswith("Get") or m == "CheckMode" for _, m, _, _ in calls)
    lock = f"flock -w {LOCK_WAIT_S} -E {LOCK_BUSY} {LOCK} " if queue and not reads_only else ""
    # PY is operator config, left unquoted so waldo's shell expands its ~; everything else is quoted.
    remote = lock + PY + " " + " ".join(shlex.quote(x) for x in ["-c", REMOTE, IFACE, json.dumps(calls)])
    r = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", "-i", KEY, HOST, remote],
                       capture_output=True, text=True, timeout=LOCK_WAIT_S + 120)
    if r.returncode == LOCK_BUSY:
        raise ToolError(f"robot busy: another session held it for {LOCK_WAIT_S}s; try again")
    line = next((l for l in r.stdout.splitlines() if l.startswith("G1JSON")), None)
    if line is None:
        raise ToolError(f"waldo call failed (exit {r.returncode}): {(r.stderr or r.stdout)[-600:]}")
    return json.loads(line[len("G1JSON"):])


def current_fsm():
    code, fid = sdk([("loco", "GetFsmId", [], 0)])[0] if not DRY else (0, 4)
    return fid if code == 0 else None


@mcp.tool()
def status() -> dict:
    """Robot mode, FSM state, volume, and the gestures this robot offers (with which FSMs they need)."""
    if DRY:
        return {"dry_run": True, "calls": sdk([("switch", "CheckMode", [], 0), ("loco", "GetFsmId", [], 0),
                                                 ("audio", "GetVolume", [], 0), ("arm", "GetActionList", [], 0)])}
    mode, (fc, fid), vol, (ac, actions) = sdk([("switch", "CheckMode", [], 0), ("loco", "GetFsmId", [], 0),
                                               ("audio", "GetVolume", [], 0), ("arm", "GetActionList", [], 0)])
    gestures = [{"name": a["name"], **({"needs_fsm": a["fsm"]} if "fsm" in a else {}),
                 **({"needs_mode_machine": a["mode_machine"]} if "mode_machine" in a else {})}
                for a in (actions[0] if ac == 0 else []) if a["name"] != "release_arm"]
    return {"mode": mode[1].get("name") if mode[0] == 0 else None, "fsm": fid,
            "fsm_name": FSM_NAMES.get(fid, "other"), "wave_works_now": fid in WAVE_FSM,
            "gestures_work_now": fid in GESTURE_FSM,
            "volume": vol[1].get("volume") if vol[0] == 0 else None, "gestures": gestures}


@mcp.tool()
def set_mode(mode: Literal["stand", "damp", "zero_torque"]) -> str:
    """Change the FSM. 'stand' powers ALL joints including legs (robot is in a harness) and is
    needed for gestures. 'damp' is the soft safe stop. 'zero_torque' turns the motors off.
    Transitions pass through damp, as Unitree's examples do."""
    if not DRY and current_fsm() == FSM[mode]:
        return f"already in {mode}"
    calls = [("loco", "SetFsmId", [FSM["damp"]], 3)] if mode != "damp" else []
    calls += [("loco", "SetFsmId", [FSM[mode]], 3), ("loco", "GetFsmId", [], 0)]
    res = sdk(calls)
    return "\n".join(res) if DRY else f"fsm now {res[-1][1]} ({FSM_NAMES.get(res[-1][1], 'other')})"


@mcp.tool()
def gesture(name: str, hold_seconds: float = 4) -> str:
    """Run one of the robot's built-in arm gestures by name (see status for the list),
    hold it, then release the arms back to the controller."""
    if not 0 <= hold_seconds <= 15:
        return "refused: hold_seconds must be 0-15"
    if DRY:
        return "\n".join(sdk([("arm", "ExecuteAction", [f"<id of {name}>"], hold_seconds), ("arm", "ExecuteAction", [99], 0)]))
    fid = current_fsm()
    if fid not in GESTURE_FSM:
        return (f"refused: robot is in fsm {fid} ({FSM_NAMES.get(fid, 'other')}); built-in gestures need fsm 801 "
                "(balance stand), which a person enters with the robot's remote")
    code, actions = sdk([("arm", "GetActionList", [], 0)])[0]
    match = next((a for a in (actions[0] if code == 0 else []) if a["name"] == name), None)
    if match is None or name == "release_arm":
        return f"refused: unknown gesture {name!r}; call status for the list"
    if "fsm" in match and fid not in match["fsm"]:
        return f"refused: {name} needs fsm {match['fsm']}, robot is in {fid}"
    rc, _ = sdk([("arm", "ExecuteAction", [match["id"]], hold_seconds), ("arm", "ExecuteAction", [99], 0)])
    return f"{name}: code {rc}" + ("" if rc == 0 else " (non-zero: robot rejected it; mode_machine may not match)")


@mcp.tool()
def wave_hand(turn_around: bool = False) -> str:
    """The locomotion controller's own wave (LocoClient.WaveHand, sent as SetTaskId so the code comes back)."""
    fid = current_fsm()
    if fid not in WAVE_FSM:
        return f"refused: robot is in fsm {fid}; wave_hand needs fsm 801 (balance stand), entered with the robot's remote"
    rc = sdk([("loco", "SetTaskId", [1 if turn_around else 0], 0)])[0]
    return f"wave_hand: code {rc}" + ("" if rc == 0 else " (robot rejected it)")


@mcp.tool()
def say(text: str, speaker_id: int = 0) -> str:
    """Speak through the robot's speaker (Unitree TTS). speaker_id 0 is the default voice."""
    if not 0 < len(text) <= 300:
        return "refused: text must be 1-300 characters"
    return str(sdk([("audio", "TtsMaker", [text, speaker_id], 0)])[0])


@mcp.tool()
def set_led(r: int, g: int, b: int) -> str:
    """Set the head LED color, 0-255 per channel."""
    if not all(0 <= v <= 255 for v in (r, g, b)):
        return "refused: channels must be 0-255"
    return str(sdk([("audio", "LedControl", [r, g, b], 0)])[0])


@mcp.tool()
def set_volume(volume: int) -> str:
    """Speaker volume, 0-100."""
    if not 0 <= volume <= 100:
        return "refused: volume must be 0-100"
    return str(sdk([("audio", "SetVolume", [volume], 0)])[0])


@mcp.tool()
def stop() -> str:
    """Emergency soft stop: release any gesture and drop the robot to damp."""
    res = sdk([("arm", "ExecuteAction", [99], 0.3), ("loco", "SetFsmId", [FSM["damp"]], 0)], queue=False)
    return "\n".join(res) if DRY else "released arms and switched to damp"


if __name__ == "__main__":
    mcp.run()

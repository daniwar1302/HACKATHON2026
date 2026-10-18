# Waldo robot survey (2026-10-17)

## What Waldo is

`waldo-WaldOS` is an Ubuntu 24.04 workstation (16 cores, 15 GB RAM, RTX 5060 8 GB) wired to a
**Unitree G1 humanoid**. It is not the robot itself.

| Interface | Address | Purpose |
|-----------|---------|---------|
| `enp5s0` | lab-pc-lan.example | Ethernet to the G1 (robot answers at robot-controller.example / robot.example); DDS goes here |
| `wlp6s0` | camera.example | Wi-Fi LAN; camera viewer binds here |
| `tailscale0` | lab-pc.example | how we reach it |

## Control stack already on the box

- **Unitree SDK (Python)**: `~/unitree_sdk2_python`, CycloneDDS at `~/cyclonedds/install`
  (`source ~/.bashrc_g1`). High-level clients: `LocoClient` (FSM, velocity) and
  `G1ArmActionClient` (canned gestures).
- **`~/g1-astra`**: an LLM agent for the G1, already working. The parts:
  - `robot.py`: small backend interface (`fsm_id`, `set_fsm`, `set_velocity`, `arm_action`, `posture`)
  - `safety.py`: `Supervisor` clamps speeds, gates commands on FSM state, asks the operator to confirm each motion, and runs a stop watchdog
  - `tools.py`: tools the model gets: `get_state look say reset_pose stand_up lock_stand start stop move arm_action damp`
  - `agent.py`: the loop, **OpenAI** Responses API (`OPENAI_API_KEY` not set)
  - `server.py`: `g1 serve`, an HTTP/JSON robot server with a token
  - backends: `mock` (no hardware), `unitree` (DDS), `remote` (HTTP)
- **`~/stand.py`**: puts the robot into balance stand (FSM 0 → 1 → 4 → 500).
- **`~/killswitch.py`**: kills behaviour scripts, releases the arms, then StopMove and balance stand (`--damp` lets it go limp).
- **Cameras** (`~/g1-camera-viewer`, port 8766 on camera.example):
  - `GET /api/cameras`, `GET /frame/<id>.jpg`, `GET /stream/<id>.mjpg`
  - ids: `head_camera` (G1 head, live), `waldo_webcam` (room C920, live), `waldo_realsense` (unplugged)
- `~/mission-robotics`: agent plugins for this bay (WidowX arms are ready; the G1 plugin is still empty).

## State at survey time

- FSM **0 = zero torque**: powered on, motors limp.
- `g1 check` over DDS works from Waldo.

## Site rules / safety

- **No walking** on this shared robot: stand, balance and gesture only. `G1_ALLOW_WALK=0` enforces it.
- Someone keeps the hand controller in hand the whole time. **L1+A** damps the robot regardless of software.
- Use the gantry for the first runs. Keep operator confirmation on until the tools have been proven.
- Other people use this machine (other projects and an active Claude session), so coordinate before taking the robot.

## FSM reference

| id | state |
|----|-------|
| 0 | zero torque |
| 1 | damp |
| 4 | locked stand |
| 500 (200 on old firmware) | main operation / balance control |

## Agent plan

Reuse the g1-astra `Supervisor` as the only path to the robot, and put Claude in front of it:

1. **MCP server on Waldo** (`waldo-mcp`) that exposes the supervisor's tools plus `look(camera)`
   from the 8766 frame endpoint (head + room cameras). Claude Code on the Mac connects over SSH stdio,
   so no ports are opened and no keys are left on Waldo.
2. Skills/prompts for multi-step behaviours (greet, gesture sequences, "look and describe").
3. Later: our own Claude API agent loop (port `agent.py` to Anthropic tool use), voice in/out,
   and a scripted-skill library.

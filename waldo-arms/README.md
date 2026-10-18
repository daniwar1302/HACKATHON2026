# waldo-arms

MCP server that gives an agent the G1's **hands and Dex1 grippers**, through Michael's `arm_server.py` (Jetson,
ZMQ :5810). The arm server does the IK, the self-collision and table path checks, the workspace guard and the
watchdogs. This server exposes its commands as agent tools, adds a table floor, and serves camera frames so the
agent can check its own work.

| Tool | What it does |
|------|--------------|
| `status` | FSM, enabled/frozen, both hands' positions, grippers, table |
| `enable` / `release` | take the arms from Unitree's controller (2 s ramp) / hand them back (2 s fade) |
| `freeze` | hold both arms where they are |
| `check_reach(side, x, y, z)` | can the hand get there, and is the path clear? Never moves |
| `move_hand(side, x, y, z, seconds)` | move the grasp point (between the jaws) there, jaws level, and wait for it |
| `gripper(side, open\|close\|stop)` | Dex1; `close` grips with a bounded squeeze and keeps holding force |
| `set_table(z, …)` | declare the table top; paths stay above it and targets stay 3 cm over it |
| `park` | lift, then go to the ready pose `[0.28, ±0.20, 0.05]` |
| `look(camera)` | one JPEG from `head_camera` (the robot's view) or `waldo_webcam` |

**Seeing → acting (VLM-driven):**

| Tool | What it does |
|------|--------------|
| `look(camera, annotate=true)` | head image (640×480) with an 80 px pixel grid and the robot's own L/R hand positions projected on it (a live calibration check) |
| `point_to_xyz([[u, v], …], scale)` | image points → pelvis-frame xyz for `move_hand`. `scale`: `pixels`, `norm1000` (Gemini Robotics-ER style) or `norm1`. One depth frame for all points; robust median depth with a confidence score |

How it works: aligned head depth (teleimager :55556, raw z16) → pinhole intrinsics (asked live from teleimager :60000)
→ camera mount from Unitree's URDF (`d435_joint`: 43 cm up, 47.6° down) → live waist angles (`T_pelvis_torso` from
arm_server) → pelvis frame. Depth is fetched on demand (newest frame only, cached 150 ms), nothing streams in the
background; ~1–4 ms per call once a frame is cached. Math and self-check in `vision.py` (`python vision.py`).
Michael's measured `camera-pose-head.json` can replace the URDF mount once it's verified against the overlay.

Onboard VLM grounding (Cosmos-Reason2-2B on Waldo's GPU) lives in `vlm_point.py`; the full VLM → hand-task guide is [`docs/vlm-hand-tasks.md`](../docs/vlm-hand-tasks.md).

**Codifying what works into routines:**

| Tool | What it does |
|------|--------------|
| `recorded_steps` / `clear_recording` | every successful `move_hand`, `gripper`, `set_table`, `park` this session, as JSON steps |
| `save_routine(name, description, steps?)` | keep the recording (or steps the LLM writes) as `routines/<name>.json`, every step validated |
| `list_routines` / `show_routine(name)` | what's saved, and the full step list |
| `run_routine(name, dry_run=true)` | dry run checks every step's bounds and reach without moving; `dry_run=false` refuses up front if any step is unreachable, then replays in order and stops at the first refused step |

Step format (readable and hand-editable):
`{"op":"move_hand","side":"left","xyz":[0.30,0.20,0.10],"seconds":3}` · `{"op":"gripper","side":"left","action":"close"}` ·
`{"op":"wait","seconds":1}` · `{"op":"park"}` · `{"op":"set_table","z":-0.10}`.
So the loop is: try a task with the tools → when it works, `save_routine` → next time, `run_routine`.

Frame: pelvis, metres, **x forward, y left, z up**. Reachable box x 0.10–0.60, y ±0.55, z −0.35–0.45.
Refusals (unreachable, outside the box, below the table, server down) come back to the agent with the reason.

## Run

Needs the robot in FSM 500/501/801, `arm_server.py` running on the Jetson, and **teleop not running**
(see [`docs/quickstart.md`](../docs/quickstart.md) §3 and §4E).

On Waldo (reaches the Jetson and cameras directly):
```sh
claude mcp add waldo-arms -- ssh waldo@100.120.182.66 'cd ~/HACKATHON2026/waldo-arms && ~/.local/bin/uv run --with "mcp>=2.2" --with pyzmq --with numpy --with pillow server.py'
```
From a laptop with tunnels:
```sh
ssh -i ~/.ssh/id_ed25519_waldo -N -L 5810:192.168.123.164:5810 -L 8766:10.2.1.180:8766 \
  -L 55556:192.168.123.164:55556 -L 60000:192.168.123.164:60000 waldo@100.120.182.66
claude mcp add waldo-arms -e G1_ARM_ADDR=tcp://localhost:5810 -e G1_CAM_BASE=http://localhost:8766 -e G1_CAM_HOST=localhost \
  -- uv run --with "mcp>=2.2" --with pyzmq --with numpy --with pillow ~/HACKATHON2026/waldo-arms/server.py
```
For the brain (`waldo-live`), add it to `~/.waldo/brain.json` next to `g1`:
```json
"arms": {"command": "/home/waldo/.local/bin/uv", "args": ["run", "--with", "mcp>=2.2", "--with", "pyzmq", "--with", "numpy", "--with", "pillow", "/home/waldo/HACKATHON2026/waldo-arms/server.py"]}
```

A task then reads: `look` → decide a target → `check_reach` → `move_hand` (steps of ≤ 15 cm) → `look` →
`gripper close` → lift → … → `release`. Manual control stays available through the legacy
[`waldo-console`](../waldo-console/) controller page (teleop). Never run teleop and the arm server together.

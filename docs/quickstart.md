# Quickstart: everything on Waldo

The short path from zero to each piece of software in this repo. Details live in each folder's README,
[`robot-survey.md`](robot-survey.md) and [`g1-arm-commands.md`](g1-arm-commands.md).

**Safety first, every time:** someone holds the G1 hand controller (**L1+A = damp**) or stays within reach of a stop.
The robot hangs in its gantry. **No walking** (site rule). Waldo is shared, so check nobody else is driving it.

## 1. Get access (once)

All `.example` hostnames, `operator` accounts (including `/home/operator` paths), and `lab-pc` device names are placeholders. Request connection details privately from your lab administrator. Set `G1_HOST`, `G1_KEY`, `G1_ARM_ADDR`, `G1_CAM_BASE`, `G1_CAM_HOST`, and `WALDO_CAMERA_URL` for your deployment; the public defaults do not point to live hardware. Enter your own WebSocket endpoint in the browser controller. For optional full robot assets, set `WALDO_HOST` before running `FETCH_FULL_G1=1 ./fetch_assets.sh`.

1. **Tailscale:** accept your lab administrator's tailnet invite, then `sudo tailscale up` (or `sudo tailscale login` and pick the approved tailnet).
   `tailscale status` should list `lab-pc` (lab-pc.example).
2. **SSH key:** `ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519_lab` then `ssh-copy-id -i ~/.ssh/id_ed25519_lab.pub operator@lab-pc.example`.
3. **Robot Jetson** (from Waldo): `ssh operator@robot.example`. Press **Enter** at the `ros:foxy(1) noetic(2) ?` prompt.

| Machine | Address | Python for the Unitree SDK |
|---|---|---|
| Waldo (lab PC) | lab-pc.example (Tailscale), camera.example (Wi-Fi), lab-pc-lan.example (robot LAN, `enp5s0`) | `~/miniconda3/envs/unitree/bin/python` |
| G1 Jetson | robot.example | `~/miniconda3/envs/tv/bin/python` |

## 2. See the robot

```sh
ssh -i ~/.ssh/id_ed25519_lab -N -L 8766:camera.example:8766 operator@lab-pc.example
```
Open http://localhost:8766: **head camera** (the G1's only camera, a fixed RealSense D435i) plus the room webcam.
Head feed says *stalled*? On Waldo: `systemctl --user restart g1-camera-viewer`.

## 3. Put the robot in the right mode

| You want | FSM | Command |
|---|---|---|
| Soft / safe | 1 damp | console STOP / `g1-console` `stop`, `waldo-mcp` `damp`, or `SetFsmId(1)` |
| Gestures, teleop, arm_server, everything | **500** | 0 → 1 → 4 → wait a few seconds → 500 (or 801 from the remote) |

On Waldo:
```sh
~/miniconda3/envs/unitree/bin/python -c "import time; from unitree_sdk2py.core.channel import ChannelFactoryInitialize as I; from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient as L; I(0,'enp5s0'); c=L(); c.SetTimeout(10.0); c.Init(); c.SetFsmId(1); time.sleep(3); c.SetFsmId(4); time.sleep(6); print(c.SetFsmId(500)); time.sleep(4); print(c.GetFsmId())"
```
Expect `(0, 500)`. Still 4? Send `SetFsmId(500)` again. (`waldo-mcp`'s `stand_up(supported=true)` does the same with checks.)

## 4. Pick a way to drive it

Only **one** arm controller at a time: teleop, `arm_server`, and gesture tools all command the arms.

### A. Phone / agent loop: `waldo-live` + `waldo-ask` + `waldo-mcp`
```sh
ssh -t operator@lab-pc.example '~/HACKATHON2026/waldo-live/start.sh'     # bus, watcher, bridge, brain in tmux
ssh operator@lab-pc.example 'cd ~/HACKATHON2026/waldo-live && ./run.sh brain task "lift the red cup"'
```
Text **`call me`** to Waldo's iMessage line for phone command mode ("wave" → "go ahead?" → "yes").
Keys and settings: [`waldo-live/README.md`](../waldo-live/README.md).

### B. Claude Code with robot tools
```sh
claude mcp add waldo -- ssh operator@lab-pc.example /home/operator/HACKATHON2026/waldo-mcp/run.sh       # state, look, stand_up, gesture, stop, damp
claude mcp add g1-console -- uv run --with "mcp>=2.2" ~/HACKATHON2026/waldo-console/g1_mcp.py     # + say, LED, volume, queue
```

### C. Browser console (laptop)
```sh
uv run --with "mcp>=2.2" ~/HACKATHON2026/waldo-console/g1_webui.py
```
http://localhost:8780: status, modes, built-in gestures (need FSM 500/801), speech, LED, volume, **STOP**.
Rehearse with `G1_DRY_RUN=1`.

### D. Agent drives the hands: `waldo-arms` (the main path)
Robot in **500**, `arm_server` running (E below, steps 1–2), teleop **not** running. Then add the MCP server
([`waldo-arms/README.md`](../waldo-arms/README.md)) to Claude Code or the brain and give it a task:
`look` → `check_reach` → `move_hand` → `gripper` → … → `release`. VLM-driven targets: [`vlm-hand-tasks.md`](vlm-hand-tasks.md).
Teaching a real desk's devices and handing them to agents: [`real-cell-bringup.md`](real-cell-bringup.md).

### D2. Legacy manual mode: teleop (headset or browser controller page)
1. Robot in **500**, `arm_server` **not** running.
2. On Waldo:
   ```sh
   cd ~/xr_teleoperate/teleop && CYCLONEDDS_HOME=/home/operator/cyclonedds/install LD_LIBRARY_PATH=/home/operator/cyclonedds/install/lib \
     ~/miniconda3/envs/tv/bin/python teleop_hand_and_arm.py --input-mode=controller --arm=G1_29 --ee=dex1 \
     --img-server-ip=robot.example --network-interface=enp5s0 --frequency=30 --motion
   ```
   (add `--record --task-dir ~/datasets --task-name <name> --task-goal "<goal>"` to record demos)
3. **Controller**, pick one:
   * **Headset** (Quest/Pico): browser to `https://camera.example:8012/?ws=wss://camera.example:8012`, accept the certificate, tap Virtual Reality. **Keep off the thumbsticks** (they walk).
   * **Browser:** open https://lab-pc.example:8012 once and accept the certificate, then http://localhost:8780/controller (console from C running) → **Connect**. It streams the neutral pose.
4. Focus the teleop terminal and press **`r`**. The log must print `🚀start Tracking🚀`; the hands glide to the neutral pose
   (25 cm forward, 8 cm up), then follow the controller.
5. Browser keys: `1`/`2` hand · `W/S` `A/D` `R/F` move 2 cm · `Q/E` `T/G` wrist · hold `Space` close gripper · `H` home · **`Esc` = DAMP**.
6. Done: `q` in the teleop terminal. Never connect a headset and the browser page at the same time.

### E. Precise hand moves: Michael's `arm_server` (IK + collision checks)
1. Robot in **500**, teleop **not** running.
2. On the Jetson:
   ```sh
   cd ~/g1-grasp && CYCLONEDDS_HOME=/home/unitree/cyclonedds_ws/install/cyclonedds LD_LIBRARY_PATH=/home/unitree/cyclonedds_ws/install/cyclonedds/lib \
     ~/miniconda3/envs/tv/bin/python -u arm_server.py           # wait for: ready on :5810
   ```
3. On the laptop: `ssh -i ~/.ssh/id_ed25519_lab -N -L 5810:robot.example:5810 operator@lab-pc.example`, then
   `uv run --with pyzmq ~/HACKATHON2026/waldo-console/arms_face_test.py --dry` (no motion), then without `--dry`.
4. Every command (`move_ee`, `ik`, `check`, `grasp`, …): [`g1-arm-commands.md`](g1-arm-commands.md).
   Finish with `release` **before** Ctrl+C: Ctrl+C freezes the arms where they are.

## 5. Stop (gentlest first)

1. `release` (arm_server) / `q` (teleop): hand the arms back. `waldo-mcp` `stop`: StopMove and hold a balanced stand.
2. `freeze` (arm_server): hold in place.
3. **Damp**: console STOP / `g1-console` `stop`, the controller page's DAMP or `Esc`, both headset thumbsticks pressed in, `waldo-mcp` `damp`.
4. **Hand controller L1+A** / e-stop.

# Waldo (Unitree G1): arm commands and controls

Everything you can command on the G1's arms and Dex1 grippers, where it runs, and how to stop it.
Checked against the code on the robot on 2026-10-17.

---

## 0. Machines and access

| Machine | Address | Notes |
|---|---|---|
| **waldo** (base PC on the robot's LAN) | Tailscale `100.120.182.66`, robot LAN `192.168.123.222`, Wi-Fi `10.2.1.180` | user `waldo`. Unitree Python: `~/miniconda3/envs/unitree/bin/python` (miniconda base is broken for the SDK) |
| **Robot Jetson** (Orin NX, inside the G1) | `192.168.123.164`, reach it from waldo | user `unitree`. On login it asks `ros:foxy(1) noetic(2) ?`: press **Enter** to skip |
| DDS network interface on waldo | `enp5s0` | pass to every SDK call |

```bash
ssh waldo@100.120.182.66        # from anywhere on the tailnet
ssh unitree@192.168.123.164     # from waldo
```

## 1. Robot modes (FSM): read this first

Arm commands are **ignored** unless Unitree's controller is in an arm-capable state.

| FSM | Name | Arms | How to enter |
|---|---|---|---|
| 0 | zero torque (limp) | ignored | `SetFsmId(0)` |
| 1 | damp (soft): **the safe stop** | ignored | `SetFsmId(1)` |
| 4 | stand | custom `rt/arm_sdk` works; built-in gestures/WaveHand do **not** | `SetFsmId(1)`, wait 3 s, `SetFsmId(4)` |
| **500** | start / balance | **everything works**; this is what `arm_server` and teleop `--motion` want | from 4, wait a few seconds, then `SetFsmId(500)` |
| 801 | balance stand | everything works | the **remote** (Michael's usual mode) |

On waldo:

```bash
# read the mode
~/miniconda3/envs/unitree/bin/python -c "from unitree_sdk2py.core.channel import ChannelFactoryInitialize as I; from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient as L; I(0,'enp5s0'); c=L(); c.SetTimeout(5.0); c.Init(); print(c.GetFsmId())"

# 0 -> damp -> stand -> 500 (the robot hangs in its harness; legs power up)
~/miniconda3/envs/unitree/bin/python -c "import time; from unitree_sdk2py.core.channel import ChannelFactoryInitialize as I; from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient as L; I(0,'enp5s0'); c=L(); c.SetTimeout(10.0); c.Init(); c.SetFsmId(1); time.sleep(3); c.SetFsmId(4); time.sleep(6); print(c.SetFsmId(500)); time.sleep(4); print(c.GetFsmId())"

# STOP: damp
~/miniconda3/envs/unitree/bin/python -c "from unitree_sdk2py.core.channel import ChannelFactoryInitialize as I; from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient as L; I(0,'enp5s0'); c=L(); c.SetTimeout(5.0); c.Init(); print(c.SetFsmId(1))"
```

> If `Start()`/500 doesn't take (it stays at 4), it was sent too soon after standing. Send `SetFsmId(500)` again.

## 2. `arm_server.py`: full arm control (Michael's, on the Jetson)

`~/g1-grasp/arm_server.py` is a **ZMQ REQ/REP JSON server on port 5810**. It drives both arms through Unitree's
`rt/arm_sdk` blend (the robot keeps balancing) plus the two Dex1 grippers. It does the IK (Unitree's official
G1 + Dex1 URDF, pinocchio + pink), a self-collision and table path check on every move, a workspace guard, and
watchdogs (speed, tracking error, torque, feedback age) that **freeze** the arms on a fault.

### Start it (Jetson, robot in FSM 500 or 801)

```bash
cd ~/g1-grasp && CYCLONEDDS_HOME=/home/unitree/cyclonedds_ws/install/cyclonedds \
  LD_LIBRARY_PATH=/home/unitree/cyclonedds_ws/install/cyclonedds/lib \
  ~/miniconda3/envs/tv/bin/python -u arm_server.py          # --port 5810 --iface <dds iface>
```

You should see `[server] ready on :5810  fsm=(500, 0)  q=[...]`.

**Stopping it:** send `release` first. Ctrl+C makes the server **FREEZE and hold** ("firmware keeps the last
packet; arms stay where they are"). After a `release` that's harmless, because Unitree's controller already
has the arms.

### Talk to it

From a laptop, tunnel through waldo:

```bash
ssh -N -L 5810:192.168.123.164:5810 waldo@100.120.182.66
```

```python
import json, zmq
s = zmq.Context().socket(zmq.REQ); s.connect("tcp://localhost:5810")
def arm(cmd, **kw):
    s.send(json.dumps(dict(cmd=cmd, **kw)).encode()); return json.loads(s.recv())

arm("state")
tok = arm("prepare_enable")["token"]; arm("enable", prepared_token=tok, ramp_s=2.0)
arm("move_ee", side="left", xyz=[0.30, 0.15, 0.20], duration=3.0)
arm("grasp", side="left"); arm("open", side="left")
arm("release", ramp_s=2.0)
```

Every reply has `ok` (true/false). On error: `{"ok": false, "error": "...", "trace": "..."}`.

### Conventions

* **Frame:** pelvis frame, metres. **x** forward, **y** left, **z** up.
* **Workspace guard:** x 0.10–0.60, y −0.55–0.55, z −0.35–0.45. Targets outside are refused.
  Hanging hands sit around `[0.04, ±0.25, −0.21]`; z 0.42 is about chin height.
* **Joint vector `q` (14):** left 7 then right 7, radians, each in the order
  `shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll, wrist_pitch, wrist_yaw`.
* **Grasp point:** Dex1 jaw centre. Jaws close along the hand's y axis.
* **Gripper `q`:** motor radians. **5.0 = open**, 0.0 = closed, hard max 5.4.

### Command reference

**Lifecycle**

| cmd | Args (defaults) | What it does |
|---|---|---|
| `state` | | Everything: `arm` status (enabled, frozen, freeze_reason, weight, moving, fsm, q, dq, tau, q_cmd, q_goal, track_err_deg, …), both grippers, `fk`/`fk_right` hand positions, `T_pelvis_torso` |
| `prepare_enable` | | Builds the IK model while idle. Returns `{token, expires_s: 15}` |
| `enable` | `prepared_token`, `ramp_s` (2.0) | Takes the arms from Unitree's controller, ramping from the **measured** pose (no snap). Needs FSM 500/501/801 |
| `release` | `ramp_s` (2.0) | Fades back to Unitree's own arm hold. **The normal way to finish** |
| `freeze` | `reason` | Holds the current measured pose immediately (keeps control) |
| `unfreeze` | | Clears a freeze |
| `hard_release` | | Drops control at once (no fade) |
| `native` | | Snapshot of Unitree's own arm command, plus which sides are "native" |

**Motion** (all gated by the path check; add `force: true` to override it, which is logged; `block` defaults to true)

| cmd | Args (defaults) | What it does |
|---|---|---|
| `move_ee` | `side` (left), `xyz`, `rpy` [0,0,0] or `R` 3×3, `duration` (3.0), `jaws_horizontal` (true), `pitch`, `yaw`, `max_pos_err` (0.01), `max_jaw_tilt` (0.15), `dry` | **Move a hand to a 3D point.** Solves IK, then moves. `dry: true` = solve only |
| `ik` | same as `move_ee` | IK only, never moves. Returns `q7`, `q7_deg`, `pos_err`, `jaw_tilt` (`ok: false` if it didn't converge) |
| `move_q` | `q` (14), `duration` (3.0) | Both arms to a joint pose (smoothstep, velocity-capped) |
| `move_left` | `q7`, `duration` (3.0) | Left arm only, joint space |
| `trajectory` | `q` [n×14], `times` [n], `max_track_err_deg`, `label` | Timed joint trajectory, every sample checked. Non-blocking |
| `park` | `side` (both) | Lifts the hand(s) 15 cm, then goes to the ready pose `[0.28, ±0.20, 0.05]` |
| `fk` | `side`, `q7` (current) | Forward kinematics: `grasp_xyz`, `grasp_R`, `wrist_xyz` |
| `check` | `q` (14) **or** `q7` + `side`, `check_step_deg` (3) | Sweeps the path from the current command to `q` for self-collision and table. **The verdict is in `check.ok` / `check.reason`**. The top-level `ok` only means the request worked |
| `set_table` | `z`, `x` [lo,hi], `y` [lo,hi] (`z: null` clears) | Declares a table so paths keep the hands above it |

**Grippers** (`side` defaults to left)

| cmd | Args (defaults) | What it does |
|---|---|---|
| `open` | `q` (5.0), `speed` (1.5) | Open |
| `grasp` | `q` (0.0), `speed` (1.0), `squeeze` (0.08) | Close with a bounded squeeze; grip-hold keeps force as the object settles |
| `gripper` | `q`, `speed` (1.0), `squeeze` (0.08), `phase` | Go to any opening |
| `gripper_stop` | | Stop the gripper motor |
| `gripper_state` | | Position, torque, age, phase |

### Safety behaviour to know

* **Heartbeat:** if a move is running and no client has talked to the server for **2 s**, it freezes. Long
  moves from a script should keep polling `state`.
* **FSM watchdog:** if the robot leaves 500/501/801 while the server has the arms, it hard-releases.
* **Frozen arm still moving** → it fades back to Unitree's hold.
* **Arm limits:** speed 3 rad/s, velocity cap 2 rad/s, arm torque 20 Nm, wrist 5 Nm, tracking error 20° (35° while moving), 250 Hz control.
* **Never run `arm_server` and teleop at the same time.** Both command `rt/arm_sdk` and they will fight.

### Python API underneath (`g1_arm.py`, class `ArmSdk`)

`enable(ramp_s)`, `release(ramp_s)`, `hard_release()`, `freeze(reason)`, `unfreeze()`,
`move_to(q_goal, duration)`, `start_trajectory(q, times)`, `wait_move(tol_deg, timeout)`, `status()`,
`arm_q()` / `arm_dq()` / `arm_tau()` / `waist_q()`, `fsm()`, `use_side(side)`, `native_snapshot()`.
It's the same controller the server wraps. Use the server unless you're editing it.

## 3. VR teleop (`xr_teleoperate`, on waldo)

Robot in FSM 500, `arm_server` **not** running. On waldo:

```bash
cd ~/xr_teleoperate/teleop && CYCLONEDDS_HOME=/home/waldo/cyclonedds/install \
  LD_LIBRARY_PATH=/home/waldo/cyclonedds/install/lib ~/miniconda3/envs/tv/bin/python teleop_hand_and_arm.py \
  --input-mode=controller --arm=G1_29 --ee=dex1 --img-server-ip=192.168.123.164 \
  --network-interface=enp5s0 --frequency=30 --motion
# record demos: add  --record --task-dir ~/datasets --task-name <name> --task-goal "<goal>"
```

Headset browser (Quest / Pico / Vision Pro), accept the self-signed certificate, tap **Virtual Reality**:
`https://10.2.1.180:8012/?ws=wss://10.2.1.180:8012` (robot site) or `https://100.120.182.66:8012/?ws=wss://100.120.182.66:8012` (Tailscale).
A desktop browser at the same URL shows the head-camera view, but it can't drive the arms.

**Keyboard** (in the teleop terminal): `r` start following, `s` start/stop recording, `q` quit.
With `--ipc` instead: JSON `CMD_START` / `CMD_RECORD_TOGGLE` / `CMD_STOP` on `ipc://@xr_teleoperate_data.ipc` (local to waldo).

**What each controller input does** (`--input-mode=controller --motion`):

| Input | Effect on the robot |
|---|---|
| Controller **pose** (each hand) | That hand's target. Taken relative to the head, mapped to the robot's waist frame (+0.15 m x, +0.45 m z), solved with IK |
| **Trigger** (analog) | That side's Dex1 gripper: released = open, pulled = closed |
| Squeeze / grip | read and passed to the gripper controller, alongside the trigger |
| **Left thumbstick** | **walk** forward/back and sideways (capped 0.3 m/s). **Never: no walking is a site rule** |
| **Right thumbstick** (x) | **turn** (capped 0.3 rad/s). **Never: site rule** |
| **Both thumbsticks pressed in** | **Damp: emergency soft stop** |
| **Right A** | quit teleop |

Neutral pose (what teleop assumes before any controller data arrives): each hand about **0.25 m forward, ±0.15 m
out, 0.08 m up** from the waist, jaws level.

## 4. Built-in Unitree commands (no custom code)

[`waldo-console`](../waldo-console/): `g1_mcp.py` (an MCP server for agents) and `g1_webui.py` (a browser UI at `localhost:8780`, plus the virtual teleop controllers at `/controller`) wrap the SDK's own calls:
`status`, `set_mode`, `gesture` (22 built-ins, e.g. high_five, hug, clap; they need FSM 801/500), `wave_hand`,
`say` (TTS), `set_led`, `set_volume`, `stop` (release arms, then damp). Robot commands queue on a lock on waldo, so several
sessions can share the robot; `stop` skips the queue.

## 5. Cameras

* **Head:** RealSense D435i (serial 243722075296), the robot's only camera, fixed (the G1 has no neck motors).
  Served by `teleimager` on the Jetson (ZMQ 55555 RGB, 55556 depth). **No wrist cameras on these hands.**
* **Room:** Logitech C920 on waldo.
* Viewer: `http://<waldo 10.2.1.180>:8766/` or tunnel `ssh -L 8766:10.2.1.180:8766 waldo@100.120.182.66`.
  If the head feed says *stalled*, the Jetson's camera server restarted; on waldo run `systemctl --user restart g1-camera-viewer`.

## 6. Stopping things (in order of force)

1. `release` (arm_server) / `q` (teleop): hand the arms back gently.
2. `freeze` (arm_server): hold where they are.
3. Damp (`SetFsmId(1)`, web UI **STOP**, or both thumbsticks in): whole robot goes soft.
4. The **remote** / e-stop.

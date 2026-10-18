# waldo-sim

The desk-sorting G1 in MuJoCo, **speaking the real robot's protocols**, so `waldo-arms`, agents, `point_to_xyz`,
routines and VLMs run against it unchanged. Use it to make picking and sorting accurate before the robot.

**Model:** Unitree's official `g1_29dof_mode_15_with_dex1_1.urdf` (the model Michael's arm IK uses) with the real
Dex1-1 grippers: Unitree's finger meshes, 24.5 mm travel per finger (6–95 mm gap), 20 N per finger. Pelvis fixed
like the gantry; arms track commands through stiff position actuators with gravity compensation; grasps hold or slip
by contact and friction. Head camera at the URDF's D435 mount with the real focal length.

**Scene:** a 74 cm desk; containers from Google Scanned Objects (pen cup, wooden tray, desk caddy); items: two
markers, pen, pencil, eraser, sticky-note pad, sticker sheet (real sizes and masses) plus scanned tape, scissors and
crayons. Sorting: writing → pen cup, supplies → tray, tools → caddy. Items randomized per seed.

| Real service | Sim stand-in (own ports, no clash with robot tunnels) |
|---|---|
| arm_server (ZMQ :5810) | `:15810`: state, fk, enable, release, freeze, ik, check, move_ee (+ `R`, `contact`), park, set_table, open, grasp, gripper, + `sim_reset {seed, n_items}`, `sim_objects`, `sim_point {item, noise_px}` |
| teleimager config / depth | `:16000` / `:15556`: head intrinsics, metric depth (raw z16) |
| camera viewer (HTTP :8766) | `:18766`: `/frame/head_camera.jpg`, `/frame/waldo_webcam.jpg` |

## Files

| File | |
|---|---|
| `scene.py` | builds the model: URDF + actuators + Dex1 pads + head camera + desk + containers + items |
| `sim_server.py` | the protocol server: IK (jaw centre, level jaws or a given `R`), path collision sweep, grasping, scoring |
| `../waldo-arms/planner.py` | pick-and-place over the **arm_server protocol** (works on the real robot too): candidate grasps screened by IK, grip verified, recovery + retries, level-jaw tilted approach, contact stage, fingertip and rim grasps |
| `../waldo-arms/motion.py` | **collision-free motion planning** on a MuJoCo digital twin (RRT-Connect + shortcutting) and **smooth min-jerk trajectories** sent through arm_server's `trajectory` command |
| `media/waldo_sim_sort.gif` | ~30 s of the sim sorting in real time |
| `bench_sort.py` | N seeded desks, every item picked and dropped in its container, scored on where it lands |
| `fetch_assets.sh` | pulls the URDF/meshes from Waldo and the scanned objects (assets aren't in git) |
| `teach.py` | auto-teach + replay for fixed stations (IT desk and lab): search presses on the twin, keep the fastest the sim verifies, replay with phases (tap, press-and-hold, release, hover, poke) |
| `agent_lab.py` | the lab driven only through waldo-arms MCP tools, every step verified by sensing |
| `stations_sim.json`, `gauges_sim.json` | the sim cell's registered containers and gauge (one-time setup, as on a real cell) |
| `agent_cell.py`, `taught_cell.json`, `stations_cell.json`, `media/cell_agent_demo.gif` | the machine-tending + QC cell: agent loop, taught transfers, registered nest/bins, recording |
| `media/lab_agent_demo.gif` | room view + head camera: firmware update with a two-hand BOOT+RESET, then machine tending with a STOP on overpressure |

## Run

```sh
./waldo-sim/fetch_assets.sh
MUJOCO_GL=glfw uv run --with mujoco --with pyzmq --with pillow --with numpy waldo-sim/sim_server.py --speed 0   # 1 = real time
uv run --with mujoco --with pyzmq --with numpy waldo-sim/bench_sort.py --trials 10 --items 3 --truth   # or --noise-px 4 for the vision path
```
Any MCP client can drive it through `waldo-arms` with `G1_ARM_ADDR=tcp://localhost:15810 G1_CAM_BASE=http://localhost:18766
G1_CAM_HOST=localhost G1_DEPTH_PORT=15556 G1_CAMCFG_PORT=16000`. Rendering needs a GL context (`glfw` in a desktop session).

## Scenes

`--scene cell` is the industrial one: **machine tending + quality check**. Raw aluminium parts stand in an infeed
fixture; the robot loads one into the machine's nest, presses START (the machine only runs with a part seated:
that's the load check), watches the stack light, reads the QC lamp (green pass / red reject; one part per seed is
out of tolerance), and unloads the finished part into the PASS or REJECT bin. Parts are fixtured, so the transfers
are **taught** like a production robot's points: `teach.py teach` on a cell sim searches each transfer
(`load_slot0..2`, `unload_pass`, `unload_reject`) with the planner, keeps the first the sim verifies, and records its
exact commands; `run_task` replays them.

| cell, agent through MCP tools only (`agent_cell.py`), 3 px VLM-box noise | |
|---|---|
| taught transfers (how production cells run) | **15/15** parts machined, QC-read and binned correctly; ~60 s sim per part |
| free pick/place from VLM boxes (no teaching) | 2/9 |

Taught on a real cell by **demonstration** instead (`bench_teach_demo.py`): the planner only knows registered
containers/keep-outs (desk-only twin), a watched pick → place → home is recorded and kept if it worked: both
transfers kept on the first demonstration, replays 5/5 + 5/5 on other seeds.

What it took: the planner's twin must contain the fixtured parts (it was sweeping the arm through them), the
gripper opens only item width + 25 mm (fully open fingers straddled the nest walls), and the rest pose moved 9 cm
behind the desk edge (it parked the hands 7 mm under it, so every path out of rest grazed the desk). Recording:
`media/cell_agent_demo.gif`.

**An LLM agent runs it** (`runs/claude_cell_run.md`, `media/claude_runs_cell.gif`): Claude Code headless, given a
plain-English work order and only the waldo-arms MCP server, found the lamps with `look`, then for each part loaded
it, confirmed it was seated, started the cycle, polled the stack light and QC lamp with `read_light`, unloaded to the
matching bin and checked with `look`: **3/3 correct vs ground truth**, no faults, 37 turns, 216 s, $0.48. It also
flagged a real bug (status called FSM 500 not arm-capable; fixed).
```sh
MUJOCO_GL=glfw uv run --with mujoco --with pyzmq --with pillow --with numpy waldo-sim/sim_server.py --speed 1 --scene cell --port-offset 1000
cd waldo-sim && claude -p "$(cat work_order.txt)" --mcp-config mcp_cell.json --strict-mcp-config --allowedTools mcp__waldo-arms
```
(`mcp_cell.json` has this laptop's absolute paths; edit them.) In real-time mode the sim's clock now keeps running
while agents think, so machines finish cycles and routers boot between tool calls.

The others: `--scene sort` (desk + containers + items), `--scene devices` (IT desk: power strip, router, 2FA key,
start button, light switch), `--scene lab`:

* **Dev board (coding agents, hardware-in-the-loop):** BOOT and RESET buttons 13 cm apart (one per hand), status LED
  (green app / red hung / blue bootloader), 4-digit 7-segment display showing a pairing code. Seeds divisible by 3
  start hung. `sim_flash` is the USB side (esptool / dfu-util on a real board) and only works in the bootloader.
* **Machine tending (manufacturing):** START/STOP mushrooms, red/yellow/green stack light, 0-10 bar gauge facing the
  camera. A cycle runs ~10 s at ~6 bar; odd seeds have a fault that keeps climbing and trips at 9.5 bar unless
  someone presses STOP.

Results (2026-10-17):

| | |
|---|---|
| IT desk tasks, taught + replayed (`teach.py bench --rounds 5`) | **20/20** |
| lab tasks incl. two-hand bootloader combo, hover, poke (`teach.py bench --rounds 5`) | **30/30** |
| agent firmware update via MCP tools only (hung-board recovery, BOOT+RESET, flash, reset, code read off the display) | **6/6**, display codes read exactly |
| agent machine tending via MCP tools only (start, hover over STOP, poll the gauge, stop over 8 bar) | **6/6**; every fault stopped at 8.0 bar, before the 9.5 bar trip; gauge error mean ~0.1 bar |

Lessons: indicators must be emissive (lit by scene light they read wrong from their shaded side); the camera sees
a lamp's near face, not its axis; a gauge is read in image angles, so register it with where its min/max ticks
appear; teaching the fastest of several verified candidates plus a "hover" over STOP is what makes stopping fast.

## What it has taught us (2026-10-17)

* **Where to put things on the real desk:** at 74 cm desk height the head camera can't see closer than ~26 cm in front
  of the pelvis and the hands can't reach the desk beyond ~38 cm. **Objects 26–38 cm out**, keep **~7 cm clear around
  containers**; containers can sit further back (the hand only needs to reach above them).
* **Grasp geometry:** jaw centre is 15.2 cm ahead of the wrist. Straight-down grasps only reach 32 cm out; **tilt ~60°**.
  The approach must come in **perpendicular to the jaw direction** or the jaws tilt and one finger hits the desk first.
  Thin sheets need the **fingertips** aimed at them. Hands start below the desk top, so **lift clear before reaching**.
* **Motion:** joint-space jumps swing the elbow through the torso/desk; straight-line hand paths in ~3 cm steps don't.
* **Tape roll is 109 mm wide**, more than the Dex1's 95 mm opening: needs a rim pinch (not implemented yet).
* **Success rates on the physics desk (ground-truth positions, 30 items each):**

  | How the robot is driven | Result |
  |---|---|
  | `waldo-arms` MCP tools as shipped (`move_hand` above → down → `gripper close` → up) | **0/30**: every first move refused by the path check (a direct move out of the hanging pose would cross the desk edge) |
  | `planner.py` over the same arm_server protocol | **8/30 (27%) sorted into the right container** |

  The safety layer works (refuses instead of crashing); good movement comes from the planner. Next: expose the
  planner as MCP tools (`pick`, `place`) so agents get it directly.
* **Earlier sort rate: 10/30 (33%)** on the small-item desk, after motion planning + smooth trajectories (was 8/30).
* **Big-object desk (2026-10-17), 12 desks x 2 items, ground-truth positions** (`bench_sort.py --truth --trials 12 --items 4`):

  | Version | Sorted |
  |---|---|
  | planner as of f0aac34 (free-path carry, drop at the box centre) | 2/24 (8%) |
  | + no stacked spawns, tall items gripped 3 cm under the top, rim-graze tolerance, other arm sent to rest | 3/24 (12%) |
  | + **up-over-down carry** (item lifted clear of every rim, carried level, lowered in), **drop spot** = open floor point farthest from walls/dividers, fingers slide off before the arm moves | **8-12/24 (33-50%)** over three runs (RRT is randomized) |
  | same, rehearsed on a twin sim first (`--twin`, rehearse.py) | 11/24 (46%) |

  What was wrong, from recorded frames: the free-path carry swung the held box into the robot and knocked it to the
  floor (the twin doesn't model the held item); the caddy's centre is a divider; a tape roll hangs on the finger
  inside its core and got flung by the move to rest. Left: label reel (5/5 miss: grasp closes on nothing), tape slips
  out of the rim pinch during the carry, a few unreachable pre-grasps near the body.
* **Device stations, taught + replayed (`teach.py bench --rounds 5`): 20/20** (power-cycle router, approve 2FA key,
  start button, light switch). Teaching searches candidate presses on the twin and keeps the first that the sim
  verifies; housing/part contact excludes were the fix that made the power strip teachable.

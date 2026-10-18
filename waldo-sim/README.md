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
| `planner.py` | pick-and-place over the **arm_server protocol** (works on the real robot too): straight-line hand paths, lift-clear route, level-jaw tilted approach, wrist turned gradually, contact stage for the final descent |
| `bench_sort.py` | N seeded desks, every item picked and dropped in its container, scored on where it lands |
| `fetch_assets.sh` | pulls the URDF/meshes from Waldo and the scanned objects (assets aren't in git) |

## Run

```sh
./waldo-sim/fetch_assets.sh
MUJOCO_GL=glfw uv run --with mujoco --with pyzmq --with pillow --with numpy waldo-sim/sim_server.py --speed 0   # 1 = real time
uv run --with pyzmq --with numpy waldo-sim/bench_sort.py --trials 10 --items 3 --truth   # or --noise-px 4 for the vision path
```
Any MCP client can drive it through `waldo-arms` with `G1_ARM_ADDR=tcp://localhost:15810 G1_CAM_BASE=http://localhost:18766
G1_CAM_HOST=localhost G1_DEPTH_PORT=15556 G1_CAMCFG_PORT=16000`. Rendering needs a GL context (`glfw` in a desktop session).

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
* **Current sort rate: 8/30 (27%)** with ground-truth positions. Main remaining failures: wrist-turn collisions with the
  torso (elbow in), grasps that miss thin items, the tape.

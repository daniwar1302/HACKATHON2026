# waldo-sim

A MuJoCo G1 that **speaks the real robot's protocols**, so `waldo-arms`, agents, `point_to_xyz`, routines and VLMs run
against it **unchanged**. Use it to rehearse tasks and to optimize them on measured success rates before the robot.

| Real service | Sim stand-in (own ports, no clash with robot tunnels) |
|---|---|
| arm_server (ZMQ :5810) | `:15810`: state, enable, release, freeze, ik, check, move_ee, park, set_table, open, grasp, + `sim_reset {seed}`, `sim_objects` |
| teleimager config / depth | `:16000` / `:15556`: head camera at the URDF d435 mount, metric depth (raw z16) |
| camera viewer (HTTP :8766) | `:18766`: `/frame/head_camera.jpg`, `/frame/waldo_webcam.jpg` |

Model: MuJoCo Menagerie `unitree_g1`, pelvis fixed like the gantry, arms kinematic (commanded angles exactly),
table + three cubes with physics, cubes randomized per seed. IK = damped least squares on the jaw centre; path check =
contacts of each arm with the body, the other arm and the table along the joint path. **Grasping is simplified**: a
closed gripper within 5 cm of a cube carries it. Poses, reach limits and collisions are real; grip physics isn't.

## Run

```sh
MUJOCO_GL=glfw uv run --with mujoco --with robot_descriptions --with pyzmq --with pillow --with numpy waldo-sim/sim_server.py --speed 0
#   --speed 1 = real time (for watching), 0 = as fast as possible (for benchmarks)
uv run --with "mcp>=2.2" --with pyzmq --with numpy --with pillow waldo-sim/bench_pick.py --trials 20
```
Point any MCP client at the sim:
```sh
claude mcp add waldo-arms-sim -e G1_ARM_ADDR=tcp://localhost:15810 -e G1_CAM_BASE=http://localhost:18766 -e G1_CAM_HOST=localhost \
  -e G1_DEPTH_PORT=15556 -e G1_CAMCFG_PORT=16000 -- uv run --with "mcp>=2.2" --with pyzmq --with numpy --with pillow ~/HACKATHON2026/waldo-arms/server.py
```
Rendering needs a GL context: `glfw` works in a desktop session. Headless boxes need EGL or OSMesa.

## What it has measured so far (2026-10-17, laptop, 20 trials in 4.8 s)

* **Vision chain is consistent**: `point_to_xyz` on a cube lands within ~1-2 mm in x/y of the true position, at the
  cube's top surface. This is the reference the real robot's calibration check (the `look` overlay) is held to.
* **Reach**: at table height (z ≈ −0.10 m in the pelvis frame) the hands only reach to x ≈ 0.35 m. **Put the real table
  close: its near edge ~20 cm and objects 25–35 cm in front of the pelvis.**
* **Pick baseline: 12/20 (60%)**. Failures: objects half out of the head camera's view (the point lands on the table:
  4), the arm path hitting the torso near the midline (3), not visible (1). Those are the next things to optimize.

`bench_pick.py` uses a colour threshold as the target finder; swap in `waldo-arms/vlm_point.ground` to benchmark the VLM.

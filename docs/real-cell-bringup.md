# Real cell bring-up: teach the desk, then let agents run it

Goal: the sim-proven skills (taught device presses, pick/place, sensing, receipts) on the real G1 in about 30 minutes.
Everything is a `waldo-arms` MCP tool, so Claude Code (or the onboard brain) can do the steps; a human watches every
first motion with a hand near STOP.

## 0. Bring the robot and servers up
1. Robot in **FSM 500** (0 → 1 → 4 → wait → 500), teleop **not** running ([quickstart](quickstart.md) §3).
2. `arm_server` on the Jetson ([quickstart](quickstart.md) §E steps 1–2): wait for `ready on :5810`.
3. Laptop tunnels (arm, depth, camera config, camera viewer):
   ```sh
   ssh -i ~/.ssh/id_ed25519_waldo -N -L 5810:192.168.123.164:5810 -L 55556:192.168.123.164:55556 \
       -L 60000:192.168.123.164:60000 -L 8766:localhost:8766 waldo@100.120.182.66
   ```
4. MCP server on the laptop (needs `waldo-sim/fetch_assets.sh` once: the planner's twin uses the URDF):
   ```sh
   G1_ARM_ADDR=tcp://localhost:5810 G1_CAM_BASE=http://localhost:8766 G1_CAM_HOST=localhost \
   G1_JOINT_SPEED=0.6 G1_TAUGHT_DIR=~/HACKATHON2026/waldo-arms/cell \
   uv run --with "mcp>=2.2" --with mujoco --with pyzmq --with numpy --with pillow waldo-arms/server.py
   ```
   `G1_JOINT_SPEED=0.6` halves replay speed until the cell is trusted. Keep `G1_TAUGHT_DIR` separate from the sim's
   taught files (sim millimetres are not the real desk's).

## 1. Check the eyes
* `status` (FSM, arm_server up), `look` (annotated): the **L/R dots must sit on the real grippers**. If they're off,
  stop here: the camera model or the waist angles are wrong, and every point → xyz will be off by the same amount.
* `set_table(z)` with the desk top in the pelvis frame: `point_to_xyz` on a bare patch of desk gives it.

## 2. Teach each device (one per minute)
For each button / switch / key:
```
teach_control("router_off", u, v, push="down", depth_mm=6, indicator=[u_led, v_led])
```
* (u, v): where the fingertip should land, from `look` (or the VLM's point). push: `down` for buttons, keys and
  rocker ends; `forward` for a switch facing the robot.
* Start from `home("both")`: the press is replayed from where it was taught.
* The result says which hand, and whether the indicator changed. Watch it. If it did the job:
  `save_control("router_off")`. If not: teach again with `side=`, a different `depth_mm` (too shallow: no click;
  too deep: arm_server refuses), or a better point.
* Multi-step tasks (power cycle, bootloader) are the taught controls in order: `run_task` with phases (`press` to
  hold BOOT, `tap` RESET with the other hand, `release`). Save the sequences you need in `teach.py`'s task tables.

## 2b. Fixtured pick/place (machine tending, kitting)
In the sim, transfers between fixed places (infeed pocket → machine nest → pass/reject bin) are taught once and
replayed (15/15 vs 2/9 for free pick/place). On the real cell, teach them supervised: `pick(u, v)` → `place(...,
container=...)` → `home`, watched, slow (`G1_JOINT_SPEED`); a tool that records a watched success as a transfer, like
`save_control`, is the next step (not built yet). Always start tasks from `home("both")`.

## 3. Register what the agent will read and fill
* Containers: `register_container("tray", u, v, rim_z=...)` (point at the open spot to drop into; give `rim_z` if a
  tall wall is outside the image).
* Gauges: `register_gauge("pressure", box, 0, 10, min_deg, max_deg, unit="bar", alarm_above=8)` with the angles where
  the 0 and max ticks appear in the image (clockwise from straight up).
* Lights and displays need no registration: `read_light([[u, v]])`, `read_display(box)`.

## 4. Hand it to the agent
Act → sense → confirm, every time:
* `run_task("router_off")` → `read_light` (LED dark) → `run_task("router_on")` → wait → `read_light` (green).
* `pick(box=...)` → `place(side, container="tray")` → `home(side)` → `look`.
* `receipts()` / `show_receipt(id)`: the before/after frames of everything the robot did.

## Sim reference (what to expect)
Teach by pointing, 3 px noise: 12/12 controls on the first candidate (~1.3 s each). Taught replays: IT desk 20/20,
lab 30/30. Agent flows through MCP only: firmware update 6/6, machine tending 6/6. Big-object sorting ~44%.

## Handled for the real server
* `trajectory` is non-blocking there: skills poll `state` until it finishes (keeping the 2 s heartbeat alive) and
  fail if the arm froze. `contact` (sim-only) is dropped; `sim_wait` becomes a sleep. Checked by
  `waldo-arms/test_real_protocol.py` against a fake server.

## Unverified on hardware
* Whether arm_server's path check accepts the last millimetres of a press (it knows the table plane, not the
  device); if it refuses, re-teach with a smaller `depth_mm` or a press point a little higher.
* Camera calibration on the real mount (step 1 checks it).
* Indicator colours under room light (`read_light` thresholds were tuned on rendered LEDs).

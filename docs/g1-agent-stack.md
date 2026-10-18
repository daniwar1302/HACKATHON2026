# G1 agent stack: agents run physical work cells through MCP

**An AI agent gets a humanoid's two arms as a set of verified skills: load the machine, read the gauge, flash the
board, stop on overpressure, and prove what it did.** Built on a Unitree G1 with Dex1 grippers (hanging in a gantry,
arms only) and a MuJoCo digital twin of it that speaks the robot's real protocol.

## What agents need from a robot, and what we built for each

| An agent is... | So it needs | Built |
|---|---|---|
| slow (seconds per decision) | limits watched faster than it can poll | `guard(gauge, above, then_task)`: a background interlock, ~0.2 s reaction |
| blind between calls | cheap, exact sensing | `read_light`, `read_display` (7-segment), `read_gauge`: ~50 ms each, no VLM |
| unreliable at raw motion | skills that are planned, collision-checked and verified | `pick`, `place`, `home`, `run_task` over a MuJoCo twin + arm_server's checks |
| new to every cell | to teach stations cheaply | `teach_control` (point at a button), `record_transfer` (one watched demonstration), `register_keepout` / `register_container` |
| accountable to humans | proof of what it did | receipts: before/after head-camera frames + args + result for every physical action |

Fixtured parts and taught moves are how production cells reach high reliability; the agent works above them, deciding,
sensing, verifying and reacting.

**Videos** ([`waldo-sim/media/demos/`](../waldo-sim/media/demos/)), all unscripted Claude runs in the MuJoCo sim, 720p
room view + the robot's head camera, captioned with the agent's own reasoning and tool calls as they happened:
* [`all_highlights.mp4`](../waldo-sim/media/demos/all_highlights.mp4) (2 min): every demo's key actions and result.
* per demo (`it`, `rack`, `qa`, `returns`, `lab`, `cell`): `<demo>_highlights.mp4` (8-31 s, each action at ~4-16x)
  and `<demo>_full.mp4` (3x, the whole run).

## Demos (one command each, in the sim)

```sh
./waldo-sim/demo.sh cell   # machine tending + QC
./waldo-sim/demo.sh lab    # hardware-in-the-loop firmware update + guarded machine cycle
./waldo-sim/demo.sh it     # IT ticket: "the office internet is down"
./waldo-sim/demo.sh rack   # SRE page: datacenter remote hands
./waldo-sim/demo.sh qa     # QA agent tests an app on a physical tablet
./waldo-sim/demo.sh returns # logistics agent triages warehouse returns
```

* **Machine tending + QC (cell):** load a raw part from the infeed fixture into the machine, press START, watch the
  stack light, read the QC lamp, unload to the PASS or REJECT bin. Claude Code, given a plain-English work order
  and only the MCP tools, found the lamps with `look` and processed all 3 parts: **3/3 vs ground truth**, no faults
  (37 turns, 216 s, $0.48). [log](../waldo-sim/runs/claude_cell_run.md) · [gif](../waldo-sim/media/claude_runs_cell.gif)
* **Test bench (lab):** a hung dev board and a faulty machine. Claude recovered the board, had the robot hold BOOT
  and tap RESET, flashed v2 from its own shell, read the new pairing code off the 7-segment display, then ran the
  machine with a guard armed: the guard pressed STOP at 8.3 bar, before the 9.5 bar trip.
  [log](../waldo-sim/runs/claude_lab_run.md) · [gif](../waldo-sim/media/claude_runs_lab.gif)
* **IT ticket (devices desk):** "the office internet has been down since this morning." Claude found the router and
  its power strip with `look`, read power green / internet amber, power-cycled the router, waited and confirmed both
  LEDs green, touching nothing else on the desk (13 turns, 89 s, $0.28; ground truth: internet up).
* **SRE page (datacenter remote hands):** "storage on db-07 degraded". Claude read the server's panel (Er42), found
  the drive bay with the red LED, reseated just that drive per the runbook, and declined to hard-reset a host that
  was only degraded; verified all LEDs green (13 turns, 41 s, $0.26; ground truth: healthy).
* **App QA on a physical device:** a test tablet running a shop app. Claude tapped through the purchase flow with the
  robot's fingertip (`tap(u, v)`, no teaching), read every screen, and filed a FAIL ticket: test card declined with
  E402, with the screen at each step and the receipt of each tap (15 turns, 108 s, $0.30).
* **Warehouse returns:** Claude put each returned item on the scanner, read its disposition code off the console,
  and binned it (restock / refurbish / return to vendor): 3/3 correct, each unload verified (30 turns, 184 s, $0.40).
* **The agent found two design flaws we had built in:** START and STOP taught on the same hand (it refused to run
  the cycle unguarded and asked for START on the other hand), and a blocking guard that couldn't cover a start (it
  reported its own failed run and proposed a background guard). Both fixed; the third run passed.

## Numbers (sim, ground truth)

| | |
|---|---|
| Machine tending + QC, scripted agent over MCP, taught transfers | **15/15** parts (free pick/place without teaching: 2/9) |
| Firmware update (hung-board recovery, BOOT+RESET, flash, read code) | **6/6** |
| Machine tending with overpressure guard | **6/6**, every fault stopped at ~8 bar, gauge error ~0.1 bar |
| IT desk: power-cycle router, 2FA key, start button, light switch (taught) | **20/20** |
| Datacenter runbook: hard reset (hold 4 s), reseat drive 0 / 1 (taught) | **15/15** |
| Warehouse returns triage, scripted agent over MCP | **12/12** |
| Device QA: purchase flow via `tap`, 0 / 4 / 8 px pointing noise | **25/25 · 25/25 · 24/25** |
| Lab buttons incl. two-hand BOOT+RESET (taught) | **18/18** |
| Teach a button by pointing (3 px noise, desk-only twin) | **12/12**, ~1.3 s each |
| Teach a pick/place by one demonstration, replay on new seeds | **5/5 + 5/5** |
| 7-segment reads (tight/normal box) | **23/24** |
| Unfixtured clutter sorting (desk, big objects) | ~44%: why cells fixture parts |

## On the real robot

Everything goes through Michael's `arm_server` protocol (IK + path checks on the Jetson); the skills handle its
non-blocking trajectories, heartbeat and freezes (`waldo-arms/test_real_protocol.py`). Bring-up and teaching a real
desk: [real-cell-bringup.md](real-cell-bringup.md). The robot was offline overnight, so the stations above have not
yet been taught on hardware.

## Where things are

`waldo-arms/server.py` (MCP: 37 tools), `planner.py` (grasp/place/press), `motion.py` (twin, RRT-Connect, min-jerk),
`vision.py` (depth segmentation, camera model), `rehearse.py`; `waldo-sim/` (MuJoCo sim on the real protocol, scenes
`cell` · `lab` · `devices` · `sort`, `teach.py`, agents and benchmarks, `demo.sh`).

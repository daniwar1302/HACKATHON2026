# Claude runs the hardware test bench (2026-10-18)

Claude Code headless with only the waldo-arms MCP server (`mcp_lab.json`) plus one shell command (`usb_flash.py`, the
sim's stand-in for esptool), real-time lab sim, seed 3: the dev board starts **hung** and the machine is **faulty** (its
pressure climbs past 8 bar and trips at 9.5 unless STOP is pressed). Work order: `work_order_lab.txt`.

## Three runs, two design flaws found by the agent

1. **Firmware right (v2, code 0856 = ground truth); refused to run the cycle.** It noticed START and STOP were both
   taught on the right hand, so parking a finger over STOP and pressing START conflict: an unguarded gap. It asked
   for START to be taught on the left hand. Done (`teach.py`: machine_start is left-handed in the lab and the cell).
2. **Firmware right; the machine tripped (9.5 bar).** It sent `guard` and `start_cycle` together expecting them to
   run concurrently; MCP calls run one at a time, so the (blocking) guard watched an idle machine, then the cycle
   ran unguarded. It said so plainly and proposed a background guard. Done: `guard` now arms in the background
   and returns at once; `guard_status(id, wait_s)` reports; the arm socket is locked for the guard's STOP.
3. **Both jobs done, ground truth matches:** board v2 running, code 0856; guard armed before START fired at a read
   8.06 bar and pressed STOP (sim: stopped at 8.3 bar, no trip). It then recommended triggering at ~7.5 bar, since
   a guard at the limit reacts after crossing it.

## Run 3 tool calls

- `ToolSearch` {"query": "select:mcp__waldo-arms__status,mcp__waldo-arms__enable,mcp__waldo-arms__look,mcp__waldo-arms__read_light,mcp__waldo-arms__read_display,mcp__waldo-arm
- `Read` {"file_path": "/home/lain_iwakura/HACKATHON2026/waldo-sim/usb_flash.py"}
- `status` {}
- `list_controls` {}
- `look` {}
  - _Status LED is the small square near (316, 389). Display shows dashes. Reading LED and display._
- `read_light` {"points": [[316, 389]]}
- `read_display` {"box": [280, 406, 355, 438]}
  - _Board is hung (red LED, "----"). Entering bootloader recovers it._
- `run_task` {"name": "enter_bootloader"}
- `read_light` {"points": [[316, 389]]}
- `read_display` {"box": [280, 406, 355, 438]}
  - _Blue LED, display "boot" (low decoder confidence, so confirming visually)._
- `look` {}
  - _Confirmed: bootloader. Flashing._
- `Bash` {"command": "uv run --no-project --with pyzmq usb_flash.py --offset 1100", "description": "Flash firmware over USB to the dev board"}
- `run_task` {"name": "reset_board"}
- `read_light` {"points": [[316, 389]]}
- `look` {}
- `read_display` {"box": [279, 405, 355, 437]}
  - _Decoder says "0856", matching the image, but confidence is low. Trying a tighter box._
- `read_display` {"box": [282, 408, 353, 435]}
  - _A tighter box cuts off segments. The first box's "0856" matches the image, so I'll take that. Now the machine: checking the gauge first._
- `read_gauge` {"name": "pressure"}
  - _Gauge reads about 0 bar at rest. Parking the finger over STOP._
- `run_task` {"name": "arm_stop"}
- `look` {}
  - _The right hand is parked over the red STOP. Arming the guard._
- `guard` {"gauge": "pressure", "above": 8, "then_task": "stop_machine", "timeout_s": 90}
- `run_task` {"name": "start_cycle"}
- `guard_status` {"guard_id": "g1", "wait_s": 60}
  - _The guard fired at 8.06 bar. Checking that the machine actually stopped._
- `read_gauge` {"name": "pressure"}
- `look` {}
  - _Pressure is back to 0 and the stack light is idle. Returning the arm._
- `run_task` {"name": "disarm_stop"}
- `status` {}
  - _Both jobs ran, but the pressure went slightly over the limit. The peak was **8.06 bar**, not under 8._

## Run 3 final report (28 turns, 94 s, $0.41)

Both jobs ran, but the pressure went slightly over the limit. The peak was **8.06 bar**, not under 8.

- **Pairing code: 0856.** The read I accepted said "0856" and matches what the camera shows. The display reader marked it low-confidence, though, and a tighter box gave "?85E" because it clipped the segments. So 0856 rests on the camera image, not a clean decode.
- **Firmware: v2 is installed and running.** The board was hung when I started (red LED, display "----"). I put it in its bootloader and confirmed the LED was blue and the display read "boot". The flash script reported `firmware v2 written, verify OK`. After `reset_board` the LED is green and the display shows the code.
- **Peak pressure: 8.06 bar.** It climbed -0.08 → 3.77 → 6.03 → 7.79 → 7.95 → 8.06 in about 15 s.
- **How the cycle ended:** I parked a finger over STOP and armed the guard (above 8.0, run `stop_machine`) before starting. It fired at 8.06 bar and pressed STOP, which cut the cycle short. Afterwards the gauge read 0 bar and the camera showed the machine idle. The arm is back at rest and both hands are in their resting position.

The guard only checks the gauge 5 times a second and acts after the reading has already passed 8.0. With pressure rising about 0.2 bar per reading near the top, a trigger at 8.0 can't keep the peak under 8. For a real "stay below 8" run, set the trigger lower, around 7.5.

# Claude runs the machine-tending cell (2026-10-18)

Claude Code headless (`claude -p`), only the waldo-arms MCP server (`mcp_cell.json`, `--strict-mcp-config`), real-time cell sim.
Ground truth afterwards: part0 pass -> pass_bin, part1 reject -> reject_bin, part2 pass -> pass_bin. **3/3 correct, no faults.**

## Work order

```
You are the operator agent for a machine-tending cell: a Unitree G1's arms, driven through the waldo-arms tools.

Work order: machine all 3 raw parts waiting on the infeed fixture and sort each into the PASS or REJECT bin by its QC result.

Cell SOP:
- Start from home (home "both"). Fixtured transfers are taught: load_slot0 / load_slot1 / load_slot2 (infeed pocket -> machine nest), unload_pass / unload_reject (nest -> bin). Start a cycle with run_task("start_cycle"). Use run_task for all motion; do not use pick, place or move_hand.
- Stack light: green = running, yellow = idle / cycle done, red = fault. A cycle takes about 8 s. The machine only starts with a part seated in the nest.
- The QC lamp is the small lamp on the machine control panel, between the START and STOP buttons. After a cycle it lights green (pass) or red (reject) while the part is still in the nest; it goes dark when the nest is empty.
- Find the lamps once with look, then use read_light to check them (fast). Verify every step by sensing before moving on. If something goes wrong, stop and report rather than guess.

When finished, report per part: QC result, which bin it went to, and the receipt id of its unload.
```

## Tool calls

- `ToolSearch` {"query": "select:mcp__waldo-arms__status,mcp__waldo-arms__enable,mcp__waldo-arms__home,mcp__waldo-arms__look,mcp__waldo-arms__read_light,mcp__waldo-arms__run_task,mcp__waldo-arms__list_controls,mcp__waldo-arms__receipts,mcp__waldo-arms__show_receipt,mcp__waldo-arms__point_to_xyz", "max_results": 10}
- `status` {}
- `list_controls` {}
- `home` {"side": "both"}
- `look` {}
- `read_light` {"points": [[582, 198], [570, 229], [560, 250], [550, 350], [360, 405]]}
  - _Idle (yellow), QC lamp dark, nest empty. Loading part 1._
- `run_task` {"name": "load_slot0"}
- `look` {}
  - _Part seated in the nest (front-right pocket emptied). Starting cycle._
- `run_task` {"name": "start_cycle"}
- `read_light` {"points": [[582, 198], [570, 229], [560, 250], [550, 350]]}
  - _Green = running. Polling for cycle end._
- `read_light` {"points": [[582, 198], [570, 229], [560, 250], [550, 350]]}
  - _Cycle done, QC = PASS. Unloading to PASS bin._
- `run_task` {"name": "unload_pass"}
- `read_light` {"points": [[570, 229], [560, 250], [550, 350]]}
- `look` {}
  - _Part 1 in green PASS bin, nest empty, QC dark. Grabbing receipt id, then part 2._
- `receipts` {"last": 1}
- `run_task` {"name": "load_slot1"}
- `look` {}
  - _Part 2 seated. Starting cycle._
- `run_task` {"name": "start_cycle"}
- `read_light` {"points": [[582, 198], [570, 229], [560, 250], [550, 350]]}
- `read_light` {"points": [[582, 198], [570, 229], [560, 250], [550, 350]]}
- `read_light` {"points": [[582, 198], [570, 229], [560, 250], [550, 350]]}
  - _Cycle done, QC = REJECT. Unloading to REJECT bin._
- `run_task` {"name": "unload_reject"}
- `read_light` {"points": [[570, 229], [560, 250], [550, 350]]}
- `look` {}
- `receipts` {"last": 1}
  - _Part 2 in red REJECT bin, nest empty. Loading part 3._
- `run_task` {"name": "load_slot2"}
- `look` {}
  - _Part 3 in the nest, infeed empty. Starting cycle._
- `run_task` {"name": "start_cycle"}
- `read_light` {"points": [[582, 198], [570, 229], [560, 250], [550, 350]]}
- `read_light` {"points": [[582, 198], [570, 229], [560, 250], [550, 350]]}
- `read_light` {"points": [[582, 198], [570, 229], [560, 250], [550, 350]]}
  - _QC = PASS. Unloading to PASS bin._
- `run_task` {"name": "unload_pass"}
- `read_light` {"points": [[582, 198], [570, 229], [560, 250], [550, 350]]}
- `look` {}
- `receipts` {"last": 1}
- `home` {"side": "both"}
  - _All 3 parts are machined and sorted. The infeed fixture and the machine nest are now empty, the stack light is yellow (idle) and the arms are back home._

## Final report (37 turns, 216 s, $0.48)

All 3 parts are machined and sorted. The infeed fixture and the machine nest are now empty, the stack light is yellow (idle) and the arms are back home.

| Part | Infeed slot | QC result | Bin | Unload receipt |
|---|---|---|---|---|
| 1 | slot0 | PASS (green lamp) | PASS (green) | `20261018-012543-723-run_task` |
| 2 | slot1 | REJECT (red lamp) | REJECT (red) | `20261018-012648-571-run_task` |
| 3 | slot2 | PASS (green lamp) | PASS (green) | `20261018-012750-728-run_task` |

I checked every step by sensing. Each load showed the part in the nest. Each `start_cycle` turned the stack light from yellow to green and back to yellow, and I read the QC lamp before every unload. After each unload I confirmed with `look` that the part was in the right bin and with `read_light` that the QC lamp had gone dark. The red stack light (fault) never came on.

Two things to look at:
- **Part 2 moved inside the REJECT bin** during the `load_slot2` transfer. It's still in the bin, but the left arm's path may pass close to that bin.
- **`status` showed `arm_capable_fsm: false` at FSM 500**, even though the server docs list 500 as an arm-capable mode. Every motion ran normally anyway, so the flag is probably reporting the wrong value.

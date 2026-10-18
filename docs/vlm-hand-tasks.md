# VLM-driven hand tasks: integration guide

How an agent goes from "pick up the red cup" to the G1's hands doing it, using the **onboard** models
(on Waldo's GPU), the head camera, and `waldo-arms`. Written for teammates wiring this into their own agents.

## The loop

```
 agent (Claude / brain)                           Waldo                                   G1
 ──────────────────────                           ─────                                   ──
 "pick up the red cup"
   │  look(annotate=true) ───────────────► head frame + 80 px grid + L/R hand marks ◄──── head D435i
   │  ground("the red cup") ─────────────► Cosmos-Reason2-2B (onboard GPU) → bbox, point (0-1000)
   │  point_to_xyz([[x, y]], "norm1000") ► depth (teleimager) + intrinsics + mount + waist → xyz (m)
   │  check_reach / move_hand(above) ────► arm_server: IK, collision + table checks ──────► left arm
   │  move_hand(down) → gripper(close) → move_hand(up) → look ─ verify ──────────────────► Dex1
   └► save_routine("pick-red-cup") ───────► routines/pick-red-cup.json  (instant replay next time)
```

The agent decides *what*; the onboard model finds *where* (pixels); `point_to_xyz` makes it *metres*; the arm
server makes it *safe joint motion*. Each layer is swappable.

## The pieces

| Piece | Where | What it gives you |
|---|---|---|
| `waldo-arms/server.py` (MCP) | Waldo or laptop | `look`, `point_to_xyz`, `check_reach`, `move_hand`, `gripper`, `set_table`, `freeze`/`release`, routines |
| `waldo-arms/vlm_point.py` | Waldo GPU | `ground(query)` → boxes + center points on the head image, 0-1000 scale |
| `waldo-arms/vision.py` | anywhere | the pixel ↔ pelvis-frame math (self-check: `python vision.py`) |
| arm_server (Michael's) | robot Jetson :5810 | IK, path checks, watchdogs, Dex1 control |
| teleimager | robot Jetson :55555/55556/60000 | head color, aligned depth, camera config |

### Onboard models (already on Waldo, `~/.cache/huggingface/hub`)

| Model | Use it for | Notes |
|---|---|---|
| **Cosmos-Reason2-2B** (NVIDIA, Qwen3-VL based) | "where is X", including descriptions ("the cup nearest the left hand") | Built-in 2D grounding prompt; outputs `bbox_2d` on 0-1000. **Measured: 2.5 s load, 4-4.5 GiB VRAM, ~5 s/query today** |
| **OWLv2** (`google/owlv2-base-patch16-ensemble`) | fast plain-noun detection ("cup", "block") | Open-vocabulary detector; much lighter than a VLM. Not wired yet |
| **UnifoLM-VLM-Base** (Qwen2.5-VL-7B, Unitree) | harder scenes | Needs 4-bit to fit 8 GB (`unifolm-vla` env has bitsandbytes). Not wired yet |
| SmolVLM2-500M | quick yes/no checks ("is the cup in the gripper?") | Small and fast; weak at localization |

GPU: RTX 5060, 8 GB. One resident 2B model (~4.5 GB) leaves room for the rest; don't load two big ones.

## Integrate it

**1. Claude Code (or any MCP client):** add `waldo-arms` (see [`waldo-arms/README.md`](../waldo-arms/README.md)),
then just ask for a task. Claude can point from the annotated `look` image itself, or you run `vlm_point.py` for
onboard grounding.

**2. The brain (`waldo-live`):** add `waldo-arms` to `~/.waldo/brain.json` next to `g1`
(snippet in the waldo-arms README). The brain gets every tool above.

**3. Your own Python agent on Waldo:**
```python
import vlm_point                      # keeps Cosmos-Reason2 resident after the first call
hit = vlm_point.ground("the red cup") # {"found": [{"bbox_2d": [...], "point": [x, y]}], "scale": "norm1000", ...}
# then call the waldo-arms MCP tools (or its functions) with:
#   point_to_xyz([hit["found"][0]["point"]], scale="norm1000") -> xyz
#   move_hand("left", x, y, z + 0.08); move_hand("left", x, y, z); gripper("left", "close"); move_hand("left", x, y, z + 0.10)
```

## A precise grasp, step by step

1. `set_table(z)`: measure once (`point_to_xyz` on the table surface, take z), so no move goes through it.
2. `enable`.
3. `look` → ground the object → `point_to_xyz` → **xyz**. Check `confidence` (≥ 0.5 is solid; 0 means it had to
   search wider, usually a dark or shiny surface) and `in_reach_box`.
4. `check_reach(side, x, y, z + 0.08)`: pre-grasp 8 cm above.
5. `move_hand` to the pre-grasp, then down to z (or z + 0.01 for small objects), `seconds` 2-3.
6. `look`: the L/R hand mark should sit on the object. If it's off, correct with small `move_hand` steps.
7. `gripper close` → `move_hand` up 10 cm → `look` to confirm it lifted.
8. Worked? `save_routine("pick-red-cup", "...")`.

Precision to expect: IK hits targets within ~0.4 mm (measured); real-world error is dominated by depth noise, arm
sag and the camera calibration: plan for **~1-2 cm**, verify with `look`, and approach from above.

## Measured, and not yet

| | Status |
|---|---|
| `point_to_xyz` math, both depth formats, normalized input | ✅ tested against fake teleimager + arm_server (1-4 ms/call with a cached frame) |
| Cosmos-Reason2 grounding on a real frame | ✅ correct box for whole objects; 0-1000 scale confirmed |
| Small-part grounding ("the robot's left hand") | ❌ unusable with the 2B model: ask for whole objects, or try the 7B |
| Query latency | ⚠️ ~5 s today. Fixes: prefill the answer so it emits ~10 tokens instead of 40; static KV cache + `torch.compile`; confirm flash-attn really runs on this GPU (the slow per-token rate suggests a fallback); OWLv2 for plain nouns |
| Camera calibration on the real robot | ⚠️ unverified: the `look` overlay is the check. If the L/R marks miss the real grippers, swap in Michael's measured `camera-pose-head.json` (on the Jetson) |
| Depth message format from the Jetson's teleimager | ⚠️ coded for raw z16 or PNG; the tool reports the byte count if it's neither |

## Safety (unchanged)

Hand controller in reach (L1+A damps). Gantry on. No walking. Only one arm controller at a time (arm_server *or*
teleop *or* gestures). `freeze` if anything looks wrong; `release` when done.

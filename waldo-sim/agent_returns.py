#!/usr/bin/env python3
"""Warehouse returns triage through MCP tools only: load each returned item onto the scanner, read the console's
disposition code (b1 restock, b2 refurbish, b3 return to vendor), unload to that bin. Scored on ground truth.
  (returns sim on --port-offset 1700)  uv run --with "mcp>=2.2" --with mujoco --with pyzmq --with numpy --with pillow agent_returns.py --rounds 4
"""
import argparse
import os
import re
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "waldo-arms"))
sys.path.insert(0, str(HERE))
BINS = {"1": "restock", "2": "refurb", "3": "rma"}

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--rounds", type=int, default=4)
ap.add_argument("--offset", type=int, default=1700)
a = ap.parse_args()
o = a.offset
os.environ.update(G1_ARM_ADDR=f"tcp://127.0.0.1:{15810 + o}", G1_CAM_BASE=f"http://127.0.0.1:{18766 + o}", G1_CAM_HOST="127.0.0.1",
                  G1_DEPTH_PORT=str(15556 + o), G1_CAMCFG_PORT=str(16000 + o), G1_TWIN="returns",
                  G1_TAUGHT_DIR=str(HERE / "taught_returns.json"), G1_RECEIPTS=tempfile.mkdtemp())
import scene  # noqa: E402
import server as S  # noqa: E402
import vision  # noqa: E402
from planner import Arm  # noqa: E402

arm = Arm(os.environ["G1_ARM_ADDR"])
ok = n = 0
for rnd in range(a.rounds):
    arm("sim_reset", seed=rnd)
    arm("enable")
    S.home("both")
    K = S._head_config()["color_intrinsics"]
    T = vision.pelvis_from_cam(np.array(arm("state")["T_pelvis_torso"]))
    (cx, cy), _ = scene.RETURNS["console"]
    z = arm("sim_objects")["desk_top_z"] + 0.012 + 0.0132
    c = [vision.pelvis_to_pixel([cx - 0.004 + dx, cy + dy, z], K, T) for dx in (-0.0135, 0.0135) for dy in (0.0345, -0.0345)]
    box = [min(p[0] for p in c), min(p[1] for p in c), max(p[0] for p in c), max(p[1] for p in c)]
    for k in range(3):
        part = f"part{k}"
        try:
            S.run_task(f"load_slot{k}")
            arm("sim_wait", seconds=1.5)
            code = re.search(r"[123]", S.read_display(box)["text"])
            if not code:
                raise RuntimeError("no disposition code on the console")
            dest = BINS[code.group()]
            S.run_task(f"unload_{dest}")
            why = f"b{code.group()} -> {dest}"
        except Exception as e:  # noqa: BLE001
            dest, why = None, str(e)[:120]
        scan = [h[1] for h in arm("sim_devices")["history"] if h[1].startswith(f"scanned {part}:")]
        truth = BINS[scan[-1].split(": b")[1][0]] if scan else None
        where = arm("sim_objects")["objects"][part]["in_container"]
        good = truth is not None and where == f"{truth}_bin"
        ok += good
        n += 1
        print(f"round {rnd} {part} {'OK ' if good else 'MISS'} scanned {truth}, in {where}: {why}", flush=True)
print(f"\nreturns triaged correctly: {ok}/{n}")

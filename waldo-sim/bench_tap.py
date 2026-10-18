#!/usr/bin/env python3
"""Device QA bench: tap through ShopApp's purchase flow (Browse, Add to cart, Checkout, Pay, OK) with pointing noise
like a VLM's, via the MCP tap skill. Scored on the screen reached after every tap. (tablet sim on --port-offset 1500)
  uv run --with "mcp>=2.2" --with mujoco --with pyzmq --with numpy --with pillow bench_tap.py [--noise 0 4 8] [--rounds 5]
"""
import argparse
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "waldo-arms"))
sys.path.insert(0, str(HERE))
FLOW = [("Browse", "product"), ("Add to cart", "cart"), ("Checkout", "checkout"), ("Pay $12", "error"), ("OK", "cart")]

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--noise", type=float, nargs="+", default=[0, 4, 8])
ap.add_argument("--rounds", type=int, default=5)
ap.add_argument("--offset", type=int, default=1500)
a = ap.parse_args()
o = a.offset
os.environ.update(G1_ARM_ADDR=f"tcp://127.0.0.1:{15810 + o}", G1_CAM_BASE=f"http://127.0.0.1:{18766 + o}", G1_CAM_HOST="127.0.0.1",
                  G1_DEPTH_PORT=str(15556 + o), G1_CAMCFG_PORT=str(16000 + o), G1_TWIN="tablet", G1_RECEIPTS=tempfile.mkdtemp())
import server as S  # noqa: E402
import vision  # noqa: E402
from planner import Arm  # noqa: E402

arm = Arm(os.environ["G1_ARM_ADDR"])
K = S._head_config()["color_intrinsics"]
for noise in a.noise:
    ok = n = 0
    for rnd in range(a.rounds):
        rng = np.random.default_rng(rnd)
        arm("sim_reset", seed=rnd)
        arm("enable")
        for button, expect in FLOW:
            T = vision.pelvis_from_cam(np.array(arm("state")["T_pelvis_torso"]))
            u, v = vision.pelvis_to_pixel(arm("sim_devices")["targets"][button]["point"], K, T)
            u, v = u + rng.normal(0, noise), v + rng.normal(0, noise)
            S._depth_cache = (0.0, None)
            try:
                S.tap(u, v)
            except Exception as e:  # noqa: BLE001
                print(f"  noise {noise} round {rnd} {button}: {str(e)[:100]}")
            got = arm("sim_devices")["state"]["screen"]
            ok += got == expect
            n += 1
            if got != expect:
                print(f"  noise {noise} round {rnd} {button}: on '{got}', expected '{expect}'")
                break
    print(f"noise {noise:3.0f} px: {ok}/{a.rounds * len(FLOW)} taps reached the expected screen")

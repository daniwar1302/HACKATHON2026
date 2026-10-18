#!/usr/bin/env python3
"""Teach by pointing, as on a real cell: each lab control is taught with teach_control from a (noisy) pixel only,
planned against a desk-only twin that knows nothing about the devices; scored on whether the device registered the
press. (lab sim on --port-offset 700)  uv run --with "mcp>=2.2" --with mujoco --with pyzmq --with numpy --with pillow bench_teach_point.py
"""
import os, sys, tempfile
TMP = tempfile.mkdtemp()
o = 700
os.environ.update(G1_ARM_ADDR=f"tcp://127.0.0.1:{15810 + o}", G1_CAM_BASE=f"http://127.0.0.1:{18766 + o}", G1_CAM_HOST="127.0.0.1",
                  G1_DEPTH_PORT=str(15556 + o), G1_CAMCFG_PORT=str(16000 + o), G1_GAUGES=TMP + "/g.json",
                  G1_TAUGHT_DIR=TMP, G1_RECEIPTS=TMP + "/receipts")
sys.path.insert(0, '/home/lain_iwakura/HACKATHON2026/waldo-arms'); sys.path.insert(0, '/home/lain_iwakura/HACKATHON2026/waldo-sim')
import server as S, vision, numpy as np, time
from planner import Arm
arm = Arm(os.environ["G1_ARM_ADDR"])
DEPTH = {"dev_boot": 4, "dev_reset": 4, "machine_start": 9, "machine_stop": 9}
ok = n = 0
for trial in range(3):
    rng = np.random.default_rng(trial)
    for ctl, depth in DEPTH.items():
        arm("sim_reset", seed=2); arm("enable")              # seed 2: board running, healthy machine
        K = S._head_config()["color_intrinsics"]; T = vision.pelvis_from_cam(np.array(arm("state")["T_pelvis_torso"]))
        u, v = vision.pelvis_to_pixel(arm("sim_devices")["targets"][ctl]["point"], K, T)
        u, v = u + rng.normal(0, 3), v + rng.normal(0, 3)
        before = arm("sim_devices")["presses"][ctl]
        t = time.time()
        try:
            r = S.teach_control(ctl + "_pt", u, v, depth_mm=depth)
            why = f"{r['side']} tried {r['tried']}"
        except Exception as e:
            why = str(e)[:110]
        good = arm("sim_devices")["presses"][ctl] == before + 1
        ok += good; n += 1
        print(f"{trial} {ctl:14s} {'OK ' if good else 'MISS'} {time.time()-t:4.1f}s {why}", flush=True)
print(f"teach by pointing (3 px noise): {ok}/{n}")

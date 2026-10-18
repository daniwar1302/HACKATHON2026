#!/usr/bin/env python3
"""Desk-sorting benchmark: N seeded desks, each item picked and dropped into its category's container, with real
grasp physics. Scored on where items actually end up. Uses planner.py over the arm_server protocol.

Perception: --truth uses ground-truth positions and grip axes; the default goes through the real vision path for
position (oracle pixel + noise -> point_to_xyz math over the sim's depth) with the grip axis from ground truth.

  (sim running)  uv run --with pyzmq --with numpy bench_sort.py --trials 10 --items 3 [--noise-px 4] [--truth]
"""
import argparse
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "waldo-arms"))
from planner import Arm, PlanError, Planner  # noqa: E402

DROP = {"writing": "pen_cup", "supplies": "tray", "tools": "caddy"}


def locate(arm, item, noise_px, seed, K, depth_fn):
    """The vision path: a (noisy) point on the item in the head image -> pelvis-frame xyz via depth + camera model."""
    import vision
    pt = arm("sim_point", item=item, noise_px=noise_px, seed=seed)
    z, conf = vision.depth_at(depth_fn(), pt["u"], pt["v"])
    if z is None:
        return None
    T = vision.pelvis_from_cam(np.array(arm("state")["T_pelvis_torso"]))
    return vision.pixel_to_pelvis(pt["u"], pt["v"], z, K, T)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--items", type=int, default=3)
    ap.add_argument("--noise-px", type=float, default=0.0)
    ap.add_argument("--truth", action="store_true")
    a = ap.parse_args()
    arm = Arm("tcp://127.0.0.1:15810")
    info = arm("sim_objects")
    p = Planner(arm, info["desk_top_z"])
    if not a.truth:
        import zmq
        cfg_sock = zmq.Context.instance().socket(zmq.REQ)
        cfg_sock.connect("tcp://127.0.0.1:16000")
        cfg_sock.send(b"cfg")
        K = cfg_sock.recv_json()["head_camera"]["color_intrinsics"]
        sub = zmq.Context.instance().socket(zmq.SUB)
        sub.setsockopt(zmq.CONFLATE, 1)
        sub.setsockopt_string(zmq.SUBSCRIBE, "")
        sub.connect("tcp://127.0.0.1:15556")
        depth_fn = lambda: np.frombuffer(sub.recv(), np.uint16).reshape(480, 640).astype(np.float32) / 1000
    stats, fails, t0, n = Counter(), Counter(), time.time(), 0
    for seed in range(a.trials):
        objs = arm("sim_reset", seed=seed, n_items=a.items)["objects"]
        for name in objs:
            o = arm("sim_objects")["objects"][name]
            if o["in_container"]:
                continue
            n += 1
            target = DROP[o["category"]]
            box = info["containers"][target]
            xyz = np.array(o["xyz"]) if a.truth else locate(arm, name, a.noise_px, seed * 100 + n, K, depth_fn)
            why = ""
            if xyz is None:
                why = "vision: no depth at the point"
            elif o["width_mm"] > 90:
                why = f"too wide for the gripper ({o['width_mm']} mm)"
            else:
                # jaw centre at the item's centre height, 2 mm lower for thin items (fingertips may brush the desk);
                # truth xyz is the centre, the vision path returns the top surface
                h = o["height_mm"] / 1000
                grip_dz = (-0.002 if h < 0.025 else 0.0) - (0 if a.truth else h / 2)
                order = ["left", "right"] if xyz[1] > 0 else ["right", "left"]
                tried = []
                for side in order:
                    try:
                        p.pick(side, xyz, o["grip_jaw_deg"], grip_dz=grip_dz, thin=o["height_mm"] < 6)
                        p.place(side, box["box_min"], box["box_max"])
                        p.home(side)
                        why = ""
                        break
                    except PlanError as e:
                        tried.append(f"{side}: {e}")
                        why = tried[0]          # the preferred hand's failure is the informative one
                        try:
                            arm("open", side=side)
                            p.home(side)
                        except PlanError:
                            pass
            final = arm("sim_objects")["objects"][name]
            ok = final["in_container"] == target
            stats["sorted" if ok else "missed"] += 1
            if not ok:
                fails[(why.split(":")[1].strip() if ":" in why else why) or f"landed in {final['in_container']}"] += 1
            print(f"seed {seed:2d} {name:14s} -> {target:8s} {'OK ' if ok else 'MISS'} {why[:95]}", flush=True)
    print(f"\nsorted {stats['sorted']}/{n} ({100 * stats['sorted'] / max(n, 1):.0f}%) in {time.time() - t0:.0f} s"
          f"  [{'ground-truth positions' if a.truth else f'vision path, {a.noise_px} px noise'}]")
    for k, v in fails.most_common():
        print(f"  {v:3d}  {k}")


if __name__ == "__main__":
    main()

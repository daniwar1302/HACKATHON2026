#!/usr/bin/env python3
"""Machine tending + QC, driven the way an onboard agent would: only waldo-arms MCP tools, every step verified by
sensing, receipts for each action. Scored against the sim's ground truth.

  for each raw part on the infeed fixture:
    pick it (a VLM's box around it) -> place in the machine nest -> START -> stack light green (the machine only
    starts with a part loaded: that's the load check) -> wait for yellow -> read the QC lamp -> pick the part out of
    the nest -> PASS bin (green lamp) or REJECT bin (red) -> home

  (cell sim on --port-offset 900)  uv run --with "mcp>=2.2" --with mujoco --with pyzmq --with numpy --with pillow \
      agent_cell.py --rounds 4 [--noise-px 3]
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "waldo-arms"))
sys.path.insert(0, str(HERE))


def stations(S, arm):
    """One-time cell setup: containers from the cell layout (what an operator measures or a VLM points at once),
    and the pixels of the QC lamp and the stack light lamps."""
    import scene
    import vision
    c = arm("sim_objects")["containers"]
    json.dump({n: {k: c[n][k] for k in ("box_min", "box_max", "drop_xy")} for n in ("nest", "pass_bin", "reject_bin")},
              open(os.environ["G1_STATIONS"], "w"), indent=1)
    K = S._head_config()["color_intrinsics"]
    T = vision.pelvis_from_cam(np.array(arm("state")["T_pelvis_torso"]))
    top = arm("sim_objects")["desk_top_z"]
    px = lambda p: list(vision.pelvis_to_pixel(p, K, T))
    (mx, my), _ = scene.CELL["machine"]
    (sx, sy), _ = scene.CELL["stack_light"]
    cam = T[:3, 3]
    toward = (cam[:2] - np.array([sx, sy])) / np.linalg.norm(cam[:2] - np.array([sx, sy]))
    return {"qc": px([mx + 0.03, my, top + 0.02 + 0.022 + 0.004]),
            "stack": {c: px([sx + 0.012 * toward[0], sy + 0.012 * toward[1], top + 0.097 + 0.025 * k])
                      for k, c in enumerate(("green", "yellow", "red"))}}


def stack(S, st):
    ls = S.read_light([st["stack"][c] for c in ("green", "yellow", "red")])["lights"]
    lit = [c for c, l in zip(("green", "yellow", "red"), ls) if l["on"]]
    return lit[0] if len(lit) == 1 else "/".join(lit) or "dark"


def tend(S, arm, st, part, noise, rng, log, taught=True):
    """taught: the cell's taught transfers (fixtured parts: how production cells run). Otherwise free pick/place
    from a VLM's box (for parts that aren't in a fixture)."""
    box = lambda: arm("sim_point", item=part, noise_px=noise, seed=int(rng.integers(1 << 30)))["box"]
    if taught:
        S.run_task(f"load_slot{part[-1]}")
    else:
        r = S.pick(box=box())
        S.place(r["side"], container="nest")
        S.home(r["side"])
    S.run_task("start_cycle")
    light = stack(S, st)
    qc_lit = lambda: S.read_light([st["qc"]])["lights"][0]["on"]
    if light != "green" and not (light == "yellow" and qc_lit()):     # running, or already done with a result
        raise RuntimeError(f"machine did not start (stack {light}): part not seated in the nest")
    for _ in range(40):
        if light != "green":
            break
        arm("sim_wait", seconds=0.5)
        light = stack(S, st)
    if light != "yellow":
        raise RuntimeError(f"cycle ended {light}")
    qc = S.read_light([st["qc"]])["lights"][0]
    verdict = {"green": "pass", "red": "reject"}.get(qc["colour"]) if qc["on"] else None
    if verdict is None:
        raise RuntimeError(f"QC lamp unreadable: {qc}")
    if taught:
        S.run_task(f"unload_{verdict}")
    else:
        r = S.pick(box=box())
        S.place(r["side"], container="pass_bin" if verdict == "pass" else "reject_bin")
        S.home(r["side"])
    if S.read_light([st["qc"]])["lights"][0]["on"]:     # the lamp only lights with a part in the nest
        raise RuntimeError("part still in the nest after unloading")
    log(f"{part}: QC {verdict} -> {'pass_bin' if verdict == 'pass' else 'reject_bin'}")
    return verdict


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--offset", type=int, default=900, help="the cell sim's --port-offset")
    ap.add_argument("--noise-px", type=float, default=3.0, help="pixel noise on the VLM's boxes")
    ap.add_argument("--free", action="store_true", help="free pick/place from VLM boxes instead of taught transfers")
    a = ap.parse_args()
    o = a.offset
    os.environ.update(G1_ARM_ADDR=f"tcp://127.0.0.1:{15810 + o}", G1_CAM_BASE=f"http://127.0.0.1:{18766 + o}",
                      G1_CAM_HOST="127.0.0.1", G1_DEPTH_PORT=str(15556 + o), G1_CAMCFG_PORT=str(16000 + o),
                      G1_TWIN="cell", G1_STATIONS=str(HERE / "stations_cell.json"),
                      G1_TAUGHT_DIR=str(HERE / "taught_cell.json"))
    import server as S
    from planner import Arm
    arm = Arm(os.environ["G1_ARM_ADDR"])
    ok = n = 0
    t_all = time.time()
    for rnd in range(a.rounds):
        arm("sim_reset", seed=rnd)
        arm("enable")
        S.home("both")                      # every task starts from the cell's home (taught transfers start there)
        st = stations(S, arm)
        log = lambda m: print(f"  [{rnd}] {m}", flush=True)
        for part in ("part0", "part1", "part2"):
            t0, sim0 = time.time(), arm("sim_devices")["time"]
            try:
                verdict = tend(S, arm, st, part, a.noise_px, np.random.default_rng(rnd * 10 + int(part[-1])), log, not a.free)
            except Exception as e:
                verdict = None
                log(f"{part} FAILED: {str(e)[:150]}")
                try:
                    S.home("both")
                except Exception:
                    pass
            truth = arm("sim_devices")["state"]["done"].get(part)
            where = arm("sim_objects")["objects"][part]["in_container"]
            good = truth is not None and verdict == truth and where == ("pass_bin" if truth == "pass" else "reject_bin")
            ok += good
            n += 1
            print(f"round {rnd} {part} {'OK ' if good else 'MISS'} truth {truth} read {verdict} in {where}  "
                  f"{arm('sim_devices')['time'] - sim0:5.1f} s sim, {time.time() - t0:4.1f} s wall", flush=True)
    print(f"\nparts machined, QC-read and binned correctly: {ok}/{n}  ({time.time() - t_all:.0f} s wall)")


if __name__ == "__main__":
    main()

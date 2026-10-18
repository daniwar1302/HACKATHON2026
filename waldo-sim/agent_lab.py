#!/usr/bin/env python3
"""The lab stations driven the way an onboard agent would: only waldo-arms MCP tools (run_task, read_light,
read_display, read_gauge, receipts), every step verified by sensing. Scored against the sim's ground truth.

  firmware: board hung? reset it -> hold BOOT + tap RESET -> LED blue, display 'boot' -> flash over USB ->
            reset -> LED green, read the new code off the display
  tending:  start the cycle -> stack light green -> park a fingertip over STOP -> poll the gauge; over 8 bar press
            STOP (odd seeds have a fault that trips at 9.5 bar), else wait for the cycle to finish

  (lab sim on --port-offset 700)  uv run --with "mcp>=2.2" --with mujoco --with pyzmq --with numpy --with pillow \
      agent_lab.py --rounds 6
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "waldo-arms"))
sys.path.insert(0, str(HERE))


def stations(S, arm):
    """One-time cell setup, what an operator (or a VLM pointing once) registers: pixel boxes for the board LED,
    display, gauge and stack light, from the scene layout through the head-camera model."""
    import scene
    import vision
    K = S._head_config()["color_intrinsics"]
    T = vision.pelvis_from_cam(np.array(arm("state")["T_pelvis_torso"]))
    px = lambda p: vision.pelvis_to_pixel(p, K, T)
    top = arm("sim_objects")["desk_top_z"]
    (bx, by), _ = scene.LAB["dev_board"]
    bz = top + 0.008 + 0.0092
    corners = [px([bx - 0.008 + dx, by + dy, bz]) for dx in (-0.0135, 0.0135) for dy in (0.0345, -0.0345)]
    us, vs = [c[0] for c in corners], [c[1] for c in corners]
    gc, GR = scene.gauge_frame(top)                     # calibrate the dial: where its 0 and 10 bar ticks appear
    gu, gv = px(gc)
    r_px = K["fx"] * 0.045 / np.linalg.norm(gc - T[:3, 3])
    def img_deg(a):
        tu, tv = px(gc + GR @ np.array([0.034 * np.sin(np.radians(a)), 0.034 * np.cos(np.radians(a)), 0.005]))
        return float(np.degrees(np.arctan2(tu - gu, -(tv - gv))))
    S.register_gauge("pressure", [gu - r_px, gv - r_px, gu + r_px, gv + r_px], 0, 10, min_deg=img_deg(-135),
                     max_deg=img_deg(135), unit="bar", alarm_above=8.0)
    (sx, sy), _ = scene.LAB["stack_light"]
    cam = T[:3, 3]
    toward = (cam[:2] - np.array([sx, sy])) / np.linalg.norm(cam[:2] - np.array([sx, sy]))   # the lamp face we see
    lamp = lambda k: list(px([sx + 0.012 * toward[0], sy + 0.012 * toward[1], top + 0.097 + 0.025 * k]))
    return {"led": list(px([bx + 0.024, by, top + 0.017])), "display": [min(us), min(vs), max(us), max(vs)],
            "stack": {c: lamp(k) for k, c in enumerate(("green", "yellow", "red"))}}


def led(S, st):
    l = S.read_light([st["led"]])["lights"][0]
    return l["colour"] if l["on"] else "off"


def stack(S, st):
    ls = S.read_light([st["stack"][c] for c in ("green", "yellow", "red")])["lights"]
    lit = [c for c, l in zip(("green", "yellow", "red"), ls) if l["on"]]
    return lit[0] if len(lit) == 1 else "/".join(lit) or "dark"


def firmware(S, arm, st, log):
    if led(S, st) == "red":
        log("board hung (LED red): reset_board")
        S.run_task("reset_board")
    assert led(S, st) == "green", f"board not running after reset (LED {led(S, st)})"
    S.run_task("enter_bootloader")
    l, shown = led(S, st), S.read_display(st["display"])["text"]
    log(f"bootloader: LED {l}, display '{shown}'")
    assert l == "blue" and shown == "boot", "not in bootloader"
    arm("sim_flash")                                     # the USB side: esptool / dfu-util on the real board
    S.run_task("reset_board")
    code = S.read_display(st["display"])["text"]
    truth = arm("sim_devices")["state"]
    log(f"flashed: LED {led(S, st)}, code read '{code}' (true '{truth['display']}'), firmware v{truth['firmware']}")
    return led(S, st) == "green" and truth["firmware"] == 2 and code == truth["display"]


def tending(S, arm, st, log):
    assert stack(S, st) == "yellow", f"machine not idle ({stack(S, st)})"
    S.run_task("arm_stop")                      # right fingertip over STOP first; START is on the left hand
    gid = S.guard("pressure", 8.0, "stop_machine", timeout_s=60)["guard"]   # armed BEFORE the start, runs in background
    S.run_task("start_cycle")
    assert stack(S, st) in ("green", "yellow"), "machine did not start"
    errs = [abs(S.read_gauge("pressure")["value"] - arm("sim_devices")["state"]["pressure_bar"])]
    g = S.guard_status(gid, wait_s=60)
    peak = g["peak"]
    log(f"guard {'fired at ' + str(g['at']) + ' bar: STOP' if g['fired'] else 'quiet: cycle ran to the end'}")
    S.run_task("disarm_stop")
    m = arm("sim_devices")["state"]["machine"]
    log(f"machine {m}, peak read {peak:.1f} bar, gauge error mean {np.mean(errs):.2f} / max {np.max(errs):.2f} bar")
    tripped = any("FAULT" in e for _, e in arm("sim_devices")["history"])
    return m in ("done", "stopped") and not tripped, m


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--offset", type=int, default=700, help="the lab sim's --port-offset")
    a = ap.parse_args()
    o = a.offset
    os.environ.update(G1_ARM_ADDR=f"tcp://127.0.0.1:{15810 + o}", G1_CAM_BASE=f"http://127.0.0.1:{18766 + o}",
                      G1_CAM_HOST="127.0.0.1", G1_DEPTH_PORT=str(15556 + o), G1_CAMCFG_PORT=str(16000 + o),
                      G1_GAUGES=str(HERE / "gauges_sim.json"))
    import server as S
    from planner import Arm
    arm = Arm(os.environ["G1_ARM_ADDR"])
    score = {"firmware": 0, "tending": 0}
    for rnd in range(a.rounds):
        arm("sim_reset", seed=rnd)
        arm("enable")
        st = stations(S, arm)
        log = lambda m: print(f"  [{rnd}] {m}", flush=True)
        for name, fn in (("firmware", firmware), ("tending", tending)):
            t0, sim0 = time.time(), arm("sim_devices")["time"]
            try:
                r = fn(S, arm, st, log)
                ok = r[0] if isinstance(r, tuple) else r
            except (AssertionError, Exception) as e:
                ok = False
                log(f"{name} FAILED: {e}")
            score[name] += ok
            print(f"round {rnd} {name:9s} {'OK ' if ok else 'MISS'}  {arm('sim_devices')['time'] - sim0:5.1f} s sim, "
                  f"{time.time() - t0:4.1f} s wall", flush=True)
    print(f"\nfirmware update {score['firmware']}/{a.rounds}, machine tending {score['tending']}/{a.rounds}")
    print("last receipts:", [r["id"] for r in S.receipts(3)])


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Device tasks an autonomous agent might hand to the robot (arms only), scored from the devices' real state:

  power_cycle_router  press the strip's O end, wait 5 s, press the I end -> history shows OFF, ON, internet UP
  approve_2fa         tap the security key pad -> one press registered
  light_on            push the light switch lever up -> light ON
  start_machine       press the green start button -> one press registered

  (sim running with --scene devices)  uv run --with mujoco --with pyzmq --with numpy bench_devices.py --rounds 3
"""
import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "waldo-arms"))
from planner import Arm, PlanError, Planner  # noqa: E402


def run(task, arm, p):
    dev = arm("sim_devices")
    t0, tg = dev["time"], dev["targets"]
    def do(name, **kw):
        p.press(tg[name]["point"], tg[name]["push"], **kw)
    if task == "power_cycle_router":
        do("power_strip_off_end")
        arm("sim_wait", seconds=5)
        do("power_strip_on_end")
        arm("sim_wait", seconds=4)          # let it boot
        ev = [e for t, e in arm("sim_devices")["history"] if t >= t0]
        ok = "router power OFF" in ev and "router power ON" in ev and ev.index("router power OFF") < ev.index("router power ON") \
            and "router internet UP" in ev
    elif task == "approve_2fa":
        before = dev["presses"]["security_key"]
        do("security_key_pad", depth=0.004)
        ok = arm("sim_devices")["presses"]["security_key"] == before + 1
    elif task == "light_on":
        do("light_switch_lever_top", depth=0.012)
        ok = arm("sim_devices")["state"]["light_on"]
    elif task == "start_machine":
        before = dev["presses"]["start_button"]
        do("start_button", depth=0.009)
        ok = arm("sim_devices")["presses"]["start_button"] == before + 1
    return ok, [e for t, e in arm("sim_devices")["history"] if t >= t0]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--addr", default="tcp://127.0.0.1:15810")
    a = ap.parse_args()
    arm = Arm(a.addr)
    p = Planner(arm, arm("sim_objects")["desk_top_z"], scene_kind="devices")
    tasks = ["power_cycle_router", "approve_2fa", "light_on", "start_machine"]
    score = {t: 0 for t in tasks}
    t_start = time.time()
    for rnd in range(a.rounds):
        arm("sim_reset", seed=rnd)
        for task in tasks:
            try:
                ok, ev = run(task, arm, p)
                why = "; ".join(ev) or "no device change"
            except PlanError as e:
                ok, why = False, str(e)[:120]
            score[task] += ok
            print(f"round {rnd}  {task:20s} {'OK ' if ok else 'MISS'}  {why}", flush=True)
            for side in ("left", "right"):
                p.recover(side)
                try:
                    p.home(side)
                except PlanError:
                    arm("sim_reset_arms")
    print(f"\n{sum(score.values())}/{a.rounds * len(tasks)} tasks done in {time.time() - t_start:.0f} s")
    for t, v in score.items():
        print(f"  {t:20s} {v}/{a.rounds}")


if __name__ == "__main__":
    main()

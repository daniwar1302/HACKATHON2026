#!/usr/bin/env python3
"""Auto-teach and replay for fixed stations (switches, buttons, keys): how production cells get near-100% reliability.

teach:  for each control, search candidate motions on the twin (hand x press angle x IK solution x collision-free
        approach), execute each one in the sim from a snapshot, keep the first that really changes the device and
        passes every collision check, roll back. Saves taught.json: joint-space trajectories from the rest pose.
replay: run a taught control: rest -> taught approach -> press -> release -> rest, as smooth trajectories.
        On the real G1 the same file is replayed through arm_server's `trajectory` (re-teach there with the teleop
        page or a supervised run of this search, since the sim's millimetres aren't the real desk's).

  uv run --with mujoco --with pyzmq --with numpy teach.py teach  --addr tcp://127.0.0.1:15910
  uv run --with mujoco --with pyzmq --with numpy teach.py replay power_cycle_router --addr ...
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "waldo-arms"))
from motion import Twin, cartesian_path, timed  # noqa: E402
from planner import Arm, PlanError, Planner  # noqa: E402

TAUGHT = HERE / "taught.json"
# control -> (depth m, check(before, after) -> bool, precondition description)
CONTROLS = {
    "power_strip_off_end": (0.006, lambda b, a: b["state"]["strip_on"] and not a["state"]["strip_on"]),
    "power_strip_on_end": (0.006, lambda b, a: not b["state"]["strip_on"] and a["state"]["strip_on"]),
    "security_key_pad": (0.004, lambda b, a: a["presses"]["security_key"] == b["presses"]["security_key"] + 1),
    "start_button": (0.009, lambda b, a: a["presses"]["start_button"] == b["presses"]["start_button"] + 1),
    "light_switch_lever_top": (0.012, lambda b, a: not b["state"]["light_on"] and a["state"]["light_on"]),
}


def send(arm, side, path, contact=False):
    q, times = timed(path)
    arm("trajectory", q=q, times=times, side=side, **({"contact": True} if contact else {}))


def candidates(p: Planner, twin: Twin, q0, point, push, depth):
    point = np.asarray(point, float)
    n0 = np.asarray(push, float) / np.linalg.norm(push)
    for side in (["left", "right"] if point[1] > 0 else ["right", "left"]):
        for dirn in p.press_options(side, point, push):
            R = p.push_R(dirn)
            along = depth / max(0.5, float(dirn @ n0))
            centre = point + dirn * along - p.TIP * dirn
            pre = centre - dirn * (along + 0.06)
            for goal in twin.ik_goals(side, q0, pre, R, n=4):
                approach = twin.rrt_connect(side, q0, goal)
                if not approach:
                    continue
                press = cartesian_path(twin, side, goal, [pre + (centre - pre) * k / 6 for k in range(1, 7)], R)
                if press:
                    yield side, dirn, approach, press


def teach(arm, twin, p, only=None):
    arm("sim_reset", seed=0)
    arm("enable")
    for sd in ("left", "right"):
        arm("grasp", side=sd)                # poke with closed fingertips
    q0 = np.array(arm("state")["arm"]["q"], float)
    taught = json.loads(TAUGHT.read_text()) if TAUGHT.exists() else {}
    for name, (depth, ok) in CONTROLS.items():
        if only and name not in only:
            continue
        t0, tried, found = time.time(), 0, None
        tg = arm("sim_devices")["targets"][name]
        arm("sim_save", name="teach")
        for side, dirn, approach, press in candidates(p, twin, q0, tg["point"], tg["push"], depth):
            tried += 1
            before = arm("sim_devices")
            try:
                send(arm, side, approach)
                send(arm, side, press, contact=True)
                arm("sim_wait", seconds=0.3)
                send(arm, side, press[::-1], contact=True)
                send(arm, side, approach[::-1])
                good = ok(before, arm("sim_devices"))
            except PlanError as e:
                good = False
                why = str(e)[:80]
            if good:
                found = {"side": side, "push": np.round(dirn, 4).tolist(),
                         "approach": [list(map(float, q)) for q in approach], "press": [list(map(float, q)) for q in press]}
                break
            arm("sim_restore", name="teach")
        if found:
            taught[name] = found
            arm("sim_save", name="teach")    # keep the device's new state (e.g. strip OFF before teaching ON)
            print(f"{name:24s} TAUGHT  {found['side']} push {found['push']}  ({tried} tried, {time.time() - t0:.0f} s)", flush=True)
        else:
            print(f"{name:24s} FAILED  ({tried} tried, {time.time() - t0:.0f} s)", flush=True)
            arm("sim_restore", name="teach")
    TAUGHT.write_text(json.dumps(taught))
    return taught


def play(arm, name, taught):
    """Run one taught control from the rest pose (where teaching started)."""
    t = taught[name]
    side = t["side"]
    q_now = np.array(arm("state")["arm"]["q"], float)
    start = np.array(t["approach"][0], float)
    sl = slice(0, 7) if side == "left" else slice(7, 14)
    if np.abs(q_now[sl] - start[sl]).max() > 0.05:         # not at rest: go there first (straight joint move)
        mid = q_now.copy()
        mid[sl] = start[sl]
        send(arm, side, [q_now, mid])
    q_now = np.array(arm("state")["arm"]["q"], float)
    other = slice(7, 14) if side == "left" else slice(0, 7)
    fix = lambda path: [np.concatenate([np.asarray(q)[:7], q_now[7:]]) if side == "left" else np.concatenate([q_now[:7], np.asarray(q)[7:]]) for q in path]
    arm("grasp", side=side)
    send(arm, side, fix(t["approach"]))
    send(arm, side, fix(t["press"]), contact=True)
    arm("sim_wait", seconds=0.3)
    send(arm, side, fix(t["press"][::-1]), contact=True)
    send(arm, side, fix(t["approach"][::-1]))


TASKS = {   # what an agent asks for -> taught controls (+ waits)
    "power_cycle_router": [("power_strip_off_end", 5.0), ("power_strip_on_end", 4.0)],
    "approve_2fa": [("security_key_pad", 0.0)],
    "start_machine": [("start_button", 0.0)],
    "light_on": [("light_switch_lever_top", 0.0)],
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["teach", "replay", "bench"])
    ap.add_argument("task", nargs="?")
    ap.add_argument("--addr", default="tcp://127.0.0.1:15910")
    ap.add_argument("--rounds", type=int, default=5)
    a = ap.parse_args()
    arm = Arm(a.addr)
    twin = Twin(arm("sim_objects")["desk_top_z"], kind="devices")
    twin.PAD = 0.002          # the sim validates every candidate; plan with a small pad
    p = Planner(arm, arm("sim_objects")["desk_top_z"], twin=False)
    if a.mode == "teach":
        teach(arm, twin, p, [a.task] if a.task else None)
        return
    taught = json.loads(TAUGHT.read_text())
    tasks = [a.task] if a.task else list(TASKS)
    ok_n = tot = 0
    for rnd in range(a.rounds if a.mode == "bench" else 1):
        arm("sim_reset", seed=rnd)
        arm("enable")
        for task in tasks:
            t0 = arm("sim_devices")["time"]
            try:
                for ctl, wait in TASKS[task]:
                    play(arm, ctl, taught)
                    if wait:
                        arm("sim_wait", seconds=wait)
                d = arm("sim_devices")
                ev = [e for t, e in d["history"] if t >= t0]
                good = {"power_cycle_router": "router power OFF" in ev and "router power ON" in ev and "router internet UP" in ev,
                        "approve_2fa": "security_key pressed" in ev, "start_machine": "start_button pressed" in ev,
                        "light_on": d["state"]["light_on"]}[task]
                why = "; ".join(ev)
            except PlanError as e:
                good, why = False, str(e)[:100]
            ok_n += good
            tot += 1
            print(f"round {rnd} {task:20s} {'OK ' if good else 'MISS'} {why}", flush=True)
    print(f"\n{ok_n}/{tot} tasks")


if __name__ == "__main__":
    main()

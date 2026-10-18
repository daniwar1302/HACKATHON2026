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
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "waldo-arms"))
from motion import Twin, cartesian_path, timed  # noqa: E402
from planner import Arm, PlanError, Planner  # noqa: E402

# control -> (depth m, check(before, after) -> bool)
pressed = lambda k: (lambda b, a: a["presses"][k] == b["presses"][k] + 1)
CONTROLS = {
    "power_strip_off_end": (0.006, lambda b, a: b["state"]["strip_on"] and not a["state"]["strip_on"]),
    "power_strip_on_end": (0.006, lambda b, a: not b["state"]["strip_on"] and a["state"]["strip_on"]),
    "security_key_pad": (0.004, lambda b, a: a["presses"]["security_key"] == b["presses"]["security_key"] + 1),
    "start_button": (0.009, lambda b, a: a["presses"]["start_button"] == b["presses"]["start_button"] + 1),
    "light_switch_lever_top": (0.012, lambda b, a: not b["state"]["light_on"] and a["state"]["light_on"]),
}
LAB_CONTROLS = {
    "dev_boot": (0.004, pressed("dev_boot")),
    "dev_reset": (0.004, pressed("dev_reset")),
    "machine_start": (0.009, pressed("machine_start"), "left"),   # left: the right hand stays parked over STOP
    "machine_stop": (0.009, pressed("machine_stop")),
}


SPEED = float(os.environ.get("G1_JOINT_SPEED", 1.2))     # peak joint speed (rad/s); run the real robot slower at first


def send(arm, side, path, contact=False):
    q, times = timed(path, max_joint_speed=SPEED)
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


def teach(arm, twin, p, controls, path, only=None):
    arm("sim_reset", seed=0)
    arm("enable")
    for sd in ("left", "right"):
        arm("grasp", side=sd)                # poke with closed fingertips
    q0 = np.array(arm("state")["arm"]["q"], float)
    taught = json.loads(path.read_text()) if path.exists() else {}
    for name, (depth, ok, *only_side) in controls.items():
        if only and name not in only:
            continue
        t0, tried, found, verified = time.time(), 0, None, 0
        tg = arm("sim_devices")["targets"][name]
        arm("sim_save", name="teach")
        for side, dirn, approach, press in candidates(p, twin, q0, tg["point"], tg["push"], depth):
            if only_side and side != only_side[0]:
                continue
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
                dur = timed(approach)[1][-1] + timed(press)[1][-1]
                if found is None or dur < found["seconds"]:
                    found = {"side": side, "push": np.round(dirn, 4).tolist(), "seconds": round(dur, 2),
                             "approach": [list(map(float, q)) for q in approach], "press": [list(map(float, q)) for q in press]}
                    arm("sim_save", name="teach_best")
                verified += 1
                if verified >= 3 or tried >= 12:
                    break
            arm("sim_restore", name="teach")
        if found:
            taught[name] = found
            arm("sim_restore", name="teach_best")
            arm("sim_save", name="teach")    # keep the device's new state (e.g. strip OFF before teaching ON)
            print(f"{name:24s} TAUGHT  {found['side']} push {found['push']}  {found['seconds']} s motion  ({tried} tried, {time.time() - t0:.0f} s)", flush=True)
        else:
            print(f"{name:24s} FAILED  ({tried} tried, {time.time() - t0:.0f} s)", flush=True)
            arm("sim_restore", name="teach")
    path.write_text(json.dumps(taught))
    return taught


PHASES = {"tap": "A P W p a", "press": "A P W", "release": "p a", "hover": "A", "poke": "P W p", "rest": "a"}


def play(arm, name, taught, phase="tap"):
    """Run a taught control. phase: tap (from rest: press and let go, back to rest), press (press and stay: hold BOOT
    while the other hand works), release (let go, back to rest), hover (fingertip parked just above it, ready),
    poke (press and let go from the hover: a fast stop), rest (from the hover back to rest). Segments: A approach,
    P press in, W dwell, p press out, a approach back. Missing preconditions are fixed first (poke from rest hovers)."""
    t = taught[name]
    if "log" in t:                                       # a taught transfer: replay its exact commands
        replay(arm, t["log"])
        return
    side = t["side"]
    sl = slice(0, 7) if side == "left" else slice(7, 14)
    approach, press = np.array(t["approach"], float), np.array(t["press"], float)

    def go(path, contact=False):
        q_now = np.array(arm("state")["arm"]["q"], float)
        P = np.array(path, float)
        if np.abs(q_now[sl] - P[-1, sl]).max() < 0.03:         # already there (e.g. hover while hovering)
            return
        P[:, slice(7, 14) if side == "left" else slice(0, 7)] = q_now[slice(7, 14) if side == "left" else slice(0, 7)]
        if np.abs(q_now[sl] - P[0, sl]).max() > 0.05:          # not where this segment starts: straight joint move there
            P = np.vstack([q_now, P])
        send(arm, side, list(P), contact=contact)

    segs = PHASES[phase].split()
    q_now = np.array(arm("state")["arm"]["q"], float)
    if segs[0] == "P" and np.abs(q_now[sl] - press[0][sl]).max() > 0.05:   # poke while not hovering: hover first
        segs = ["A"] + segs
    for sg in segs:
        if sg == "A":
            arm("grasp", side=side)
            go(approach)
        elif sg == "P":
            go(press, contact=True)
        elif sg == "W":
            arm("sim_wait", seconds=0.25)
        elif sg == "p":
            go(press[::-1], contact=True)
        elif sg == "a":
            go(approach[::-1])


# ---- taught transfers (fixtured cell): pick/place moves found by the planner, verified in the sim, replayed exactly ----
def replay(arm, log):
    """Send a recorded command log. The first trajectory starts where teaching started; get there first."""
    for cmd, kw in log:
        if cmd == "trajectory":
            q_now = np.array(arm("state")["arm"]["q"], float)
            q0 = np.array(kw["q"][0], float)
            if np.abs(q_now - q0).max() > 0.05:
                send(arm, kw.get("side", "left"), [q_now, q0])
        arm(cmd, **kw)


def teach_transfers(arm, path, kind="cell", dests=("pass", "reject"), tries=6):
    """Fixtured cell: load_slot{k} (infeed pocket k -> nest, then rest) and unload_<dest> (nest -> <dest>_bin, rest).
    Each: plan with the cell twin (fixture parts as obstacles), run it in the sim from a snapshot, keep the first whose
    part really ends where it should, record its commands."""
    import scene
    from rehearse import Recorder
    p = Planner(arm, arm("sim_objects")["desk_top_z"], scene_kind=kind)
    cont = arm("sim_objects")["containers"]
    # finished parts stand in the bins taller than the walls: plan 6 cm above the rims (a load swept one aside)
    p.twin.set_keepouts([(cont[f"{b}_bin"]["box_min"], cont[f"{b}_bin"]["box_max"][:2] + [cont[f"{b}_bin"]["box_max"][2] + 0.06]) for b in dests])
    taught = json.loads(path.read_text()) if path.exists() else {}
    arm("sim_reset", seed=0)
    arm("enable")
    for sd in ("left", "right"):
        p.home(sd)
    arm("sim_save", name="cell_rest")

    def attempt(name, fn, ok, side):
        rec = Recorder(arm)
        p.arm, p.R, p.held = rec, {"left": None, "right": None}, {}
        try:
            fn(side)
            good = ok()
        except PlanError as e:
            good, rec.why = False, str(e)[:90]
        p.arm = arm
        return good, rec

    def search(name, fn, ok, start):
        t0 = time.time()
        for k in range(tries):
            arm("sim_restore", name=start)
            good, rec = attempt(name, fn, ok, [None, "left", "right"][k % 3])
            if good:
                taught[name] = {"log": rec.log, "side": next((kw.get("side") for c, kw in rec.log if kw.get("side")), "left")}
                print(f"{name:16s} TAUGHT  {len(rec.log)} commands  (try {k + 1}, {time.time() - t0:.0f} s)", flush=True)
                return True
            print(f"   {name} try {k + 1}: {getattr(rec, 'why', 'did not verify')}", flush=True)
        print(f"{name:16s} FAILED", flush=True)
        return False

    in_nest = lambda part: (lambda: arm("sim_devices")["state"]["part_in_nest"] == part)
    for k, part in enumerate(("part0", "part1", "part2")):
        def load(side, part=part):
            o = arm("sim_objects")["objects"][part]
            r = p.pick_item(np.array(o["xyz"]), o["grip_jaw_deg"], height_m=o["height_mm"] / 1000, prefer_side=side,
                            round_item=kind == "cell", width_m=o["width_mm"] / 1000)
            p.place(r["side"], cont["nest"]["box_min"], cont["nest"]["box_max"], at=cont["nest"]["drop_xy"])
            p.home(r["side"])
        if search(f"load_slot{k}", load, in_nest(part), "cell_rest") and k == 0:
            arm("sim_save", name="cell_loaded")          # part0 sits in the nest: teach the unloads from here
    for verdict in dests:
        turn = [0]
        def unload(side, verdict=verdict, turn=turn):
            o = arm("sim_objects")["objects"]["part0"]
            jaw = o["grip_jaw_deg"] + 90 * (turn[0] // 3 % 2)      # square boxes grip across either pair of faces
            turn[0] += 1
            r = p.pick_item(np.array(o["xyz"]), jaw, height_m=o["height_mm"] / 1000, prefer_side=side,
                            round_item=kind == "cell", width_m=o["width_mm"] / 1000)
            b = cont[f"{verdict}_bin"]
            p.place(r["side"], b["box_min"], b["box_max"], at=b["drop_xy"])
            p.home(r["side"])
        search(f"unload_{verdict}", unload, lambda v=verdict: arm("sim_objects")["objects"]["part0"]["in_container"] == f"{v}_bin",
               "cell_loaded")
    path.write_text(json.dumps(taught))


TASKS = {   # what an agent asks for -> taught controls (+ wait after, + phase)
    "power_cycle_router": [("power_strip_off_end", 5.0), ("power_strip_on_end", 4.0)],
    "approve_2fa": [("security_key_pad", 0.0)],
    "start_machine": [("start_button", 0.0)],
    "light_on": [("light_switch_lever_top", 0.0)],
}
LAB_TASKS = {
    "reset_board": [("dev_reset", 1.5)],
    "enter_bootloader": [("dev_boot", 0.0, "press"), ("dev_reset", 0.3), ("dev_boot", 0.0, "release")],
    "arm_stop": [("machine_stop", 0.0, "hover")],         # park the right fingertip over STOP before starting
    "start_cycle": [("machine_start", 0.0)],              # left hand: STOP stays covered the whole cycle
    "stop_machine": [("machine_stop", 0.0, "poke")],      # from the hover if armed (fast), else from rest
    "disarm_stop": [("machine_stop", 0.0, "rest")],
}
RACK_CONTROLS = {   # order matters: a latch pops its caddy out, then the caddy push is taught from there
    "power_button": (0.004, pressed("power_button")),
    "drive0_latch": (0.004, lambda b, a: a["state"]["caddy_out_mm"][0] > 20),
    "drive0_caddy": (0.033, lambda b, a: b["state"]["caddy_out_mm"][0] > 20 and a["state"]["caddy_out_mm"][0] < 3),
    "drive1_latch": (0.004, lambda b, a: a["state"]["caddy_out_mm"][1] > 20),
    "drive1_caddy": (0.033, lambda b, a: b["state"]["caddy_out_mm"][1] > 20 and a["state"]["caddy_out_mm"][1] < 3),
}
CELL_CONTROLS = {"machine_start": LAB_CONTROLS["machine_start"][:2],   # any hand: no STOP watch in the cell, and the
                 "machine_stop": LAB_CONTROLS["machine_stop"]}           # right hand is twice as fast to START
CELL_TASKS = {k: LAB_TASKS[k] for k in ("start_cycle", "stop_machine", "arm_stop", "disarm_stop")}
RACK_TASKS = {      # the runbook: Er01 -> hard_reset; Er42 -> reseat the drive whose LED is red
    "hard_reset": [("power_button", 4.6, "press"), ("power_button", 1.0, "release"), ("power_button", 5.0, "tap")],
    "reseat_drive0": [("drive0_latch", 1.0, "tap"), ("drive0_caddy", 3.0, "tap")],
    "reseat_drive1": [("drive1_latch", 1.0, "tap"), ("drive1_caddy", 3.0, "tap")],
}
SCENES = {"returns": ({}, {}, HERE / "taught_returns.json"), "rack": (RACK_CONTROLS, RACK_TASKS, HERE / "taught_rack.json"), "devices": (CONTROLS, TASKS, HERE / "taught.json"), "lab": (LAB_CONTROLS, LAB_TASKS, HERE / "taught_lab.json"),
          "cell": (CELL_CONTROLS, CELL_TASKS, HERE / "taught_cell.json")}
ALL_TASKS = {**TASKS, **LAB_TASKS, **RACK_TASKS}


def run(arm, task, taught):
    """Do an agent-level task: its taught controls in order, with waits and hold phases."""
    for step in ALL_TASKS[task]:
        ctl, wait, phase = (step + ("tap",))[:3]
        play(arm, ctl, taught, phase)
        if wait:
            arm("sim_wait", seconds=wait)


CHECKS = {   # task -> did it work, from the sim's device state and the events since the task began
    "power_cycle_router": lambda d, ev: "router power OFF" in ev and "router power ON" in ev and "router internet UP" in ev,
    "approve_2fa": lambda d, ev: "security_key pressed" in ev,
    "start_machine": lambda d, ev: "start_button pressed" in ev,
    "light_on": lambda d, ev: d["state"]["light_on"],
    "reset_board": lambda d, ev: d["state"]["board"] == "app" and any(e.startswith("board up") for e in ev),
    "enter_bootloader": lambda d, ev: d["state"]["board"] == "boot",
    "start_cycle": lambda d, ev: d["state"]["machine"] == "running",
    "stop_machine": lambda d, ev: d["state"].get("machine") in ("stopped", "done") and not any("FAULT" in e for e in ev),
    "hard_reset": lambda d, ev: "hard power off (held 4 s)" in ev and any(e.startswith("boot complete") for e in ev),
    "reseat_drive0": lambda d, ev: d["state"]["drives"][0] == "ok" and "drive0 seated: rebuilding" in ev,
    "reseat_drive1": lambda d, ev: d["state"]["drives"][1] == "ok" and "drive1 seated: rebuilding" in ev,
    "arm_stop": lambda d, ev: True,          # motion-only: a failure raises
    "disarm_stop": lambda d, ev: True,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["teach", "replay", "bench"])
    ap.add_argument("task", nargs="?")
    ap.add_argument("--addr", default="tcp://127.0.0.1:15910")
    ap.add_argument("--rounds", type=int, default=5)
    a = ap.parse_args()
    arm = Arm(a.addr)
    kind = arm("sim_devices")["scene"]
    controls, tasks_all, path = SCENES[kind]
    twin = Twin(arm("sim_objects")["desk_top_z"], kind=kind)
    twin.PAD = 0.002          # the sim validates every candidate; plan with a small pad
    if kind == "cell":        # bins hold finished parts taller than their walls: keep presses 6 cm above the rims
        cont = arm("sim_objects")["containers"]
        twin.set_keepouts([(cont[b]["box_min"], cont[b]["box_max"][:2] + [cont[b]["box_max"][2] + 0.06]) for b in ("pass_bin", "reject_bin")])
    p = Planner(arm, arm("sim_objects")["desk_top_z"], twin=False)
    if a.mode == "teach":
        if kind in ("cell", "returns") and a.task in (None, "transfers"):
            teach_transfers(arm, path, kind, ("pass", "reject") if kind == "cell" else ("restock", "refurb", "rma"))
        if a.task != "transfers":
            teach(arm, twin, p, controls, path, [a.task] if a.task else None)
        return
    taught = json.loads(path.read_text())
    tasks = [a.task] if a.task else list(tasks_all)
    ok_n = tot = 0
    for rnd in range(a.rounds if a.mode == "bench" else 1):
        arm("sim_reset", seed=rnd)
        arm("enable")
        for task in tasks:
            t0 = arm("sim_devices")["time"]
            try:
                run(arm, task, taught)
                d = arm("sim_devices")
                ev = [e for t, e in d["history"] if t >= t0]
                good = CHECKS[task](d, ev)
                why = "; ".join(ev)
            except PlanError as e:
                good, why = False, str(e)[:100]
            ok_n += good
            tot += 1
            print(f"round {rnd} {task:20s} {'OK ' if good else 'MISS'} {why}", flush=True)
    print(f"\n{ok_n}/{tot} tasks")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Teach a cell by demonstration, as on the real robot: the planner's twin is desk-only (G1_TWIN=bare) and knows the
cell only through registered containers and keep-out zones; a watched pick -> place -> home is recorded
(record_transfer) and kept only if it worked without disturbing anything (save_transfer); then replayed on other
seeds with run_task. (cell sim on --port-offset 900)
  uv run --with "mcp>=2.2" --with mujoco --with pyzmq --with numpy --with pillow bench_teach_demo.py
"""
import os, sys, json, shutil, tempfile, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
TMP = tempfile.mkdtemp()
shutil.copy(os.path.join(HERE, "stations_cell.json"), TMP + "/stations_demo.json")
sys.argv = [sys.argv[0], TMP, "bare"]
o = 900; T = sys.argv[1] + "/taught_demo.json"
if os.path.exists(T): os.remove(T)
open(T, "w").write("{}")
os.environ.update(G1_ARM_ADDR=f"tcp://127.0.0.1:{15810 + o}", G1_CAM_BASE=f"http://127.0.0.1:{18766 + o}", G1_CAM_HOST="127.0.0.1",
                  G1_DEPTH_PORT=str(15556 + o), G1_CAMCFG_PORT=str(16000 + o), G1_TWIN=sys.argv[2],
                  G1_STATIONS=sys.argv[1] + "/stations_demo.json", G1_TAUGHT_DIR=T,
                  G1_RECEIPTS=sys.argv[1] + "/receipts")
sys.path.insert(0, '/home/lain_iwakura/HACKATHON2026/waldo-arms'); sys.path.insert(0, '/home/lain_iwakura/HACKATHON2026/waldo-sim')
import server as S
from planner import Arm
arm = Arm(os.environ["G1_ARM_ADDR"])
top = -0.06
S.register_keepout("infeed", box_min=[0.265, 0.09, top], box_max=[0.365, 0.24, top + 0.056])
S.register_keepout("panel", box_min=[0.29, -0.27, top], box_max=[0.37, -0.13, top + 0.06])
S.register_keepout("stack_light", box_min=[0.425, -0.245, top], box_max=[0.475, -0.195, top + 0.16])
print("registered:", sorted(S.list_containers()))
pos = lambda: {k: np.array(v["xyz"]) for k, v in arm("sim_objects")["objects"].items()}
def demo(name, part, container, setup, ok, tries=5):
    for k in range(tries):
        setup()
        before = pos()
        S.record_transfer(name)
        try:
            r = S.pick(box=arm("sim_point", item=part, noise_px=3, seed=k)["box"], then_place_in=container)
            S.place(r["side"], container=container); S.home(r["side"])
            after = pos()
            undisturbed = all(np.linalg.norm(after[n] - before[n]) < 0.005 for n in after if n != part)
            good = ok() and undisturbed
            why = "" if good else f"result wrong (target {ok()}, others undisturbed {undisturbed})"
        except Exception as e:
            good, why = False, str(e)[:90]
        if good:
            print(f"{name}: demo {k + 1} kept", S.save_transfer(name)["commands"], "commands"); return True
        S.save_transfer(name, discard=True); print(f"   {name}: demo {k + 1} discarded: {why}")
    return False
def fresh(seed=0):
    arm("sim_reset", seed=seed); arm("enable"); S.home("both")
in_nest = lambda p: (lambda: arm("sim_devices")["state"]["part_in_nest"] == p)
demo("my_load_slot0", "part0", "nest", fresh, in_nest("part0"))
arm("sim_save", name="demo_loaded")
demo("my_unload_pass", "part0", "pass_bin", lambda: (arm("sim_restore", name="demo_loaded"), S.home("both")),
     lambda: arm("sim_objects")["objects"]["part0"]["in_container"] == "pass_bin")
# replay across seeds
taught = json.load(open(T)); print("taught:", sorted(taught))
okl = oku = 0
for seed in range(5):
    fresh(seed)
    try: S.run_task("my_load_slot0")
    except Exception as e: print("load replay error", str(e)[:80])
    l = arm("sim_devices")["state"]["part_in_nest"] == "part0"; okl += l
    try: S.run_task("my_unload_pass")
    except Exception as e: print("unload replay error", str(e)[:80])
    u = arm("sim_objects")["objects"]["part0"]["in_container"] == "pass_bin"; oku += u
    print(f"seed {seed}: load {'OK' if l else 'MISS'}, unload {'OK' if u else 'MISS'}")
print(f"replays: load {okl}/5, unload {oku}/5")

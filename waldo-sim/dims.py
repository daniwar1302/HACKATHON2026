#!/usr/bin/env python3
"""Measure the sim's physical dimensions from the compiled model and compare with real-world references."""
import math
import sys

import mujoco
import numpy as np

import scene
from sim_server import Sim

REF = {   # real-world references (mm) and where they come from
    "G1 standing height": (1320, "Unitree G1 spec"),
    "pelvis height (floor to pelvis)": (790, "G1 standing, = FLOOR_Z"),
    "desk height": (740, "standard desk"),
    "Dex1 max jaw gap": (95, "URDF: 2 x (23 + 24.5) mm"),
    "Dex1 min jaw gap": (6, "URDF: 2 x (23 - 20) mm"),
    "jaw centre ahead of wrist": (152, "measured from Dex1 finger pads"),
    "marker length": (136, "Sharpie-size marker"),
    "marker diameter": (17, "Sharpie-size marker"),
    "pen length": (140, "ballpoint pen"),
    "pencil length": (176, "standard pencil, sharpened"),
    "eraser": (50, "block eraser length"),
    "sticky-note pad": (76, "3x3 in Post-it"),
    "tape roll diameter": (111, "3M vinyl tape roll (scanned)"),
    "scissors length": (216, "scanned"),
    "pen cup diameter": (100, "scanned"),
    "tray length": (307, "scanned organizer tray"),
}

sim = Sim(0, scene.Desk())
m, d = sim.m, sim.d
mujoco.mj_forward(m, d)
robot = [g for g in range(m.ngeom) if m.geom_bodyid[g] not in set(sim.items.values()) | set(sim.containers.values()) and m.geom_bodyid[g] != 0]
top = max(d.geom_xpos[g][2] + m.geom_rbound[g] * 0.6 for g in robot)
head_top = max(d.xpos[b][2] for b in range(m.nbody) if "torso" in m.body(b).name) 
ext = sim.extent
meas = {
    "G1 standing height": (top - scene.FLOOR_Z) * 1000,
    "pelvis height (floor to pelvis)": -scene.FLOOR_Z * 1000,
    "desk height": (sim.desk.top_z - scene.FLOOR_Z) * 1000,
    "Dex1 max jaw gap": 2 * (0.023 + scene.FINGER[1]) * 1000,
    "Dex1 min jaw gap": 2 * (0.023 + scene.FINGER[0]) * 1000,
    "jaw centre ahead of wrist": scene.GRASP_X * 1000,
    "marker length": max(ext["marker_black"]) * 1000,
    "marker diameter": sorted(ext["marker_black"])[0] * 1000,
    "pen length": max(ext["pen"]) * 1000,
    "pencil length": max(ext["pencil"]) * 1000,
    "eraser": max(ext["eraser"]) * 1000,
    "sticky-note pad": max(ext["sticky_notes"]) * 1000,
    "tape roll diameter": max(ext["tape"]) * 1000,
    "scissors length": max(ext["scissors"]) * 1000,
    "pen cup diameter": (sim.box["pen_cup"][1][0] - sim.box["pen_cup"][0][0]) * 1000,
    "tray length": (sim.box["tray"][1][1] - sim.box["tray"][0][1]) * 1000,
}
print(f"{'dimension':34s} {'sim':>7s} {'real':>7s}  {'diff':>6s}  source")
for k, (ref, src) in REF.items():
    v = meas[k]
    flag = "" if abs(v - ref) <= max(5, 0.08 * ref) else "  <-- off"
    print(f"{k:34s} {v:7.0f} {ref:7.0f}  {100 * (v - ref) / ref:+5.0f}%  {src}{flag}")

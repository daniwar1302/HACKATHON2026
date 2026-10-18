#!/usr/bin/env python3
"""waldo-sim: the desk-sorting G1 in MuJoCo, speaking the real robot's protocols so waldo-arms, agents, VLMs and
routines run against it unchanged.

Serves, on its own ports (no clash with tunnels to the real robot):
  arm_server protocol   ZMQ REP  :15810   state enable release freeze ik check move_ee park set_table open grasp
                                          gripper ... + sim_reset {seed, n_items}, sim_objects, sim_point {item, noise_px}
  teleimager config     ZMQ REP  :16000   head camera intrinsics + depth scale
  head depth            ZMQ PUB  :15556   raw z16 (mm), aligned to color
  camera viewer         HTTP     :18766   /frame/head_camera.jpg  /frame/waldo_webcam.jpg  /api/cameras

Physics, not shortcuts: Unitree's G1+Dex1 URDF; arms track commands through stiff position actuators with gravity
compensation; each Dex1 finger squeezes with at most 20 N, so a grasp holds or slips by contact and friction.
IK targets the jaw centre (15.2 cm ahead of the wrist) with either level jaws (the real arm_server's default) or a
top-down approach at a chosen jaw angle (`approach: "down", jaw_deg`), which pens and markers need.

  MUJOCO_GL=glfw uv run --with mujoco --with pyzmq --with pillow --with numpy waldo-sim/sim_server.py [--speed 1]
"""
from __future__ import annotations

import argparse
import copy
import io
import json
import math
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import mujoco
import numpy as np
import zmq
from PIL import Image

import scene
from scene import ARM, FINGER, SIDES

ARM_PORT, CFG_PORT, DEPTH_PORT, HTTP_PORT = 15810, 16000, 15556, 18766
W, H = scene.W, scene.H
WS = {"x": (0.10, 0.60), "y": (-0.55, 0.55), "z": (-0.35, 0.45)}
READY = {"left": [0.28, 0.20, 0.05], "right": [0.28, -0.20, 0.05]}
HANG = {"shoulder_pitch": 0.28, "shoulder_roll": 0.22, "elbow": 0.98}
SPAWN = {"x": (0.26, 0.38), "y": (-0.20, 0.20)}     # visible from the head camera AND reachable at desk height
RADIUS = {"marker_black": 0.064, "marker_blue": 0.064, "pen": 0.07, "pencil": 0.088, "eraser": 0.028,
          "sticky_notes": 0.054, "sticker_sheet": 0.057, "tape": 0.056, "scissors": 0.108, "crayons": 0.07,
          "pencil_case": 0.113, "label_reel": 0.048, "ink_box": 0.049}


def motor_q(finger):   # finger slide (m) -> Dex1 motor angle reported by the real service (0 closed .. 5.4 open)
    return round((finger - FINGER[0]) / (FINGER[1] - FINGER[0]) * 5.4, 3)


def down_R(jaw_deg: float) -> np.ndarray:
    """Grasp frame for a top-down approach: hand x points down, jaws close along (cos a, sin a, 0)."""
    a = math.radians(jaw_deg)
    x, y = np.array([0.0, 0.0, -1.0]), np.array([math.cos(a), math.sin(a), 0.0])
    return np.column_stack([x, y, np.cross(x, y)])


class Devices:
    """Switch/button state for the devices scene: on/off, presses (rising edges), history, and status LEDs."""
    LED_OFF, GREEN, AMBER, LAMP_ON = [0.15, 0.15, 0.15, 1], [0.1, 0.95, 0.2, 1], [1.0, 0.7, 0.1, 1], [1.0, 0.95, 0.6, 1]
    BOOT_S = 3.0                       # router internet LED comes on this long after power-on (like a real reboot)

    def __init__(self, m, d):
        self.m, self.d = m, d
        j = lambda n: m.joint(n).qposadr[0]
        self.q = {"rocker": j("power_strip_rocker"), "key": j("security_key_pad"), "button": j("start_button_cap"),
                  "lever": j("light_switch_lever")}
        self.g = {"power": m.geom("router_led_power").id, "internet": m.geom("router_led_internet").id, "lamp": m.geom("light_switch_lamp").id}
        self.bodies = {b for b in range(m.nbody) if any(m.body(b).name.startswith(n) for n in scene.DEVICES)}
        self.reset()

    def reset(self):
        self.d.qpos[self.q["rocker"]] = 0.24          # strip ON: the router is running
        self.d.qpos[self.q["lever"]] = -0.30          # light OFF (lever down)
        self.d.qpos[self.q["key"]] = self.d.qpos[self.q["button"]] = 0.0
        self.history = []                             # (t, event)
        self.presses = {"security_key": 0, "start_button": 0}
        self._pressed = {"security_key": False, "start_button": False}
        self.power_on_t = 0.0
        self.hung = getattr(self, "seed", 0) % 2 == 1      # odd seeds: router up but its internet is down until power-cycled
        self.state = self.read()
        self.state["internet"] = not self.hung

    def read(self):
        q = self.d.qpos
        return {"strip_on": bool(q[self.q["rocker"]] > 0.0), "light_on": bool(q[self.q["lever"]] > 0.0),
                "rocker_rad": round(float(q[self.q["rocker"]]), 3), "lever_rad": round(float(q[self.q["lever"]]), 3)}

    def update(self):
        t = self.d.time
        now = self.read()
        if now["strip_on"] != self.state["strip_on"]:
            self.history.append((round(t, 2), "router power ON" if now["strip_on"] else "router power OFF"))
            if now["strip_on"]:
                self.power_on_t = t
            else:
                self.hung = False                           # a power cycle clears the hang
        if now["light_on"] != self.state["light_on"]:
            self.history.append((round(t, 2), "light ON" if now["light_on"] else "light OFF"))
        internet = not self.hung and now["strip_on"] and (t - self.power_on_t >= self.BOOT_S or self.state.get("internet") and now["strip_on"] == self.state["strip_on"])
        if internet and not self.state.get("internet"):
            self.history.append((round(t, 2), "router internet UP"))
        now["internet"] = bool(internet)
        for name, key, th in (("security_key", "key", -0.0018), ("start_button", "button", -0.005)):
            down = self.d.qpos[self.q[key]] < th
            if down and not self._pressed[name]:
                self.presses[name] += 1
                self.history.append((round(t, 2), f"{name} pressed"))
            self._pressed[name] = down
        self.state = now
        rgba = self.m.geom_rgba
        rgba[self.g["power"]] = self.GREEN if now["strip_on"] else self.LED_OFF
        rgba[self.g["internet"]] = self.GREEN if now["internet"] else (self.AMBER if now["strip_on"] else self.LED_OFF)
        rgba[self.g["lamp"]] = self.LAMP_ON if now["light_on"] else self.LED_OFF

    def targets(self):
        """Where to touch each control (pelvis frame) and which way to push, from the live model."""
        m, d = self.m, self.d
        def wpt(body, local):
            b = m.body(body).id
            return (d.xpos[b] + d.xmat[b].reshape(3, 3) @ np.asarray(local, float)).round(4).tolist()
        return {
            # pressing the rocker's +y end tips it to OFF, the -y end to ON (measured in the sim)
            "power_strip_off_end": {"point": wpt("power_strip_rocker", [0, 0.012, 0.006]), "push": [0, 0, -1]},
            "power_strip_on_end": {"point": wpt("power_strip_rocker", [0, -0.012, 0.006]), "push": [0, 0, -1]},
            "security_key_pad": {"point": wpt("security_key_pad", [0, 0, 0.0015]), "push": [0, 0, -1]},
            "start_button": {"point": wpt("start_button_cap", [0, 0, 0.006]), "push": [0, 0, -1]},
            "light_switch_lever_bottom": {"point": wpt("light_switch_lever", [-0.014, 0, -0.012]), "push": [1, 0, 0]},
            "light_switch_lever_top": {"point": wpt("light_switch_lever", [-0.014, 0, 0.012]), "push": [1, 0, 0]},
        }


class LabDevices:
    """Lab scene: a dev board (BOOT/RESET, status LED, 4-digit display) and a machine (START/STOP, stack light,
    pressure gauge). Odd seeds have a faulty machine whose pressure keeps climbing: over 8 bar the operator must
    press STOP before it trips at 9.5 bar. Seeds divisible by 3 start with the board's firmware hung."""
    OFF, GREEN, RED, BLUE, YELLOW = [0.04, 0.04, 0.04, 1], [0.1, 0.95, 0.2, 1], [1.0, 0.12, 0.08, 1], [0.02, 0.1, 1.0, 1], [1.0, 0.85, 0.1, 1]
    SEG_ON, SEG_OFF = [1.0, 0.12, 0.08, 1], [0.05, 0.01, 0.01, 1]      # emissive: off must be near black
    BUTTONS = {"dev_boot": "dev_board_boot", "dev_reset": "dev_board_reset", "machine_start": "machine_start", "machine_stop": "machine_stop"}
    REBOOT_S, CYCLE_S, TRIP_BAR = 1.0, 10.0, 9.5

    def __init__(self, m, d):
        self.m, self.d = m, d
        self.q = {k: m.joint(j).qposadr[0] for k, j in self.BUTTONS.items()}
        self.q_needle, self.v_needle = m.joint("gauge_needle").qposadr[0], m.joint("gauge_needle").dofadr[0]
        self.g = {"led": m.geom("dev_board_led").id, **{l: m.geom(f"stack_light_{l}").id for l in ("green", "yellow", "red")},
                  "seg": {(k, s): m.geom(f"dev_board_seg{k}{s}").id for k in range(4) for s in "abcdefg"}}
        self.bodies = {b for b in range(m.nbody) if any(m.body(b).name.startswith(n) for n in scene.LAB)}
        self.seed = 0
        self.reset()

    def reset(self):
        for a in self.q.values():
            self.d.qpos[a] = 0.0
        rng = np.random.default_rng(self.seed)
        self.rng = rng
        self.history, self.presses = [], {k: 0 for k in self.BUTTONS}
        self._pressed = {k: False for k in self.BUTTONS}
        self.board = "hung" if self.seed % 3 == 0 else "app"
        self.firmware, self.code, self.board_t = 1, f"{rng.integers(0, 10000):04d}", 0.0
        self.machine, self.machine_t, self.pressure, self.faulty = "idle", 0.0, 0.0, self.seed % 2 == 1
        self.shown = None
        self.state = self.read()

    def read(self):
        return {"board": self.board, "firmware": self.firmware, "display": self.display(), "machine": self.machine,
                "pressure_bar": round(self.pressure, 2)}

    def display(self):
        return {"app": self.code, "hung": "----", "boot": "boot", "rebooting": "    "}[self.board]

    def flash(self):
        if self.board != "boot":
            raise ValueError(f"board is not in the bootloader (state {self.board}); hold BOOT and tap RESET first")
        self.firmware += 1
        self.history.append((round(self.d.time, 2), f"flashed firmware v{self.firmware}"))

    def update(self):
        d, t = self.d, self.d.time
        for k, a in self.q.items():
            down = d.qpos[a] < -0.0018 if k.startswith("dev") else d.qpos[a] < -0.005
            if down and not self._pressed[k]:
                self.presses[k] += 1
                self.history.append((round(t, 2), f"{k} pressed"))
                self._on_press(k, t)
            self._pressed[k] = down
        if self.board == "rebooting" and t - self.board_t >= self.REBOOT_S:
            self.board, self.code = "app", f"{self.rng.integers(0, 10000):04d}"
            self.history.append((round(t, 2), f"board up: firmware v{self.firmware}, code {self.code}"))
        # machine: pressure follows a first-order rise while running; a faulty one keeps climbing
        dt = self.m.opt.timestep
        if self.machine == "running":
            r = t - self.machine_t
            target = 6.0 + (0.5 * (r - 4.0) if self.faulty and r > 4.0 else 0.0)
            self.pressure += (target - self.pressure) * dt / 0.8
            if self.pressure >= self.TRIP_BAR:
                self.machine = "fault"
                self.history.append((round(t, 2), "machine FAULT: overpressure trip"))
            elif r >= self.CYCLE_S and not self.faulty:
                self.machine = "done"
                self.history.append((round(t, 2), "machine cycle complete"))
        else:
            self.pressure += (0.0 - self.pressure) * dt / 0.6
        d.qpos[self.q_needle] = -np.radians(-135 + 27 * self.pressure)      # clockwise from up as the camera sees it
        d.qvel[self.v_needle] = 0.0
        self.state = self.read()
        self._paint()

    def _on_press(self, k, t):
        if k == "dev_reset":
            if self._pressed["dev_boot"]:
                self.board = "boot"
                self.history.append((round(t, 2), "board in bootloader"))
            else:
                self.board, self.board_t = "rebooting", t
        elif k == "machine_start" and self.machine in ("idle", "done", "stopped"):
            self.machine, self.machine_t = "running", t
            self.history.append((round(t, 2), "machine running"))
        elif k == "machine_stop" and self.machine == "running":
            self.machine = "stopped"
            self.history.append((round(t, 2), f"machine stopped by operator at {self.pressure:.1f} bar"))

    def _paint(self):
        key = (self.board, self.code, self.machine)
        if key == self.shown:
            return
        self.shown = key
        rgba = self.m.geom_rgba
        rgba[self.g["led"]] = {"app": self.GREEN, "hung": self.RED, "boot": self.BLUE, "rebooting": self.OFF}[self.board]
        for k, ch in enumerate(self.display()):
            for sname in "abcdefg":
                rgba[self.g["seg"][(k, sname)]] = self.SEG_ON if sname in scene.SEG[ch] else self.SEG_OFF
        lamp = {"running": "green", "fault": "red"}.get(self.machine, "yellow")
        for l in ("green", "yellow", "red"):
            rgba[self.g[l]] = {"green": self.GREEN, "yellow": self.YELLOW, "red": self.RED}[l] if l == lamp else self.OFF

    def targets(self):
        m, d = self.m, self.d
        def top(body, dz):
            b = m.body(body).id
            return (d.xpos[b] + d.xmat[b].reshape(3, 3) @ np.array([0, 0, dz])).round(4).tolist()
        return {"dev_boot": {"point": top("dev_board_boot", 0.003), "push": [0, 0, -1]},
                "dev_reset": {"point": top("dev_board_reset", 0.003), "push": [0, 0, -1]},
                "machine_start": {"point": top("machine_start", 0.006), "push": [0, 0, -1]},
                "machine_stop": {"point": top("machine_stop", 0.006), "push": [0, 0, -1]}}


class CellDevices:
    """Machine-tending cell: START only runs with a part in the nest; a cycle finishes the part (raw grey -> bright)
    and the QC lamp shows PASS (green) or REJECT (red) while it sits in the nest. Pulling a part mid-cycle faults.
    One part per seed is out of tolerance (part{seed % 3})."""
    OFF, GREEN, RED, YELLOW = LabDevices.OFF, LabDevices.GREEN, LabDevices.RED, LabDevices.YELLOW
    CYCLE_S = 12.0

    def __init__(self, m, d, items):
        self.m, self.d, self.items = m, d, items
        self.q = {k: m.joint(k).qposadr[0] for k in ("machine_start", "machine_stop")}
        self.g = {"qc": m.geom("qc_light").id, **{l: m.geom(f"stack_light_{l}").id for l in ("green", "yellow", "red")},
                  **{n: m.geom(f"{n}_geom").id for n in items}}
        self.nest = m.body("nest").id
        self.bodies = {b for b in range(m.nbody) if m.body(b).name in ("machine", "machine_start", "machine_stop", "stack_light", "infeed")}
        self.seed = 0
        self.reset()

    def reset(self):
        for a in self.q.values():
            self.d.qpos[a] = 0.0
        self.history, self.presses = [], {k: 0 for k in self.q}
        self._pressed = {k: False for k in self.q}
        self.machine, self.t0, self.part, self.qc = "idle", 0.0, None, None
        self.bad = f"part{self.seed % 3}"
        self.done = {}                                   # part -> "pass" / "reject"
        self.shown = None
        for n in self.items:
            self.m.geom_rgba[self.g[n]] = scene.PART_RAW
        self.state = self.read()

    def in_nest(self):
        c = self.d.xpos[self.nest]
        for n, b in self.items.items():
            p = self.d.xpos[b]
            if np.hypot(*(p[:2] - c[:2])) < 0.016 and p[2] < c[2] + 0.045:
                return n
        return None

    def read(self):
        return {"machine": self.machine, "part_in_nest": self.in_nest(), "qc": self.qc, "done": dict(self.done)}

    def update(self):
        d, t = self.d, self.d.time
        for k, a in self.q.items():
            down = d.qpos[a] < -0.005
            if down and not self._pressed[k]:
                self.presses[k] += 1
                self.history.append((round(t, 2), f"{k} pressed"))
                self._on_press(k, t)
            self._pressed[k] = down
        nest = self.in_nest()
        if self.machine == "running":
            if nest != self.part:
                self.machine = "fault"
                self.history.append((round(t, 2), "machine FAULT: part moved during the cycle"))
            elif t - self.t0 >= self.CYCLE_S:
                self.machine, self.qc = "done", "reject" if self.part == self.bad else "pass"
                self.done[self.part] = self.qc
                self.m.geom_rgba[self.g[self.part]] = scene.PART_DONE
                self.history.append((round(t, 2), f"cycle complete: {self.part} QC {self.qc.upper()}"))
        if self.qc and nest is None:
            self.qc = None                               # lamp shows the result of the part in the nest only
        self.state = self.read()
        self._paint()

    def _on_press(self, k, t):
        nest = self.in_nest()
        if k == "machine_start" and self.machine in ("idle", "done", "stopped"):
            if nest is None:
                self.history.append((round(t, 2), "START ignored: nest empty"))
            elif nest in self.done:
                self.history.append((round(t, 2), f"START ignored: {nest} already machined"))
            else:
                self.machine, self.t0, self.part, self.qc = "running", t, nest, None
                self.history.append((round(t, 2), f"machine running {nest}"))
        elif k == "machine_stop" and self.machine == "running":
            self.machine = "stopped"
            self.history.append((round(t, 2), "machine stopped by operator"))

    def _paint(self):
        key = (self.machine, self.qc)
        if key == self.shown:
            return
        self.shown = key
        rgba = self.m.geom_rgba
        rgba[self.g["qc"]] = {"pass": self.GREEN, "reject": self.RED}.get(self.qc, self.OFF)
        lamp = {"running": "green", "fault": "red"}.get(self.machine, "yellow")
        for l in ("green", "yellow", "red"):
            rgba[self.g[l]] = {"green": self.GREEN, "yellow": self.YELLOW, "red": self.RED}[l] if l == lamp else self.OFF

    def targets(self):
        m, d = self.m, self.d
        def top(body):
            b = m.body(body).id
            return (d.xpos[b] + d.xmat[b].reshape(3, 3) @ np.array([0, 0, 0.006])).round(4).tolist()
        return {k: {"point": top(k), "push": [0, 0, -1]} for k in ("machine_start", "machine_stop")}


class RackDevices:
    """Datacenter server: seed % 3 == 0 hung (Er01: hold power 4 s, then tap to boot), 1 / 2: drive 0 / 1 offline (Er42:
    press its latch, the caddy pops out, push it home: it rebuilds). Status: good / Er01 / Er42 / boot / blank (off)."""
    OFF, GREEN, RED, AMBER, BLUE = LabDevices.OFF, LabDevices.GREEN, LabDevices.RED, [1.0, 0.6, 0.05, 1], LabDevices.BLUE
    SEG_ON, SEG_OFF = LabDevices.SEG_ON, LabDevices.SEG_OFF
    HOLD_OFF_S, BOOT_S, REBUILD_S, EJECT_M = 4.0, 4.0, 2.0, 0.03

    def __init__(self, m, d):
        self.m, self.d = m, d
        j = lambda n: (m.joint(n).qposadr[0], m.joint(n).dofadr[0])
        self.q = {k: j(f"server_{k}") for k in ("power", "drive0_latch", "drive1_latch", "drive0", "drive1")}
        self.g = {"led": m.geom("server_led").id, "d0": m.geom("server_drive0_led").id, "d1": m.geom("server_drive1_led").id,
                  "seg": {(k, sn): m.geom(f"server_seg{k}{sn}").id for k in range(4) for sn in "abcdefg"}}
        self.bodies = {b for b in range(m.nbody) if m.body(b).name.startswith("server")}
        self.seed = 0
        self.reset()

    def reset(self):
        for a, _ in self.q.values():
            self.d.qpos[a] = 0.0
        f = self.seed % 3
        self.server = "hung" if f == 0 else "degraded"
        self.drives = ["ok", "ok"]
        if f:
            self.drives[f - 1] = "fault"
        self.history, self.presses = [], {"power_button": 0, "drive0_latch": 0, "drive1_latch": 0}
        self._pressed = {k: False for k in self.presses}
        self.power_t, self.boot_t, self.rebuild_t, self.out = None, 0.0, [None, None], [False, False]
        self.shown = None
        self.state = self.read()

    def read(self):
        return {"server": self.server, "display": self.display(), "drives": list(self.drives),
                "caddy_out_mm": [round(float(self.d.qpos[self.q[f"drive{k}"][0]]) * 1000) for k in (0, 1)]}

    def display(self):
        return {"good": "good", "hung": "Er01", "degraded": "Er42", "booting": "boot", "off": "    "}[self.server]

    def update(self):
        d, t = self.d, self.d.time
        for k, key, th in (("power_button", "power", 0.002), ("drive0_latch", "drive0_latch", 0.0015), ("drive1_latch", "drive1_latch", 0.0015)):
            down = d.qpos[self.q[key][0]] > th
            if down and not self._pressed[k]:
                self.presses[k] += 1
                self.history.append((round(t, 2), f"{k} pressed"))
                if k == "power_button":
                    self.power_t = t
                else:
                    self._eject(int(k[5]), t)
            if not down and self._pressed[k] and k == "power_button":
                held = t - (self.power_t or t)
                if self.server == "off" and held < 2.0:
                    self.server, self.boot_t = "booting", t
                    self.history.append((round(t, 2), "power on: booting"))
                self.power_t = None
            self._pressed[k] = down
        if self.power_t is not None and self.server != "off" and t - self.power_t >= self.HOLD_OFF_S:
            self.server = "off"
            self.history.append((round(t, 2), "hard power off (held 4 s)"))
        if self.server == "booting" and t - self.boot_t >= self.BOOT_S:
            self.server = "good" if all(s == "ok" for s in self.drives) else "degraded"
            self.history.append((round(t, 2), f"boot complete: {self.display()}"))
        for k in (0, 1):
            q = d.qpos[self.q[f"drive{k}"][0]]
            if self.out[k] and q < 0.003:                           # pushed home: rebuild
                self.out[k], self.rebuild_t[k] = False, t
                self.drives[k] = "rebuild"
                self.history.append((round(t, 2), f"drive{k} seated: rebuilding"))
            if self.drives[k] == "rebuild" and t - self.rebuild_t[k] >= self.REBUILD_S:
                self.drives[k] = "ok"
                self.history.append((round(t, 2), f"drive{k} online"))
        if self.server in ("good", "degraded"):
            self.server = "good" if all(s == "ok" for s in self.drives) else "degraded"
        self.state = self.read()
        self._paint()

    def _eject(self, k, t):
        a, v = self.q[f"drive{k}"]
        self.d.qpos[a], self.d.qvel[v] = self.EJECT_M, 0.0
        self.out[k], self.drives[k] = True, "out"
        self.history.append((round(t, 2), f"drive{k} ejected"))

    def _paint(self):
        key = (self.server, tuple(self.drives))
        if key == self.shown:
            return
        self.shown = key
        rgba = self.m.geom_rgba
        rgba[self.g["led"]] = {"good": self.GREEN, "hung": self.AMBER, "degraded": self.AMBER, "booting": self.BLUE}.get(self.server, self.OFF)
        for k in (0, 1):
            rgba[self.g[f"d{k}"]] = self.OFF if self.server == "off" else {"ok": self.GREEN, "rebuild": self.AMBER}.get(self.drives[k], self.RED)
        for k, ch in enumerate(self.display()):
            for sn in "abcdefg":
                rgba[self.g["seg"][(k, sn)]] = self.SEG_ON if sn in scene.SEG[ch] else self.SEG_OFF

    def targets(self):
        m, d = self.m, self.d
        def face(body, local):
            b = m.body(body).id
            return (d.xpos[b] + d.xmat[b].reshape(3, 3) @ np.asarray(local, float)).round(4).tolist()
        out = {"power_button": {"point": face("server_power", [0, 0, -0.004]), "push": [1, 0, 0]}}
        for k in (0, 1):
            out[f"drive{k}_latch"] = {"point": face(f"server_drive{k}_latch", [0, 0, -0.003]), "push": [1, 0, 0]}
            out[f"drive{k}_caddy"] = {"point": face(f"server_drive{k}", [-0.003, 0, 0]), "push": [1, 0, 0]}
        return out


class TabletDevices:
    """A test tablet running ShopApp: a fingertip touching the glass is a tap at that spot; buttons switch screens.
    The planted bug: paying with the test card 4242 (which must succeed) fails with E402."""
    FLOW = {("home", "Browse"): "product", ("home", "Cart"): "cart?", ("product", "Add to cart"): "cart+",
            ("product", "Back"): "home", ("cart_empty", "Back"): "home", ("cart", "Checkout"): "checkout",
            ("cart", "Back"): "home", ("checkout", "Pay $12"): "error", ("checkout", "Back"): "cart", ("error", "OK"): "cart"}

    def __init__(self, m, d):
        self.m, self.d = m, d
        self.glass = m.geom("tablet_glass").id
        self.body = m.body("tablet").id
        self.scr = {n: m.geom(f"screen_{n}").id for n in scene.SCREENS}
        self.z_on = float(m.geom_pos[self.scr["home"]][2])
        self.bodies = {self.body}
        self.seed = 0
        self.reset()

    def reset(self):
        self.screen, self.cart, self.history, self.taps = "home", 0, [], 0
        self.presses = {}
        self._touch, self._released_t = False, -1.0
        self._show()
        self.state = self.read()

    def read(self):
        return {"screen": self.screen, "cart_items": self.cart, "taps": self.taps}

    def _show(self):
        for n, g in self.scr.items():
            self.m.geom_pos[g][2] = self.z_on if n == self.screen else -5.0

    def _local(self, p):
        R = self.d.xmat[self.body].reshape(3, 3)
        q = R.T @ (np.asarray(p) - self.d.xpos[self.body])
        return (scene.TABLET_W / 2 - q[1]) / scene.TABLET_W, (scene.TABLET_H / 2 - q[0]) / scene.TABLET_H   # u right, v down

    def update(self):
        m, d, t = self.m, self.d, self.d.time
        touch = None
        for c in d.contact[:d.ncon]:
            if self.glass in (c.geom1, c.geom2):
                other = c.geom2 if c.geom1 == self.glass else c.geom1
                if "dex1" in m.body(m.geom_bodyid[other]).name:
                    touch = c.pos.copy()
                    break
        if touch is not None and not self._touch and t - self._released_t > 0.15:     # touch-down = a tap
            self._tap(*self._local(touch), t)
        if touch is None and self._touch:
            self._released_t = t
        self._touch = touch is not None
        self.state = self.read()

    def _tap(self, u, v, t):
        self.taps += 1
        hit = next((lab for lab, (u0, v0, u1, v1) in scene.SCREENS[self.screen][1] if u0 <= u <= u1 and v0 <= v <= v1), None)
        self.history.append((round(t, 2), f"tap ({u:.2f}, {v:.2f}) on {self.screen}: {hit or 'no button'}"))
        if hit is None:
            return
        nxt = self.FLOW[(self.screen, hit)]
        if nxt == "cart?":
            nxt = "cart" if self.cart else "cart_empty"
        elif nxt == "cart+":
            self.cart, nxt = self.cart + 1, "cart"
        self.screen = nxt
        self._show()

    def targets(self):
        """Centre of each button on the current screen (pelvis frame): for benches, not for the agent."""
        R = self.d.xmat[self.body].reshape(3, 3)
        out = {}
        for lab, (u0, v0, u1, v1) in scene.SCREENS[self.screen][1]:
            u, v = (u0 + u1) / 2, (v0 + v1) / 2
            local = np.array([scene.TABLET_H / 2 - v * scene.TABLET_H, scene.TABLET_W / 2 - u * scene.TABLET_W, 0.006])
            out[lab] = {"point": (self.d.xpos[self.body] + R @ local).round(4).tolist(), "push": [0, 0, -1]}
        return out


class Sim:
    def __init__(self, speed: float, desk: scene.Desk, item_set: str = "big", kind: str = "sort"):
        self.desk, self.item_set, self.kind = desk, item_set, kind
        self.item_defs = scene.CELL_PARTS if kind == "cell" else scene.ITEMS
        self.m = (scene.build(desk, items={}, containers={}, devices=kind) if kind in ("devices", "lab", "rack", "tablet") else
                  scene.build(desk, items=scene.CELL_PARTS, containers={}, devices="cell") if kind == "cell" else scene.build(desk))
        self.d = mujoco.MjData(self.m)
        self.speed = speed
        m = self.m
        self.qadr = {s: [m.joint(f"{s}_{j}_joint").qposadr[0] for j in ARM] for s in SIDES}
        self.dadr = {s: [m.joint(f"{s}_{j}_joint").dofadr[0] for j in ARM] for s in SIDES}
        self.act = {s: [m.actuator(f"{s}_{j}_joint").id for j in ARM] for s in SIDES}
        self.lim = {s: np.array([m.joint(f"{s}_{j}_joint").range for j in ARM]) for s in SIDES}
        self.fing_act = {s: [m.actuator(f"{s}_dex1_finger_joint_{k}").id for k in (1, 2)] for s in SIDES}
        self.fing_q = {s: [m.joint(f"{s}_dex1_finger_joint_{k}").qposadr[0] for k in (1, 2)] for s in SIDES}
        self.fing_body = {s: {m.body(f"{s}_dex1_finger_link_{k}").id for k in (1, 2)} for s in SIDES}
        self.site = {s: m.site(f"{s}_grasp").id for s in SIDES}
        self.arm_bodies = {s: {b for b in range(m.nbody) if m.body(b).name.startswith(s) and any(
            k in m.body(b).name for k in ("shoulder", "elbow", "wrist", "dex1"))} for s in SIDES}
        self.items = {} if kind in ("devices", "lab", "rack", "tablet") else {n: scene.item_body(m, n) for n in self.item_defs}
        self.dev = (Devices(m, self.d) if kind == "devices" else LabDevices(m, self.d) if kind == "lab" else
                    CellDevices(m, self.d, self.items) if kind == "cell" else RackDevices(m, self.d) if kind == "rack" else
                    TabletDevices(m, self.d) if kind == "tablet" else None)
        self.item_qadr = {n: m.jnt_qposadr[m.body_jntadr[b]] for n, b in self.items.items()}
        self.item_dadr = {n: m.jnt_dofadr[m.body_jntadr[b]] for n, b in self.items.items()}
        self.containers = ({} if kind in ("devices", "lab", "rack", "tablet") else {n: m.body(n).id for n in ("nest", "pass_bin", "reject_bin")}
                           if kind == "cell" else {n: m.body(f"{n}_model").id for n in scene.CONTAINERS})
        self.static_geoms = {g for g in range(m.ngeom) if m.geom_bodyid[g] == 0 or m.geom_bodyid[g] in self.containers.values()}
        mujoco.mj_forward(m, self.d)
        self.box = {n: self._aabb(b) for n, b in self.containers.items()}
        self.drop = None        # per-container drop spots, mapped at the first reset (items then sit outside them)
        ec = {n: self._body_extent(b) for n, b in self.items.items()}
        self.extent = {n: e for n, (e, _) in ec.items()}
        self.centre = {n: c for n, (_, c) in ec.items()}     # scanned meshes aren't centred on the body origin
        self.rest_z = {}
        self.enabled = self.frozen = self.moving = False
        self.freeze_reason, self.table = "", None
        self.snapshots = {}
        self.reset(0)

    def _aabb(self, body):
        pts = []
        for g in range(self.m.ngeom):
            if self.m.geom_bodyid[g] == body and self.m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and self.m.geom_group[g] == 3:
                mid = self.m.geom_dataid[g]
                v = self.m.mesh_vert[self.m.mesh_vertadr[mid]:self.m.mesh_vertadr[mid] + self.m.mesh_vertnum[mid]]
                pts.append((self.d.geom_xmat[g].reshape(3, 3) @ v.T).T + self.d.geom_xpos[g])
            elif self.m.geom_bodyid[g] == body and self.m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:   # built bins/fixtures
                sz = self.m.geom_size[g]
                v = np.array([[a * sz[0], b * sz[1], c * sz[2]] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)])
                pts.append((self.d.geom_xmat[g].reshape(3, 3) @ v.T).T + self.d.geom_xpos[g])
        p = np.vstack(pts)
        return p.min(0), p.max(0)

    def _body_extent(self, body):
        """Item size along its own x, y, z axes (m), from its collision shapes."""
        m, d = self.m, self.d
        Rb, pb, pts = d.xmat[body].reshape(3, 3), d.xpos[body], []
        for g in range(m.ngeom):
            if m.geom_bodyid[g] != body or not (m.geom_contype[g] or m.geom_conaffinity[g]):
                continue
            Rg, pg, sz, t = d.geom_xmat[g].reshape(3, 3), d.geom_xpos[g], m.geom_size[g], m.geom_type[g]
            if t == mujoco.mjtGeom.mjGEOM_MESH:
                mid = m.geom_dataid[g]
                local = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
            elif t == mujoco.mjtGeom.mjGEOM_BOX:
                local = np.array([[a, b, c] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)]) * sz
            else:   # capsule / cylinder: axis z, radius sz[0], half-length sz[1]
                r, h = sz[0], sz[1] + (sz[0] if t == mujoco.mjtGeom.mjGEOM_CAPSULE else 0)
                local = np.array([[a * r, b * r, c * h] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)])
            pts.append(((Rg @ local.T).T + pg - pb) @ Rb)   # into the body frame
        p = np.vstack(pts)
        return p.max(0) - p.min(0), (p.max(0) + p.min(0)) / 2     # size, geometric centre (body frame)

    def _drop_spot(self, lo, hi, cell=0.005):
        """Where to drop into a container: the open floor point farthest from any wall or divider (raycast height
        map + brute-force distance transform). On the robot the same comes from the head camera's depth."""
        xs, ys = np.arange(lo[0] + cell / 2, hi[0], cell), np.arange(lo[1] + cell / 2, hi[1], cell)
        gid = np.zeros(1, np.int32)
        z = np.array([[0.5 - mujoco.mj_ray(self.m, self.d, np.array([x, y, 0.5]), np.array([0, 0, -1.0]), None, 1, -1, gid)
                       for y in ys] for x in xs])
        floor = z.min()
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        open_ = z < floor + 0.02
        P, Q = np.stack([X[open_], Y[open_]], 1), np.stack([X[~open_], Y[~open_]], 1)
        border = np.minimum.reduce([P[:, 0] - lo[0], hi[0] - P[:, 0], P[:, 1] - lo[1], hi[1] - P[:, 1]])
        clear = border if not len(Q) else np.minimum(border, np.sqrt(((P[:, None] - Q[None]) ** 2).sum(-1)).min(1))
        k = int(np.argmax(clear))
        return [round(float(P[k, 0]), 3), round(float(P[k, 1]), 3), round(float(clear[k]), 3)]

    # ----- world -----
    def reset(self, seed: int, n_items: int | None = None):
        m, d = self.m, self.d
        mujoco.mj_resetData(m, d)
        rng = np.random.default_rng(seed)
        for s in SIDES:
            for j, a in zip(ARM, self.qadr[s]):
                d.qpos[a] = HANG.get(j, 0.0) * (-1 if (s == "right" and j == "shoulder_roll") else 1)
            for a in self.fing_q[s]:
                d.qpos[a] = FINGER[1]
        names = list(self.item_defs) if self.items else []
        pool = (scene.ITEM_SETS.get(self.item_set) or names) if names and self.kind != "cell" else names
        chosen = list(rng.choice(pool, size=min(n_items or len(pool), len(pool)), replace=False)) if self.kind != "cell" else list(names)
        placed = []
        if self.kind == "cell":              # fixtured: each part stands in its infeed pocket (+-2 mm)
            for n, (x, y) in zip(names, scene.INFEED_SLOTS):
                qa = self.item_qadr[n]
                d.qpos[qa:qa + 3] = [x + rng.uniform(-0.002, 0.002), y + rng.uniform(-0.002, 0.002), self.desk.top_z + 0.006 + 0.0255]
                d.qpos[qa + 3:qa + 7] = [1, 0, 0, 0]
            names = []
        for n in names:
            qa = self.item_qadr[n]
            if n in chosen:
                for _ in range(500):
                    xy = np.array([rng.uniform(*SPAWN["x"]), rng.uniform(*SPAWN["y"])])
                    clear_of_containers = all(not (lo[0] - 0.04 - RADIUS[n] < xy[0] < hi[0] + 0.04 + RADIUS[n] and
                                                   lo[1] - 0.04 - RADIUS[n] < xy[1] < hi[1] + 0.04 + RADIUS[n])
                                              for lo, hi in self.box.values())
                    if clear_of_containers and all(np.linalg.norm(xy - p) > RADIUS[n] + RADIUS[o] + 0.01 for o, p in placed):
                        break
                else:                              # no room left on this desk: leave it off rather than stack it
                    chosen.remove(n)
            if n not in chosen:                    # parked out of the way, behind the robot
                d.qpos[qa:qa + 3] = [-0.8, 0.15 * names.index(n) - 0.7, scene.FLOOR_Z + 0.05]
                continue
            placed.append((n, xy))
            yaw = rng.uniform(-math.pi, math.pi)
            c = self.centre[n]                         # place the item's geometric centre (not its origin) at xy
            d.qpos[qa:qa + 2] = xy - np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]]) @ c[:2]
            q0, qz, out = d.qpos[qa + 3:qa + 7].copy(), np.array([math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]), np.zeros(4)
            mujoco.mju_mulQuat(out, qz, q0)
            d.qpos[qa + 3:qa + 7] = out
            d.qpos[qa + 2] = self.desk.top_z + 0.03   # dropped from 3 cm, settles below
        self.chosen = chosen
        if self.drop is None:
            mujoco.mj_forward(m, d)
            self.drop = {n: self._drop_spot(lo, hi) for n, (lo, hi) in self.box.items()}
        if self.dev:
            self.dev.seed = seed
            self.dev.reset()
        mujoco.mj_forward(m, d)
        self._hold_ctrl()
        for s in SIDES:
            for a in self.fing_act[s]:
                d.ctrl[a] = FINGER[1]
        self.enabled = self.frozen = self.moving = False
        self.table = None
        self.step(600)                             # 1.2 s to settle

    def _hold_ctrl(self):
        for i in range(self.m.nu):
            self.d.ctrl[i] = self.d.qpos[self.m.jnt_qposadr[self.m.actuator_trnid[i, 0]]]

    viewer = None          # a mujoco.viewer window, if --viewer
    _last_sync = 0.0

    def step(self, n=1):
        for _ in range(n):
            mujoco.mj_step(self.m, self.d)
            if self.dev:
                self.dev.update()
            if self.viewer is not None and time.monotonic() - self._last_sync > 1 / 60:
                self._last_sync = time.monotonic()
                self.viewer.sync()

    # ----- arms -----
    def arm_q(self, s):
        return self.d.qpos[self.qadr[s]].copy()

    def q14(self):
        return np.concatenate([self.arm_q("left"), self.arm_q("right")])

    def grasp_pose(self, s, q7=None):
        save = None
        if q7 is not None:
            save = self.d.qpos[self.qadr[s]].copy()
            self.d.qpos[self.qadr[s]] = q7
        mujoco.mj_kinematics(self.m, self.d)
        p, R = self.d.site_xpos[self.site[s]].copy(), self.d.site_xmat[self.site[s]].reshape(3, 3).copy()
        if save is not None:
            self.d.qpos[self.qadr[s]] = save
            mujoco.mj_kinematics(self.m, self.d)
        return p, R

    def ik(self, s, target, R_target=None, iters=300):
        """Damped least squares on the jaw centre. Orientation: R_target if given, else level jaws (hand y horizontal,
        like the real arm_server's jaws_horizontal default). Tries the current pose and two natural seeds."""
        m, d = self.m, self.d
        save = d.qpos.copy()
        tgt = np.asarray(target, float)
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        seeds = [self.arm_q(s), np.array([-0.3, 0.3, 0, 1.2, 0, 0, 0]), np.array([-0.8, 0.2, 0, 1.4, 0, -0.6, 0])]
        if s == "right":
            seeds[1][1] *= -1
            seeds[2][1] *= -1
        best = None
        for q in seeds:
            q = np.clip(q.copy(), self.lim[s][:, 0] + 0.01, self.lim[s][:, 1] - 0.01)
            for _ in range(iters):
                d.qpos[self.qadr[s]] = q
                mujoco.mj_kinematics(m, d)
                mujoco.mj_comPos(m, d)
                p, R = d.site_xpos[self.site[s]], d.site_xmat[self.site[s]].reshape(3, 3)
                mujoco.mj_jacSite(m, d, jp, jr, self.site[s])
                Jp, Jr = jp[:, self.dadr[s]], jr[:, self.dadr[s]]
                ep = tgt - p
                if R_target is not None:
                    er = 0.5 * sum(np.cross(R[:, i], R_target[:, i]) for i in range(3))
                    J, e = np.vstack([Jp, 0.3 * Jr]), np.concatenate([ep, 0.3 * er])
                else:
                    y = R[:, 1]
                    J = np.vstack([Jp, 0.3 * (np.cross(y, [0, 0, 1.0]) @ Jr)[None, :]])
                    e = np.concatenate([ep, [-0.3 * y[2]]])
                if np.linalg.norm(ep) < 3e-4 and np.linalg.norm(e[3:]) < 2e-3:
                    break
                dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(len(e)), e)
                q = np.clip(q + np.clip(dq, -0.15, 0.15), self.lim[s][:, 0] + 0.01, self.lim[s][:, 1] - 0.01)
            d.qpos[self.qadr[s]] = q
            mujoco.mj_kinematics(m, d)
            pe = float(np.linalg.norm(tgt - d.site_xpos[self.site[s]]))
            R = d.site_xmat[self.site[s]].reshape(3, 3)
            oe = float(np.linalg.norm(0.5 * sum(np.cross(R[:, i], R_target[:, i]) for i in range(3)))) if R_target is not None else abs(float(R[2, 1]))
            score = pe + 0.05 * oe
            if best is None or score < best[3]:
                best = (q.copy(), pe, oe, score)
            if pe < 2e-3 and oe < 0.05:      # continuity first (like the real arm_server): keep the first good answer
                break
        d.qpos[:] = save
        mujoco.mj_forward(m, d)
        return best[0], best[1], best[2]

    def collisions(self, q14, contact_ok=()):
        """Arm contacts with the body, the other arm, the desk or a container at joint angles q14 (items excluded).
        contact_ok: sides whose finger pads may touch the desk (a grasp's final descent onto a flat item)."""
        m, d = self.m, self.d
        save = d.qpos.copy()
        d.qpos[self.qadr["left"]], d.qpos[self.qadr["right"]] = q14[:7], q14[7:]
        mujoco.mj_forward(m, d)
        bad = []
        item_bodies = set(self.items.values())
        for c in d.contact[:d.ncon]:
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            for s in SIDES:
                arm, other = self.arm_bodies[s], self.arm_bodies["right" if s == "left" else "left"]
                hit, hg = (b2, c.geom2) if b1 in arm else (b1, c.geom1) if b2 in arm else (None, None)
                if hit is None or hit in arm or hit in item_bodies or (self.dev and hit in self.dev.bodies):
                    continue
                mine = b1 if b1 in arm else b2
                depth = -c.dist
                if s in contact_ok and "dex1" in m.body(mine).name and (hit == 0 or (hit in self.containers.values() and depth < 0.004)):
                    continue    # a grasp's descent/lift may touch the desk and graze a container rim (rigid, harmless)
                own_body = hit not in self.containers.values() and hit != 0
                # Convex-hull collision shapes are bulkier than the real links: tolerate <5 mm overlap against the
                # robot's own body / other arm, ~1 mm (numerical) against the desk and containers.
                if depth < (0.005 if own_body else 0.001):
                    continue
                what = ("the desk" if hg in self.static_geoms and hit == 0 else
                        next((n for n, b in self.containers.items() if b == hit), None) or
                        ("the other arm" if hit in other else m.body(hit).name))
                bad.append(f"{m.body(mine).name} / {what} ({-c.dist * 1000:.0f} mm deep)")
        d.qpos[:] = save
        mujoco.mj_forward(m, d)
        return bad

    def sweep(self, q_from, q_to, step_deg=3.0, contact_ok=()):
        start = set(self.collisions(q_from, contact_ok))   # contacts that already exist don't block moving away
        n = max(2, int(np.abs(np.asarray(q_to) - q_from).max() / math.radians(step_deg)) + 1)
        for k in range(1, n + 1):
            new = [b for b in self.collisions(q_from + (np.asarray(q_to) - q_from) * k / n, contact_ok) if b not in start]
            if new:
                return {"ok": False, "reason": f"collision {new[0]} at sample {k}/{n}", "samples": k}
        return {"ok": True, "reason": "", "samples": n}

    def move(self, q14_goal, duration):
        q0, q1 = self.q14(), np.asarray(q14_goal, float)
        steps = max(1, int(duration / self.m.opt.timestep))
        self.moving, t_wall = True, time.monotonic()
        for k in range(1, steps + 1):
            a = k / steps
            a = a * a * (3 - 2 * a)
            q = q0 + (q1 - q0) * a
            for s, sl in (("left", slice(0, 7)), ("right", slice(7, 14))):
                self.d.ctrl[self.act[s]] = q[sl]
            self.step()
            if self.speed > 0:
                lag = k * self.m.opt.timestep / self.speed - (time.monotonic() - t_wall)
                if lag > 0:
                    time.sleep(lag)
        self.step(100)                              # 0.2 s settle
        self.moving = False

    def follow(self, qs, times):
        """Track a timed joint trajectory, interpolating linearly between samples (no per-segment easing)."""
        qs, times = np.asarray(qs, float), np.asarray(times, float)
        dt, t_wall = self.m.opt.timestep, time.monotonic()
        self.moving = True
        steps = max(1, int(times[-1] / dt))
        for k in range(1, steps + 1):
            t = k * dt
            i = min(int(np.searchsorted(times, t, side="right")) - 1, len(times) - 2)
            f = 1.0 if times[i + 1] == times[i] else min(1.0, (t - times[i]) / (times[i + 1] - times[i]))
            q = qs[i] + (qs[i + 1] - qs[i]) * f
            self.d.ctrl[self.act["left"]], self.d.ctrl[self.act["right"]] = q[:7], q[7:]
            self.step()
            if self.speed > 0:
                lag = t / self.speed - (time.monotonic() - t_wall)
                if lag > 0:
                    time.sleep(lag)
        self.step(75)
        self.moving = False

    # ----- grippers -----
    def holding(self, s):
        """The item both finger pads are touching, if any."""
        touch = {1: set(), 2: set()}
        fl = sorted(self.fing_body[s])
        for c in self.d.contact[:self.d.ncon]:
            b1, b2 = self.m.geom_bodyid[c.geom1], self.m.geom_bodyid[c.geom2]
            for k, fb in zip((1, 2), fl):
                if fb in (b1, b2):
                    touch[k].add(b2 if b1 == fb else b1)
        both = touch[1] & touch[2] & set(self.items.values())
        return next((n for n, b in self.items.items() if b in both), None)

    def set_fingers(self, s, q, seconds):
        for a in self.fing_act[s]:
            self.d.ctrl[a] = q
        for _ in range(int(seconds / self.m.opt.timestep)):
            self.step()

    def gstat(self, s):
        q = float(np.mean(self.d.qpos[self.fing_q[s]]))
        h = self.holding(s)
        return {"q": motor_q(q), "gap_mm": round(2 * (0.023 + q) * 1000, 1), "phase": "holding" if h else "idle",
                "holding": bool(h), "object": h}

    # ----- scoring -----
    def objects(self):
        out = {}
        for n in self.chosen:
            b = self.items[n]
            R = self.d.xmat[b].reshape(3, 3)
            p = self.d.xpos[b] + R @ self.centre[n]      # geometric centre, where a grasp should aim
            inside = next((c for c, (lo, hi) in self.box.items()
                           if lo[0] <= p[0] <= hi[0] and lo[1] <= p[1] <= hi[1] and p[2] <= hi[2] + 0.01), None)
            ex = self.extent[n]
            yaw = math.degrees(math.atan2(R[1, 0], R[0, 0]))
            jaw = yaw + (90 if ex[0] > ex[1] else 0)            # jaws close across the narrow horizontal axis
            out[n] = {"xyz": [round(float(v), 4) for v in p], "yaw_deg": round(yaw, 1),
                      "grip_jaw_deg": round((jaw + 90) % 180 - 90, 1), "width_mm": round(float(min(ex[0], ex[1])) * 1000, 1),
                      "height_mm": round(float(ex[2]) * 1000, 1),
                      "category": self.item_defs[n][4], "in_container": inside,
                      **({"rim_radius_m": 0.0465} if n == "tape" else {}),     # 3M roll: 76 mm core, 111 mm outside
                      "held_by": next((s for s in SIDES if self.holding(s) == n), None),
                      "lifted_m": round(float(p[2] - self.desk.top_z), 4)}
        return out

    def T_pelvis_torso(self):
        mujoco.mj_kinematics(self.m, self.d)
        t = self.m.body("torso_link").id
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = self.d.xmat[t].reshape(3, 3), self.d.xpos[t]   # world frame == pelvis frame
        return T


def handle(sim: Sim, r: dict, cam_project) -> dict:
    cmd, side = r.get("cmd"), r.get("side", "left")
    if cmd == "state":
        return {"arm": {"fsm": [500, 0], "enabled": sim.enabled, "frozen": sim.frozen, "freeze_reason": sim.freeze_reason,
                        "moving": sim.moving, "weight": 1.0 if sim.enabled else 0.0, "q": sim.q14().round(4).tolist()},
                "fk": {"grasp_xyz": sim.grasp_pose("left")[0].round(4).tolist()},
                "fk_right": {"grasp_xyz": sim.grasp_pose("right")[0].round(4).tolist()},
                "gripper": sim.gstat("left"), "gripper_right": sim.gstat("right"), "table": sim.table,
                "T_pelvis_torso": sim.T_pelvis_torso().round(6).tolist()}
    if cmd == "fk":                                # same reply shape as the real arm_server
        q7 = np.asarray(r["q7"], float) if "q7" in r else None
        pz, Rz = sim.grasp_pose(side, q7)
        return {"grasp_xyz": pz.round(4).tolist(), "grasp_R": Rz.round(4).tolist()}
    if cmd == "prepare_enable":
        return {"token": "sim", "expires_s": 15.0}
    if cmd == "enable":
        sim.enabled, sim.frozen = True, False
        return {"status": {"enabled": True}}
    if cmd == "release":
        sim.enabled = False
        return {}
    if cmd == "freeze":
        sim.frozen, sim.freeze_reason = True, r.get("reason", "client")
        return {}
    if cmd == "unfreeze":
        sim.frozen = False
        return {}
    if cmd == "set_table":
        sim.table = None if r.get("z") is None else {"z": float(r["z"]), "x": r.get("x"), "y": r.get("y")}
        return {"table": sim.table or {}}
    if cmd in ("ik", "move_ee"):
        xyz = np.asarray(r["xyz"], float)
        for k, v in zip("xyz", xyz):
            if not WS[k][0] <= v <= WS[k][1]:
                raise ValueError(f"target {k}={v:.3f} outside workspace {WS[k]}")
        R = np.asarray(r["R"], float) if "R" in r else (down_R(float(r.get("jaw_deg", 0))) if r.get("approach") == "down" else None)
        q7, pe, oe = sim.ik(side, xyz, R)
        out = {"q7": q7.round(5).tolist(), "q7_deg": np.degrees(q7).round(1).tolist(), "pos_err": round(pe, 4),
               "jaw_tilt": round(oe, 3)}
        if pe > r.get("max_pos_err", 0.01) or oe > r.get("max_jaw_tilt", 0.15):
            return {**out, "ok": False, "error": "IK did not converge within tolerance"}
        if cmd == "ik" or r.get("dry"):
            return out
        if not sim.enabled:
            raise RuntimeError("not enabled")
        if sim.frozen:
            raise RuntimeError(f"frozen: {sim.freeze_reason}")
        goal = sim.q14()
        goal[(slice(0, 7) if side == "left" else slice(7, 14))] = q7
        chk = sim.sweep(sim.q14(), goal, contact_ok=(side,) if r.get("contact") else ())
        if not chk["ok"] and not r.get("force"):
            raise ValueError(f"path check: {chk['reason']} (send force=true to override)")
        sim.move(goal, float(r.get("duration", 3.0)))
        p, _ = sim.grasp_pose(side)
        return {**out, "duration": r.get("duration", 3.0), "check": chk, "reached_err": round(float(np.linalg.norm(p - xyz)), 4)}
    if cmd == "trajectory":                          # like the real arm_server: q [n,14], times [n]; every sample checked
        if not sim.enabled:
            raise RuntimeError("not enabled")
        q, times = np.asarray(r["q"], float), np.asarray(r["times"], float)
        if q.ndim != 2 or q.shape[1] != 14 or times.shape != (len(q),):
            raise ValueError("trajectory needs q [n,14] and times [n]")
        ok_sides = (side,) if r.get("contact") else ()
        cur = sim.q14()
        for k, qk in enumerate(q):
            chk = sim.sweep(cur, qk, contact_ok=ok_sides)
            if not chk["ok"] and not r.get("force"):
                raise ValueError(f"path check: {chk['reason']} (trajectory sample {k})")
            cur = qk
        sim.follow(q, times)                       # timing is already shaped by the client: follow it as given
        return {"duration": float(times[-1]), "samples": int(len(q))}
    if cmd == "check":
        q = np.asarray(r["q"], float) if "q" in r else None
        if q is None:
            q = sim.q14()
            q[(slice(0, 7) if side == "left" else slice(7, 14))] = r["q7"]
        return {"check": sim.sweep(sim.q14(), q), "q_start": sim.q14().round(5).tolist()}
    if cmd == "park":
        for s in SIDES:
            if r.get("side", "both") in ("both", s):
                p, _ = sim.grasp_pose(s)
                for xyz in ([max(p[0], 0.13), p[1], p[2] + 0.15], READY[s]):
                    handle(sim, {"cmd": "move_ee", "side": s, "xyz": xyz, "duration": 2.5, "max_pos_err": 0.03}, cam_project)
        return {}
    if cmd == "grasp":
        sim.set_fingers(side, FINGER[0], 0.6)
        return sim.gstat(side)
    if cmd == "open":
        sim.set_fingers(side, FINGER[1], 0.5)
        return sim.gstat(side)
    if cmd == "gripper":                            # motor angle 0..5.4 like the real service
        sim.set_fingers(side, FINGER[0] + float(r["q"]) / 5.4 * (FINGER[1] - FINGER[0]), 0.5)
        return sim.gstat(side)
    if cmd in ("gripper_stop", "gripper_state"):
        return sim.gstat(side)
    if cmd == "sim_reset":
        sim.reset(int(r.get("seed", 0)), r.get("n_items"))
        return {"objects": sim.objects()}
    if cmd == "sim_reset_arms":                     # sim-only: put both arms back to hanging (benchmark isolation)
        for sd in SIDES:
            for j, a in zip(ARM, sim.qadr[sd]):
                sim.d.qpos[a] = HANG.get(j, 0.0) * (-1 if (sd == "right" and j == "shoulder_roll") else 1)
            for a in sim.fing_q[sd]:
                sim.d.qpos[a] = FINGER[1]
        sim.d.qvel[:sim.m.nv] = 0
        mujoco.mj_forward(sim.m, sim.d)
        sim._hold_ctrl()
        for sd in SIDES:
            for a in sim.fing_act[sd]:
                sim.d.ctrl[a] = FINGER[1]
        sim.step(100)
        return {}
    if cmd == "sim_devices":
        if not sim.dev:
            raise ValueError("not the devices scene (start with --scene devices)")
        notes = ({n: note for n, (_p, note) in scene.LAB.items()} if sim.kind == "lab" else
                 {n: note for n, (_p, note) in scene.CELL.items()} if sim.kind == "cell" else
                 {n: note for n, (_p, note) in scene.RACK.items()} if sim.kind == "rack" else
                 {n: note for n, (_p, note) in scene.TABLET.items()} if sim.kind == "tablet" else
                 {n: note for n, (_k, _p, note) in scene.DEVICES.items()})
        return {"scene": sim.kind, "state": sim.dev.state, "presses": sim.dev.presses, "history": sim.dev.history[-30:],
                "targets": sim.dev.targets(), "time": round(sim.d.time, 2), "devices": notes}
    if cmd == "sim_flash":                          # the agent's USB side: flash firmware while the board is in its bootloader
        sim.dev.flash()
        return {"firmware": sim.dev.firmware}
    if cmd == "sim_save":                           # snapshot the whole world (try something, then roll back)
        snap = {"qpos": sim.d.qpos.copy(), "qvel": sim.d.qvel.copy(), "act": sim.d.act.copy(), "ctrl": sim.d.ctrl.copy(),
                "time": sim.d.time, "enabled": sim.enabled}
        if sim.dev:
            snap["dev"] = {k: copy.deepcopy(v) for k, v in vars(sim.dev).items() if k not in ("m", "d", "rng")}
        sim.snapshots[r.get("name", "default")] = snap
        return {}
    if cmd == "sim_restore":
        snap = sim.snapshots[r.get("name", "default")]
        sim.d.qpos[:], sim.d.qvel[:], sim.d.act[:], sim.d.ctrl[:] = snap["qpos"], snap["qvel"], snap["act"], snap["ctrl"]
        sim.d.time, sim.enabled = snap["time"], snap["enabled"]
        if sim.dev:
            for k, v in snap["dev"].items():
                setattr(sim.dev, k, copy.deepcopy(v))
            if hasattr(sim.dev, "shown"):
                sim.dev.shown = None
        mujoco.mj_forward(sim.m, sim.d)
        return {}
    if cmd == "sim_get_world":                      # whole-world state, to seed a twin sim (the real robot: from perception)
        return {"qpos": sim.d.qpos.tolist(), "items": {str(n): int(sim.item_qadr[n]) for n in sim.chosen}, "enabled": sim.enabled}
    if cmd == "sim_set_world":
        sim.d.qpos[:] = r["qpos"]
        sim.d.qvel[:] = 0
        sim.enabled = r.get("enabled", sim.enabled)
        mujoco.mj_forward(sim.m, sim.d)
        sim._hold_ctrl()
        sim.step(int(r.get("settle", 0)))
        return {}
    if cmd == "sim_wait":                          # let sim time pass (e.g. "wait 5 s before powering back on")
        sim.step(int(float(r.get("seconds", 1.0)) / sim.m.opt.timestep))
        return {"time": round(sim.d.time, 2)}
    if cmd == "sim_objects":
        return {"objects": sim.objects(), "desk_top_z": sim.desk.top_z,
                "containers": {n: {"category": c[3], "box_min": sim.box[n][0].round(3).tolist(), "box_max": sim.box[n][1].round(3).tolist(),
                                   "drop_xy": sim.drop[n][:2], "drop_clear_m": sim.drop[n][2]}
                               for n, c in {**scene.CONTAINERS, **{k: (None, None, None, k) for k in ("nest", "pass_bin", "reject_bin")}}.items()
                               if n in sim.box}}
    if cmd == "sim_point":
        # Oracle pointer: the item's top-centre projected into the head image, with VLM-like pixel noise.
        n = r["item"]
        b = sim.items[n]
        top = sim.d.xpos[b] + sim.d.xmat[b].reshape(3, 3) @ sim.centre[n]
        top[2] += sim.extent[n][2] / 2 - 0.002
        u, v = cam_project(top)
        rng = np.random.default_rng(r.get("seed"))
        nz = float(r.get("noise_px", 0))
        R = sim.d.xmat[b].reshape(3, 3)          # and its 2D box, like a detector / VLM bbox_2d
        uv = np.array([cam_project(sim.d.xpos[b] + R @ (sim.centre[n] + sim.extent[n] * np.array([i, j, k]) / 2))
                       for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)])
        box = np.concatenate([uv.min(0), uv.max(0)]) + rng.normal(0, nz, 4)
        return {"item": n, "u": round(u + rng.normal(0, nz), 1), "v": round(v + rng.normal(0, nz), 1), "noise_px": nz,
                "box": [round(float(c), 1) for c in box]}
    raise ValueError(f"unknown cmd {cmd}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--speed", type=float, default=1.0, help="sim-time / wall-time while moving; 0 = as fast as possible")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--desk-top", type=float, default=scene.Desk.top_z, help="desk top z in the pelvis frame (m)")
    ap.add_argument("--viewer", action="store_true", help="open the interactive MuJoCo viewer window (watch it live)")
    ap.add_argument("--items", default="big", choices=sorted(scene.ITEM_SETS), help="item set desks are drawn from")
    ap.add_argument("--scene", default="sort", choices=["sort", "devices", "lab", "cell", "rack", "tablet"],
                    help="desk sorting, IT switches/buttons, or lab (dev board + machine tending)")
    ap.add_argument("--port-offset", type=int, default=0, help="add to every port (run several sims side by side)")
    a = ap.parse_args()
    global ARM_PORT, CFG_PORT, DEPTH_PORT, HTTP_PORT
    ARM_PORT, CFG_PORT, DEPTH_PORT, HTTP_PORT = (p + a.port_offset for p in (ARM_PORT, CFG_PORT, DEPTH_PORT, HTTP_PORT))
    sim = Sim(a.speed, scene.Desk(top_z=a.desk_top), a.items, a.scene)
    sim.reset(a.seed)
    if a.viewer:
        from mujoco import viewer as mj_viewer   # (a plain `import mujoco.viewer` here would shadow `mujoco` in main)
        sim.viewer = mj_viewer.launch_passive(sim.m, sim.d, show_left_ui=False, show_right_ui=False)
        sim.viewer.cam.lookat[:] = [0.25, 0.0, -0.05]
        sim.viewer.cam.distance, sim.viewer.cam.azimuth, sim.viewer.cam.elevation = 1.6, 200, -25
    m = sim.m
    head, fy = m.cam("head_camera").id, scene.FY
    K = {"fx": fy, "fy": fy, "ppx": W / 2, "ppy": H / 2}
    cfg = {"head_camera": {"image_shape": [H, W], "depth_scale": 0.001, "enable_depth": True, "type": "sim", "color_intrinsics": K}}

    def cam_project(p_world):
        mujoco.mj_kinematics(m, sim.d)
        mujoco.mj_camlight(m, sim.d)
        pos, R = sim.d.cam_xpos[head], sim.d.cam_xmat[head].reshape(3, 3)
        pc = R.T @ (np.asarray(p_world) - pos)          # MuJoCo camera: x right, y up, looking along -z
        return K["ppx"] + fy * pc[0] / -pc[2], K["ppy"] - fy * pc[1] / -pc[2]

    rgb_r, dep_r = mujoco.Renderer(m, H, W), mujoco.Renderer(m, H, W)
    dep_r.enable_depth_rendering()
    room = mujoco.MjvCamera()
    room.lookat[:] = [0.3, 0, -0.1]
    room.distance, room.azimuth, room.elevation = 1.4, 210, -28

    def render(which, depth=False):
        r = dep_r if depth else rgb_r
        r.update_scene(sim.d, camera=head if which == "head" else room)
        return r.render()

    ctx = zmq.Context.instance()
    arm_sock, cfg_sock, pub = ctx.socket(zmq.REP), ctx.socket(zmq.REP), ctx.socket(zmq.PUB)
    arm_sock.bind(f"tcp://127.0.0.1:{ARM_PORT}")
    cfg_sock.bind(f"tcp://127.0.0.1:{CFG_PORT}")
    pub.bind(f"tcp://127.0.0.1:{DEPTH_PORT}")
    jobs: queue.Queue = queue.Queue()

    class Http(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/api/cameras":
                body, ctype = json.dumps({"connected": True, "cameras": [{"id": "head_camera", "state": "live"}, {"id": "waldo_webcam", "state": "live"}]}).encode(), "application/json"
            elif self.path in ("/frame/head_camera.jpg", "/frame/waldo_webcam.jpg"):
                done, box = threading.Event(), {}
                jobs.put(("head" if "head" in self.path else "room", box, done))
                done.wait(5)
                body, ctype = box.get("jpg", b""), "image/jpeg"
            else:
                self.send_response(404)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("content-type", ctype)
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    threading.Thread(target=ThreadingHTTPServer(("127.0.0.1", HTTP_PORT), Http).serve_forever, daemon=True).start()
    print(f"waldo-sim (desk) up: arm :{ARM_PORT}  config :{CFG_PORT}  depth :{DEPTH_PORT}  http :{HTTP_PORT}  "
          f"(speed {a.speed}x, desk top z {sim.desk.top_z})", flush=True)
    poller = zmq.Poller()
    poller.register(arm_sock, zmq.POLLIN)
    poller.register(cfg_sock, zmq.POLLIN)
    last_depth = last_idle = 0.0
    last_idle = time.monotonic()
    while True:
        socks = dict(poller.poll(5))
        if cfg_sock in socks:
            cfg_sock.recv()
            cfg_sock.send_json(cfg)
        if arm_sock in socks:
            try:
                out = handle(sim, json.loads(arm_sock.recv()), cam_project)
                out.setdefault("ok", True)
            except Exception as e:
                out = {"ok": False, "error": str(e)}
            arm_sock.send(json.dumps(out).encode())
        while not jobs.empty():
            which, box, done = jobs.get()
            buf = io.BytesIO()
            Image.fromarray(render(which)).save(buf, "JPEG", quality=88)
            box["jpg"] = buf.getvalue()
            done.set()
        if time.monotonic() - last_depth > 1 / 15:
            last_depth = time.monotonic()
            pub.send((np.clip(render("head", depth=True), 0, 65.0) * 1000).astype(np.uint16).tobytes())
        if a.speed > 0:          # idle: keep sim time with the wall clock (machines run, LEDs boot while agents think)
            now = time.monotonic()
            n = min(100, int((now - last_idle) * a.speed / sim.m.opt.timestep))
            if n:
                sim.step(n)
                last_idle += n * sim.m.opt.timestep / a.speed
            if now - last_idle > 0.5:
                last_idle = now
        else:
            sim.step()


if __name__ == "__main__":
    main()

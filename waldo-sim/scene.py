"""Desk-sorting scene: Unitree's official G1 + Dex1-1 URDF, a desk, sorting containers and stationery.

World frame = pelvis frame (x forward, y left, z up), the frame every waldo-arms tool uses. The pelvis is fixed
(gantry). Arms, waist and legs have stiff position actuators with gravity compensation, so commanded poses are
tracked dynamically; each Dex1 finger is a force-limited position actuator (20 N, Unitree's spec), so grasps hold
or slip by contact and friction, not by magic.

Assets (fetch with fetch_assets.sh, not in git): assets/g1_description (Unitree URDF + meshes) and
assets/gso_repo/models (Google Scanned Objects in MuJoCo format, kevinzakka/mujoco_scanned_objects).
Pens, markers, pencil, eraser, sticky notes and sticker sheets are primitives with real dimensions and masses
(Google Scanned Objects has none of them).
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
URDF = HERE / "assets" / "g1_description" / "g1_29dof_mode_15_with_dex1_1.urdf"
GSO = HERE / "assets" / "gso_repo" / "models"
sys.path.insert(0, str(HERE.parent / "waldo-arms"))
import vision  # noqa: E402  (same URDF head-camera mount the real pipeline uses)

SIDES = ("left", "right")
ARM = ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw"]
GRASP_X = 0.152            # Dex1 pad centre ahead of wrist_yaw_link (measured from the finger collision meshes)
FINGER = (-0.02, 0.0245)   # each finger's prismatic range: closed .. open (gap 6 mm .. 95 mm)
FLOOR_Z = -0.79            # floor below the pelvis (G1 standing pelvis height)
H, W, FY = 480, 640, 604.4344


@dataclass
class Desk:
    top_z: float = -0.06            # a standard 74 cm desk under a 79 cm pelvis
    x: tuple = (0.16, 0.62)         # near edge 16 cm in front of the pelvis
    y: tuple = (-0.45, 0.45)


# Items: name -> (kind, size/params, mass kg, rgba or gso model, category)
ITEMS = {
    "marker_black": ("capsule", (0.0085, 0.0595), 0.015, (0.08, 0.08, 0.08, 1), "writing"),
    "marker_blue": ("capsule", (0.0085, 0.0595), 0.015, (0.1, 0.25, 0.85, 1), "writing"),
    "pen": ("capsule", (0.005, 0.065), 0.010, (0.9, 0.9, 0.95, 1), "writing"),
    "pencil": ("cylinder", (0.0042, 0.088), 0.006, (0.95, 0.75, 0.1, 1), "writing"),
    "eraser": ("box", (0.025, 0.011, 0.006), 0.020, (0.95, 0.55, 0.65, 1), "supplies"),
    "sticky_notes": ("box", (0.038, 0.038, 0.008), 0.045, (1.0, 0.92, 0.3, 1), "supplies"),
    "sticker_sheet": ("box", (0.045, 0.035, 0.0015), 0.008, (0.4, 0.85, 0.95, 1), "supplies"),
    "tape": ("gso", "3M_Vinyl_Tape_Green_1_x_36_yd", 0.060, None, "supplies"),
    "scissors": ("gso", "Diamond_Visions_Scissors_Red", 0.060, None, "tools"),
    "crayons": ("gso", "Crayola_Crayons_24_count", 0.120, None, "tools"),
    "pencil_case": ("gso", "Big_Dot_Aqua_Pencil_Case", 0.150, None, "supplies"),
    "label_reel": ("gso", "Epson_LabelWorks_LC4WBN9_Tape_reel_labels_047_x_295_Roll_Black_on_White", 0.090, None, "supplies"),
    "ink_box": ("gso", "Office_Depot_HP_950XL_Ink_Cartridge_Black_CN045AN", 0.070, None, "supplies"),
}
# Which items a desk is drawn from. "big" (default): no sticky notes, stickers, erasers, pens or pencils.
ITEM_SETS = {
    "big": ["marker_black", "marker_blue", "crayons", "scissors", "tape", "pencil_case", "label_reel", "ink_box"],
    "all": None,   # every item above
}
# Containers (fixed on the desk): name -> (gso model, (x, y) pelvis frame, yaw deg, category it receives)
CONTAINERS = {
    "pen_cup": ("Ecoforms_Cup_B4_SAN", (0.30, -0.27), 0, "writing"),
    "tray": ("Design_Ideas_Drawer_Store_Organizer", (0.47, 0.0), 90, "supplies"),
    "caddy": ("Markings_Desk_Caddy", (0.36, 0.29), 90, "tools"),
}


def _gso_body(spec, parent, model, name, pos, quat, mass, free):
    """Attach a Google Scanned Object (visual mesh + convex collision pieces) as a body."""
    src = mujoco.MjSpec.from_file(str(GSO / model / "model.xml"))
    src.meshdir = str(GSO / model)
    src.texturedir = str(GSO / model)
    frame = parent.add_frame(pos=pos, quat=quat)
    body = frame.attach_body(src.body("model"), f"{name}_", "")
    if free:
        body.add_freejoint()
    for g in body.geoms:
        if g.contype or g.conaffinity or g.group == 3:
            g.friction = [0.9, 0.01, 0.001]
            g.condim = 4
    # GSO masses are density defaults (an 11 kg bin); use real ones
    body.explicitinertial = False
    return body


def _mesh_bottom(model_dir: str) -> float:
    m = mujoco.MjModel.from_xml_path(str(GSO / model_dir / "model.xml"))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    zmin = math.inf
    for g in range(m.ngeom):
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and m.geom_group[g] == 3:
            mid = m.geom_dataid[g]
            v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
            vw = (d.geom_xmat[g].reshape(3, 3) @ v.T).T + d.geom_xpos[g]
            zmin = min(zmin, vw[:, 2].min())
    return float(zmin)


KEEPOUTS = 16


def build(desk: Desk = Desk(), items: dict = ITEMS, containers: dict = CONTAINERS, devices: bool | str = False,
          twin: bool = False) -> mujoco.MjModel:
    """devices: False, True / "devices" (IT desk: power strip, router, key, button, light) or "lab" (dev board,
    machine panel with stack light, pressure gauge)."""
    spec = mujoco.MjSpec.from_file(str(URDF))
    spec.meshdir = str(URDF.parent)
    spec.compiler.fusestatic = False
    spec.compiler.discardvisual = False
    spec.option.timestep = 0.002
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 10
    spec.option.noslip_iterations = 4
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.visual.global_.offwidth, spec.visual.global_.offheight = 1280, 960

    # Robot: gravity compensation + stiff position actuators on every hinge, force-limited fingers.
    for b in spec.bodies:
        if b.name != "world":
            b.gravcomp = 1
    for j in spec.joints:
        if j.type == mujoco.mjtJoint.mjJNT_HINGE:
            kp = 400 if any(k in j.name for k in ("shoulder", "elbow", "waist")) else 150
            spec.add_actuator(name=j.name, target=j.name, trntype=mujoco.mjtTrn.mjTRN_JOINT,
                              gaintype=mujoco.mjtGain.mjGAIN_FIXED, gainprm=[kp] + [0] * 9,
                              biastype=mujoco.mjtBias.mjBIAS_AFFINE, biasprm=[0, -kp, -0.08 * kp] + [0] * 7,
                              ctrlrange=j.range, ctrllimited=True)
            j.armature, j.damping = 0.05, [1.0, 0, 0]
        elif j.type == mujoco.mjtJoint.mjJNT_SLIDE and "dex1_finger" in j.name:
            spec.add_actuator(name=j.name, target=j.name, trntype=mujoco.mjtTrn.mjTRN_JOINT,
                              gaintype=mujoco.mjtGain.mjGAIN_FIXED, gainprm=[2000] + [0] * 9,
                              biastype=mujoco.mjtBias.mjBIAS_AFFINE, biasprm=[0, -2000, -40] + [0] * 7,
                              ctrlrange=list(FINGER), ctrllimited=True, forcerange=[-20, 20], forcelimited=True)
            j.armature, j.damping = 0.01, [2.0, 0, 0]
    for g in spec.geoms:   # grippy finger pads; robot collision geoms keep their URDF shapes
        if g.contype or g.conaffinity:
            g.condim = 6 if "finger" in (g.parent.name if g.parent else "") else 3
            if "finger" in (g.parent.name if g.parent else ""):
                g.friction = [1.5, 0.02, 0.002]
    # The upper arm is built to move alongside the torso; their coarse URDF meshes touch in ordinary poses.
    # Exclude those pairs (as curated collision models do); elbow, wrist and gripper vs torso stay checked.
    for s in SIDES:
        for link in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw"):
            for torso_part in ("torso_link", "logo_link", "head_link", "imu_in_torso", "d435_link", "mid360_link"):
                try:
                    spec.add_exclude(bodyname1=f"{s}_{link}_link", bodyname2=torso_part)
                except Exception:
                    pass   # part not in this URDF
    for s in SIDES:
        spec.body(f"{s}_wrist_yaw_link").add_site(name=f"{s}_grasp", pos=[GRASP_X, 0, 0], size=[0.008, 0, 0],
                                                  rgba=[1, 1, 0, 0.5], group=4)

    # Head camera at the URDF d435 mount (identical to what point_to_xyz assumes).
    T = vision.T_TORSO_CAM
    xo, yo = T[:3, 0], T[:3, 1]
    spec.body("torso_link").add_camera(name="head_camera", pos=T[:3, 3].tolist(),
                                       xyaxes=[*xo.tolist(), *(-yo).tolist()], fovy=math.degrees(2 * math.atan(H / 2 / FY)))

    wb = spec.worldbody
    wb.add_light(pos=[0.5, -0.4, 1.4], dir=[-0.3, 0.3, -1], diffuse=[0.75, 0.75, 0.75], castshadow=True)
    wb.add_light(pos=[0.3, 0.6, 1.2], dir=[0, -0.4, -1], diffuse=[0.35, 0.35, 0.35], castshadow=False)
    wb.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, pos=[0, 0, FLOOR_Z], size=[4, 4, 0.1], rgba=[0.42, 0.36, 0.3, 1])
    cx, cy = sum(desk.x) / 2, sum(desk.y) / 2
    hx, hy = (desk.x[1] - desk.x[0]) / 2, (desk.y[1] - desk.y[0]) / 2
    wb.add_geom(name="desk", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[cx, cy, desk.top_z - 0.015], size=[hx, hy, 0.015],
                rgba=[0.62, 0.45, 0.3, 1], friction=[0.7, 0.005, 0.0001])
    for lx in desk.x:
        for ly in desk.y:
            sx, sy = (0.03 if lx == desk.x[0] else -0.03), (0.03 if ly == desk.y[0] else -0.03)
            wb.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=[lx + sx, ly + sy, (desk.top_z - 0.03 + FLOOR_Z) / 2],
                        size=[0.02, 0.02, (desk.top_z - 0.03 - FLOOR_Z) / 2], rgba=[0.45, 0.32, 0.2, 1])

    if devices == "returns":
        add_returns(spec, desk, twin)
    elif devices == "tablet":
        add_tablet(spec, desk)
    elif devices == "rack":
        add_rack(spec, desk)
        for housing, part in RACK_PARTS:
            spec.add_exclude(bodyname1=housing, bodyname2=part)
    elif devices == "cell":
        add_cell(spec, desk, twin)
        for housing, part in CELL_PARTS_BODIES:
            spec.add_exclude(bodyname1=housing, bodyname2=part)
    elif devices == "lab":
        add_lab(spec, desk)
        for housing, part in LAB_PARTS:
            spec.add_exclude(bodyname1=housing, bodyname2=part)
    elif devices:
        add_devices(spec, desk)
        # moving parts sit inside their housings; parent/child contact filtering isn't active for this URDF-based
        # model, so without these the housing pushes the rocker/lever back to centre (switches lose their state)
        for housing, part in DEVICE_PARTS:
            spec.add_exclude(bodyname1=housing, bodyname2=part)
    for name, (model, (x, y), yaw, _cat) in containers.items():
        quat = [math.cos(math.radians(yaw) / 2), 0, 0, math.sin(math.radians(yaw) / 2)]
        _gso_body(spec, wb, model, name, [x, y, desk.top_z - _mesh_bottom(model) + 0.001], quat, None, free=False)

    for i, (name, (kind, p, mass, rgba, _cat)) in enumerate(items.items()):
        z0 = desk.top_z + 0.05 + 0.03 * i
        if kind == "gso":
            b = _gso_body(spec, wb, p, name, [0.3, 0, z0 - _mesh_bottom(p)], [1, 0, 0, 0], mass, free=True)
            body = spec.body(f"{name}_model")
            body.mass = mass
        else:
            body = wb.add_body(name=name, pos=[0.3, 0, z0])
            body.add_freejoint()
            # long items lie along x; capsule/cylinder axis is z by default, so rotate onto the desk
            quat = [math.cos(math.pi / 4), 0, math.sin(math.pi / 4), 0] if kind in ("capsule", "cylinder") else [1, 0, 0, 0]
            gtype = {"capsule": mujoco.mjtGeom.mjGEOM_CAPSULE, "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER,
                     "upright": mujoco.mjtGeom.mjGEOM_CYLINDER, "box": mujoco.mjtGeom.mjGEOM_BOX}[kind]
            body.add_geom(name=f"{name}_geom", type=gtype, size=list(p) + [0] * (3 - len(p)), quat=quat, rgba=list(rgba),
                          mass=mass, friction=[0.9, 0.01, 0.001], condim=6 if kind == "upright" else 4)   # 6: rolling friction, so a dropped part settles instead of rolling forever
    if twin:     # spare boxes the planner switches on as registered keep-out zones (containers, fixtures, machines)
        for i in range(KEEPOUTS):
            spec.worldbody.add_geom(name=f"keepout{i}", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.001] * 3, pos=[-5, -5, -5 - i],
                                    contype=0, conaffinity=0, rgba=[1, 0.3, 0.1, 0.3])
    return spec.compile()


# ---- devices: switches and buttons an agent might need a robot to operate (all arms-only) ----
# name -> (kind, (x, y) on the desk in the pelvis frame, note). Positions are within the measured reach.
# Layout = the robot's comfortable reach zone (measured in the sim): controls 30-34 cm in front of the pelvis and
# 12-25 cm to either side (not the midline, where both arms are cramped). Copy this when setting up real props.
DEVICES = {
    "power_strip": ("rocker", (0.32, -0.30), "router power: the strip's rocker switch (press the O end = off, I end = on)"),
    "router": ("router", (0.46, -0.20), "router with power + internet LEDs (lit while the strip is on)"),
    "security_key": ("button", (0.31, 0.14), "USB security key: tap the gold pad to approve 2FA"),
    "start_button": ("big_button", (0.31, 0.25), "machine start button"),
    "light_switch": ("toggle", (0.42, 0.02), "wall light switch on a stand, facing the robot (push the lever up = on)"),
}


def add_devices(spec, desk: "Desk"):
    """Desk devices with real joints: rockers/toggles hold where they're pushed (friction), buttons spring back."""
    wb, top = spec.worldbody, desk.top_z
    box = mujoco.mjtGeom.mjGEOM_BOX
    cyl = mujoco.mjtGeom.mjGEOM_CYLINDER
    for name, (kind, (x, y), _note) in DEVICES.items():
        if kind == "rocker":            # power strip 30 x 5.5 x 4 cm, rocker on top near one end, axis across the strip
            b = wb.add_body(name=name, pos=[x, y, top + 0.02])
            b.add_geom(type=box, size=[0.028, 0.15, 0.02], rgba=[0.92, 0.92, 0.9, 1])
            for k in range(4):          # sockets
                b.add_geom(type=box, size=[0.012, 0.012, 0.0005], pos=[0, -0.08 + 0.045 * k, 0.0202], rgba=[0.2, 0.2, 0.2, 1], contype=0, conaffinity=0)
            r = b.add_body(name=f"{name}_rocker", pos=[0, 0.11, 0.022])
            r.add_joint(name=f"{name}_rocker", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[1, 0, 0], range=[-0.26, 0.26],
                        frictionloss=0.06, damping=0.01, armature=0.0005)
            r.add_geom(name=f"{name}_rocker_geom", type=box, size=[0.012, 0.018, 0.006], rgba=[0.85, 0.1, 0.1, 1], mass=0.005)
            r.add_geom(type=box, size=[0.0015, 0.004, 0.0005], pos=[0, 0.011, 0.0062], rgba=[1, 1, 1, 1], contype=0, conaffinity=0)  # "I" end mark
        elif kind == "router":          # 18 x 12 x 3 cm box, two LEDs on top front edge
            b = wb.add_body(name=name, pos=[x, y, top + 0.015])
            b.add_geom(type=box, size=[0.06, 0.09, 0.015], rgba=[0.12, 0.12, 0.14, 1])
            for k, led in enumerate(("power", "internet")):
                b.add_geom(name=f"{name}_led_{led}", type=box, size=[0.004, 0.004, 0.0015], pos=[-0.05, -0.02 + 0.02 * k, 0.0155],
                           rgba=[0.15, 0.15, 0.15, 1], contype=0, conaffinity=0)
        elif kind == "button":          # security key: 4.5 x 2 cm body, gold touch pad on a spring
            b = wb.add_body(name=name, pos=[x, y, top + 0.004])
            b.add_geom(type=box, size=[0.0225, 0.01, 0.004], rgba=[0.1, 0.1, 0.1, 1])
            pad = b.add_body(name=f"{name}_pad", pos=[0.008, 0, 0.005])
            pad.add_joint(name=f"{name}_pad", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0, 0, 1], range=[-0.003, 0],
                          stiffness=300, damping=2, springref=0)
            pad.add_geom(type=cyl, size=[0.0055, 0.0015, 0], rgba=[0.85, 0.7, 0.2, 1], mass=0.002)
        elif kind == "big_button":      # 6 x 6 x 4 cm box, 3.5 cm green mushroom button on a spring
            b = wb.add_body(name=name, pos=[x, y, top + 0.02])
            b.add_geom(type=box, size=[0.03, 0.03, 0.02], rgba=[0.95, 0.8, 0.1, 1])
            btn = b.add_body(name=f"{name}_cap", pos=[0, 0, 0.026])
            btn.add_joint(name=f"{name}_cap", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0, 0, 1], range=[-0.008, 0],
                          stiffness=250, damping=3, springref=0)
            btn.add_geom(type=cyl, size=[0.0175, 0.006, 0], rgba=[0.1, 0.75, 0.2, 1], mass=0.01)
        elif kind == "toggle":          # switch plate on a stand, facing the robot (-x), lever pivots about y
            b = wb.add_body(name=name, pos=[x, y, top + 0.10])
            b.add_geom(type=box, size=[0.004, 0.035, 0.06], rgba=[0.95, 0.95, 0.93, 1])
            b.add_geom(type=box, size=[0.02, 0.02, 0.045], pos=[0.02, 0, -0.055], rgba=[0.4, 0.4, 0.42, 1])   # stand
            lev = b.add_body(name=f"{name}_lever", pos=[-0.006, 0, 0])
            lev.add_joint(name=f"{name}_lever", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 1, 0], range=[-0.35, 0.35],
                          frictionloss=0.05, damping=0.01, armature=0.0005)
            lev.add_geom(name=f"{name}_lever_geom", type=box, size=[0.008, 0.009, 0.018], pos=[-0.006, 0, 0], rgba=[0.97, 0.97, 0.95, 1], mass=0.004)
            b.add_geom(name=f"{name}_lamp", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.008, 0, 0], pos=[-0.002, 0, 0.075],
                       rgba=[0.25, 0.25, 0.2, 1], contype=0, conaffinity=0)   # the "light"


# ---- lab: hardware-in-the-loop dev board + machine tending (start/stop, stack light, pressure gauge) ----
LAB = {
    "dev_board": ((0.31, 0.02), "dev carrier board: BOOT (blue cap, robot's left) and RESET (red cap, right) buttons "
                  "13 cm apart, status LED (green app / red hung / blue bootloader), 4-digit display (pairing code)"),
    "machine": ((0.33, -0.20), "machine control panel: START (green) and STOP (red) mushroom buttons"),
    "stack_light": ((0.45, -0.22), "stack light: green running, yellow idle/done, red fault"),
    "gauge": ((0.43, 0.20), "pressure gauge 0-10 bar facing the robot; normal run ~6 bar, over 8 bar press STOP"),
}
LAB_PARTS = [("dev_board", "dev_board_boot"), ("dev_board", "dev_board_reset"),
             ("machine", "machine_start"), ("machine", "machine_stop")]
SEG = {"0": "abcdef", "1": "bc", "2": "abged", "3": "abgcd", "4": "fgbc", "5": "afgcd", "6": "afgedc", "7": "abc",
       "8": "abcdefg", "9": "abcdfg", "-": "g", "b": "fgedc", "o": "gcde", "t": "fged", " ": "",
       "E": "adefg", "r": "eg", "g": "abcdfg", "d": "bcdeg"}
DIGIT_W, DIGIT_H = 0.013, 0.022     # one 7-segment digit on the board (y across, x up the image)


def seg_layout():
    """segment -> (centre (u right, v up) in digit units, is_horizontal). u maps to -y, v to +x on the desk."""
    return {"a": ((0, 0.5), True), "b": ((0.5, 0.25), False), "c": ((0.5, -0.25), False), "d": ((0, -0.5), True),
            "e": ((-0.5, -0.25), False), "f": ((-0.5, 0.25), False), "g": ((0, 0), True)}


def gauge_frame(top):
    """Dial centre and frame (columns x, y = 'up' on the face, z = face normal toward the head camera)."""
    (x, y), _ = LAB["gauge"]
    c = np.array([x, y, top + 0.075])
    z = vision.pelvis_from_cam(np.eye(4))[:3, 3] - c      # face the head camera (waist straight)
    z /= np.linalg.norm(z)
    yv = np.array([0.0, 0.0, 1.0]) - z[2] * z
    yv /= np.linalg.norm(yv)
    return c, np.stack([np.cross(yv, z), yv, z], 1)


def glow(spec):
    """Indicators emit light (LEDs, lamps, digits): lit colours read the same from any side, like the real thing."""
    spec.add_material(name="glow").emission = 0.9
    return "glow"


def machine_panel(wb, xy, top, qc_material=None):
    """Grey control panel with START (green) / STOP (red) mushroom buttons; optional QC result lamp."""
    box, cyl = mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_CYLINDER
    b = wb.add_body(name="machine", pos=[xy[0], xy[1], top + 0.02])
    b.add_geom(type=box, size=[0.04, 0.07, 0.02], rgba=[0.55, 0.57, 0.6, 1])
    for name, dy, col in (("start", 0.035, [0.1, 0.75, 0.2, 1]), ("stop", -0.035, [0.85, 0.1, 0.1, 1])):
        cap = b.add_body(name=f"machine_{name}", pos=[0, dy, 0.026])
        cap.add_joint(name=f"machine_{name}", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0, 0, 1], range=[-0.008, 0],
                      stiffness=250, damping=3, springref=0)
        cap.add_geom(type=cyl, size=[0.015, 0.006, 0], rgba=col, mass=0.01)
    if qc_material:              # QC lamp at the far edge, between the buttons: green pass / red reject
        b.add_geom(name="qc_light", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.009, 0, 0], pos=[0.03, 0, 0.022],
                   rgba=[0.04, 0.04, 0.04, 1], material=qc_material, contype=0, conaffinity=0)


def stack_light(wb, xy, top, material):
    cyl = mujoco.mjtGeom.mjGEOM_CYLINDER
    b = wb.add_body(name="stack_light", pos=[xy[0], xy[1], top])
    b.add_geom(type=cyl, size=[0.025, 0.005, 0], pos=[0, 0, 0.005], rgba=[0.2, 0.2, 0.2, 1])
    b.add_geom(type=cyl, size=[0.006, 0.04, 0], pos=[0, 0, 0.045], rgba=[0.6, 0.6, 0.6, 1])
    for k, lamp in enumerate(("green", "yellow", "red")):
        b.add_geom(name=f"stack_light_{lamp}", type=cyl, size=[0.017, 0.012, 0], pos=[0, 0, 0.097 + 0.025 * k],
                   rgba=[0.04, 0.04, 0.04, 1], material=material)


# ---- returns: warehouse returns triage (scan each returned item, bin it by its disposition code) ----
RETURNS = {
    "infeed": ((0.30, 0.17), "inbound tray: returned items in 3 pockets"),
    "nest": ((0.32, -0.02), "scanner window (red glow): set an item here; the console shows its code ~1 s later"),
    "console": ((0.40, 0.05), "scanner console: 4-digit display (b1 restock, b2 refurbish, b3 return to vendor) + LED"),
    "restock_bin": ((0.45, 0.17), "green bin: restock"),
    "refurb_bin": ((0.46, -0.07), "amber bin: refurbish"),
    "rma_bin": ((0.33, -0.21), "red bin: return to vendor (RMA)"),
}
RETURN_ITEMS = {f"part{k}": ("box", (0.017, 0.017, 0.0225), 0.08, c, "item")
                for k, c in enumerate([(0.75, 0.55, 0.3, 1), (0.3, 0.45, 0.75, 1), (0.85, 0.85, 0.82, 1)])}


def add_returns(spec, desk: "Desk", twin: bool = False):
    wb, top = spec.worldbody, desk.top_z
    g = glow(spec)
    box = mujoco.mjtGeom.mjGEOM_BOX
    nocol = dict(contype=0, conaffinity=0)
    (ix, iy), _ = RETURNS["infeed"]
    inf = wb.add_body(name="infeed", pos=[ix, iy, top])
    inf.add_geom(type=box, size=[0.05, 0.075, 0.003], pos=[0.015, -0.005, 0.003], rgba=[0.35, 0.3, 0.25, 1])
    if twin:
        for x, y in INFEED_SLOTS:
            inf.add_geom(type=box, size=[0.019, 0.019, 0.0225], pos=[x - ix, y - iy, 0.006 + 0.0225])
    _tray(wb, "nest", RETURNS["nest"][0], top, 0.060, 0.003, [0.2, 0.2, 0.22, 1], floor=0.010)   # a flat scan plate, low lip
    (nx, ny), _ = RETURNS["nest"]
    wb.add_geom(name="scan_window", type=box, size=[0.012, 0.02, 0.0004], pos=[nx, ny, top + 0.0105],
                rgba=[0.5, 0.02, 0.02, 1], material=g, **nocol)
    _tray(wb, "restock_bin", RETURNS["restock_bin"][0], top, 0.09, 0.045, [0.15, 0.5, 0.25, 1])
    _tray(wb, "refurb_bin", RETURNS["refurb_bin"][0], top, 0.09, 0.045, [0.85, 0.55, 0.1, 1])
    _tray(wb, "rma_bin", RETURNS["rma_bin"][0], top, 0.09, 0.045, [0.65, 0.18, 0.15, 1])
    (cx, cy), _ = RETURNS["console"]
    c = wb.add_body(name="console", pos=[cx, cy, top + 0.012])
    c.add_geom(type=box, size=[0.03, 0.045, 0.012], rgba=[0.15, 0.15, 0.17, 1])
    c.add_geom(type=box, size=[0.014, 0.036, 0.0006], pos=[-0.004, 0, 0.0125], rgba=[0.03, 0.03, 0.03, 1], **nocol)
    for dgt in range(4):
        dy = 0.0255 - dgt * 0.017
        for sname, ((u, v), horiz) in seg_layout().items():
            px, py = -0.004 + v * DIGIT_H, dy - u * DIGIT_W
            size = [0.0012, DIGIT_W * 0.38, 0.0004] if horiz else [DIGIT_H * 0.2, 0.0012, 0.0004]
            c.add_geom(name=f"console_seg{dgt}{sname}", type=box, size=size, pos=[px, py, 0.0132], rgba=[0.05, 0.01, 0.01, 1], material=g, **nocol)
    c.add_geom(name="console_led", type=box, size=[0.004, 0.004, 0.002], pos=[0.022, 0, 0.012], rgba=[0.04, 0.04, 0.04, 1], material=g, **nocol)


# ---- tablet: real-device app QA (a test tablet flat on the desk; taps are fingertip contacts on the glass) ----
TABLET = {"tablet": ((0.37, 0.03), "test tablet running ShopApp 2.3.1: tap buttons with a fingertip; the screen is the "
                     "app's real UI")}
TABLET_W, TABLET_H = 0.20, 0.13          # screen size: across the image (pelvis y) x up the image (pelvis x)
# screen -> (lines of text, [(button label, (u0, v0, u1, v1) in screen fractions, u right / v down in the image), ...])
SCREENS = {
    "home": (["ShopApp", "test build 2.3.1"], [("Browse", (0.04, 0.58, 0.48, 0.95)), ("Cart", (0.52, 0.58, 0.96, 0.95))]),
    "product": (["Blue Mug", "$12.00"], [("Add to cart", (0.04, 0.58, 0.48, 0.95)), ("Back", (0.52, 0.58, 0.96, 0.95))]),
    "cart_empty": (["Cart is empty"], [("Back", (0.52, 0.58, 0.96, 0.95))]),
    "cart": (["Cart: 1 item", "Total $12.00"], [("Checkout", (0.04, 0.58, 0.48, 0.95)), ("Back", (0.52, 0.58, 0.96, 0.95))]),
    "checkout": (["Card **** 4242", "Total $12.00"], [("Pay $12", (0.04, 0.58, 0.48, 0.95)), ("Back", (0.52, 0.58, 0.96, 0.95))]),
    "error": (["Payment failed", "E402 card declined"], [("OK", (0.28, 0.58, 0.72, 0.95))]),
}


def _screen_png(name, path):
    """Render one app screen (what the tablet displays) to a PNG texture."""
    from PIL import Image, ImageDraw, ImageFont
    W, H = 800, 520
    im = Image.new("RGB", (W, H), (245, 246, 250))
    d = ImageDraw.Draw(im)
    font = lambda n, b: ImageFont.truetype("/usr/share/fonts/TTF/DejaVuSans-Bold.ttf" if b else "/usr/share/fonts/TTF/DejaVuSans.ttf", n)
    lines, buttons = SCREENS[name]
    bad = name == "error"
    d.rectangle([0, 0, W, 70], fill=(200, 40, 40) if bad else (40, 90, 200))
    d.text((20, 10), "ShopApp", font=font(44, True), fill=(255, 255, 255))
    for i, line in enumerate(lines):
        d.text((30, 90 + i * 75), line, font=font(64 if i == 0 else 52, i == 0), fill=(200, 30, 30) if bad else (20, 20, 30))
    for label, (u0, v0, u1, v1) in buttons:
        d.rounded_rectangle([u0 * W, v0 * H, u1 * W, v1 * H], 18, fill=(30, 150, 80) if label not in ("Back", "OK") else (90, 95, 110))
        tw = d.textlength(label, font=font(52, True))
        d.text(((u0 + u1) / 2 * W - tw / 2, (v0 + v1) / 2 * H - 32), label, font=font(52, True), fill=(255, 255, 255))
    im.rotate(-90, expand=True).save(path)      # MuJoCo maps a box face's texture x along the face's y axis


def add_tablet(spec, desk: "Desk"):
    wb, top = spec.worldbody, desk.top_z
    box = mujoco.mjtGeom.mjGEOM_BOX
    (x, y), _ = TABLET["tablet"]
    b = wb.add_body(name="tablet", pos=[x, y, top + 0.005])
    b.add_geom(type=box, size=[TABLET_H / 2 + 0.012, TABLET_W / 2 + 0.012, 0.005], rgba=[0.1, 0.1, 0.12, 1])
    b.add_geom(name="tablet_glass", type=box, size=[TABLET_H / 2, TABLET_W / 2, 0.0005], pos=[0, 0, 0.0055], rgba=[0, 0, 0, 0])
    tex_dir = HERE / "assets" / "tablet"        # generated screen textures
    tex_dir.mkdir(parents=True, exist_ok=True)
    for name in SCREENS:
        png = tex_dir / f"{name}.png"
        _screen_png(name, png)
        spec.add_texture(name=f"screen_{name}", type=mujoco.mjtTexture.mjTEXTURE_2D, file=str(png))
        mat = spec.add_material(name=f"screen_{name}")
        mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = f"screen_{name}"
        mat.emission = 0.6
        b.add_geom(name=f"screen_{name}", type=box, size=[TABLET_H / 2, TABLET_W / 2, 0.0002],
                   pos=[0, 0, 0.0052 if name == "home" else -5.0], material=f"screen_{name}", contype=0, conaffinity=0)


# ---- rack: datacenter remote hands (server front panel, power button, hot-swap drive bays) ----
RACK = {"server": ((0.43, 0.0), "server chassis, front panel facing the robot: 7-segment status ('good', or an error "
                   "code: Er01 = hung, Er42 = a drive offline), status LED, power button (hold 4 s = hard off, tap = on), "
                   "two drive bays (latch button pops the caddy out; push it home to reseat), each with an LED")}
RACK_FRONT_X = 0.37                       # the panel's plane (pelvis x)
RACK_PARTS = [("server", "server_power"), ("server", "server_drive0_latch"), ("server", "server_drive1_latch"),
              ("server", "server_drive0"), ("server", "server_drive1")]


def add_rack(spec, desk: "Desk"):
    wb, top = spec.worldbody, desk.top_z
    g = glow(spec)
    box, cyl = mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_CYLINDER
    nocol = dict(contype=0, conaffinity=0)
    (x, y), _ = RACK["server"]
    hx = x - RACK_FRONT_X                 # half depth: the front face sits at RACK_FRONT_X
    b = wb.add_body(name="server", pos=[x, y, top + 0.045])
    b.add_geom(type=box, size=[hx, 0.13, 0.045], rgba=[0.22, 0.23, 0.26, 1])
    fx = -hx                              # front face, local
    along_x = [0.7071068, 0, 0.7071068, 0]   # cylinder axis along x (pointing at the robot)
    # status panel: 4 x 7-segment digits, centred on the front
    b.add_geom(type=box, size=[0.0006, 0.036, 0.016], pos=[fx - 0.0006, 0.0, 0.018], rgba=[0.03, 0.03, 0.03, 1], **nocol)
    for dgt in range(4):
        cy = 0.0255 - dgt * 0.017       # centred: the midline is for reading only (an arm can't press straight ahead there)
        for sname, ((u, v), horiz) in seg_layout().items():
            py, pz = cy - u * DIGIT_W, 0.018 + v * DIGIT_H
            size = [0.0004, DIGIT_W * 0.38, 0.0012] if horiz else [0.0004, 0.0012, DIGIT_H * 0.2]
            b.add_geom(name=f"server_seg{dgt}{sname}", type=box, size=size, pos=[fx - 0.0013, py, pz], rgba=[0.05, 0.01, 0.01, 1], material=g, **nocol)
    b.add_geom(name="server_led", type=box, size=[0.0008, 0.005, 0.005], pos=[fx - 0.0008, 0.0, -0.008], rgba=[0.04, 0.04, 0.04, 1], material=g, **nocol)
    # power button (pushes in along +x)
    pw = b.add_body(name="server_power", pos=[fx - 0.004, 0.085, 0.018], quat=along_x)       # left hand
    pw.add_joint(name="server_power", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0, 0, 1], range=[0, 0.004], stiffness=250, damping=2, springref=0)   # local z = world +x: pushed in
    pw.add_geom(type=cyl, size=[0.009, 0.004, 0], rgba=[0.75, 0.75, 0.78, 1], mass=0.003)
    # drive bays: caddy (slides out toward the robot, friction-held) + latch button + LED on the caddy face
    for k, dy in enumerate((-0.07, -0.115)):                                                   # right hand
        cad = b.add_body(name=f"server_drive{k}", pos=[fx - 0.003, dy, 0.012])
        cad.add_joint(name=f"server_drive{k}", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[-1, 0, 0], range=[0, 0.035],
                      frictionloss=0.4, damping=2, armature=0.01)
        cad.add_geom(type=box, size=[0.003, 0.021, 0.017], rgba=[0.12, 0.12, 0.13, 1], mass=0.05)
        cad.add_geom(name=f"server_drive{k}_led", type=box, size=[0.0008, 0.003, 0.003], pos=[-0.0035, 0.014, 0.010],
                     rgba=[0.04, 0.04, 0.04, 1], material=g, **nocol)
        la = b.add_body(name=f"server_drive{k}_latch", pos=[fx - 0.003, dy, -0.022], quat=along_x)
        la.add_joint(name=f"server_drive{k}_latch", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0, 0, 1], range=[0, 0.003],
                     stiffness=300, damping=2, springref=0)
        la.add_geom(type=cyl, size=[0.006, 0.003, 0], rgba=[0.95, 0.6, 0.1, 1], mass=0.002)


# ---- cell: machine tending + QC, the industrial pick-and-place (fixtured parts, not desk clutter) ----
CELL = {
    "infeed": ((0.30, 0.17), "infeed fixture: raw parts stand in 3 pockets"),
    "nest": ((0.32, -0.02), "machine fixture: load one part here, press START"),
    "machine": ((0.33, -0.20), "machine panel: START / STOP, QC lamp (green pass, red reject) after each cycle"),
    "stack_light": ((0.45, -0.22), "stack light: green running, yellow idle/done, red fault"),
    "pass_bin": ((0.45, 0.17), "outfeed: good parts"),
    "reject_bin": ((0.46, -0.06), "outfeed: rejected parts"),
}
INFEED_SLOTS = [(0.29, 0.13), (0.29, 0.20), (0.345, 0.165)]
PART_RAW, PART_DONE = (0.45, 0.45, 0.48, 1), (0.85, 0.87, 0.9, 1)
CELL_PARTS = {f"part{k}": ("upright", (0.012, 0.025), 0.06, PART_RAW, "part") for k in range(3)}
CELL_PARTS_BODIES = [("machine", "machine_start"), ("machine", "machine_stop")]


def _tray(wb, name, xy, top, inner, wall_h, rgba, floor=0.004):
    """Open box (bin / nest pocket): floor plate + four walls, inner size `inner` (m)."""
    box = mujoco.mjtGeom.mjGEOM_BOX
    b = wb.add_body(name=name, pos=[xy[0], xy[1], top])
    t, h = 0.004, inner / 2
    b.add_geom(type=box, size=[h + t, h + t, floor / 2], pos=[0, 0, floor / 2], rgba=rgba)
    for dx, dy, sx, sy in ((h + t / 2, 0, t / 2, h + t), (-h - t / 2, 0, t / 2, h + t), (0, h + t / 2, h, t / 2), (0, -h - t / 2, h, t / 2)):
        b.add_geom(type=box, size=[sx, sy, wall_h / 2], pos=[dx, dy, floor + wall_h / 2], rgba=rgba)


def add_cell(spec, desk: "Desk", twin: bool = False):
    """twin: the planner's copy also gets the fixtured parts as obstacles (their slots are known; a slot emptied by a
    pick stays an obstacle, which is only conservative)."""
    wb, top = spec.worldbody, desk.top_z
    g = glow(spec)
    box = mujoco.mjtGeom.mjGEOM_BOX
    (ix, iy), _ = CELL["infeed"]
    inf = wb.add_body(name="infeed", pos=[ix, iy, top])
    inf.add_geom(type=box, size=[0.05, 0.075, 0.003], pos=[0.015, -0.005, 0.003], rgba=[0.2, 0.25, 0.35, 1])
    if twin:
        for x, y in INFEED_SLOTS:
            inf.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.013, 0.025, 0], pos=[x - ix, y - iy, 0.006 + 0.025])
    _tray(wb, "nest", CELL["nest"][0], top, 0.040, 0.008, [0.25, 0.27, 0.3, 1], floor=0.010)
    _tray(wb, "pass_bin", CELL["pass_bin"][0], top, 0.09, 0.045, [0.15, 0.5, 0.25, 1])
    _tray(wb, "reject_bin", CELL["reject_bin"][0], top, 0.09, 0.045, [0.65, 0.18, 0.15, 1])
    machine_panel(wb, CELL["machine"][0], top, qc_material=g)
    stack_light(wb, CELL["stack_light"][0], top, g)


def add_lab(spec, desk: "Desk"):
    wb, top = spec.worldbody, desk.top_z
    g = glow(spec)
    box, cyl = mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_CYLINDER
    nocol = dict(contype=0, conaffinity=0)
    # dev board: 7 x 20 cm PCB on standoffs
    (x, y), _ = LAB["dev_board"]
    b = wb.add_body(name="dev_board", pos=[x, y, top + 0.008])
    b.add_geom(type=box, size=[0.035, 0.10, 0.008], rgba=[0.05, 0.35, 0.15, 1])
    for name, dy, col in (("boot", 0.065, [0.15, 0.35, 0.95, 1]), ("reset", -0.065, [0.9, 0.15, 0.1, 1])):
        b.add_geom(type=box, size=[0.012, 0.012, 0.003], pos=[0.0, dy, 0.011], rgba=[0.08, 0.08, 0.08, 1])
        cap = b.add_body(name=f"dev_board_{name}", pos=[0.0, dy, 0.017])
        cap.add_joint(name=f"dev_board_{name}", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0, 0, 1], range=[-0.003, 0],
                      stiffness=300, damping=2, springref=0)
        cap.add_geom(type=cyl, size=[0.008, 0.003, 0], rgba=col, mass=0.002)
    b.add_geom(name="dev_board_led", type=box, size=[0.004, 0.004, 0.002], pos=[0.024, 0.0, 0.009], rgba=[0.04, 0.04, 0.04, 1], material=g, **nocol)
    b.add_geom(type=box, size=[0.016, 0.034, 0.0006], pos=[-0.008, 0.0, 0.0085], rgba=[0.03, 0.03, 0.03, 1], **nocol)   # display window
    for dgt in range(4):
        cy = 0.0255 - dgt * 0.017                      # digit 0 is leftmost in the image (+y)
        for sname, ((u, v), horiz) in seg_layout().items():
            px, py = -0.008 + v * DIGIT_H, cy - u * DIGIT_W
            size = [0.0012, DIGIT_W * 0.38, 0.0004] if horiz else [DIGIT_H * 0.2, 0.0012, 0.0004]
            b.add_geom(name=f"dev_board_seg{dgt}{sname}", type=box, size=size, pos=[px, py, 0.0092], rgba=[0.05, 0.01, 0.01, 1], material=g, **nocol)
    machine_panel(wb, LAB["machine"][0], top)
    stack_light(wb, LAB["stack_light"][0], top, g)
    # pressure gauge: dial on a stand, face turned toward the head camera
    (x, y), _ = LAB["gauge"]
    c, R = gauge_frame(top)
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, R.flatten())
    st = wb.add_body(name="gauge", pos=[x, y, top])
    st.add_geom(type=box, size=[0.012, 0.012, 0.035], pos=[0.01, 0, 0.035], rgba=[0.3, 0.3, 0.32, 1])
    dial = st.add_body(name="gauge_dial", pos=[0, 0, 0.075], quat=quat)
    dial.add_geom(type=cyl, size=[0.045, 0.004, 0], rgba=[0.15, 0.15, 0.15, 1])
    dial.add_geom(type=cyl, size=[0.041, 0.0005, 0], pos=[0, 0, 0.0042], rgba=[0.97, 0.97, 0.95, 1], **nocol)
    for k in range(11):                                  # ticks every bar, 0 at -135 deg (clockwise from up)
        a = np.radians(-135 + 27 * k)
        r = 0.034
        dial.add_geom(type=box, size=[0.0008, 0.004 if k % 5 == 0 else 0.0025, 0.0003],
                      pos=[r * np.sin(a), r * np.cos(a), 0.0048], euler=[0, 0, -a], rgba=[0.05, 0.05, 0.05, 1], **nocol)
    nd = dial.add_body(name="gauge_needle", pos=[0, 0, 0.005])
    nd.add_joint(name="gauge_needle", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 0, 1], damping=1.0, armature=0.001)
    nd.add_geom(type=box, size=[0.0012, 0.015, 0.0004], pos=[0, 0.013, 0.0005], rgba=[0.9, 0.05, 0.05, 1], mass=0.001, **nocol)
    nd.add_geom(type=cyl, size=[0.004, 0.0008, 0], rgba=[0.05, 0.05, 0.05, 1], mass=0.001, **nocol)


DEVICE_PARTS = [("power_strip", "power_strip_rocker"), ("light_switch", "light_switch_lever"),
                ("security_key", "security_key_pad"), ("start_button", "start_button_cap")]


def item_body(m: mujoco.MjModel, name: str) -> int:
    for n in (name, f"{name}_model"):
        try:
            return m.body(n).id
        except KeyError:
            pass
    raise KeyError(name)


if __name__ == "__main__":
    import time
    t = time.time()
    m = build()
    print(f"built in {time.time() - t:.1f}s: {m.nbody} bodies, {m.njnt} joints, {m.nu} actuators, {m.ngeom} geoms")

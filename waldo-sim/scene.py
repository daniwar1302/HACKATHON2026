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


def build(desk: Desk = Desk(), items: dict = ITEMS, containers: dict = CONTAINERS, devices: bool = False) -> mujoco.MjModel:
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

    if devices:
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
                     "box": mujoco.mjtGeom.mjGEOM_BOX}[kind]
            body.add_geom(name=f"{name}_geom", type=gtype, size=list(p) + [0] * (3 - len(p)), quat=quat, rgba=list(rgba),
                          mass=mass, friction=[0.9, 0.01, 0.001], condim=4)
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

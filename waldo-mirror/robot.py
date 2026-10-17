"""G1 side of mirroring: Unitree's arm IK, and a small, careful arm_sdk controller.

ArmIK wraps G1_29_ArmIK from ~/xr_teleoperate (legs and waist locked in the model, so it only
ever solves for the 14 arm joints) and reads the robot's shoulder positions and reach from it.

ArmSdk follows the SDK's g1_arm*_sdk_dds_example: it publishes rt/arm_sdk at 250 Hz, holds the
waist yaw where it is, and hands the arms over through the weight on motor slot 29, ramped in
and out. On top of that: a joint speed cap, joint limits, and a return to the starting pose
before handing the arms back to the stock controller.
"""
from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

import numpy as np

XR = Path.home() / "xr_teleoperate"
ARM_JOINTS = list(range(15, 29))   # left shoulder pitch .. right wrist yaw (G1 29-DoF)
WRISTS = {19, 20, 21, 26, 27, 28}
WAIST_YAW = 12
WEIGHT_SLOT = 29                   # arm_sdk weight: 1 = our commands, 0 = stock controller
LIMIT_MARGIN = 0.10                # rad kept away from each joint's hard limit

# Comfortable ranges for mirroring, per arm joint (degrees): shoulder pitch, roll, yaw, elbow,
# wrist roll, pitch, yaw. Tighter than the hardware limits so the IK can never choose a contorted
# pose (arm swung far back, twisted upper arm, hyperextended elbow). Roll is outward-positive on
# the left arm and outward-negative on the right.
_LEFT = [(-160, 40), (-10, 110), (-60, 60), (-5, 115), (-40, 40), (-40, 40), (-40, 40)]
_RIGHT = [(-160, 40), (-110, 10), (-60, 60), (-5, 115), (-40, 40), (-40, 40), (-40, 40)]
COMFORT_LOWER = np.radians([lo for lo, _ in _LEFT + _RIGHT])
COMFORT_UPPER = np.radians([hi for _, hi in _LEFT + _RIGHT])
# Where the G1's own controller rests the arms (measured on Waldo): the "natural" posture the IK prefers.
NATURAL = np.radians([16, 12, 0, 56, 0, 0, 0, 16, -12, 0, 56, 0, 0, 0])


class _NoFilter:
    """Stands in for the IK's internal moving-average filter: we smooth targets and cap speed ourselves."""

    def add_data(self, q):
        self.filtered_data = np.asarray(q)


class ArmIK:
    def __init__(self):
        import pinocchio as pin

        sys.path.insert(0, str(XR))
        cwd = os.getcwd()
        os.chdir(XR / "teleop")  # the IK finds its URDF and model cache relative to this
        try:
            from teleop.robot_control.robot_arm_ik import G1_29_ArmIK

            self.ik = G1_29_ArmIK(Unit_Test=False, Visualization=False)
        finally:
            os.chdir(cwd)
        self.ik.smooth_filter = _NoFilter()
        model = self.ik.reduced_robot.model
        data = model.createData()
        q0 = np.zeros(model.nq)
        pin.forwardKinematics(model, data, q0)
        pin.updateFramePlacements(model, data)
        ee = {"left": data.oMf[model.getFrameId("L_ee")], "right": data.oMf[model.getFrameId("R_ee")]}
        self.shoulder = {s: data.oMi[model.getJointId(f"{s}_shoulder_pitch_joint")].translation.copy() for s in ("left", "right")}
        # The G1's zero pose has the elbow bent, so a straight arm is shoulder->elbow + elbow->hand.
        elbow = {s: data.oMi[model.getJointId(f"{s}_elbow_joint")].translation for s in ("left", "right")}
        self.reach = {s: float(np.linalg.norm(elbow[s] - self.shoulder[s]) + np.linalg.norm(ee[s].translation - elbow[s]))
                      for s in ("left", "right")}
        self.lower = np.maximum(model.lowerPositionLimit + LIMIT_MARGIN, COMFORT_LOWER)
        self.upper = np.minimum(model.upperPositionLimit - LIMIT_MARGIN, COMFORT_UPPER)

    def solve(self, left: np.ndarray, right: np.ndarray, current_q: np.ndarray) -> np.ndarray:
        """14 arm joint angles that put the hands at `left` / `right` (waist frame), within limits.

        Only the hand position matters: each target keeps the hand's current orientation, so the
        solver never twists the shoulder to hold a wrist angle the person didn't ask for. It solves
        from the current pose and from the natural one, and keeps the more natural answer among
        those that reach the targets, so a run of frames can't drift into a contorted pose.
        """
        targets = {"left": left, "right": right}
        best, best_score = None, None
        for seed in (current_q, NATURAL):
            rotation = self.hand_rotations(seed)
            tf = {}
            for side, p in targets.items():
                m = np.eye(4)
                m[:3, :3], m[:3, 3] = rotation[side], p
                tf[side] = m
            q, _ = self.ik.solve_ik(tf["left"], tf["right"], seed)
            q = np.clip(q, self.lower, self.upper)
            hands = self.hand_positions(q)
            miss = sum(np.linalg.norm(hands[s] - targets[s]) for s in targets)
            score = 10.0 * miss + 0.1 * np.linalg.norm(q - NATURAL)
            if best_score is None or score < best_score:
                best, best_score = q, score
        return best

    def _hands(self, q: np.ndarray):
        import pinocchio as pin

        model = self.ik.reduced_robot.model
        data = model.createData()
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        return {"left": data.oMf[model.getFrameId("L_ee")], "right": data.oMf[model.getFrameId("R_ee")]}

    def hand_positions(self, q: np.ndarray) -> dict[str, np.ndarray]:
        return {s: m.translation.copy() for s, m in self._hands(q).items()}

    def hand_rotations(self, q: np.ndarray) -> dict[str, np.ndarray]:
        return {s: m.rotation.copy() for s, m in self._hands(q).items()}


def fsm_id(iface: str) -> int:
    """Current locomotion state (500 = balancing). Initialises DDS on first use."""
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient

    _init_dds(iface)
    loco = LocoClient()
    loco.SetTimeout(3.0)
    loco.Init()
    code, value = loco.GetFsmId()
    if code != 0 or value is None:
        raise RuntimeError(f"could not read FSM state (sdk code {code})")
    return int(value)


_dds_ready = False


def _init_dds(iface: str) -> None:
    global _dds_ready
    if not _dds_ready:
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize

        ChannelFactoryInitialize(0, iface)
        _dds_ready = True


class ArmSdk:
    """Drives the 14 arm joints through rt/arm_sdk with a speed cap and ramped hand-over."""

    def __init__(self, iface: str, max_vel: float = 1.5, kp: float = 60.0, kd: float = 1.5,
                 kp_wrist: float = 40.0, kd_wrist: float = 1.5):
        from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
        from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
        from unitree_sdk2py.utils.crc import CRC

        _init_dds(iface)
        self.max_vel, self.dt = max_vel, 1.0 / 250.0
        self.gains = {j: (kp_wrist, kd_wrist) if j in WRISTS else (kp, kd) for j in ARM_JOINTS}
        self.gains[WAIST_YAW] = (kp, kd)
        self._state = None
        self.pub = ChannelPublisher("rt/arm_sdk", LowCmd_)
        self.pub.Init()
        self.sub = ChannelSubscriber("rt/lowstate", LowState_)
        self.sub.Init(self._on_state, 10)
        deadline = time.monotonic() + 5
        while self._state is None:
            if time.monotonic() > deadline:
                raise RuntimeError("no rt/lowstate from the robot in 5 s")
            time.sleep(0.05)
        self.crc, self.msg = CRC(), unitree_hg_msg_dds__LowCmd_()
        self.msg.mode_pr = 0
        self.msg.mode_machine = self._state.mode_machine
        self.start = self.measured()          # arms go back here before hand-back
        self.waist = self._state.motor_state[WAIST_YAW].q
        self.cmd = self.start.copy()          # what we command right now (speed-limited)
        self.target = self.start.copy()       # where we want to go
        self.weight = 0.0
        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _on_state(self, msg) -> None:
        self._state = msg

    def measured(self) -> np.ndarray:
        return np.array([self._state.motor_state[j].q for j in ARM_JOINTS])

    def set_target(self, q: np.ndarray) -> None:
        with self._lock:
            self.target = np.asarray(q, dtype=float).copy()

    def _loop(self) -> None:
        step = self.max_vel * self.dt
        while self._running:
            t0 = time.monotonic()
            with self._lock:
                self.cmd = self.cmd + np.clip(self.target - self.cmd, -step, step)
                weight, cmd = self.weight, self.cmd.copy()
            for i, j in enumerate(ARM_JOINTS):
                m = self.msg.motor_cmd[j]
                m.mode, m.q, m.dq, m.tau = 1, float(cmd[i]), 0.0, 0.0
                m.kp, m.kd = self.gains[j]
            w = self.msg.motor_cmd[WAIST_YAW]
            w.mode, w.q, w.dq, w.tau = 1, float(self.waist), 0.0, 0.0
            w.kp, w.kd = self.gains[WAIST_YAW]
            self.msg.motor_cmd[WEIGHT_SLOT].q = weight
            self.msg.crc = self.crc.Crc(self.msg)
            self.pub.Write(self.msg)
            time.sleep(max(0.0, self.dt - (time.monotonic() - t0)))

    def _ramp(self, to: float, seconds: float) -> None:
        start = self.weight
        steps = max(1, int(seconds / 0.02))
        for k in range(1, steps + 1):
            with self._lock:
                self.weight = start + (to - start) * k / steps
            time.sleep(0.02)

    def engage(self, seconds: float = 2.0) -> None:
        """Take the arms over gradually, holding them exactly where they are."""
        with self._lock:
            self.cmd = self.target = self.measured()
        self._ramp(1.0, seconds)

    def release(self, seconds: float = 2.0, settle_timeout: float = 6.0) -> None:
        """Glide back to the starting pose, then hand the arms back to the stock controller."""
        self.set_target(self.start)
        deadline = time.monotonic() + settle_timeout
        while time.monotonic() < deadline and np.max(np.abs(self.cmd - self.start)) > 0.02:
            time.sleep(0.05)
        self._ramp(0.0, seconds)
        time.sleep(0.1)
        self._running = False
        self._thread.join(timeout=1)

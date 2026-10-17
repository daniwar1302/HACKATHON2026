"""Human arm pose -> robot hand targets. Pure numpy, no robot, so it can be tested anywhere.

Input is MediaPipe world landmarks (metres, origin between the hips). Each arm becomes a
"reach" vector: wrist minus shoulder, in the person's own torso frame (x forward, y to the
person's left, z up, the robot's convention), divided by that arm's length. The robot hand
target is its shoulder plus reach times its own arm length, clamped to a safe workspace.

Mirror mode (default): facing the robot, your right arm drives its left arm, like a mirror.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# MediaPipe pose landmark indices (left/right are the person's own).
L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST, L_HIP, R_HIP = 11, 12, 13, 14, 15, 16, 23, 24
ARM_POINTS = {"left": (L_SHOULDER, L_ELBOW, L_WRIST), "right": (R_SHOULDER, R_ELBOW, R_WRIST)}
MIN_VISIBILITY = 0.6


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-6:
        raise ValueError("degenerate pose")
    return v / n


def torso_frame(pts: np.ndarray) -> np.ndarray:
    """3x3 matrix whose rows are the person's forward, left and up axes in landmark coordinates."""
    left = _unit(pts[L_SHOULDER] - pts[R_SHOULDER])
    up = (pts[L_SHOULDER] + pts[R_SHOULDER]) / 2 - (pts[L_HIP] + pts[R_HIP]) / 2
    up = _unit(up - up.dot(left) * left)
    forward = np.cross(left, up)
    return np.stack([forward, left, up])


def arm_reaches(pts: np.ndarray, visibility: np.ndarray) -> dict[str, np.ndarray | None]:
    """Per human arm: wrist-minus-shoulder in the torso frame, as a fraction of arm length.

    None for an arm the camera can't see well enough (that arm should hold still).
    """
    frame = torso_frame(pts)
    out: dict[str, np.ndarray | None] = {}
    for side, (s, e, w) in ARM_POINTS.items():
        if min(visibility[s], visibility[e], visibility[w]) < MIN_VISIBILITY:
            out[side] = None
            continue
        length = np.linalg.norm(pts[e] - pts[s]) + np.linalg.norm(pts[w] - pts[e])
        reach = frame @ (pts[w] - pts[s]) / max(length, 1e-3)
        n = np.linalg.norm(reach)
        out[side] = reach / n if n > 1.0 else reach  # can't reach further than a straight arm
    return out


def robot_reaches(human: dict[str, np.ndarray | None], mirror: bool = True) -> dict[str, np.ndarray | None]:
    """Which human arm drives which robot arm. Mirror swaps sides and flips left/right."""
    if not mirror:
        return dict(human)
    flip = np.array([1.0, -1.0, 1.0])
    return {"left": None if human["right"] is None else human["right"] * flip,
            "right": None if human["left"] is None else human["left"] * flip}


@dataclass(frozen=True)
class Workspace:
    """Where the robot's hands may go, in its waist frame (x forward, y left, z up), metres."""
    min_forward: float = 0.08    # always in front of the torso
    min_side: float = 0.10       # each hand stays on its own side of the midline
    min_height: float = -0.05    # not below the waist
    max_height: float = 0.75     # not far over the head

    def clamp(self, side: str, p: np.ndarray) -> np.ndarray:
        p = p.copy()
        p[0] = max(p[0], self.min_forward)
        p[1] = max(p[1], self.min_side) if side == "left" else min(p[1], -self.min_side)
        p[2] = min(max(p[2], self.min_height), self.max_height)
        return p


def hand_targets(reaches: dict[str, np.ndarray | None], shoulders: dict[str, np.ndarray],
                 arm_length: dict[str, float], workspace: Workspace = Workspace()) -> dict[str, np.ndarray | None]:
    return {side: None if r is None else workspace.clamp(side, shoulders[side] + r * arm_length[side])
            for side, r in reaches.items()}


class Smoother:
    """Exponential smoothing per arm; a missing arm keeps its last value."""

    def __init__(self, alpha: float = 0.5):
        self.alpha, self.value = alpha, {}

    def update(self, side: str, x: np.ndarray | None) -> np.ndarray | None:
        if x is None:
            return self.value.get(side)
        prev = self.value.get(side)
        self.value[side] = x if prev is None else self.alpha * x + (1 - self.alpha) * prev
        return self.value[side]

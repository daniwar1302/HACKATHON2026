import numpy as np
import pytest

from pose import (L_ELBOW, L_HIP, L_SHOULDER, L_WRIST, R_ELBOW, R_HIP, R_SHOULDER, R_WRIST, Smoother, Workspace,
                  arm_reaches, hand_targets, robot_reaches, torso_frame)

# A person facing the camera, in MediaPipe world coordinates: x = image right (= the person's left),
# y = down, z = away from the camera. So the person's forward (towards the camera) is -z.
FWD, LEFT, UP = np.array([0, 0, -1.0]), np.array([1.0, 0, 0]), np.array([0, -1.0, 0])


def person(left_wrist_dir, right_wrist_dir):
    pts = np.zeros((33, 3))
    pts[L_HIP], pts[R_HIP] = 0.1 * LEFT, -0.1 * LEFT
    pts[L_SHOULDER], pts[R_SHOULDER] = 0.5 * UP + 0.2 * LEFT, 0.5 * UP - 0.2 * LEFT
    for s, e, w, d in ((L_SHOULDER, L_ELBOW, L_WRIST, left_wrist_dir), (R_SHOULDER, R_ELBOW, R_WRIST, right_wrist_dir)):
        d = np.asarray(d, float) / np.linalg.norm(d)
        pts[e], pts[w] = pts[s] + 0.3 * d, pts[s] + 0.6 * d  # straight arm, 0.6 m
    return pts, np.ones(33)


def test_torso_frame_is_forward_left_up():
    pts, _ = person(-UP, -UP)
    assert np.allclose(torso_frame(pts), np.stack([FWD, LEFT, UP]))


def test_arm_straight_forward_reaches_forward():
    pts, vis = person(FWD, -UP)
    r = arm_reaches(pts, vis)
    assert np.allclose(r["left"], [1, 0, 0]) and np.allclose(r["right"], [0, 0, -1])


def test_mirror_swaps_arms_and_flips_sideways():
    pts, vis = person(-UP, -LEFT)  # right arm straight out to the person's right
    mirrored = robot_reaches(arm_reaches(pts, vis), mirror=True)
    assert np.allclose(mirrored["left"], [0, 1, 0])   # robot's LEFT arm goes out to ITS left
    copied = robot_reaches(arm_reaches(pts, vis), mirror=False)
    assert np.allclose(copied["right"], [0, -1, 0])


def test_hidden_arm_is_none():
    pts, vis = person(FWD, FWD)
    vis[R_WRIST] = 0.2
    r = arm_reaches(pts, vis)
    assert r["right"] is None and r["left"] is not None


def test_workspace_keeps_hands_in_front_and_on_their_own_side():
    ws = Workspace()
    shoulders = {"left": np.array([0.0, 0.15, 0.4]), "right": np.array([0.0, -0.15, 0.4])}
    crossing = {"left": np.array([0.0, -1.0, 0.0]), "right": np.array([-1.0, 0.0, -1.0])}  # across the body / behind and low
    t = hand_targets(crossing, shoulders, {"left": 0.5, "right": 0.5}, ws)
    assert t["left"][1] >= ws.min_side and t["left"][0] >= ws.min_forward
    assert t["right"][0] >= ws.min_forward and t["right"][2] >= ws.min_height


def test_degenerate_pose_raises():
    pts = np.zeros((33, 3))
    with pytest.raises(ValueError):
        torso_frame(pts)


def test_smoother_holds_when_arm_missing():
    s = Smoother(0.5)
    s.update("left", np.array([1.0, 0, 0]))
    assert np.allclose(s.update("left", None), [1, 0, 0])
    assert np.allclose(s.update("left", np.array([0.0, 0, 0])), [0.5, 0, 0])

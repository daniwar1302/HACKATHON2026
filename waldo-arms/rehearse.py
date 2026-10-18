"""Rehearse in a twin sim, then execute: try each strategy on a copy of the world, keep the first that verifiably
works, and send exactly its commands to the real arm. Production cells do this so a failed idea costs sim time,
not a knocked-over part.

The twin is a second sim_server seeded with sim_set_world. Against the real G1 the world comes from perception
(item poses from the head camera); in the benchmark it is copied from the 'real' sim plus position noise.
"""
import numpy as np

from planner import PlanError

ACTUATE = {"trajectory", "move_ee", "grasp", "open", "gripper", "enable", "park"}


class Recorder:
    """Arm wrapper: forwards to the twin, remembers what moved it."""

    def __init__(self, arm):
        self.arm, self.log = arm, []

    def __call__(self, cmd, **kw):
        r = self.arm(cmd, **kw)
        if cmd in ACTUATE:
            self.log.append((cmd, kw))
        return r


def copy_world(real, twin, noise_mm=0.0, rng=None):
    """Twin <- real world, with each item's xy shifted by perception-like noise."""
    w = real("sim_get_world")
    q = np.array(w["qpos"])
    if noise_mm:
        rng = rng or np.random.default_rng()
        for qa in w["items"].values():
            q[qa:qa + 2] += rng.normal(0, noise_mm / 1000, 2)
    twin("sim_set_world", qpos=q.tolist(), enabled=w["enabled"], settle=50 if noise_mm else 0)


def rehearse(planner, real, twin, strategies, check, noise_mm=0.0, rng=None):
    """strategies: [(label, fn(planner))]; check(twin) -> bool. Returns (label, tried) after executing on `real`,
    or raises PlanError listing why each rehearsal failed."""
    tried = []
    for label, fn in strategies:
        copy_world(real, twin, noise_mm, rng)
        rec = Recorder(twin)
        planner.arm, planner.R = rec, {"left": None, "right": None}
        try:
            fn(planner)
            ok = check(twin)
            why = "" if ok else "rehearsal: did not verify"
        except PlanError as e:
            ok, why = False, str(e).split("; ")[-1][:80]
        tried.append(f"{label}: {'ok' if ok else why}")
        if ok:
            planner.arm = real
            for cmd, kw in rec.log:
                real(cmd, **kw)
            return label, tried
    planner.arm = real
    raise PlanError("; ".join(tried))

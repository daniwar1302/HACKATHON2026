#!/usr/bin/env python3
"""Smoothness of every taught motion: peak joint speed, acceleration and jerk of each trajectory as sent, and the
velocity jumps at waypoint corners (a polyline timed along its length kinks at each waypoint).
  uv run --with numpy smoothness.py [taught*.json]
"""
import glob
import json
import sys

import numpy as np


def metrics(q, t):
    q, t = np.asarray(q, float), np.asarray(t, float)
    if len(q) < 4:
        return None
    dt = np.diff(t)
    v = np.diff(q, axis=0) / dt[:, None]
    a = np.diff(v, axis=0) / dt[1:, None]
    j = np.diff(a, axis=0) / dt[2:, None]
    return np.abs(v).max(), np.abs(a).max(), np.abs(j).max()


def trajectories(entry):
    if "log" in entry:
        return [kw for c, kw in entry["log"] if c == "trajectory"]
    sys.path.insert(0, "../waldo-arms")
    from motion import timed
    out = []
    for key in ("approach", "press"):
        q, t = timed(entry[key])
        out.append({"q": q, "times": t})
    return out


if __name__ == "__main__" and "--efficiency" not in sys.argv:
    files = sys.argv[1:] or sorted(glob.glob("taught*.json"))
    worst = []
    for f in files:
        for name, e in json.load(open(f)).items():
            for k, tr in enumerate(trajectories(e)):
                m = metrics(tr["q"], tr["times"])
                if m:
                    worst.append((m[1], f, name, k, m))
    worst.sort(reverse=True)
    acc = np.array([w[0] for w in worst])
    print(f"{len(worst)} trajectories: peak joint accel median {np.median(acc):.1f}, p90 {np.percentile(acc, 90):.1f}, "
          f"max {acc.max():.1f} rad/s^2")
    for a, f, name, k, (v, aa, j) in worst[:8]:
        print(f"  {f:18s} {name:16s} #{k}: speed {v:4.2f} rad/s  accel {aa:6.1f} rad/s^2  jerk {j:8.0f} rad/s^3")


def efficiency(files):
    """Path economy: joint-space distance travelled per trajectory and the widest sideways arm swing
    (|shoulder roll| beyond the rest pose's 0.2 rad means the elbow is thrown out to the side)."""
    import numpy as np
    lens, rolls = [], []
    for f in files:
        for name, e in json.load(open(f)).items():
            for tr in trajectories(e):
                q = np.asarray(tr["q"], float)
                lens.append(float(np.abs(np.diff(q, axis=0)).sum()))
                rolls.append(float(max(np.abs(q[:, 1]).max(), np.abs(q[:, 8]).max())))
    lens, rolls = np.array(lens), np.array(rolls)
    print(f"joint travel per trajectory: median {np.median(lens):.1f}, total {lens.sum():.0f} rad; "
          f"shoulder roll max {rolls.max():.2f} rad, trajectories with roll > 0.8: {(rolls > 0.8).sum()}/{len(rolls)}")


if __name__ == "__main__" and "--efficiency" in sys.argv:
    efficiency(sorted(glob.glob("taught*.json")))

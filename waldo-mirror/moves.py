"""Dance moves for the G1's arms (and a little waist sway). Pure numpy, testable without a robot.

A move is an 8-beat pattern: given the position within it (in beats, 0..8) and an energy (0..1),
it returns the 14 arm joint angles and a waist-yaw offset. Every pose is clipped to the same
comfort ranges as mirroring, and `fit_energy` lowers the energy until the move's fastest joint
stays under the speed cap at the current tempo.

Joint order per arm: shoulder pitch (negative = forward/up), roll (outward), yaw, elbow (bent
positive), wrist roll, pitch, yaw. Left arm first. Right-arm roll and yaw are mirrored here, so
moves are written as if both arms were left arms.
"""
from __future__ import annotations

import math

import numpy as np

from robot import COMFORT_LOWER, COMFORT_UPPER

BEATS_PER_MOVE = 8  # long enough that travelling between poses is a small part of each move
MAX_WAIST = 0.15  # rad, ~9 degrees of hip sway at most


def arms(left: tuple, right: tuple) -> np.ndarray:
    """Build 14 joint angles from (pitch, roll, yaw, elbow) per arm, written left-arm style."""
    lp, lr, ly, le = left
    rp, rr, ry, re = right
    return np.array([lp, lr, ly, le, 0, 0, 0, rp, -rr, -ry, re, 0, 0, 0], dtype=float)


def bump(b: float) -> float:
    """0 -> 1 -> 0 once per beat, peaking on the beat."""
    return 0.5 + 0.5 * math.cos(2 * math.pi * b)


def swing(b: float, beats: float = 2) -> float:
    """-1..1 sinusoid with the given period in beats."""
    return math.sin(2 * math.pi * b / beats)


BASE = (-0.35, 0.25, 0.0, 1.4)  # ready pose: arms a little forward, elbows bent


def pump(b, e):
    up = 0.55 * e * bump(b / 2)  # half-time: one pump every two beats, so it can be bigger under the speed cap
    a = (BASE[0] - up, BASE[1], 0.0, BASE[3] - 0.3 * e * bump(b / 2))
    return arms(a, a), 0.0


def sway(b, e):
    s = swing(b, 2)
    return arms((BASE[0], BASE[1] + 0.3 * e * s, 0.0, BASE[3]),
                (BASE[0], BASE[1] - 0.3 * e * s, 0.0, BASE[3])), MAX_WAIST * e * s


def wave(b, e):
    lift = 0.7 + 0.5 * e
    return arms((-1.2, lift, 0.0, 0.6 + 0.6 * e * (0.5 + 0.5 * swing(b, 2))),
                (-1.2, lift, 0.0, 0.6 + 0.6 * e * (0.5 - 0.5 * swing(b, 2)))), 0.0


def raise_roof(b, e):
    a = (-1.6 - 0.6 * e, 0.35, 0.0, 1.3 - 0.6 * e * bump(b / 2))
    return arms(a, a), 0.0


def disco(b, e):
    point = 0.5 - 0.5 * math.cos(math.pi * b / 2)  # up for two beats, down for two
    right = (-0.3 - (0.4 + 1.6 * e) * point, 0.45, 0.0, 0.25)
    left = (0.0, 0.3, 0.0, 1.6)  # hand by the hip
    return arms(left, right), MAX_WAIST * 0.5 * e * swing(b, 4)


def circles(b, e):
    t = 2 * math.pi * b / 2
    a = (-0.9 + 0.45 * e * math.cos(t), 0.6 + 0.3 * e * math.sin(t), 0.0, 0.7)
    return arms(a, a), 0.0


def hands_up(b, e):
    wiggle = 0.25 * e * swing(b, 1)
    return arms((-2.0, 0.6, 0.0, 0.5 + wiggle), (-2.0, 0.6, 0.0, 0.5 - wiggle)), 0.0


MOVES = {"pump": pump, "sway": sway, "wave": wave, "raise_roof": raise_roof,
         "disco": disco, "circles": circles, "hands_up": hands_up}
CALM = ("sway", "circles", "wave")             # good for slow / quiet music
HYPE = ("pump", "raise_roof", "disco", "hands_up")


def pose(move: str, beat: float, energy: float) -> tuple[np.ndarray, float]:
    """Joint angles (clipped to comfort ranges) and waist yaw for `move` at `beat` within it."""
    q, waist = MOVES[move](beat % BEATS_PER_MOVE, float(np.clip(energy, 0.0, 1.0)))
    return np.clip(q, COMFORT_LOWER, COMFORT_UPPER), float(np.clip(waist, -MAX_WAIST, MAX_WAIST))


def ready_pose() -> np.ndarray:
    return np.clip(arms(BASE, BASE), COMFORT_LOWER, COMFORT_UPPER)


def max_speed(move: str, bpm: float, energy: float, samples: int = 200) -> float:
    """Fastest joint speed (rad/s) of one pass through the move at this tempo."""
    beat_s = 60.0 / bpm
    db = BEATS_PER_MOVE / samples
    qs = np.array([pose(move, i * db, energy)[0] for i in range(samples + 1)])
    return float(np.max(np.abs(np.diff(qs, axis=0))) / (db * beat_s))


def fit_energy(move: str, bpm: float, energy: float, max_vel: float) -> float:
    """The highest energy <= `energy` whose fastest joint stays under `max_vel` at this tempo."""
    if max_speed(move, bpm, energy) <= max_vel:
        return energy
    lo, hi = 0.0, energy
    for _ in range(12):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if max_speed(move, bpm, mid) <= max_vel else (lo, mid)
    return lo

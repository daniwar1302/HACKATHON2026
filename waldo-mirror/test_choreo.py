import random

import numpy as np
import pytest

from choreo import ENV_RATE, Candidate, align, onset_envelope, score, speed_limit
from dance import Choreo


def test_score_prefers_the_real_just_dance_routine():
    q = "levitating dua lipa"
    cands = [
        Candidate("a", "Dua Lipa - Levitating (Official Music Video)", 231, ""),
        Candidate("b", "Levitating (Extreme) - Just Dance 2022 - Megastar", 226, ""),
        Candidate("c", "Just Dance 2022 - Levitating - All Perfects", 253, ""),
        Candidate("d", "Just Dance Kids Megamix 2 hours", 7200, ""),
    ]
    assert max(cands, key=lambda c: score(c, q)).id == "c"


def test_speed_limit_caps_every_joint():
    q = np.zeros((50, 14))
    q[10:, 0] = 2.0  # an instant 2 rad jump
    out = speed_limit(q, dt=1 / 15, max_vel=1.0)
    assert np.max(np.abs(np.diff(out, axis=0))) <= 1.0 / 15 + 1e-9
    assert out[-1, 0] == pytest.approx(2.0)  # still gets there


def song(seconds=120, rate=16000, seed=0):
    """A fake song: irregular drum hits and a little noise, so every 10 s window is unique."""
    rng = np.random.default_rng(seed)
    x = 0.01 * rng.standard_normal(seconds * rate)
    t = 0.0
    while t < seconds - 0.1:
        i = int(t * rate)
        n = int(0.03 * rate)
        x[i:i + n] += rng.uniform(0.3, 0.9) * np.exp(-np.arange(n) / (0.005 * rate)) * rng.standard_normal(n)
        t += rng.choice([0.25, 0.5, 0.5, 0.75])
    return x.astype(np.float32)


def test_align_finds_where_a_phone_clip_sits_in_the_song():
    full = song()
    song_env = onset_envelope(full, 16000)
    # the clip as heard on the call: 10 s ending at 70 s, a different sample rate, quieter, noisier
    clip = full[60 * 16000:70 * 16000]
    clip24 = np.interp(np.arange(0, len(clip), 16000 / 24000), np.arange(len(clip)), clip).astype(np.float32)
    clip24 = 0.5 * clip24 + 0.02 * np.random.default_rng(1).standard_normal(len(clip24)).astype(np.float32)
    end_t, strength = align(onset_envelope(clip24, 24000), song_env)
    assert end_t == pytest.approx(70.0, abs=2 / ENV_RATE)
    assert strength > 0.1


def test_align_reports_weak_match_for_a_different_song():
    end_t, strength = align(onset_envelope(song(seed=5)[:160000], 16000), onset_envelope(song(seed=0), 16000))
    assert strength < 0.1


def test_routine_plays_in_time_then_hands_back_to_freestyle():
    c = Choreo(max_vel=1.0, offset=0.0, rng=random.Random(0))
    c.on_tempo({"bpm": 120, "beat_t": 0.0, "confidence": 0.6, "loudness": 0.3}, now=0.0)
    times = np.arange(0, 10, 1 / 15)
    qs = np.zeros((len(times), 14))
    qs[:, 0] = times / 10  # shoulder pitch ramps 0 -> 1 rad over the routine
    c.play(times, qs, video_t=4.0, wall_t=100.0)  # 4 s into the routine at wall time 100
    move, q, _ = c.target(101.0)
    assert move == "choreography" and q[0] == pytest.approx(0.5, abs=0.01)
    move, _, _ = c.target(107.0)  # routine ended (4 + 7 > 10 s): back to freestyle moves
    assert move != "choreography"

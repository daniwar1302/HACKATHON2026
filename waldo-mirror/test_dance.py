import random

import numpy as np
import pytest

from dance import Choreo
from moves import BEATS_PER_MOVE, MAX_WAIST, MOVES, fit_energy, max_speed, pose, ready_pose
from robot import COMFORT_LOWER, COMFORT_UPPER


@pytest.mark.parametrize("move", sorted(MOVES))
def test_every_pose_stays_in_comfort_ranges(move):
    for energy in (0.0, 0.5, 1.0):
        for b in np.linspace(0, BEATS_PER_MOVE, 41):
            q, waist = pose(move, b, energy)
            assert np.all(q >= COMFORT_LOWER - 1e-9) and np.all(q <= COMFORT_UPPER + 1e-9)
            assert abs(waist) <= MAX_WAIST


@pytest.mark.parametrize("move", sorted(MOVES))
@pytest.mark.parametrize("bpm", [90, 128, 170])
def test_fit_energy_respects_the_speed_cap(move, bpm):
    e = fit_energy(move, bpm, 1.0, max_vel=1.0)
    assert 0.0 <= e <= 1.0
    if e > 0:
        assert max_speed(move, bpm, e) <= 1.0 + 1e-6


def test_ready_pose_is_valid():
    q = ready_pose()
    assert q.shape == (14,) and np.all(q >= COMFORT_LOWER) and np.all(q <= COMFORT_UPPER)


def tempo(bpm=120.0, beat_t=0.0, confidence=0.6, loudness=0.3):
    return {"bpm": bpm, "beat_t": beat_t, "confidence": confidence, "loudness": loudness}


def test_waits_on_ready_pose_until_the_next_beat_then_follows_the_phrase():
    c = Choreo(max_vel=1.0, offset=0.0, rng=random.Random(0))
    c.on_tempo(tempo(beat_t=0.0), now=10.2)  # 120 BPM: beats at 10.0, 10.5, ...
    c.on_phrase(["wave", "disco", "nope"], 0.8)
    move, q, _ = c.target(10.2)
    assert move is None and np.allclose(q, ready_pose())  # first move starts on beat 21 (t=10.5)
    assert c.target(10.6)[0] == "wave"
    assert c.target(10.5 + BEATS_PER_MOVE * 0.5 + 0.1)[0] == "disco"  # next slot
    nxt = c.target(10.5 + 2 * BEATS_PER_MOVE * 0.5 + 0.1)[0]
    assert nxt in MOVES  # queue empty: improvises


def test_improvises_calm_moves_for_quiet_music():
    from moves import CALM

    c = Choreo(max_vel=1.0, offset=0.0, rng=random.Random(1))
    c.energy = 0.3
    c.on_tempo(tempo(loudness=0.1), now=0.0)
    picked = set()
    for slot in range(8):
        picked.add(c.target(0.6 + slot * BEATS_PER_MOVE * 0.5)[0])
    picked.discard(None)  # before the first beat it holds the ready pose
    assert picked and picked <= set(CALM)


def test_low_confidence_tempo_is_ignored_and_music_gone_after_silence():
    c = Choreo(max_vel=1.0, offset=0.0)
    c.on_tempo(tempo(confidence=0.05), now=0.0)
    assert c.bpm is None and c.music_gone(0.0)
    c.on_tempo(tempo(), now=1.0)
    assert not c.music_gone(2.0)
    assert c.music_gone(1.0 + 6.5)  # no fresh tempo for >6 s
    c.on_tempo(tempo(loudness=0.0), now=8.0)
    c.on_tempo(tempo(loudness=0.0), now=15.0)
    assert c.music_gone(15.0)  # quiet for >6 s


def test_fresh_tempo_updates_keep_the_dance_moving():
    """The tracker reports the latest beat each second; the move sequence must keep advancing."""
    c = Choreo(max_vel=1.0, offset=0.0, rng=random.Random(2))
    c.on_phrase(["wave", "disco", "sway", "circles"], 0.7)
    seen = []
    for second in range(18):  # 120 BPM: a new 8-beat move every 4 s
        now = 100.0 + second
        c.on_tempo(tempo(beat_t=now - 0.01), now=now)  # slightly jittery latest-beat reports
        seen.append(c.target(now + 0.3)[0])
    order = [m for i, m in enumerate(seen) if m and (i == 0 or m != seen[i - 1])]
    assert order[:4] == ["wave", "disco", "sway", "circles"]


def test_offset_leads_the_beat():
    c = Choreo(max_vel=1.0, offset=0.25)
    c.on_tempo(tempo(beat_t=0.0), now=0.0)
    assert c.beats(1.0) == pytest.approx((1.0 + 0.25) * 2.0)

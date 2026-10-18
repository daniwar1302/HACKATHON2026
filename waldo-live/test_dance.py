import numpy as np
import pytest

from beats import BeatTracker
from dance_call import DanceCall


def click_track(bpm: float, seconds: float, sr: int = 24000, noise: float = 0.02, seed: int = 0) -> bytes:
    """Short percussive clicks on every beat (beat 0 at t=0), plus noise, as 16-bit PCM."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * sr)) / sr
    x = noise * rng.standard_normal(len(t))
    for b in np.arange(0, seconds, 60.0 / bpm):
        i = int(b * sr)
        n = min(len(x) - i, int(0.03 * sr))
        x[i:i + n] += 0.8 * np.exp(-np.arange(n) / (0.004 * sr)) * rng.standard_normal(n)
    return (np.clip(x, -1, 1) * 32767).astype("<i2").tobytes()


@pytest.mark.parametrize("bpm", [96.0, 128.0, 150.0])
def test_beat_tracker_finds_tempo_and_phase(bpm):
    clock = {"t": 0.0}
    tracker = BeatTracker(24000, clock=lambda: clock["t"])
    pcm = click_track(bpm, 8.0)
    chunk = 24000 // 50 * 2  # 20 ms
    for i in range(0, len(pcm), chunk):
        tracker.feed(pcm[i:i + chunk])
        clock["t"] = (i + chunk) / 2 / 24000
    beat = tracker.estimate()
    assert beat is not None and abs(beat.bpm - bpm) < 3, beat
    period = 60.0 / bpm
    phase_error = (beat.beat_t + period / 2) % period - period / 2  # distance to the nearest true beat
    assert abs(phase_error) < 0.05, beat
    assert beat.confidence > 0.2


def test_beat_tracker_waits_for_enough_audio():
    tracker = BeatTracker(24000)
    tracker.feed(click_track(120, 1.0))
    assert tracker.estimate() is None


class FakeSession:
    def __init__(self):
        self.sent, self.said = [], []

    async def send_openai(self, payload):
        self.sent.append(payload)

    async def respond(self, instructions=None):
        self.said.append(instructions)


def make(clock=None):
    published = []

    async def publish(event):
        published.append(event)

    s = FakeSession()
    return DanceCall(s, publish, clock=clock or (lambda: 0.0)), s, published


class SteadyTracker:
    def __init__(self, beat):
        self.beat = beat

    def feed(self, pcm):
        pass

    def estimate(self):
        return self.beat


async def test_listening_turns_off_auto_responses_and_asks_once_the_beat_is_steady():
    from beats import Beat

    d, s, published = make()
    await d.run_tool("listen_for_music", {})
    td = s.sent[-1]["session"]["audio"]["input"]["turn_detection"]
    assert td["create_response"] is False and td["interrupt_response"] is False
    d.tracker = SteadyTracker(Beat(124.0, 10.0, 0.6, 0.3))
    for _ in range(3):
        await d.tick()
    assert d.state == "confirming" and "go Waldo" in s.said[-1]
    assert published[-1]["type"] == "dance_tempo" and published[-1]["bpm"] == 124.0


async def test_lyrics_saying_yes_do_not_start_the_dance_but_go_waldo_does():
    d, s, published = make()
    d.state = "confirming"
    await d.on_transcript("remote", "yeah yeah yeah, go go go")
    assert d.state == "confirming" and not any(e["type"] == "dance_start" for e in published)
    await d.on_transcript("local", "go Waldo")  # Waldo's own words never count
    assert d.state == "confirming"
    await d.on_transcript("remote", "Go, Waldo!")
    assert d.state == "starting" and published[-1]["type"] == "dance_start"


async def test_texted_go_starts_and_texted_stop_stops():
    d, s, published = make()
    d.state = "confirming"
    await d.on_text("Go")
    assert published[-1]["type"] == "dance_start"
    await d.on_bus({"type": "dance_started"})
    await d.on_text("stop")
    assert published[-1]["type"] == "dance_stop" and d.state == "done"


async def test_stop_word_ends_the_dance_and_turns_responses_back_on():
    d, s, published = make()
    d.state = "dancing"
    await d.on_transcript("remote", "ok that's enough")
    assert published[-1]["type"] == "dance_stop" and d.state == "done"
    assert any(p.get("session", {}).get("audio", {}).get("input", {}).get("turn_detection", {}).get("create_response")
               for p in s.sent)


async def test_dance_tool_only_while_dancing_and_filters_moves():
    d, s, published = make()
    assert not (await d.run_tool("dance", {"moves": ["pump"], "energy": 0.8}))["ok"]
    d.state = "dancing"
    out = await d.run_tool("dance", {"moves": ["pump", "backflip", "disco"], "energy": 0.8})
    assert out["ok"] and published[-1] == {"type": "dance_phrase", "moves": ["pump", "disco"], "energy": 0.8}


async def test_phrase_requests_every_16_beats():
    from beats import Beat

    clock = {"t": 100.0}
    d, s, published = make(clock=lambda: clock["t"])
    d.state = "dancing"
    d.tracker = SteadyTracker(Beat(120.0, 99.5, 0.6, 0.3))
    await d.tick()
    assert "dance tool" in s.said[-1]
    n = len(s.said)
    clock["t"] += 4.0
    await d.tick()
    assert len(s.said) == n  # 16 beats at 120 BPM = 8 s, not yet
    clock["t"] += 4.1
    await d.tick()
    assert len(s.said) == n + 1


async def test_engine_stopping_ends_the_dance():
    d, s, published = make()
    d.state = "dancing"
    await d.on_bus({"type": "dance_stopped", "reason": "music ended"})
    assert d.state == "done" and "music ended" in s.said[-1]

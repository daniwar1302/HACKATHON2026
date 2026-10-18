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
    assert d.state == "countdown" and "3, 2, 1" in s.said[-1] and "in my joints" in s.said[-1] and "BPM" not in s.said[-1]
    assert published[-1]["type"] == "dance_tempo" and published[-1]["bpm"] == 124.0


async def test_countdown_starts_the_dance_by_itself_unless_stopped():
    from beats import Beat

    clock = {"t": 0.0}
    d, s, published = make(clock=lambda: clock["t"])
    await d.run_tool("listen_for_music", {})
    d.tracker = SteadyTracker(Beat(124.0, 0.0, 0.6, 0.3))
    for _ in range(3):
        await d.tick()
    clock["t"] += 2.0
    await d.tick()
    assert d.state == "countdown"
    clock["t"] += 2.5
    await d.tick()
    assert d.state == "starting" and any(e["type"] == "dance_start" for e in published)

    d2, s2, published2 = make(clock=lambda: clock["t"])
    d2.state, d2.go_at = "countdown", clock["t"] + 4
    await d2.on_transcript("remote", "wait, stop!")
    clock["t"] += 5
    await d2.tick()
    assert d2.state == "done" and not any(e["type"] == "dance_start" for e in published2)


async def test_confirm_mode_still_waits_for_go_waldo(monkeypatch):
    from beats import Beat

    monkeypatch.setenv("WALDO_DANCE_CONFIRM", "1")
    d, s, published = make()
    await d.run_tool("listen_for_music", {})
    d.tracker = SteadyTracker(Beat(124.0, 10.0, 0.6, 0.3))
    for _ in range(3):
        await d.tick()
    assert d.state == "confirming" and "go Waldo" in s.said[-1]


async def test_lyrics_saying_yes_do_not_start_the_dance_but_go_waldo_does():
    d, s, published = make()
    d.state = "confirming"
    await d.on_transcript("remote", "yeah yeah yeah, go go go")
    assert d.state == "confirming" and not any(e["type"] == "dance_start" for e in published)
    await d.on_transcript("local", "go Waldo")  # Waldo's own words never count
    assert d.state == "confirming"
    await d.on_transcript("remote", "Go, Waldo!")
    assert d.state == "starting" and published[-1]["type"] == "dance_start"


@pytest.mark.parametrize("heard", ["Go well, though.", "go wall", "Go, Walter!", "go world", "Go Waldo.", "Go.", "okay go"])
async def test_go_waldo_as_transcription_hears_it(heard):
    d, s, published = make()
    d.state = "confirming"
    await d.on_transcript("remote", heard)
    assert published and published[-1]["type"] == "dance_start", heard


@pytest.mark.parametrize("lyric", ["let's go", "go go go", "yeah yeah", "here we go again tonight", "I gotta go"])
async def test_lyrics_with_go_do_not_start(lyric):
    d, s, published = make()
    d.state = "confirming"
    await d.on_transcript("remote", lyric)
    assert d.state == "confirming", lyric


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


async def test_phrase_requests_every_32_beats():
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
    assert len(s.said) == n  # 32 beats at 120 BPM = 16 s, not yet
    clock["t"] += 12.1
    await d.tick()
    assert len(s.said) == n + 1


async def test_asks_to_text_the_song_then_requests_the_routine_and_syncs(tmp_path, monkeypatch):
    import dance_call
    from beats import Beat

    monkeypatch.setattr(dance_call, "CALL_AUDIO", str(tmp_path / "call.raw"))
    clock = {"t": 50.0}
    d, s, published = make(clock=lambda: clock["t"])
    d.state = "dancing"
    d.tracker = SteadyTracker(Beat(120.0, 49.5, 0.6, 0.3))
    d.on_audio(b"\x01\x00" * 24000 * 9)  # 9 s of music
    await d.tick()
    assert d.song_asked and any("text me the song" in (x or "") for x in s.said)
    await d.on_text("Levitating Dua Lipa")
    assert published[-1] == {"type": "choreo_request", "query": "Levitating Dua Lipa"}
    await d.on_bus({"type": "choreo_ready", "query": "Levitating Dua Lipa", "path": "/x/abc.npz", "title": "JD Levitating"})
    align = published[-1]
    assert align["type"] == "choreo_align" and align["path"] == "/x/abc.npz" and align["rate"] == 24000
    assert (tmp_path / "call.raw").stat().st_size == 24000 * 2 * 9
    await d.on_bus({"type": "choreo_sync", "path": "/x/abc.npz", "video_t": 30.0, "wall_t": 50.0})
    assert d.synced and "Just Dance" in s.said[-1]
    said = len(s.said)
    clock["t"] += 31
    before = len(published)
    await d.tick()  # re-sync every 30 s, and no more freestyle phrase requests
    assert any(e["type"] == "choreo_align" for e in published[before:]) and len(s.said) == said


async def test_identifies_the_song_itself_when_it_can():
    from beats import Beat

    async def identify(wav: bytes):
        assert wav[:4] == b"RIFF"
        return "Levitating Dua Lipa"

    published = []

    async def publish(e):
        published.append(e)

    s = FakeSession()
    d = DanceCall(s, publish, clock=lambda: 0.0, identify=identify)
    d.state = "dancing"
    d.tracker = SteadyTracker(Beat(120.0, 0.0, 0.6, 0.3))
    d.on_audio(b"\x01\x00" * 24000 * 9)
    await d.tick()
    import asyncio
    await asyncio.sleep(0.01)
    assert {"type": "choreo_request", "query": "Levitating Dua Lipa"} in published


@pytest.mark.parametrize("lyric", ["I got you, moonlight, you're my starlight, stop and wait for me tonight",
                                   "don't stop me now I'm having such a good time"])
async def test_lyrics_containing_stop_words_do_not_stop(lyric):
    d, s, published = make()
    d.state = "dancing"
    await d.on_transcript("remote", lyric)
    assert d.state == "dancing" and not published


async def test_voice_agent_has_no_stop_tool():
    from dance_call import DANCE_TOOLS

    assert "stop_dance" not in {t["name"] for t in DANCE_TOOLS}


async def test_engine_stopping_ends_the_dance():
    d, s, published = make()
    d.state = "dancing"
    await d.on_bus({"type": "dance_stopped", "reason": "music ended"})
    assert d.state == "done" and "music ended" in s.said[-1]

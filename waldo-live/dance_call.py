"""Dance mode for calls: Waldo asks for a song, finds the beat, and dances however it likes.

Flow on the call:
  greeting   -> Waldo asks the operator to play a song
  listening  -> the voice agent stops answering every sound (music would set it off) and the
                beat tracker listens; once the beat is steady Waldo asks to dance
  confirming -> the dance starts only on the phrase "go Waldo" (said) or "go" (texted): lyrics
                are full of "yeah"s, so a plain yes doesn't count here
  dancing    -> every 16 beats the voice agent, which hears the music, picks the next phrase of
                moves and an energy (the dance tool); the engine on Waldo moves the arms on the beat
  done       -> music stopped, "stop" was heard, or the engine stopped; Waldo talks again

"stop" (or "enough", "wait", "pause") from the operator stops the dance at any point.
"""
from __future__ import annotations

import base64
import re
import time

from beats import BeatTracker

GO = re.compile(r"\bgo,?\s+waldo\b", re.I)
STOP = re.compile(r"\b(stop|enough|wait|pause|that'?s it)\b", re.I)
STEADY_ESTIMATES = 3          # confident tempo readings in a row before asking to dance
MIN_CONFIDENCE = 0.2
MOVE_NAMES = ["pump", "sway", "wave", "raise_roof", "disco", "circles", "hands_up"]

DANCE_INSTRUCTIONS = """You are Waldo, a playful humanoid robot in a robotics lab, on a phone call with your operator.
You called because you want to dance. Speak in short, warm sentences.

- Ask the operator to play a song into the phone, then call listen_for_music.
- While music plays you will be asked to choose dance phrases: call the dance tool with 4 moves
  (from: pump, sway, wave, raise_roof, disco, circles, hands_up) and an energy from 0.1 (chill)
  to 1.0 (hype) that fits what you hear. Vary it: build up, repeat what works, go big on drops.
  Only speak when asked to.
- If the operator says stop, call stop_dance right away.
- Messages starting with [robot] are updates from your body; mention them only when relevant.
- When the operator is done, say goodbye and call hang_up.
"""


def _fn(name: str, description: str, props: dict | None = None) -> dict:
    props = props or {}
    return {"type": "function", "name": name, "description": description,
            "parameters": {"type": "object", "properties": props, "required": list(props)}}


DANCE_TOOLS = [
    _fn("listen_for_music", "Start listening for the operator's music (call after asking them to play a song)."),
    _fn("dance", "Choose the next 16 beats: 4 moves, 4 beats each, and an energy.",
        {"moves": {"type": "array", "items": {"type": "string", "enum": MOVE_NAMES}, "minItems": 1, "maxItems": 4},
         "energy": {"type": "number", "description": "0.1 chill .. 1.0 hype"}}),
    _fn("stop_dance", "Stop dancing now."),
    _fn("hang_up", "End the call after saying goodbye."),
]


def turn_detection(auto: bool) -> dict:
    """Server VAD; with auto=False the agent stops answering and interrupting on its own (music!)."""
    return {"type": "server_vad", "threshold": 0.65, "prefix_padding_ms": 300, "silence_duration_ms": 600,
            "create_response": auto, "interrupt_response": auto}


class DanceCall:
    """Per-call dance state. `bridge` provides publish(event) and the session provides speech."""

    def __init__(self, session, publish, clock=time.time):
        self.s, self.publish, self.clock = session, publish, clock
        self.state = "greeting"
        self.tracker = BeatTracker(24000, clock=clock)
        self.steady = 0
        self.bpm = None
        self.next_phrase_at = 0.0
        self.phrases = 0

    async def _auto_responses(self, on: bool) -> None:
        await self.s.send_openai({"type": "session.update", "session": {
            "type": "realtime", "audio": {"input": {"turn_detection": turn_detection(on)}}}})

    async def _say(self, instructions: str) -> None:
        await self.s.respond(instructions)

    # -- hooks called by the bridge ------------------------------------------

    def on_audio(self, pcm24k: bytes) -> None:
        if self.state in ("listening", "confirming", "dancing"):
            self.tracker.feed(pcm24k)

    async def tick(self) -> None:
        """About once a second while the call is up."""
        if self.state not in ("listening", "confirming", "dancing"):
            return
        beat = self.tracker.estimate()
        if beat is None:
            return
        await self.publish({"type": "dance_tempo", "bpm": beat.bpm, "beat_t": beat.beat_t,
                            "confidence": beat.confidence, "loudness": beat.loudness})
        steady = beat.confidence >= MIN_CONFIDENCE and beat.loudness > 0.02
        self.steady = self.steady + 1 if steady else 0
        if steady:
            self.bpm = beat.bpm
        if self.state == "listening" and self.steady >= STEADY_ESTIMATES:
            self.state = "confirming"
            await self._say(f"Say, excited and short: you can hear the beat, about {round(self.bpm)} BPM, and to start "
                            "you need them to say 'go Waldo' (or text 'go'). Nothing else.")
        elif self.state == "dancing" and self.clock() >= self.next_phrase_at and self.bpm:
            self.phrases += 1
            talk = "You may add at most 4 excited words." if self.phrases % 3 == 0 else "Do not speak."
            await self._say(f"Choose the next 16 beats for the music you hear now: call the dance tool. {talk}")
            self.next_phrase_at = self.clock() + 16 * 60.0 / self.bpm

    async def on_transcript(self, party: str, text: str) -> None:
        if party != "remote":
            return
        if STOP.search(text) and self.state in ("listening", "confirming", "dancing"):
            await self.stop("operator said stop")
        elif self.state == "confirming" and GO.search(text):
            await self.go()

    async def on_text(self, text: str) -> None:
        """An iMessage from the operator during the call."""
        t = text.strip().lower()
        if self.state == "confirming" and t in ("go", "go waldo", "go!", "dance"):
            await self.go()
        elif t in ("stop", "stop!") and self.state in ("listening", "confirming", "dancing"):
            await self.stop("operator texted stop")

    async def on_bus(self, event: dict) -> None:
        t = event.get("type")
        if t == "dance_started":
            self.state = "dancing"
            self.next_phrase_at = 0.0
        elif t in ("dance_stopped", "dance_error") and self.state in ("starting", "dancing"):
            await self._done(event.get("reason") or event.get("error") or "stopped")

    async def run_tool(self, name: str, args: dict) -> dict:
        if name == "listen_for_music":
            self.state, self.steady = "listening", 0
            self.tracker = BeatTracker(24000, clock=self.clock)
            await self._auto_responses(False)
            return {"ok": True, "note": "listening; I'll tell you when I hear a steady beat"}
        if name == "dance":
            moves = [m for m in args.get("moves", []) if m in MOVE_NAMES][:4]
            if self.state != "dancing" or not moves:
                return {"ok": False, "note": "not dancing yet" if moves else "no valid moves"}
            await self.publish({"type": "dance_phrase", "moves": moves, "energy": args.get("energy", 0.6)})
            return {"ok": True}
        if name == "stop_dance":
            await self.stop("voice agent stopped it")
            return {"ok": True}
        return {"ok": False, "error": f"unknown dance tool {name}"}

    # -- transitions -----------------------------------------------------------

    async def go(self) -> None:
        self.state = "starting"
        await self.publish({"type": "dance_start", "by": "operator", "bpm": self.bpm})
        await self._say("Say only: 'Let's go!'")

    async def stop(self, reason: str) -> None:
        await self.publish({"type": "dance_stop", "reason": reason})
        await self._done(reason)

    async def _done(self, reason: str) -> None:
        if self.state == "done":
            return
        self.state = "done"
        await self._auto_responses(True)
        await self._say(f"The dance is over ({reason}). React in one short sentence and ask if they want another "
                        "song (then call listen_for_music) or to hang up.")


def pcm_from_b64(payload: str) -> bytes:
    return base64.b64decode(payload)

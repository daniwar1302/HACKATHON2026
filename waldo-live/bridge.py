"""Call bridge: Inkbox phone call <-> OpenAI Realtime voice agent, reacting to the bus.

    python bridge.py serve                          tunnel + call server; places calls on `call_requested`
    python bridge.py call "reason" "question"       publish a call_requested (needs `serve` running)
    python bridge.py setup-signing-key              create Waldo's Inkbox signing key into ~/.waldo-ask/env

Flow: something publishes `call_requested` -> the bridge places an Inkbox call whose
audio WebSocket points at this server through the Inkbox tunnel -> audio is relayed to
OpenAI Realtime -> while the call is up, watcher/brain events are injected into the
conversation, and what the operator says flows back to the brain.

Patterns ported from inkbox-ai/hermes-agent-plugin realtime.py (audio framing, raw-media
headers, barge-in clear, stop frame, filler speech while a consult runs).

Only the call WebSocket is reachable from the internet, and it checks both Inkbox's
signature and a one-time token. Placing calls is triggered only from the local bus.
"""
from __future__ import annotations

import argparse
import asyncio
import audioop  # removed in Python 3.13; this venv is 3.12
import base64
import collections
import json
import os
import re
import secrets
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import aiohttp
from aiohttp import web

from commands import Commands, find_confirmation
from dance_call import DANCE_INSTRUCTIONS, DANCE_TOOLS, DanceCall, turn_detection
from phonecam import LOCAL_PORT as PHONE_LOCAL_PORT, PhoneCam, token as phone_token

CONFIRM_WAIT_S = 4.0
from common import BRAIN_URL, BUS_URL, CAMERA_URL, ENV_FILE, apublish, load_env, publish, run_stop_command, subscribe

CALL_ME = re.compile(r"\bcall me\b", re.I)

PORT = 8771
WS_PATH = "/ws/call"
REALTIME_URL = "wss://api.openai.com/v1/realtime"
SPEAK_GAP_S = 3.0  # at most one spoken camera alert per this many seconds
FORWARDED = {"check_failed", "check_cleared", "stop_requested", "robot_stopped", "brain_progress", "call_requested",
             "dance_started", "dance_stopped", "dance_error", "choreo_ready", "choreo_failed", "choreo_sync"}
DANCE_EVENTS = {"dance_started", "dance_stopped", "dance_error", "choreo_ready", "choreo_failed", "choreo_sync"}


def audd_identify(http: aiohttp.ClientSession, token: str):
    """Song recognition with AudD (audd.io): wav bytes -> "title artist", or None."""
    async def identify(wav: bytes) -> str | None:
        form = aiohttp.FormData()
        form.add_field("api_token", token)
        form.add_field("file", wav, filename="clip.wav", content_type="audio/wav")
        async with http.post("https://api.audd.io/", data=form, timeout=20) as resp:
            data = await resp.json(content_type=None)
        result = data.get("result") if data.get("status") == "success" else None
        return f"{result['title']} {result['artist']}" if result else None
    return identify
DANCE_ME = re.compile(r"\bdance\b", re.I)

INSTRUCTIONS = """You are Waldo, a robot in a robotics lab, on a phone call with your operator.
You called because you need help with a task. Speak in short, natural sentences.

- Messages starting with [camera] or [robot] are live updates from your cameras and controller.
  Mention them when they matter; a [camera] alert marked URGENT must be said right away.
- When the operator tells you what to do, call give_instruction with their words, then call
  consult_robot to get the plan, read the plan back in one sentence and ask "go ahead?".
  Only after they confirm, call give_instruction with "confirmed".
- If the operator says stop, wait, or anything like it, call stop_robot immediately.
- Use robot_status when asked what's going on. Never claim the robot did something unless a
  tool result or [robot] update says so.
- When the operator is done, say a short goodbye and call hang_up.
"""


COMMAND_INSTRUCTIONS = """You are Waldo, a robot in a robotics lab, on a phone call with your operator, who gives you commands.
Speak in short, natural sentences.

- You can only do what run_command offers. If asked for something else, say so and list a few options.
- For a command that moves the robot: name it and ask "go ahead?" (for example "I'll wave. Go ahead?").
  Only after the operator says yes, call run_command. If it answers needs_confirmation, ask again the same way.
- Commands that don't move the robot (like status) run right away.
- After a command, say the result in one sentence.
- If the operator says stop, wait, or anything like it, call stop_robot immediately, no confirmation.
- Messages starting with [camera] or [robot] are live updates; mention them when they matter.
- When the operator is done, say a short goodbye and call hang_up.
"""


def _fn(name: str, description: str, props: dict | None = None) -> dict:
    props = props or {}
    return {"type": "function", "name": name, "description": description,
            "parameters": {"type": "object", "properties": props, "required": list(props)}}


TOOLS = [
    _fn("robot_status", "Current task, step and recent camera/robot events."),
    _fn("consult_robot", "Ask the robot's planner (takes a few seconds). Use for the plan after an instruction, or questions about what the robot can do.",
        {"query": {"type": "string", "description": "The question, with enough context to stand alone."}}),
    _fn("give_instruction", "Pass the operator's instruction to the robot's planner, in their words.",
        {"text": {"type": "string"}}),
    _fn("stop_robot", "Stop the robot immediately. Use whenever the operator says stop or wait."),
    _fn("hang_up", "End the call after saying goodbye."),
]


def command_tools(menu: list[dict]) -> list[dict]:
    lines = "; ".join(f"{m['command']} = {m['does']}{' (moves the robot)' if m['moves_robot'] else ''}" for m in menu)
    run = _fn("run_command", f"Run one robot command. Available: {lines or 'none configured'}.",
              {"command": {"type": "string", "enum": [m["command"] for m in menu] or ["none"]}})
    keep = [t for t in TOOLS if t["name"] in ("stop_robot", "hang_up", "robot_status")]
    return [run, *keep]


class AudioConverter:
    """One continuous mono PCM16 stream between two sample rates, optionally mu-law on the call side."""

    def __init__(self, src_rate: int, dst_rate: int, decode_ulaw: bool = False, encode_ulaw: bool = False):
        self.src_rate, self.dst_rate = src_rate, dst_rate
        self.decode_ulaw, self.encode_ulaw = decode_ulaw, encode_ulaw
        self.reset()

    def reset(self) -> None:
        self._state, self._pending = None, b""

    def convert(self, payload_b64: str) -> str:
        data = base64.b64decode(payload_b64)
        if self.decode_ulaw:
            data = audioop.ulaw2lin(data, 2)
        else:
            data = self._pending + data
            whole = len(data) - len(data) % 2
            self._pending, data = data[whole:], data[:whole]
        if not data:
            return ""
        out, self._state = audioop.ratecv(data, 2, 1, self.src_rate, self.dst_rate, self._state)
        if self.encode_ulaw:
            out = audioop.lin2ulaw(out, 2)
        return base64.b64encode(out).decode()


def call_audio(media_format: dict | None) -> tuple[AudioConverter, AudioConverter]:
    """(inbound call->24k, outbound 24k->call) for the format Inkbox announced in `start`."""
    fmt = media_format or {}
    if (fmt.get("encoding"), fmt.get("sample_rate")) == ("L16", 16000):
        return AudioConverter(16000, 24000), AudioConverter(24000, 16000)
    return AudioConverter(8000, 24000, decode_ulaw=True), AudioConverter(24000, 8000, encode_ulaw=True)


def system_item(text: str) -> dict:
    return {"type": "conversation.item.create",
            "item": {"type": "message", "role": "system", "content": [{"type": "input_text", "text": text}]}}


def describe(event: dict) -> tuple[str, bool] | None:
    """Turn a bus event into (text for the voice agent, urgent?), or None to skip."""
    t = event.get("type")
    if t == "check_failed":
        urgent = event.get("severity") == "critical"
        return (f"[camera]{' URGENT' if urgent else ''} {event.get('question')} -> {event.get('answer')} "
                f"(confidence {event.get('p_bad')}, step {event.get('step')})"), urgent
    if t == "check_cleared":
        return f"[camera] {event.get('check')} looks OK again", False
    if t in ("stop_requested", "robot_stopped"):
        return f"[robot] stopped ({event.get('reason') or event.get('by') or 'stop'})", True
    if t == "brain_progress":
        return f"[robot] {event.get('text')}", False
    if t == "call_requested":
        return f"[robot] new problem: {event.get('reason')} Question: {event.get('question')}", True
    return None


@dataclass
class CallSession:
    inkbox: web.WebSocketResponse
    openai: aiohttp.ClientWebSocketResponse
    http: aiohttp.ClientSession
    brief: dict
    call_id: str = ""
    mode: str = "help"  # help: Waldo needs help with a task; command: operator gives commands
    transcripts: list = field(default_factory=list)  # (monotonic t, party, text)
    used_yes: set = field(default_factory=set)  # transcript times of yeses already spent on a motion
    end_reason: str = ""
    stats: collections.Counter = field(default_factory=collections.Counter)
    dance: object = None  # DanceCall in dance mode
    inbound: AudioConverter | None = None
    outbound: AudioConverter | None = None
    stream_id: str | None = None
    response_active: bool = False
    want_response: bool = False
    greeted: bool = False
    hangup_armed: bool = False
    closed: bool = False
    last_spoken: float = 0.0
    firing: set = field(default_factory=set)
    done_calls: set = field(default_factory=set)

    async def send_openai(self, payload: dict) -> None:
        if not self.closed:
            await self.openai.send_str(json.dumps(payload))

    async def send_inkbox(self, payload: dict) -> None:
        if self.stream_id:
            payload = {**payload, "stream_id": self.stream_id}
        if not self.closed:
            await self.inkbox.send_str(json.dumps(payload))

    async def respond(self, instructions: str | None = None) -> None:
        """Ask the model to speak, or queue it if it's already talking."""
        if self.response_active:
            self.want_response = True
            return
        payload: dict = {"type": "response.create"}
        if instructions:
            payload["response"] = {"instructions": instructions}
        await self.send_openai(payload)

    async def interrupt(self) -> None:
        if self.response_active:
            await self.send_openai({"type": "response.cancel"})
        await self.send_inkbox({"event": "clear"})  # drop audio already queued on the call
        self.outbound and self.outbound.reset()

    async def inject(self, event: dict) -> None:
        described = describe(event)
        if not described:
            return
        text, urgent = described
        check = event.get("check")
        if event.get("type") == "check_cleared":
            self.firing.discard(check)
        elif event.get("type") == "check_failed":
            if check in self.firing:
                return  # still the same problem; don't nag
            self.firing.add(check)
        await self.send_openai(system_item(text))
        now = time.monotonic()
        if urgent:
            was_talking = self.response_active
            await self.interrupt()
            self.last_spoken = now
            if was_talking:
                self.want_response = True  # speak as soon as the cancelled response is done
            else:
                await self.respond()
        elif event.get("type") == "check_failed" and now - self.last_spoken >= SPEAK_GAP_S:
            self.last_spoken = now
            await self.respond()

    async def hang_up(self, delay: float = 1.5) -> None:
        await asyncio.sleep(delay)
        if not self.closed:
            await self.send_inkbox({"event": "stop", "reason": "goodbye complete"})
            self.closed = True
            await self.inkbox.close()
            await self.openai.close()


class Bridge:
    def __init__(self, identity, public_host: str, signing_key: str | None):
        self.identity = identity
        self.public_host = public_host
        self.signing_key = signing_key
        self.operator = os.environ["WALDO_OPERATOR_PHONE"]
        self.openai_key = os.environ["OPENAI_API_KEY"]
        self.model = os.environ.get("WALDO_REALTIME_MODEL", "gpt-realtime-2.1")
        self.voice = os.environ.get("WALDO_REALTIME_VOICE", "cedar")
        self.briefs: dict[str, dict] = {}
        self.active: CallSession | None = None
        self.http: aiohttp.ClientSession | None = None
        self.commands = Commands.load(builtins={"snapshot": self.snapshot})

    # -- placing calls ----------------------------------------------------

    async def place_call(self, brief: dict) -> None:
        from inkbox import CallOrigin
        from inkbox.phone.types import CallMode

        token = secrets.token_urlsafe(16)
        self.briefs[token] = brief
        url = f"wss://{self.public_host}{WS_PATH}?s={token}"
        call = await asyncio.to_thread(
            self.identity.place_call,
            to_number=self.operator,
            origination=CallOrigin.SHARED_IMESSAGE_NUMBER,
            mode=CallMode.CLIENT_WEBSOCKET,
            client_websocket_url=url,
            on_voicemail="hang_up",
        )
        await apublish(self.http, {"type": "call_placed", "call_id": str(call.id), "reason": brief.get("reason")})
        print(f"call placed {call.id}")

    async def bus_loop(self) -> None:
        async for event in subscribe(self.http, FORWARDED):
            if event["type"] == "call_requested" and self.active is None:
                try:
                    await self.place_call(event)
                except Exception as e:
                    await apublish(self.http, {"type": "call_failed", "error": str(e)})
                    print(f"call failed: {e}")
            elif self.active is not None and event["type"] in DANCE_EVENTS:
                if self.active.dance:
                    await self.active.dance.on_bus(event)
            elif self.active is not None:
                await self.active.inject(event)

    # -- the call WebSocket -------------------------------------------------

    def _authorized(self, request: web.Request) -> dict | None:
        brief = self.briefs.pop(request.query.get("s", ""), None)
        if brief is None:
            return None
        if self.signing_key:
            from inkbox import verify_webhook

            payload = (request.headers.get("X-Call-Context") or "").encode()
            if not verify_webhook(payload=payload, headers=dict(request.headers), secret=self.signing_key):
                return None
        elif os.environ.get("WALDO_ALLOW_UNSIGNED") != "1":
            return None
        return brief

    async def handle_call(self, request: web.Request) -> web.StreamResponse:
        brief = self._authorized(request)
        if brief is None:
            return web.Response(status=401, text="unauthorized")
        try:
            openai = await self.http.ws_connect(
                f"{REALTIME_URL}?model={self.model}", headers={"Authorization": f"Bearer {self.openai_key}"}, heartbeat=20,
            )
        except Exception as e:
            await apublish(self.http, {"type": "call_failed", "error": f"OpenAI Realtime: {e}"})
            return web.Response(status=503, text="voice agent unavailable")

        inkbox = web.WebSocketResponse()
        inkbox.headers["x-inkbox-audio-format"] = "pcm_s16le_16000"
        inkbox.headers["x-use-inkbox-text-to-speech"] = "false"
        inkbox.headers["x-use-inkbox-speech-to-text"] = "false"
        await inkbox.prepare(request)

        try:
            ctx = json.loads(request.headers.get("X-Call-Context") or "{}")
        except ValueError:
            ctx = {}
        mode = brief.get("mode") if brief.get("mode") in ("command", "dance") else "help"
        s = CallSession(inkbox, openai, self.http, brief, call_id=str(ctx.get("call_id") or ctx.get("id") or ""), mode=mode)
        self.active = s
        await apublish(self.http, {"type": "call_started", "call_id": s.call_id, "mode": mode})
        if mode == "command":
            instructions, tools = COMMAND_INSTRUCTIONS, command_tools(self.commands.menu())
        elif mode == "dance":
            instructions, tools = DANCE_INSTRUCTIONS, DANCE_TOOLS
            token = os.environ.get("AUDD_API_TOKEN")
            s.dance = DanceCall(s, lambda event: apublish(self.http, event),
                                identify=audd_identify(self.http, token) if token else None)
        else:
            instructions, tools = INSTRUCTIONS, TOOLS
        await s.send_openai({
            "type": "session.update",
            "session": {
                "type": "realtime", "model": self.model, "instructions": instructions,
                "output_modalities": ["audio"],
                "audio": {
                    "input": {"format": {"type": "audio/pcm", "rate": 24000},
                              "transcription": {"model": "gpt-4o-mini-transcribe"},
                              # A bit less sensitive than the default so line echo of Waldo's own voice
                              # doesn't count as the operator interrupting.
                              "turn_detection": turn_detection(auto=True)},
                    "output": {"format": {"type": "audio/pcm", "rate": 24000}, "voice": self.voice},
                },
                "tools": tools, "tool_choice": "auto",
            },
        })
        ticker = asyncio.create_task(self._dance_ticks(s)) if s.dance else None
        try:
            await asyncio.gather(self._from_inkbox(s), self._from_openai(s))
        except Exception as e:
            s.end_reason = f"bridge error: {type(e).__name__}: {e}"
            import traceback
            traceback.print_exc()
        finally:
            if ticker:
                ticker.cancel()
            s.closed = True
            self.active = None
            await openai.close()
            print(f"call ended: {s.end_reason} | stats {dict(s.stats)}")
            await apublish(self.http, {"type": "call_ended", "call_id": s.call_id, "reason": s.end_reason, "stats": dict(s.stats)})
        return inkbox

    async def _dance_ticks(self, s: CallSession) -> None:
        while not s.closed:
            await asyncio.sleep(1.0)
            try:
                await s.dance.tick()
            except Exception as e:
                print(f"dance tick: {e}")

    async def _greet(self, s: CallSession) -> None:
        if s.greeted:
            return
        s.greeted = True
        b = s.brief
        if s.mode == "dance":
            await s.send_openai(system_item("Call connected. Greet the operator as Waldo, say you'd love to dance, "
                                            "and ask them to play a song into the phone."))
            await s.respond()
            return
        if s.mode == "command":
            await s.send_openai(system_item("Call connected. Greet the operator as Waldo and ask what you should do."))
            await s.respond()
            return
        await s.send_openai(system_item(
            f"Call connected. Task: {b.get('task') or 'unknown'}. What happened: {b.get('reason')}. "
            f"Greet the operator as Waldo, explain in one or two sentences, then ask: {b.get('question')}"
        ))
        await s.respond()

    async def _from_inkbox(self, s: CallSession) -> None:
        why = "inkbox socket ended"
        async for msg in s.inkbox:
            if msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                why = f"inkbox socket {msg.type.name}"
                break
            if msg.type != aiohttp.WSMsgType.TEXT:
                continue  # ignore anything that isn't a JSON frame
            frame = json.loads(msg.data)
            event = (frame.get("event") or "").lower()
            if event == "start":
                start = frame.get("start") or {}
                s.stream_id = frame.get("stream_id") or start.get("stream_id")
                fmt = start.get("media_format") or frame.get("media_format")
                print(f"call start: media_format={fmt} stream_id={s.stream_id}")
                s.inbound, s.outbound = call_audio(fmt)
                await self._greet(s)
            elif event == "media":
                if s.inbound is None:
                    print("call media before start: assuming 8 kHz mu-law")
                    s.inbound, s.outbound = call_audio(None)
                await self._greet(s)
                payload = (frame.get("media") or {}).get("payload")
                if payload and (converted := s.inbound.convert(payload)):
                    s.stats["in_frames"] += 1
                    await s.send_openai({"type": "input_audio_buffer.append", "audio": converted})
                    if s.dance:
                        s.dance.on_audio(base64.b64decode(converted))
            elif event in ("stop", "closed", "hangup"):
                why = f"inkbox sent {event} ({frame.get('reason') or 'no reason'})"
                break
        s.end_reason = s.end_reason or why
        s.closed = True
        await s.openai.close()

    async def _from_openai(self, s: CallSession) -> None:
        why = "openai socket ended"
        async for msg in s.openai:
            if msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                why = f"openai socket {msg.type.name} {s.openai.close_code}"
                break
            if msg.type != aiohttp.WSMsgType.TEXT:
                continue
            frame = json.loads(msg.data)
            t = frame.get("type", "")
            if t in ("response.output_audio.delta", "response.audio.delta"):
                if s.outbound and (converted := s.outbound.convert(frame.get("delta") or "")):
                    s.stats["out_frames"] += 1
                    await s.send_inkbox({"event": "media", "media": {"payload": converted, "track": "outbound"}})
            elif t in ("response.output_audio.done", "response.audio.done"):
                s.outbound and s.outbound.reset()
                await s.send_inkbox({"event": "audio_done"})
                if s.hangup_armed:
                    asyncio.create_task(s.hang_up())
            elif t == "input_audio_buffer.speech_started":
                # Server VAD already cancels the response; we only drop audio still queued on the call.
                s.hangup_armed = False  # operator barged in during the goodbye
                s.stats["barge_ins"] += 1
                await s.send_inkbox({"event": "clear"})
                s.outbound and s.outbound.reset()
            elif t == "response.created":
                s.response_active = True
            elif t == "response.done":
                s.response_active = False
                if s.want_response:
                    s.want_response = False
                    await s.respond()
            elif t == "conversation.item.input_audio_transcription.completed":
                await self._transcript(s, "remote", frame.get("transcript", ""))
            elif t in ("response.output_audio_transcript.done", "response.audio_transcript.done"):
                await self._transcript(s, "local", frame.get("transcript", ""))
            elif t == "response.output_item.done" and (frame.get("item") or {}).get("type") == "function_call":
                item = frame["item"]
                if item["call_id"] not in s.done_calls:
                    s.done_calls.add(item["call_id"])
                    asyncio.create_task(self._tool(s, item["name"], item["call_id"], item.get("arguments") or "{}"))
            elif t == "error":
                print(f"realtime error: {frame.get('error')}")
        s.end_reason = s.end_reason or why
        s.closed = True
        await s.inkbox.close()

    async def _transcript(self, s: CallSession, party: str, text: str) -> None:
        text = text.strip()
        if not text:
            return
        s.transcripts.append((time.monotonic(), party, text))
        await s.send_inkbox({"event": "transcript", "party": party, "text": text, "is_final": True})
        await apublish(self.http, {"type": "call_transcript", "party": "operator" if party == "remote" else "waldo", "text": text})
        if s.dance:
            await s.dance.on_transcript(party, text)

    async def _tool(self, s: CallSession, name: str, call_id: str, raw_args: str) -> None:
        try:
            args = json.loads(raw_args)
        except ValueError:
            args = {}
        if name == "consult_robot":
            await s.respond("Say one very short filler like 'one sec, let me check' and nothing else.")
        elif name == "run_command" and getattr(self.commands.commands.get(args.get("command", "")), "slow", False):
            await s.respond("Say in one short sentence that you're starting it and it takes a few seconds "
                            "because it runs a safety check first. Nothing else.")
        try:
            output = await self._run_tool(s, name, args)
        except Exception as e:
            output = {"ok": False, "error": str(e)}
        await s.send_openai({"type": "conversation.item.create",
                             "item": {"type": "function_call_output", "call_id": call_id, "output": json.dumps(output)}})
        if name not in ("hang_up", "listen_for_music", "dance", "stop_dance"):  # dance tools: stay quiet over the music
            await s.respond()

    async def _run_tool(self, s: CallSession, name: str, args: dict) -> dict:
        if s.dance and name in ("listen_for_music", "dance", "stop_dance"):
            return await s.dance.run_tool(name, args)
        if name == "robot_status":
            try:
                async with self.http.get(f"{BRAIN_URL}/status", timeout=3) as resp:
                    return await resp.json()
            except Exception:
                async with self.http.get(f"{BUS_URL}/recent?n=10", timeout=3) as resp:
                    return {"recent_events": await resp.json()}
        if name == "consult_robot":
            async with self.http.post(f"{BRAIN_URL}/consult", json={"query": args.get("query", "")}, timeout=45) as resp:
                return await resp.json()
        if name == "give_instruction":
            await apublish(self.http, {"type": "operator_instruction", "text": args.get("text", ""), "call_id": s.call_id})
            return {"ok": True, "note": "passed to the planner"}
        if name == "run_command":
            return await self.run_command(s, args.get("command", ""))
        if name == "stop_robot":
            await apublish(self.http, {"type": "stop_requested", "by": "operator"})
            s.used_yes.update(t for t, party, _ in s.transcripts if party == "remote")  # no earlier yes survives a stop
            results = []
            if self.commands.stop_tool:
                try:
                    results.append((await self.commands.stop())["ok"])
                except Exception as e:
                    print(f"stop tool failed: {e}")
                    results.append(False)
            if os.environ.get("WALDO_STOP_CMD"):
                results.append(await run_stop_command() is None)
            if not results:
                return {"ok": False, "note": "no stop is configured; use the hand controller"}
            return {"ok": all(results), "note": "robot stopped" if all(results) else "stop may have failed; use the hand controller"}
        if name == "hang_up":
            s.hangup_armed = True
            asyncio.create_task(s.hang_up(delay=10))  # fallback if no goodbye audio follows
            await s.respond("Say a short goodbye.")
            return {"ok": True}
        return {"ok": False, "error": f"unknown tool {name}"}

    async def run_command(self, s: CallSession, name: str) -> dict:
        cmd = self.commands.commands.get(name)
        if cmd is None:
            return {"ok": False, "error": f"unknown command {name!r}", "available": [m["command"] for m in self.commands.menu()]}
        if cmd.motion:
            # The operator's words are transcribed a moment after the voice agent acts on them,
            # so give the transcript a few seconds to catch up before refusing.
            deadline = time.monotonic() + CONFIRM_WAIT_S
            while (yes_at := find_confirmation(s.transcripts, name, time.monotonic(), s.used_yes)) is None:
                if time.monotonic() >= deadline:
                    return {"ok": False, "needs_confirmation": True,
                            "note": f"Ask the operator whether to {name} (say '{name}' or 'go ahead?'), wait for a clear yes, "
                                    "then call run_command again."}
                await asyncio.sleep(0.25)
            s.used_yes.add(yes_at)
        await apublish(self.http, {"type": "command_started", "command": name, "call_id": s.call_id})
        try:
            result = await self.commands.run(name)
        except Exception as e:
            result = {"ok": False, "error": str(e)}
        await apublish(self.http, {"type": "command_done", "command": name, "ok": result.get("ok"), "call_id": s.call_id})
        return result

    async def snapshot(self) -> dict:
        try:
            async with self.http.get(f"{CAMERA_URL}/frame/waldo_webcam.jpg", timeout=3) as resp:
                jpeg = await resp.read()
            upload = await asyncio.to_thread(self.identity.upload_imessage_media, content=jpeg, filename="waldo.jpg", content_type="image/jpeg")
            await asyncio.to_thread(self.identity.send_imessage, to=self.operator, text="Room camera, just now", media_urls=[upload.media_url])
            return {"ok": True, "note": "photo sent by iMessage"}
        except Exception as e:
            return {"ok": False, "error": f"couldn't send the photo: {e}"}

    async def imessage_loop(self, poll_s: float = 3.0) -> None:
        """Text Waldo "call me" over iMessage and it calls you back in command mode."""
        started, seen = datetime.now(timezone.utc), set()
        operator = re.sub(r"\D", "", self.operator)
        while True:
            try:
                msgs = await asyncio.to_thread(self.identity.list_imessages, limit=10)
                for m in msgs:
                    if (m.id in seen or m.direction != "inbound" or m.created_at <= started
                            or re.sub(r"\D", "", m.remote_number or "") != operator):
                        continue
                    seen.add(m.id)
                    text = m.content or ""
                    if self.active is not None and self.active.dance:
                        await self.active.dance.on_text(text)
                    elif self.active is None and DANCE_ME.search(text):
                        await apublish(self.http, {"type": "call_requested", "mode": "dance", "reason": "operator wants a dance"})
                    elif self.active is None and CALL_ME.search(text):
                        await apublish(self.http, {"type": "call_requested", "mode": "command", "reason": "operator asked by text"})
            except Exception as e:
                print(f"imessage poll: {e}")
            await asyncio.sleep(poll_s)


def open_tunnel(client, handle: str):
    from inkbox.tunnels.client import connect

    state_dir = Path.home() / ".waldo" / "tunnel"
    state_dir.mkdir(parents=True, exist_ok=True)
    listener = connect(client, name=handle, forward_to=f"http://127.0.0.1:{PORT}", state_dir=str(state_dir))
    threading.Thread(target=listener.wait, name="inkbox-tunnel", daemon=True).start()
    return listener


async def serve() -> None:
    from inkbox import Inkbox

    handle = os.environ.get("WALDO_ASK_IDENTITY", "waldo-agent")
    client = Inkbox()
    identity = client.get_identity(handle)
    listener = await asyncio.to_thread(open_tunnel, client, handle)
    signing_key = os.environ.get("WALDO_INKBOX_SIGNING_KEY")
    if not signing_key:
        print("WARNING: no WALDO_INKBOX_SIGNING_KEY; run `bridge.py setup-signing-key` (calls are refused until then "
              "unless WALDO_ALLOW_UNSIGNED=1)")
    bridge = Bridge(identity, listener.tunnel.public_host, signing_key)
    phone = PhoneCam(phone_token())
    async with aiohttp.ClientSession() as http:
        bridge.http = http
        app = web.Application(client_max_size=3 * 1024 * 1024)
        app.router.add_get(WS_PATH, bridge.handle_call)
        phone.add_public_routes(app)
        runner = web.AppRunner(app)
        await runner.setup()
        await web.TCPSite(runner, "127.0.0.1", PORT).start()
        local = web.AppRunner(phone.local_app())  # latest phone frame, for this machine only
        await local.setup()
        await web.TCPSite(local, "127.0.0.1", PHONE_LOCAL_PORT).start()
        print(f"bridge ready: wss://{listener.tunnel.public_host}{WS_PATH} -> 127.0.0.1:{PORT}")
        print(f"phone camera: open `./run.sh bridge phone-link` on the phone; frames at 127.0.0.1:{PHONE_LOCAL_PORT}/frame/phone.jpg")
        print(f"command menu: {[m['command'] for m in bridge.commands.menu()]}; text 'call me' to Waldo's iMessage to get a call")
        try:
            await asyncio.gather(bridge.bus_loop(), bridge.imessage_loop())
        finally:
            await bridge.commands.close()


def setup_signing_key() -> None:
    from inkbox import Inkbox

    handle = os.environ.get("WALDO_ASK_IDENTITY", "waldo-agent")
    with Inkbox() as client:
        key = client.get_identity(handle).create_signing_key()
    secret = getattr(key, "signing_key", None) or getattr(key, "secret", None) or getattr(key, "key", None)
    if not secret:
        sys.exit(f"unexpected signing key response fields: {sorted(vars(key))}")
    lines = [l for l in ENV_FILE.read_text().splitlines() if not l.startswith("WALDO_INKBOX_SIGNING_KEY=")]
    ENV_FILE.write_text("\n".join([*lines, f"WALDO_INKBOX_SIGNING_KEY={secret}"]) + "\n")
    print("signing key saved to", ENV_FILE)


if __name__ == "__main__":
    load_env()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("serve")
    sub.add_parser("setup-signing-key")
    sub.add_parser("phone-link", help="print the link that turns a phone into Waldo's 'phone' camera")
    c = sub.add_parser("call", help='help mode: call "reason" "question"; command mode: call --mode command')
    c.add_argument("reason", nargs="?", default="operator asked for a call")
    c.add_argument("question", nargs="?", default="")
    c.add_argument("--task", default="")
    c.add_argument("--mode", choices=["help", "command", "dance"], default="help")
    args = parser.parse_args()
    if args.cmd == "serve":
        asyncio.run(serve())
    elif args.cmd == "setup-signing-key":
        setup_signing_key()
    elif args.cmd == "phone-link":
        from inkbox import Inkbox

        with Inkbox() as client:
            host = client.get_identity(os.environ.get("WALDO_ASK_IDENTITY", "waldo-agent")).tunnel.public_host
        print(f"https://{host}/phone-cam?t={phone_token()}")
    else:
        ok = publish({"type": "call_requested", "reason": args.reason, "question": args.question, "task": args.task, "mode": args.mode})
        print("call requested" if ok else "bus not running (python bus.py)")

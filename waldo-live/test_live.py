import asyncio
import json
import time
from types import SimpleNamespace

import pytest
from aiohttp.test_utils import TestClient, TestServer

import bridge
import brain as brain_mod
from bus import Bus, make_app
from deciders import Check, JevDecider
from watch import Debounce, Task, Watcher

GRIP = Check("grip", "How is the cup held?", {"secure": "", "slipping": "", "missed": ""}, ("slipping", "missed"))
PERSON = Check("person", "Person near?", {"yes": "", "no": ""}, ("yes",), "critical")


# -- watcher -----------------------------------------------------------------

def test_debounce_needs_consecutive_bad_frames_and_hysteresis():
    d = Debounce()
    assert [d.update(p) for p in (0.9, 0.1, 0.9, 0.9)] == [None, None, None, None]  # noise resets the streak
    assert d.update(0.9) == "fire"
    assert d.update(0.6) is None  # between clear_at and fire_at: stays firing
    assert d.update(0.0) == "clear"


def test_watcher_fires_once_and_ignores_stale_frames():
    w = Watcher(Task("lift", "cam", {"grasp": [GRIP]}), decider=None)
    bad = {"grip": {"secure": 0.1, "slipping": 0.85, "missed": 0.05}}
    events = [e for t in (1, 2, 3, 4) for e in w.apply(float(t), bad)]
    assert [e["type"] for e in events] == ["check_failed"]
    assert events[0]["answer"] == "slipping" and events[0]["p_bad"] == 0.9
    assert w.apply(2.5, {"grip": {"secure": 1.0}}) == []  # older than the last applied frame


def test_step_change_resets_state():
    w = Watcher(Task("lift", "cam", {"grasp": [GRIP], "lift": [PERSON]}), decider=None)
    assert w.set_step("lift") and not w.set_step("lift") and not w.set_step("nope")
    assert w.step == "lift" and w.states == {}


def test_jev_request_and_parse():
    req = JevDecider(api_key="k").request(b"\xff\xd8jpg", [GRIP, PERSON])
    ref = req["state"]["camera"]
    assert req["assets"][ref]["format"] == "jpg"
    assert req["questions"]["grip"]["type"] == "choice" and req["questions"]["person"]["type"] == "noul"
    answers = {"grip": {"type": "choice", "choice": "secure", "probabilities": {"secure": 0.9, "slipping": 0.1}},
               "person": {"type": "noul", "noul": 0.2}}
    parsed = JevDecider.parse(answers, [GRIP, PERSON])
    assert parsed["grip"]["secure"] == 0.9 and parsed["person"] == {"yes": 0.2, "no": 0.8}


def test_lift_cup_task_loads():
    task = Task.load("tasks/lift_cup.json")
    assert list(task.steps) == ["approach", "grasp", "lift"]
    assert any(c.severity == "critical" for c in task.steps["grasp"])


# -- bus -------------------------------------------------------------------

async def test_bus_publish_stream_and_recent(tmp_path):
    bus = Bus(log=tmp_path / "events.jsonl")
    async with TestClient(TestServer(make_app(bus))) as client:
        resp = await client.get("/stream?types=check_failed")
        await asyncio.sleep(0.05)
        await client.post("/events", json={"type": "brain_progress", "text": "hi"})
        await client.post("/events", json={"type": "check_failed", "check": "grip"})
        line = b""
        while not line.startswith(b"data:"):
            line = await resp.content.readline()
        assert json.loads(line[5:])["check"] == "grip"  # filtered: brain_progress skipped
        recent = await (await client.get("/recent?n=5")).json()
        assert [e["type"] for e in recent] == ["brain_progress", "check_failed"]
        assert len((tmp_path / "events.jsonl").read_text().splitlines()) == 2
        resp.close()


# -- bridge ----------------------------------------------------------------

def test_audio_converter_resamples_16k_to_24k():
    inbound, outbound = bridge.call_audio({"encoding": "L16", "sample_rate": 16000, "channels": 1})
    import base64
    pcm16k = base64.b64encode(b"\x00\x01" * 1600).decode()  # 100 ms
    out = base64.b64decode(inbound.convert(pcm16k))
    assert abs(len(out) - 4800) <= 4  # 100 ms at 24 kHz, 2 bytes/sample
    assert outbound.convert(base64.b64encode(out).decode())


class FakeWS:
    def __init__(self):
        self.sent = []

    async def send_str(self, s):
        self.sent.append(json.loads(s))

    async def close(self):
        pass


def session():
    return bridge.CallSession(FakeWS(), FakeWS(), None, {"reason": "slipped", "question": "handle or rim?"})


async def test_inject_warn_speaks_and_dedupes_while_firing():
    s = session()
    e = {"type": "check_failed", "check": "grip", "question": "How is the cup held?", "answer": "slipping", "p_bad": 0.9, "severity": "warn"}
    await s.inject(e)
    await s.inject(e)  # same check still failing: not repeated
    kinds = [m["type"] for m in s.openai.sent]
    assert kinds == ["conversation.item.create", "response.create"]
    await s.inject({"type": "check_cleared", "check": "grip"})
    s.last_spoken = 0
    await s.inject(e)
    assert sum(m["type"] == "response.create" for m in s.openai.sent) == 2


async def test_inject_urgent_interrupts_speech():
    s = session()
    s.response_active = True
    await s.inject({"type": "check_failed", "check": "person", "question": "Person near?", "answer": "yes", "p_bad": 0.95, "severity": "critical"})
    assert {"type": "response.cancel"} in s.openai.sent
    assert {"event": "clear"} in s.inkbox.sent
    assert "URGENT" in s.openai.sent[0]["item"]["content"][0]["text"]
    assert s.want_response  # speaks once the cancelled response finishes


async def test_tool_stop_robot_publishes_and_runs_stop_cmd(monkeypatch):
    published = []

    async def fake_publish(http, event, bus=None):
        published.append(event)
        return True

    async def fake_stop():
        return None

    monkeypatch.setattr(bridge, "apublish", fake_publish)
    monkeypatch.setattr(bridge, "run_stop_command", fake_stop)
    b = bridge.Bridge.__new__(bridge.Bridge)
    b.http = None
    out = await b._run_tool(session(), "stop_robot", {})
    assert out["ok"] and published == [{"type": "stop_requested", "by": "operator"}]


def test_authorized_requires_token_and_signature(monkeypatch):
    b = bridge.Bridge.__new__(bridge.Bridge)
    b.briefs, b.signing_key = {"tok": {"reason": "r"}}, None
    req = SimpleNamespace(query={"s": "bad"}, headers={})
    assert b._authorized(req) is None
    monkeypatch.delenv("WALDO_ALLOW_UNSIGNED", raising=False)
    assert b._authorized(SimpleNamespace(query={"s": "tok"}, headers={})) is None  # unsigned refused by default
    b.briefs["tok2"] = {"reason": "r"}
    monkeypatch.setenv("WALDO_ALLOW_UNSIGNED", "1")
    assert b._authorized(SimpleNamespace(query={"s": "tok2"}, headers={})) == {"reason": "r"}
    assert b._authorized(SimpleNamespace(query={"s": "tok2"}, headers={})) is None  # one-time token


# -- brain -----------------------------------------------------------------

class FakeMessages:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    async def create(self, **kw):
        self.calls.append(json.loads(json.dumps(kw, default=lambda o: o.__dict__)))
        return self.replies.pop(0)


def block(**kw):
    return SimpleNamespace(**kw)


def reply(stop, *blocks):
    return SimpleNamespace(stop_reason=stop, content=list(blocks))


@pytest.fixture
def quiet_bus(monkeypatch):
    published = []

    async def fake_publish(http, event, bus=None):
        published.append(event)
        return True

    monkeypatch.setattr(brain_mod, "apublish", fake_publish)
    return published


def make_brain(tmp_path, monkeypatch, replies):
    monkeypatch.setattr(brain_mod, "HINTS", tmp_path / "hints.json")
    messages = FakeMessages(replies)
    client = SimpleNamespace(beta=SimpleNamespace(messages=messages), messages=messages)
    b = brain_mod.Brain({}, client=client)
    b.tools = list(brain_mod.LOCAL_TOOLS)
    b.running = True
    return b, messages


async def test_brain_injects_watcher_notes_as_system_message(tmp_path, monkeypatch, quiet_bus):
    b, messages = make_brain(tmp_path, monkeypatch, [
        reply("tool_use", block(type="tool_use", id="t1", name="set_step", input={"step": "grasp"})),
        reply("end_turn", block(type="text", text="done")),
    ])

    async def failing_set_step(name, args):
        await b.on_event({"type": "check_failed", "question": "How is the cup held?", "answer": "slipping", "p_bad": 0.9, "severity": "warn", "step": "grasp"})
        return [{"type": "text", "text": "ok"}], False

    b.run_tool = failing_set_step
    await b.run_task("lift the red cup", ["grasp"])
    second = messages.calls[1]["messages"]
    assert second[-1]["role"] == "system" and "[watcher]" in second[-1]["content"]
    assert second[-2]["content"][0]["type"] == "tool_result"
    assert messages.calls[0]["fallbacks"] == "default"
    assert quiet_bus[-1] == {"type": "task_ended", "task": "lift the red cup", "result": "done"}


async def test_operator_instruction_becomes_hint_and_wait_returns_it(tmp_path, monkeypatch, quiet_bus):
    b, _ = make_brain(tmp_path, monkeypatch, [])
    b.task, b.last_problem, b.last_question = "lift the red cup", "grip -> slipping", "handle or rim?"
    await b.on_event({"type": "operator_instruction", "text": "use the handle"})
    content, err = await b.run_tool("wait_for_operator", {"seconds": 1})
    assert not err and "use the handle" in content[0]["text"]
    assert b.hints.hints[0].answer == "use the handle" and b.hints.hints[0].situation == "grip -> slipping"
    await b.on_event({"type": "operator_instruction", "text": "confirmed"})
    assert len(b.hints.hints) == 1  # confirmations aren't advice


async def test_call_operator_publishes_call_requested(tmp_path, monkeypatch, quiet_bus):
    b, _ = make_brain(tmp_path, monkeypatch, [])
    b.task = "lift the red cup"
    await b.run_tool("call_operator", {"reason": "slipped twice", "question": "handle or rim?"})
    assert quiet_bus[-1] == {"type": "call_requested", "task": "lift the red cup", "reason": "slipped twice", "question": "handle or rim?"}

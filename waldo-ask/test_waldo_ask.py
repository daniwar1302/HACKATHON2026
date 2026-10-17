import asyncio
from datetime import datetime, timedelta, timezone

from channels import EmailChannel, Sent, strip_quoted
from hints import HintStore
from server import Asker, build_instructions


class FakeChannel:
    name = "fake"

    def __init__(self, replies=None):
        self.sent = []
        self.queued = list(replies or [])

    def send(self, code, subject, body, image):
        self.sent.append((code, subject, body))
        return Sent(self.name, code, datetime.now(timezone.utc))

    def replies(self, sent):
        return [self.queued.pop(0)] if self.queued else []


def make(tmp_path, replies=None):
    store = HintStore(tmp_path / "hints.json")
    channel = FakeChannel(replies)
    return store, channel, Asker(store, lambda: channel, camera_url=None, poll=0.01)


def ask(asker, **kw):
    args = dict(task="lift the red cup", situation="gripper slipped at the rim", question="handle or rim?",
                camera="none", wait_seconds=1, force_human=False)
    return asyncio.run(asker.ask(**{**args, **kw}))


def test_operator_answer_is_saved_as_hint(tmp_path):
    store, channel, asker = make(tmp_path, ["handle, left hand"])
    result = ask(asker)
    assert result["status"] == "answered" and result["source"] == "operator"
    assert result["answer"] == "handle, left hand"
    assert len(channel.sent) == 1
    assert HintStore(tmp_path / "hints.json").get(result["hint_id"]).answer == "handle, left hand"


def test_trusted_hint_answers_without_messaging(tmp_path):
    store, channel, asker = make(tmp_path, ["handle, left hand"])
    first = ask(asker)
    store.report(first["hint_id"], worked=True)
    second = ask(asker, situation="gripper slipped on the rim again")
    assert second["source"] == "memory" and second["answer"] == "handle, left hand"
    assert len(channel.sent) == 1  # no second message


def test_untrusted_hint_still_asks(tmp_path):
    store, channel, asker = make(tmp_path, ["handle", "rim, slower"])
    ask(asker)  # saved but never confirmed
    second = ask(asker)
    assert second["source"] == "operator" and len(channel.sent) == 2


def test_force_human_skips_memory(tmp_path):
    store, channel, asker = make(tmp_path, ["handle", "use both hands"])
    store.report(ask(asker)["hint_id"], worked=True)
    assert ask(asker, force_human=True)["answer"] == "use both hands"


def test_failed_hints_drop_out(tmp_path):
    store, _, asker = make(tmp_path, ["handle"])
    hint_id = ask(asker)["hint_id"]
    for _ in range(2):
        store.report(hint_id, worked=False)
    assert asker.recall("lift the red cup") == []


def test_stop_reply_aborts_and_is_not_saved(tmp_path):
    store, _, asker = make(tmp_path, ["STOP, it's about to fall"])
    assert ask(asker)["status"] == "abort"
    assert store.hints == []


def test_no_reply_returns_waiting_then_wait_for_reply(tmp_path):
    store, channel, asker = make(tmp_path)
    result = ask(asker, wait_seconds=0.05)
    assert result["status"] == "waiting"
    channel.queued.append("rim")
    later = asyncio.run(asker.wait(result["ask_id"], 1))
    assert later["status"] == "answered" and later["answer"] == "rim"


def test_instructions_include_best_hints(tmp_path):
    store, _, asker = make(tmp_path, ["handle, left hand"])
    store.report(ask(asker)["hint_id"], worked=True)
    assert "handle, left hand" in build_instructions(store)


def test_strip_quoted_email():
    body = "Use the handle.\n\nOn Fri, Oct 17, 2026 at 1:00 PM Waldo <waldo@x> wrote:\n> Waldo needs help."
    assert strip_quoted(body) == "Use the handle."


class _Msg:
    def __init__(self, id, from_address, subject, created_at, thread_id=None, body="ok"):
        self.id, self.from_address, self.subject = id, from_address, subject
        self.created_at, self.thread_id, self.snippet, self.body_text = created_at, thread_id, body, body


class _Identity:
    def __init__(self, msgs):
        self.msgs = msgs

    def iter_emails(self, **kw):
        return iter(sorted(self.msgs, key=lambda m: m.created_at, reverse=True))

    def get_message(self, id):
        return next(m for m in self.msgs if str(m.id) == id)


class _Seg:
    def __init__(self, party, text):
        self.party, self.text = party, text


class _CallIdentity:
    def __init__(self, segs):
        self.segs = segs

    def list_transcripts(self, call_id):
        return self.segs


class _Calls:
    def __init__(self, status):
        self.status = status

    def get(self, call_id):
        return type("Call", (), {"status": self.status})()


def test_call_answer_is_what_the_operator_said():
    from channels import CallChannel

    segs = [_Seg("local", "Handle or rim?"), _Seg("remote", "Yes"), _Seg("remote", "By the rim")]
    sent = Sent("call", "AB12", datetime.now(timezone.utc), "call-1")
    assert CallChannel(_CallIdentity(segs), _Calls("answered"), "+1").replies(sent) == []
    assert CallChannel(_CallIdentity(segs), _Calls("completed"), "+1").replies(sent) == ["Yes By the rim"]


def test_unanswered_call_is_unreachable(tmp_path):
    from channels import ChannelError

    class NoAnswer(FakeChannel):
        def replies(self, sent):
            raise ChannelError("call no_answer without an answer")

    store = HintStore(tmp_path / "hints.json")
    asker = Asker(store, lambda: NoAnswer(), camera_url=None, poll=0.01)
    assert ask(asker)["status"] == "unreachable"


def test_email_replies_only_from_operator_after_question():
    t0 = datetime(2026, 10, 17, 13, tzinfo=timezone.utc)
    msgs = [
        _Msg(1, "dani@example.com", "Re: Waldo [AB12]", t0 + timedelta(seconds=30), body="handle"),
        _Msg(2, "stranger@example.com", "Re: Waldo [AB12]", t0 + timedelta(seconds=40), body="rim"),
        _Msg(3, "dani@example.com", "Re: Waldo [AB12]", t0 - timedelta(minutes=5), body="old"),
        _Msg(4, "Dani <DANI@example.com>", "unrelated", t0 + timedelta(seconds=50), body="lunch?"),
    ]
    channel = EmailChannel(_Identity(msgs), "dani@example.com")
    assert channel.replies(Sent("email", "AB12", t0)) == ["handle"]

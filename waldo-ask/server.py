"""MCP server that lets Waldo's Claude ask the operator for help, and learn from the answers.

Speed comes from not asking twice:
- recall_hints / ask_human check past advice first and answer in milliseconds when a
  trusted hint matches, with no message sent.
- The best hints are written into the server instructions, so Claude has them
  before its first tool call.
- When a human is needed, the reply is returned within ~2 s of arriving.
- report_outcome marks advice as working or not, so good advice gets reused and bad advice is dropped.

This server never moves a robot. Claude must stop the robot before asking.
"""
from __future__ import annotations

import asyncio
import os
import secrets
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from channels import Channel, Sent, make_channel
from hints import HintStore

HOME = Path(os.environ.get("WALDO_ASK_HOME", Path.home() / ".waldo-ask"))
CHANNEL = os.environ.get("WALDO_ASK_CHANNEL", "email")
TO = os.environ.get("WALDO_ASK_TO", "")
AGENT = os.environ.get("WALDO_ASK_IDENTITY", "waldo")
CAMERA_URL = os.environ.get("WALDO_CAMERA_URL", "http://10.2.1.180:8766")

REUSE_SCORE = 0.5   # word overlap needed to answer from memory without asking
POLL_SECONDS = 2.0
MAX_WAIT = 120
ABORT_WORDS = {"stop", "abort", "cancel", "halt"}


@dataclass
class Pending:
    sent: Sent
    task: str
    situation: str
    question: str


class Asker:
    def __init__(self, store: HintStore, channel_factory, camera_url: str | None = CAMERA_URL, poll: float = POLL_SECONDS):
        self.store = store
        self._factory = channel_factory
        self._channel: Channel | None = None
        self.camera_url = camera_url
        self.poll = poll
        self.pending: dict[str, Pending] = {}

    @property
    def channel(self) -> Channel:
        if self._channel is None:
            self._channel = self._factory()
        return self._channel

    def recall(self, query: str) -> list[dict]:
        return [{"score": score, **h.summary()} for score, h in self.store.search(query)]

    def _frame(self, camera: str) -> bytes | None:
        if not self.camera_url or camera == "none":
            return None
        try:
            with urllib.request.urlopen(f"{self.camera_url}/frame/{camera}.jpg", timeout=3) as resp:
                return resp.read()
        except Exception:
            return None  # a missing photo shouldn't stop the question

    async def ask(self, task: str, situation: str, question: str, camera: str, wait_seconds: float, force_human: bool) -> dict:
        if not force_human:
            for score, hint in self.store.search(f"{task} {situation} {question}", limit=1):
                if score >= REUSE_SCORE and hint.trusted:
                    return {
                        "status": "answered",
                        "source": "memory",
                        "answer": hint.answer,
                        "hint_id": hint.id,
                        "matched": hint.summary(),
                        "next": "Act on it, then call report_outcome. If it doesn't fit this case, call ask_human again with force_human=true.",
                    }

        code = secrets.token_hex(2).upper()
        body = (
            f"Waldo needs help.\n\nTask: {task}\nWhat happened: {situation}\n\n{question}\n\n"
            "Reply to this message. Reply STOP to abort the task."
        )
        image = await asyncio.to_thread(self._frame, camera)
        sent = await asyncio.to_thread(self.channel.send, code, f"Waldo: {question[:60]}", body, image)
        self.pending[code] = Pending(sent, task, situation, question)
        return await self.wait(code, wait_seconds)

    async def wait(self, ask_id: str, wait_seconds: float) -> dict:
        pending = self.pending.get(ask_id)
        if pending is None:
            return {"status": "error", "error": f"no open question {ask_id!r}", "open": sorted(self.pending)}
        deadline = time.monotonic() + max(0.0, min(MAX_WAIT, wait_seconds))
        while True:
            replies = await asyncio.to_thread(self.channel.replies, pending.sent)
            if replies:
                del self.pending[ask_id]
                answer = "\n".join(replies).strip()
                first = answer.split()[0].strip(".!,").lower() if answer.split() else ""
                if first in ABORT_WORDS:
                    return {"status": "abort", "answer": answer, "next": "Operator said stop. Stop all motion and end the task."}
                hint = self.store.add(pending.task, pending.situation, pending.question, answer)
                return {
                    "status": "answered",
                    "source": "operator",
                    "answer": answer,
                    "hint_id": hint.id,
                    "next": "Act on it, then call report_outcome so this advice is reused next time.",
                }
            if time.monotonic() >= deadline:
                return {
                    "status": "waiting",
                    "ask_id": ask_id,
                    "channel": pending.sent.channel,
                    "next": "No reply yet. Keep the robot stopped and call wait_for_reply, or give up the task.",
                }
            await asyncio.sleep(self.poll)

    async def notify(self, message: str, camera: str) -> dict:
        image = await asyncio.to_thread(self._frame, camera)
        code = secrets.token_hex(2).upper()
        await asyncio.to_thread(self.channel.send, code, f"Waldo: {message[:60]}", message, image)
        return {"ok": True, "sent_via": self.channel.name}


def build_instructions(store: HintStore) -> str:
    text = (
        "Ask the operator for help through ask_human when you are stuck or unsure, instead of retrying blindly. "
        "Stop the robot before asking; this server does not move it. "
        "At the start of a task call recall_hints. After acting on any advice, call report_outcome."
    )
    best = store.best()
    if best:
        text += "\n\nAdvice the operator gave before (reuse it when it fits):\n" + "\n".join(
            f"- [{h.id}] {h.task} / {h.situation}: {h.answer} (worked {h.worked}, failed {h.failed})" for h in best
        )
    return text


def _channel_factory():
    if not TO:
        raise RuntimeError("WALDO_ASK_TO is not set (operator email or +E.164 phone number)")
    return make_channel(CHANNEL, AGENT, TO)


store = HintStore(HOME / "hints.json")
asker = Asker(store, _channel_factory)
mcp = FastMCP("waldo-ask", instructions=build_instructions(store))


@mcp.tool()
def recall_hints(task: str) -> list[dict]:
    """Past operator advice for tasks like this one. Instant and local; call at the start of a task."""
    return asker.recall(task)


@mcp.tool()
async def ask_human(
    task: str,
    situation: str,
    question: str,
    camera: str = "head_camera",
    wait_seconds: float = 60,
    force_human: bool = False,
) -> dict:
    """Ask the operator a question, with a camera photo, and wait for the answer.

    task: what you are trying to do ("lift the red cup").
    situation: what went wrong ("gripper slipped twice at the rim").
    question: one specific question, ideally with options ("handle or rim?").
    camera: head_camera, waldo_webcam, or none.
    wait_seconds: how long to wait for a reply (max 120); if nothing arrives you get an ask_id for wait_for_reply.
    force_human: skip remembered advice and always ask the person.

    Stop the robot before calling. Answers from memory come back immediately.
    """
    return await asker.ask(task, situation, question, camera, wait_seconds, force_human)


@mcp.tool()
async def wait_for_reply(ask_id: str, wait_seconds: float = 60) -> dict:
    """Keep waiting for the reply to an earlier ask_human that returned status 'waiting'."""
    return await asker.wait(ask_id, wait_seconds)


@mcp.tool()
def report_outcome(hint_id: str, worked: bool, note: str = "") -> dict:
    """Record whether the advice worked, so it is reused (worked) or dropped (failed) next time."""
    hint = store.report(hint_id, worked, note)
    if hint is None:
        return {"ok": False, "error": f"unknown hint {hint_id!r}"}
    return {"ok": True, **hint.summary()}


@mcp.tool()
async def notify_human(message: str, camera: str = "none") -> dict:
    """Send the operator a one-way update (task done, gave up, etc.). Does not wait for a reply."""
    return await asker.notify(message, camera)


if __name__ == "__main__":
    mcp.run()

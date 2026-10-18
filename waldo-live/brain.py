"""Brain: a Claude agent loop that runs a robot task with the robot's MCP tools and reacts to the bus.

    python brain.py serve                      HTTP on 127.0.0.1:8772 (/task, /status, /consult)
    python brain.py task "lift the red cup"    start a task on a running brain

Robot servers come from ~/.waldo/brain.json:
    {"servers": {"g1": {"command": "/home/operator/HACKATHON2026/waldo-mcp/run.sh"}}, "stop_tool": "g1__stop"}

Between model calls the loop drains the bus. Watcher failures and operator instructions
arrive as mid-conversation system messages; a critical failure or a stop request calls the
stop tool directly before Claude is asked anything. Operator instructions are saved as
waldo-ask hints, so the same problem is solved from memory next time.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from contextlib import AsyncExitStack
from pathlib import Path

import aiohttp
from aiohttp import web
from anthropic import AsyncAnthropic

from common import BRAIN_URL, apublish, load_env, subscribe

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "waldo-ask"))
from hints import HintStore  # noqa: E402

PORT = 8772
CONFIG = Path.home() / ".waldo" / "brain.json"
HINTS = Path(os.environ.get("WALDO_ASK_HOME", Path.home() / ".waldo-ask")) / "hints.json"
PLANNER = os.environ.get("WALDO_PLANNER_MODEL", "claude-opus-5-5")
FAST = os.environ.get("WALDO_FAST_MODEL", "claude-haiku-5-5")
EVENT_TYPES = {"check_failed", "check_cleared", "stop_requested", "operator_instruction", "call_started", "call_ended", "call_failed"}

SYSTEM = """You control Waldo, a robot in a robotics lab, through the robot tools.
Work in small, verifiable steps. Before each phase of the task call set_step so the camera watcher
checks the right things (steps are listed in the task). Report short progress with progress().

System messages tagged [watcher] are live camera checks; [operator] is the human on the phone.
When a [watcher] check fails, stop and decide: retry differently once if you know a fix (for
example a hint below), otherwise call call_operator with what happened and one specific question,
then wait_for_operator. Follow operator instructions; when they say stop, stop.
After acting on an operator instruction or a hint, call report_outcome so good advice is reused.
Never move faster or further than the robot tools allow, and never work around their safety checks."""

LOCAL_TOOLS = [
    {"name": "set_step", "description": "Tell the camera watcher which phase of the task you are in.",
     "input_schema": {"type": "object", "properties": {"step": {"type": "string"}}, "required": ["step"]}},
    {"name": "progress", "description": "One short spoken-friendly status line for the operator.",
     "input_schema": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}},
    {"name": "call_operator", "description": "Phone the operator for help. Stop the robot first.",
     "input_schema": {"type": "object", "properties": {"reason": {"type": "string", "description": "What happened, one sentence."},
                                                       "question": {"type": "string", "description": "One specific question, with options if possible."}},
                      "required": ["reason", "question"]}},
    {"name": "wait_for_operator", "description": "Wait for the operator's next instruction (max 120 s).",
     "input_schema": {"type": "object", "properties": {"seconds": {"type": "number"}}, "required": ["seconds"]}},
    {"name": "report_outcome", "description": "Record whether an operator instruction or hint worked.",
     "input_schema": {"type": "object", "properties": {"hint_id": {"type": "string"}, "worked": {"type": "boolean"}, "note": {"type": "string"}},
                      "required": ["hint_id", "worked"]}},
]


def mcp_content_to_blocks(content) -> list[dict]:
    blocks = []
    for c in content or []:
        if getattr(c, "type", "") == "text":
            blocks.append({"type": "text", "text": c.text})
        elif getattr(c, "type", "") == "image":
            blocks.append({"type": "image", "source": {"type": "base64", "media_type": c.mimeType, "data": c.data}})
    return blocks or [{"type": "text", "text": "(no output)"}]


class Brain:
    def __init__(self, config: dict, client: AsyncAnthropic | None = None):
        self.config = config
        self.client = client or AsyncAnthropic()
        self.hints = HintStore(HINTS)
        self.http: aiohttp.ClientSession | None = None
        self.task: str | None = None
        self.step: str | None = None
        self.running: asyncio.Task | None = None
        self.notes: list[str] = []          # pending [watcher]/[operator] lines for the next model call
        self.operator: asyncio.Queue = asyncio.Queue()
        self.actions: list[str] = []
        self.events: list[dict] = []
        self.last_problem = ""
        self.last_question = ""
        self.mcp: dict[str, object] = {}
        self.tools: list[dict] = []

    # -- bus ---------------------------------------------------------------

    async def on_event(self, e: dict) -> None:
        self.events = (self.events + [e])[-20:]
        t = e["type"]
        if t == "check_failed":
            self.last_problem = f"{e.get('question')} -> {e.get('answer')}"
            self.notes.append(f"[watcher] {e.get('severity')}: {self.last_problem} (p={e.get('p_bad')}, step {e.get('step')})")
            if e.get("severity") == "critical":
                await self.stop_robot("watcher critical check")
        elif t == "check_cleared":
            self.notes.append(f"[watcher] {e.get('check')} is OK again")
        elif t == "stop_requested" and e.get("by") != "brain":
            await self.stop_robot(f"stop requested by {e.get('by')}")
        elif t == "operator_instruction":
            text = e.get("text", "")
            if self.task and text.lower() not in ("confirmed", "yes", "go ahead"):
                hint = self.hints.add(self.task, self.last_problem, self.last_question, text)
                text = f"{text} (saved as hint {hint.id})"
            await self.operator.put(text)
        elif t in ("call_started", "call_ended", "call_failed"):
            self.notes.append(f"[phone] {t.replace('_', ' ')} {e.get('error', '')}".strip())

    async def stop_robot(self, why: str) -> None:
        name = self.config.get("stop_tool")
        if not name or name not in self.mcp_tools():
            self.notes.append(f"[robot] stop needed ({why}) but no stop_tool is configured")
            return
        server, tool = name.split("__", 1)
        try:
            await self.mcp[server].call_tool(tool, {})
            self.notes.append(f"[robot] stopped automatically ({why})")
            await apublish(self.http, {"type": "robot_stopped", "by": "brain", "reason": why})
        except Exception as ex:
            self.notes.append(f"[robot] automatic stop failed: {ex}")

    def mcp_tools(self) -> set[str]:
        return {t["name"] for t in self.tools if "__" in t["name"]}

    # -- MCP robot servers -------------------------------------------------

    async def connect(self, stack: AsyncExitStack) -> None:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        self.tools = list(LOCAL_TOOLS)
        for name, spec in self.config.get("servers", {}).items():
            params = StdioServerParameters(command=spec["command"], args=spec.get("args", []), env={**os.environ, **spec.get("env", {})})
            read, write = await stack.enter_async_context(stdio_client(params))
            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            self.mcp[name] = session
            for t in (await session.list_tools()).tools:
                self.tools.append({"name": f"{name}__{t.name}"[:64], "description": t.description or "", "input_schema": t.inputSchema})

    # -- tools ---------------------------------------------------------------

    async def run_tool(self, name: str, args: dict) -> tuple[list[dict], bool]:
        self.actions = (self.actions + [f"{name} {json.dumps(args)[:120]}"])[-10:]
        if "__" in name:
            server, tool = name.split("__", 1)
            result = await self.mcp[server].call_tool(tool, args)
            return mcp_content_to_blocks(result.content), bool(result.isError)
        if name == "set_step":
            self.step = args["step"]
            await apublish(self.http, {"type": "step_changed", "task": self.task, "step": self.step})
            return [{"type": "text", "text": f"watcher now checking step {self.step}"}], False
        if name == "progress":
            await apublish(self.http, {"type": "brain_progress", "text": args["text"]})
            return [{"type": "text", "text": "ok"}], False
        if name == "call_operator":
            self.last_question = args["question"]
            await apublish(self.http, {"type": "call_requested", "task": self.task, "reason": args["reason"], "question": args["question"]})
            return [{"type": "text", "text": "Calling the operator. Call wait_for_operator for their answer."}], False
        if name == "wait_for_operator":
            try:
                text = await asyncio.wait_for(self.operator.get(), timeout=min(120.0, float(args.get("seconds", 60))))
                return [{"type": "text", "text": f"Operator says: {text}"}], False
            except asyncio.TimeoutError:
                return [{"type": "text", "text": "No instruction yet. Keep the robot stopped; wait again or give up."}], False
        if name == "report_outcome":
            hint = self.hints.report(args["hint_id"], args["worked"], args.get("note", ""))
            return [{"type": "text", "text": "recorded" if hint else "unknown hint"}], hint is None
        return [{"type": "text", "text": f"unknown tool {name}"}], True

    def drain_notes(self) -> str | None:
        while not self.operator.empty() and self.running:
            self.notes.append(f"[operator] {self.operator.get_nowait()}")
        notes, self.notes = self.notes, []
        return "\n".join(notes) or None

    def system_prompt(self) -> str:
        best = self.hints.best()
        if not best:
            return SYSTEM
        return SYSTEM + "\n\nAdvice the operator gave before:\n" + "\n".join(
            f"- [{h.id}] {h.task} / {h.situation}: {h.answer} (worked {h.worked}, failed {h.failed})" for h in best)

    # -- the loop ------------------------------------------------------------

    async def run_task(self, task: str, steps: list[str]) -> None:
        self.task, self.step, self.notes = task, None, []
        messages: list[dict] = [{"role": "user", "content": f"Task: {task}\nSteps for the watcher: {', '.join(steps) or 'none'}"}]
        await apublish(self.http, {"type": "task_started", "task": task})
        try:
            while True:
                resp = await self.client.beta.messages.create(
                    model=PLANNER, max_tokens=16000, system=self.system_prompt(), tools=self.tools, messages=messages,
                    output_config={"effort": "medium"},
                    betas=["server-side-fallback-2026-07-01"], fallbacks="default",
                )
                messages.append({"role": "assistant", "content": resp.content})
                if resp.stop_reason == "refusal":
                    await apublish(self.http, {"type": "task_ended", "task": task, "result": "refused"})
                    return
                uses = [b for b in resp.content if b.type == "tool_use"]
                if resp.stop_reason != "tool_use" or not uses:
                    text = " ".join(b.text for b in resp.content if b.type == "text")
                    await apublish(self.http, {"type": "task_ended", "task": task, "result": text[:500]})
                    return
                results = []
                for u in uses:
                    try:
                        content, is_error = await self.run_tool(u.name, u.input)
                    except Exception as ex:
                        content, is_error = [{"type": "text", "text": f"Error: {ex}"}], True
                    results.append({"type": "tool_result", "tool_use_id": u.id, "content": content, "is_error": is_error})
                messages.append({"role": "user", "content": results})
                if notes := self.drain_notes():
                    messages.append({"role": "system", "content": notes})
        except Exception as ex:
            print(f"task failed: {type(ex).__name__}: {ex}")
            await apublish(self.http, {"type": "task_ended", "task": task, "result": f"error: {ex}"[:500]})
        finally:
            self.running = None

    async def consult(self, query: str) -> str:
        """Fast answer for the voice agent, from the current state. Doesn't change the plan."""
        state = json.dumps(self.status())
        resp = await self.client.messages.create(
            model=FAST, max_tokens=1000, output_config={"effort": "low"},
            system="You are the planner for Waldo, a lab robot. Answer the voice agent in at most two short spoken sentences, "
                   "based only on the state given. If the operator gave an instruction, state the concrete plan for it.",
            messages=[{"role": "user", "content": f"State: {state}\n\nQuestion: {query}"}],
        )
        if resp.stop_reason == "refusal":
            return "I can't answer that one."
        return " ".join(b.text for b in resp.content if b.type == "text").strip()

    def status(self) -> dict:
        return {"task": self.task, "step": self.step, "running": self.running is not None,
                "last_actions": self.actions[-6:], "recent_events": [
                    {k: e.get(k) for k in ("type", "check", "answer", "p_bad", "text", "by")} for e in self.events[-8:]]}


def make_app(brain: Brain) -> web.Application:
    async def post_task(request):
        body = await request.json()
        if brain.running:
            return web.json_response({"error": f"already running: {brain.task}"}, status=409)
        brain.running = asyncio.create_task(brain.run_task(body["task"], body.get("steps", [])))
        return web.json_response({"ok": True})

    async def get_status(request):
        return web.json_response(brain.status())

    async def post_consult(request):
        body = await request.json()
        return web.json_response({"answer": await brain.consult(body.get("query", ""))})

    app = web.Application()
    app.router.add_post("/task", post_task)
    app.router.add_get("/status", get_status)
    app.router.add_post("/consult", post_consult)
    return app


async def serve() -> None:
    config = json.loads(CONFIG.read_text()) if CONFIG.exists() else {}
    brain = Brain(config)
    async with AsyncExitStack() as stack, aiohttp.ClientSession() as http:
        brain.http = http
        await brain.connect(stack)
        runner = web.AppRunner(make_app(brain))
        await runner.setup()
        await web.TCPSite(runner, "127.0.0.1", PORT).start()
        print(f"brain on 127.0.0.1:{PORT}; tools: {[t['name'] for t in brain.tools]}")
        async for event in subscribe(http, EVENT_TYPES):
            await brain.on_event(event)


if __name__ == "__main__":
    load_env()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("serve")
    t = sub.add_parser("task")
    t.add_argument("task")
    t.add_argument("--steps", default="approach,grasp,lift")
    args = parser.parse_args()
    if args.cmd == "serve":
        asyncio.run(serve())
    else:
        import urllib.request

        req = urllib.request.Request(f"{BRAIN_URL}/task", method="POST", headers={"Content-Type": "application/json"},
                                     data=json.dumps({"task": args.task, "steps": args.steps.split(",")}).encode())
        print(urllib.request.urlopen(req, timeout=5).read().decode())

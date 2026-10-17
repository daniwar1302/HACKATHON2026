"""Spoken command menu for command-mode calls: each name maps to one robot tool call.

Config is ~/.waldo/commands.json (copy commands.example.json). Only entries with
"enabled": true are offered to the voice agent. "motion": true entries run only after
the operator has said yes on the call (checked against the transcript, not the model).

    {"servers": {"g1": {"command": "/home/waldo/HACKATHON2026/waldo-mcp/run.sh"}},
     "stop": "g1__stop",
     "commands": {"wave": {"description": "wave hello", "tool": "g1__gesture",
                           "args": {"name": "high wave"}, "motion": true, "enabled": true}}}

A command can be {"builtin": "snapshot"} instead of a tool: text the operator a camera photo.
"""
from __future__ import annotations

import json
import os
import re
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Callable

CONFIG = Path.home() / ".waldo" / "commands.json"
YES = re.compile(r"\b(yes|yeah|yep|yup|sure|ok|okay|go|go ahead|do it|confirm(ed)?|please do|correct)\b", re.I)
CONFIRM_WINDOW_S = 60.0


@dataclass
class Command:
    name: str
    description: str
    tool: str | None = None
    args: dict = field(default_factory=dict)
    builtin: str | None = None
    motion: bool = False


class Commands:
    def __init__(self, config: dict, builtins: dict[str, Callable[[], Awaitable[dict]]] | None = None):
        self.servers = config.get("servers", {})
        self.stop_tool = config.get("stop")
        self.commands = {
            name: Command(name, c.get("description", name), c.get("tool"), c.get("args", {}), c.get("builtin"), bool(c.get("motion")))
            for name, c in config.get("commands", {}).items() if c.get("enabled")
        }
        self.builtins = builtins or {}
        self._stack: AsyncExitStack | None = None
        self._sessions: dict = {}

    @classmethod
    def load(cls, builtins=None, path: Path = CONFIG) -> "Commands":
        return cls(json.loads(path.read_text()) if path.exists() else {}, builtins)

    def menu(self) -> list[dict]:
        return [{"command": c.name, "does": c.description, "moves_robot": c.motion} for c in self.commands.values()]

    async def _session(self, server: str):
        if server not in self._sessions:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client

            spec = self.servers[server]
            self._stack = self._stack or AsyncExitStack()
            params = StdioServerParameters(command=spec["command"], args=spec.get("args", []), env={**os.environ, **spec.get("env", {})})
            read, write = await self._stack.enter_async_context(stdio_client(params))
            session = await self._stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            self._sessions[server] = session
        return self._sessions[server]

    async def call_tool(self, qualified: str, args: dict) -> dict:
        server, tool = qualified.split("__", 1)
        result = await (await self._session(server)).call_tool(tool, args)
        text = " ".join(getattr(c, "text", "") for c in result.content).strip()
        return {"ok": not result.isError, "result": text[:800]}

    async def run(self, name: str) -> dict:
        cmd = self.commands[name]
        if cmd.builtin:
            return await self.builtins[cmd.builtin]()
        return await self.call_tool(cmd.tool, cmd.args)

    async def stop(self) -> dict | None:
        return await self.call_tool(self.stop_tool, {}) if self.stop_tool else None

    async def close(self) -> None:
        if self._stack:
            await self._stack.aclose()
            self._stack, self._sessions = None, {}


def find_confirmation(transcripts: list[tuple[float, str, str]], command: str, now: float, used: set) -> float | None:
    """When the operator said yes to this command, or None.

    A yes counts if it's recent, hasn't confirmed something already, and answers Waldo:
    the last thing Waldo said before it mentioned this command or asked "go ahead".
    """
    for i in range(len(transcripts) - 1, -1, -1):
        t, party, text = transcripts[i]
        if now - t > CONFIRM_WINDOW_S:
            break
        if party != "remote" or t in used or not YES.search(text):
            continue
        asked = next((q for _, p, q in reversed(transcripts[:i]) if p == "local"), "")
        if command.lower() in asked.lower() or "go ahead" in asked.lower():
            return t
    return None

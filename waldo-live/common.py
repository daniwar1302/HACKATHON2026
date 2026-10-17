"""Shared bits: settings from ~/.waldo-ask/env and the event bus client."""
from __future__ import annotations

import asyncio
import json
import os
import shlex
import urllib.request
from pathlib import Path

ENV_FILE = Path.home() / ".waldo-ask" / "env"
BUS_URL = os.environ.get("WALDO_BUS", "http://127.0.0.1:8770")
BRAIN_URL = os.environ.get("WALDO_BRAIN", "http://127.0.0.1:8772")
CAMERA_URL = os.environ.get("WALDO_CAMERA_URL", "http://10.2.1.180:8766")
PHONE_CAMERA_URL = os.environ.get("WALDO_PHONE_CAMERA_URL", "http://127.0.0.1:8773")


def frame_url(camera: str) -> str:
    """Latest-frame URL for a camera: the viewer's cameras, or `phone` (served by the bridge)."""
    return f"{PHONE_CAMERA_URL if camera == 'phone' else CAMERA_URL}/frame/{camera}.jpg"


def load_env(path: Path = ENV_FILE) -> None:
    """KEY=VALUE lines into os.environ, without overriding what is already set."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


def publish(event: dict, bus: str = BUS_URL) -> bool:
    """Fire-and-forget publish from sync code. Returns False if the bus is down."""
    req = urllib.request.Request(
        f"{bus}/events", data=json.dumps(event).encode(), headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        urllib.request.urlopen(req, timeout=1).read()
        return True
    except Exception:
        return False


async def apublish(session, event: dict, bus: str = BUS_URL) -> bool:
    try:
        async with session.post(f"{bus}/events", json=event, timeout=2) as resp:
            return resp.status == 200
    except Exception:
        return False


async def subscribe(session, types: set[str] | None = None, bus: str = BUS_URL):
    """Yield bus events forever (Server-Sent Events), reconnecting if the bus restarts."""
    query = f"?types={','.join(sorted(types))}" if types else ""
    while True:
        try:
            async with session.get(f"{bus}/stream{query}", timeout=None) as resp:
                async for raw in resp.content:
                    line = raw.decode().strip()
                    if line.startswith("data:"):
                        yield json.loads(line[5:])
        except (asyncio.CancelledError, GeneratorExit):
            raise
        except Exception:
            await asyncio.sleep(1)


async def run_stop_command() -> str | None:
    """Run WALDO_STOP_CMD (direct robot stop, no LLM). Returns an error string or None."""
    cmd = os.environ.get("WALDO_STOP_CMD")
    if not cmd:
        return "WALDO_STOP_CMD not set"
    proc = await asyncio.create_subprocess_exec(*shlex.split(cmd), stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE)
    try:
        _, err = await asyncio.wait_for(proc.communicate(), timeout=10)
    except asyncio.TimeoutError:
        proc.kill()
        return "stop command timed out"
    return None if proc.returncode == 0 else f"stop command failed: {err.decode()[-200:]}"

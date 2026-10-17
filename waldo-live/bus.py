"""Event bus: a tiny pub/sub on 127.0.0.1 that connects watcher, brain and call bridge.

POST /events          publish one JSON event
GET  /stream?types=   Server-Sent Events, optionally filtered by type
GET  /recent?n=&types= last events, newest last

Every event is appended to ~/.waldo/events.jsonl, the record of each run.
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import itertools
import json
import time
from pathlib import Path

from aiohttp import web

LOG = Path.home() / ".waldo" / "events.jsonl"


class Bus:
    def __init__(self, log: Path | None = LOG, keep: int = 500):
        self.log = log
        self.recent: collections.deque[dict] = collections.deque(maxlen=keep)
        self.subscribers: set[tuple[asyncio.Queue, frozenset | None]] = set()
        self._ids = itertools.count(1)
        if log:
            log.parent.mkdir(parents=True, exist_ok=True)

    def publish(self, event: dict) -> dict:
        event = {"id": next(self._ids), "t": round(time.time(), 3), **event}
        self.recent.append(event)
        if self.log:
            with self.log.open("a") as f:
                f.write(json.dumps(event) + "\n")
        for queue, types in list(self.subscribers):
            if types is None or event.get("type") in types:
                try:
                    queue.put_nowait(event)
                except asyncio.QueueFull:
                    pass  # a stuck subscriber must not block the others
        return event


def _types(request: web.Request) -> frozenset | None:
    raw = request.query.get("types", "")
    return frozenset(t for t in raw.split(",") if t) or None


def make_app(bus: Bus) -> web.Application:
    async def post_event(request: web.Request) -> web.Response:
        event = await request.json()
        if not isinstance(event, dict) or "type" not in event:
            return web.json_response({"error": "event must be an object with a type"}, status=400)
        return web.json_response(bus.publish(event))

    async def stream(request: web.Request) -> web.StreamResponse:
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"})
        await resp.prepare(request)
        queue: asyncio.Queue = asyncio.Queue(maxsize=200)
        entry = (queue, _types(request))
        bus.subscribers.add(entry)
        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=15)
                    await resp.write(f"data: {json.dumps(event)}\n\n".encode())
                except asyncio.TimeoutError:
                    await resp.write(b": ping\n\n")
        except (ConnectionResetError, asyncio.CancelledError):
            pass
        finally:
            bus.subscribers.discard(entry)
        return resp

    async def recent(request: web.Request) -> web.Response:
        types, n = _types(request), int(request.query.get("n", 50))
        events = [e for e in bus.recent if types is None or e.get("type") in types]
        return web.json_response(events[-n:])

    app = web.Application()
    app.router.add_post("/events", post_event)
    app.router.add_get("/stream", stream)
    app.router.add_get("/recent", recent)
    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8770)
    args = parser.parse_args()
    web.run_app(make_app(Bus()), host="127.0.0.1", port=args.port, print=lambda *_: print(f"bus on 127.0.0.1:{args.port}"))

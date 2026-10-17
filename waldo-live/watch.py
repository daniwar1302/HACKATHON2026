"""Watcher: camera frames -> Jev checks -> debounced events on the bus.

    python watch.py run   --task tasks/lift_cup.json          live, publishes events
    python watch.py bench --task tasks/lift_cup.json --frames DIR [--labels labels.json] [--step grasp]

The brain sets the current step with a `step_changed` event; only that step's
checks run. A critical check firing publishes `stop_requested` and runs
WALDO_STOP_CMD directly, with no LLM in the loop. That is a convenience, not the
safety system: the hand controller and firmware limits stay the real e-stop.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path

import aiohttp

from common import BUS_URL, CAMERA_URL, apublish, load_env, run_stop_command, subscribe
from deciders import Check, ClaudeDecider, Decider, JevDecider

EVENTS_DIR = Path.home() / ".waldo" / "events"


@dataclass
class Task:
    name: str
    camera: str
    steps: dict[str, list[Check]]

    @classmethod
    def load(cls, path: str) -> "Task":
        d = json.loads(Path(path).read_text())
        steps = {s: [Check.from_dict(c) for c in checks] for s, checks in d["steps"].items()}
        return cls(d["task"], d.get("camera", "waldo_webcam"), steps)


@dataclass
class Debounce:
    """Smoothed P(bad) with hysteresis: fire after `need` frames above `fire_at`, clear below `clear_at`."""
    alpha: float = 0.6
    fire_at: float = 0.7
    clear_at: float = 0.4
    need: int = 3
    ema: float | None = None
    streak: int = 0
    firing: bool = False

    def update(self, p_bad: float) -> str | None:
        self.ema = p_bad if self.ema is None else self.alpha * p_bad + (1 - self.alpha) * self.ema
        if not self.firing:
            self.streak = self.streak + 1 if self.ema >= self.fire_at else 0
            if self.streak >= self.need:
                self.firing = True
                return "fire"
        elif self.ema < self.clear_at:
            self.firing, self.streak = False, 0
            return "clear"
        return None


@dataclass
class Watcher:
    task: Task
    decider: Decider
    fps: float = 5.0
    max_in_flight: int = 2
    step: str = ""
    states: dict[str, Debounce] = field(default_factory=dict)
    _latest_frame_t: float = 0.0  # newest frame whose result was applied

    def __post_init__(self):
        self.step = self.step or next(iter(self.task.steps))

    def set_step(self, step: str) -> bool:
        if step not in self.task.steps or step == self.step:
            return False
        self.step, self.states = step, {}
        return True

    def apply(self, frame_t: float, probs: dict[str, dict[str, float]]) -> list[dict]:
        """Fold one frame's answers into the debouncers; return events to publish."""
        if frame_t <= self._latest_frame_t:
            return []  # a slower, older request finished late; the present already moved on
        self._latest_frame_t = frame_t
        events = []
        for check in self.task.steps[self.step]:
            if check.id not in probs:
                continue
            p = probs[check.id]
            transition = self.states.setdefault(check.id, Debounce()).update(check.p_bad(p))
            if transition:
                answer = max(p, key=p.get)
                events.append({
                    "type": "check_failed" if transition == "fire" else "check_cleared",
                    "task": self.task.name, "step": self.step, "check": check.id, "question": check.question,
                    "answer": answer, "p_bad": round(check.p_bad(p), 2), "severity": check.severity,
                    "frame_t": round(frame_t, 3),
                })
        return events


async def grab(session, camera: str) -> bytes | None:
    try:
        async with session.get(f"{CAMERA_URL}/frame/{camera}.jpg", timeout=2) as resp:
            return await resp.read() if resp.status == 200 else None
    except Exception:
        return None


async def run(watcher: Watcher, verbose: bool = False) -> None:
    EVENTS_DIR.mkdir(parents=True, exist_ok=True)
    in_flight = asyncio.Semaphore(watcher.max_in_flight)
    async with aiohttp.ClientSession() as session:

        async def follow_steps():
            async for e in subscribe(session, {"step_changed"}):
                if e.get("task") in (None, watcher.task.name) and watcher.set_step(e.get("step", "")):
                    print(f"step -> {watcher.step}")

        async def one_frame(jpeg: bytes, frame_t: float):
            try:
                t0 = time.monotonic()
                probs = await watcher.decider.decide(session, jpeg, watcher.task.steps[watcher.step])
                latency = time.monotonic() - t0
            except Exception as e:
                print(f"decider error: {e}")
                return
            finally:
                in_flight.release()
            if verbose:
                top = {c: f"{max(p, key=p.get)} {max(p.values()):.2f}" for c, p in probs.items()}
                print(f"{time.strftime('%H:%M:%S')} {latency:.2f}s {top}")
            for event in watcher.apply(frame_t, probs):
                if event["type"] == "check_failed":
                    snap = EVENTS_DIR / f"{int(frame_t * 1000)}-{event['check']}.jpg"
                    snap.write_bytes(jpeg)
                    event["frame"] = str(snap)
                event["decider_latency_s"] = round(latency, 3)
                await apublish(session, event)
                print(json.dumps(event))
                if event["type"] == "check_failed" and event["severity"] == "critical":
                    await apublish(session, {"type": "stop_requested", "by": "watcher", "reason": f"{event['check']}={event['answer']}"})
                    if err := await run_stop_command():
                        print(f"reflex stop: {err}")

        steps_task = asyncio.create_task(follow_steps())
        period = 1.0 / watcher.fps
        print(f"watching {watcher.task.camera} at {watcher.fps} fps with {watcher.decider.name}, step {watcher.step}")
        try:
            while True:
                tick = time.monotonic()
                if not in_flight.locked():  # drop frames rather than queue them
                    jpeg = await grab(session, watcher.task.camera)
                    if jpeg:
                        await in_flight.acquire()
                        asyncio.create_task(one_frame(jpeg, time.time()))
                await asyncio.sleep(max(0.0, period - (time.monotonic() - tick)))
        finally:
            steps_task.cancel()


async def bench(task: Task, decider: Decider, frames: Path, step: str, labels: dict) -> None:
    """Phase 0: real latency and accuracy on recorded frames."""
    files = sorted(p for p in frames.iterdir() if p.suffix.lower() in (".jpg", ".jpeg"))
    checks = task.steps[step]
    latencies, correct, total = [], 0, 0
    async with aiohttp.ClientSession() as session:
        for f in files:
            t0 = time.monotonic()
            probs = await decider.decide(session, f.read_bytes(), checks)
            latencies.append(time.monotonic() - t0)
            row = {c.id: max(probs[c.id], key=probs[c.id].get) for c in checks if c.id in probs}
            for check_id, expected in labels.get(f.name, {}).items():
                total += 1
                correct += row.get(check_id) == expected
            print(f"{f.name}: {row}  ({latencies[-1]:.2f}s)")
    if latencies:
        q = statistics.quantiles(latencies, n=10) if len(latencies) > 1 else latencies * 9
        print(f"\n{len(files)} frames, latency p50 {statistics.median(latencies):.2f}s p90 {q[-1]:.2f}s")
    if total:
        print(f"accuracy {correct}/{total} = {correct / total:.0%}")


def make_decider(name: str) -> Decider:
    return ClaudeDecider() if name == "claude" else JevDecider()


if __name__ == "__main__":
    load_env()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["run", "bench"])
    parser.add_argument("--task", required=True)
    parser.add_argument("--decider", choices=["jev", "claude"], default="jev")
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--step", default="")
    parser.add_argument("--frames", type=Path)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--verbose", action="store_true", help="print every frame's answers and latency")
    args = parser.parse_args()
    task = Task.load(args.task)
    if args.mode == "run":
        asyncio.run(run(Watcher(task, make_decider(args.decider), fps=args.fps, step=args.step), verbose=args.verbose))
    else:
        labels = json.loads(args.labels.read_text()) if args.labels else {}
        asyncio.run(bench(task, make_decider(args.decider), args.frames, args.step or next(iter(task.steps)), labels))

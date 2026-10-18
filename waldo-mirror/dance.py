"""Dance engine: arms (and a little waist) on the beat, moves chosen by the voice agent or improvised.

    ./dance.sh             armed: waits on the waldo-live bus, moves nothing until `dance_start`
    ./dance.sh --dry-run   same, but never touches the robot; prints what it would do

Bus events it listens to:
    dance_tempo   {bpm, beat_t, confidence, loudness}   from the call bridge's beat tracker
    dance_phrase  {moves: [...], energy: 0..1}          the voice agent's choreography
    dance_start   only published once the operator asked for a dance (texted "dance") and music plays
    choreo_sync   {path, video_t, wall_t}: play the Just Dance routine from choreo.py, in time with the music
    dance_stop / stop_requested / call_ended            stop: glide back, hand the arms back

It publishes dance_started / dance_move / dance_stopped / dance_error. The dance ends by itself
when the music stops (no confident tempo for 6 s, or near-silence).
"""
from __future__ import annotations

import argparse
import json
import math
import queue
import random
import signal
import subprocess
import threading
import time
import urllib.request
from collections import deque

import numpy as np

from moves import BEATS_PER_MOVE, CALM, HYPE, MOVES, fit_energy, pose, ready_pose

BUS = "http://127.0.0.1:8770"
EVENTS = {"dance_tempo", "dance_phrase", "dance_start", "dance_stop", "stop_requested", "call_ended", "choreo_sync"}
MIN_CONFIDENCE = 0.2
MUSIC_GONE_S = 6.0


def publish(event: dict) -> None:
    try:
        req = urllib.request.Request(f"{BUS}/events", data=json.dumps(event).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
        urllib.request.urlopen(req, timeout=1).read()
    except Exception:
        pass


def listen(events: queue.Queue, stop: threading.Event) -> None:
    types = ",".join(sorted(EVENTS))
    while not stop.is_set():
        try:
            with urllib.request.urlopen(f"{BUS}/stream?types={types}", timeout=30) as resp:
                for raw in resp:
                    if raw.startswith(b"data:"):
                        events.put(json.loads(raw[5:]))
        except Exception:
            time.sleep(1)


class Choreo:
    """Beat clock + move sequence. No robot here, so it can be tested on its own."""

    def __init__(self, max_vel: float, offset: float, rng: random.Random | None = None):
        self.max_vel, self.offset, self.rng = max_vel, offset, rng or random.Random()
        self.bpms: deque[float] = deque(maxlen=3)
        self.bpm = None
        self.ref_t = self.ref_b = None  # continuous beat clock: beat number ref_b happened at time ref_t
        self.tempo_at = 0.0
        self.loudness = 0.0
        self.quiet_since = None
        self.queue: deque[str] = deque()
        self.energy = 0.6
        self.reset()

    def reset(self) -> None:
        self.routine = None  # (times, joint angles, video_t, wall_t) while playing a Just Dance routine
        self.start_beat = None
        self.slot = -1
        self.move = None
        self.move_energy = 0.0
        self.last_moves: deque[str] = deque(maxlen=2)

    def on_tempo(self, e: dict, now: float) -> None:
        self.loudness = float(e.get("loudness", 0.0))
        self.quiet_since = (self.quiet_since or now) if self.loudness < 0.02 else None
        if float(e.get("confidence", 0)) < MIN_CONFIDENCE or not e.get("bpm"):
            return
        self.bpms.append(float(e["bpm"]))
        beat_t = float(e["beat_t"])
        if self.ref_t is None:
            self.ref_t, self.ref_b = beat_t, 0.0
        else:
            # The tracker says a beat fell at beat_t: snap that moment to the nearest whole beat of our
            # running count. The count never restarts; its phase moves by at most half a beat.
            self.ref_b = float(round(self.ref_b + (beat_t - self.ref_t) * self.bpm / 60.0))
            self.ref_t = beat_t
        self.bpm = float(np.median(self.bpms))
        self.tempo_at = now

    def on_phrase(self, moves: list, energy) -> None:
        self.queue.clear()
        self.queue.extend(m for m in moves if m in MOVES)
        if energy is not None:
            self.energy = float(np.clip(energy, 0.1, 1.0))

    def music_gone(self, now: float) -> bool:
        no_tempo = self.bpm is None or now - self.tempo_at > MUSIC_GONE_S
        silent = self.quiet_since is not None and now - self.quiet_since > MUSIC_GONE_S
        return no_tempo or silent

    def beats(self, now: float) -> float:
        """Running beat count; `offset` leads the music to make up for call and motor lag."""
        return self.ref_b + (now + self.offset - self.ref_t) * self.bpm / 60.0

    def _pick(self) -> str:
        if self.queue:
            return self.queue.popleft()
        pool = HYPE if max(self.energy, self.loudness * 1.5) > 0.6 else CALM
        options = [m for m in pool if m not in self.last_moves] or list(pool)
        return self.rng.choice(options)

    def play(self, times: np.ndarray, qs: np.ndarray, video_t: float, wall_t: float) -> None:
        """Switch to a recorded routine; video_t (s into it) was playing at wall time wall_t."""
        self.routine = (times, qs, video_t, wall_t)

    def target(self, now: float):
        """(move or None, 14 joint angles, waist yaw) for this instant."""
        if self.routine is not None:
            times, qs, video_t, wall_t = self.routine
            vt = video_t + (now + self.offset - wall_t)
            if vt <= times[-1]:
                q = np.array([np.interp(vt, times, qs[:, j]) for j in range(qs.shape[1])])
                return "choreography", q, 0.0
            self.routine, self.slot = None, -1  # routine over: back to freestyle
        if self.bpm is None:
            return None, ready_pose(), 0.0
        b = self.beats(now)
        if self.start_beat is None:
            self.start_beat = math.ceil(b)  # first move begins on the next beat
        if b < self.start_beat:
            return None, ready_pose(), 0.0
        rel = b - self.start_beat
        slot = int(rel // BEATS_PER_MOVE)
        if slot != self.slot:
            self.slot, self.move = slot, self._pick()
            self.last_moves.append(self.move)
            self.move_energy = fit_energy(self.move, self.bpm, self.energy, self.max_vel)
        q, waist = pose(self.move, rel - slot * BEATS_PER_MOVE, self.move_energy)
        return self.move, q, waist


def mirror_running() -> bool:
    return subprocess.run(["pgrep", "-f", "/waldo-mirror/.venv/bin/python mirror.py"], capture_output=True).returncode == 0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dry-run", action="store_true", help="never touch the robot")
    p.add_argument("--max-vel", type=float, default=1.0, help="joint speed cap, rad/s (max 1.5)")
    p.add_argument("--offset", type=float, default=0.35, help="seconds to lead the music (call + motor lag)")
    p.add_argument("--iface", default="enp5s0")
    args = p.parse_args()
    if not 0.2 <= args.max_vel <= 1.5:
        raise SystemExit("--max-vel must be between 0.2 and 1.5 rad/s")

    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    events: queue.Queue = queue.Queue()
    threading.Thread(target=listen, args=(events, stop), daemon=True).start()
    choreo = Choreo(args.max_vel, args.offset)
    arms, dancing, last_move, last_print = None, False, None, 0.0
    print(f"dance engine armed ({'DRY RUN' if args.dry_run else 'LIVE'}, {args.max_vel} rad/s); waiting for dance_start", flush=True)

    def finish(reason: str) -> None:
        nonlocal arms, dancing
        if arms:
            print(f"stopping ({reason}): arms back to start, then handed to the stock controller", flush=True)
            arms.release()
            arms = None
        dancing = False
        publish({"type": "dance_stopped", "reason": reason})
        print(f"stopped: {reason}", flush=True)

    try:
        while not stop.is_set():
            now = time.time()
            while not events.empty():
                e = events.get()
                t = e["type"]
                if t == "dance_tempo":
                    choreo.on_tempo(e, now)
                elif t == "dance_phrase":
                    choreo.on_phrase(e.get("moves", []), e.get("energy"))
                elif t == "dance_start" and not dancing:
                    if mirror_running():
                        publish({"type": "dance_error", "error": "mirroring is running; stop it first"})
                        continue
                    if not args.dry_run:
                        from robot import ArmSdk, fsm_id

                        state = fsm_id(args.iface)
                        if state not in (200, 500):
                            publish({"type": "dance_error", "error": f"robot is not balancing (FSM {state})"})
                            continue
                        arms = ArmSdk(args.iface, max_vel=args.max_vel)
                        arms.engage()
                        arms.set_target(ready_pose())
                    choreo.reset()
                    dancing = True
                    publish({"type": "dance_started"})
                    print("dancing", flush=True)
                elif t == "choreo_sync" and dancing:
                    data = np.load(e["path"])
                    choreo.play(data["t"], data["q"], float(e["video_t"]), float(e["wall_t"]))
                    print(f"choreography: {data['title']} from {float(e['video_t']):.1f}s", flush=True)
                elif t in ("dance_stop", "stop_requested", "call_ended") and dancing:
                    finish(t)
            if dancing:
                if choreo.music_gone(now):
                    finish("music ended")
                    continue
                move, q, waist = choreo.target(now)
                if arms:
                    arms.set_target(q)
                    arms.set_waist_offset(waist)
                if move != last_move and move:
                    last_move = move
                    publish({"type": "dance_move", "move": move, "bpm": choreo.bpm, "energy": round(choreo.move_energy, 2)})
                if now - last_print > 1.0:
                    last_print = now
                    print(f"{choreo.bpm:5.1f} BPM | {move or 'ready':>10} | energy {choreo.move_energy:.2f} | "
                          f"L sh p/r {math.degrees(q[0]):+4.0f} {math.degrees(q[1]):+4.0f} | "
                          f"R sh p/r {math.degrees(q[7]):+4.0f} {math.degrees(q[8]):+4.0f} | waist {math.degrees(waist):+3.0f}", flush=True)
            time.sleep(0.02)
    finally:
        if dancing:
            finish("engine stopped")


if __name__ == "__main__":
    main()

"""Just Dance choreography for the G1: find the routine on YouTube, turn it into arm motion, sync it.

    ./choreo.sh prepare "levitating dua lipa"     search, download, convert; prints the result file
    ./choreo.sh worker                            serve choreo_request / choreo_align from the waldo-live bus

Pipeline: YouTube search "just dance <song>" -> best match (title says Just Dance, sane length)
-> video-only and audio-only tracks (yt-dlp) -> MediaPipe pose at 15 fps -> mirrored arm reach ->
Unitree IK with comfort limits -> forward pass that caps joint speed (so the robot can actually
follow) -> ~/.waldo/choreo/<id>.npz. Sync matches the music heard on the call against the video's
own soundtrack (onset-envelope cross-correlation) to find where in the routine the song is.

Downloading from YouTube is against its terms of service; this is for a lab demo.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MODEL = HERE / "models" / "pose_landmarker_full.task"
CACHE = Path.home() / ".waldo" / "choreo"
BUS = "http://127.0.0.1:8770"
FPS = 15.0
ENV_RATE = 50.0  # onset envelope frames per second, for sync
AUDIO_RATE = 16000


# -- finding the routine ------------------------------------------------------------

@dataclass
class Candidate:
    id: str
    title: str
    duration: float
    url: str


def score(c: Candidate, query: str) -> float:
    """Higher is better: a Just Dance upload of this song with a normal song length."""
    t = c.title.lower()
    s = 3.0 if "just dance" in t else 0.0
    words = [w for w in re.findall(r"[a-z0-9']+", query.lower()) if w not in {"just", "dance", "the", "by"}]
    s += 2.0 * sum(w in t for w in words) / max(1, len(words))
    s += 1.0 if 90 <= c.duration <= 360 else -2.0
    s -= 1.5 if any(k in t for k in ("extreme", "fanmade", "fan made", "kids", "megamix", "tutorial", "reaction")) else 0.0
    return s


def search(query: str, n: int = 8) -> list[Candidate]:
    import yt_dlp

    with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True, "extract_flat": True}) as y:
        info = y.extract_info(f"ytsearch{n}:just dance {query}", download=False)
    out = [Candidate(e["id"], e.get("title") or "", float(e.get("duration") or 0), e.get("url") or "") for e in info.get("entries", [])]
    return sorted(out, key=lambda c: score(c, query), reverse=True)


def download(c: Candidate) -> tuple[Path, Path]:
    """Video-only (<=360p) and audio-only tracks; YouTube no longer serves both in one file here."""
    import yt_dlp

    CACHE.mkdir(parents=True, exist_ok=True)
    paths = []
    for kind, fmt in (("video", "bv*[height<=360][vcodec^=avc1]/bv*[height<=360]"), ("audio", "ba[ext=m4a]/ba")):
        with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True, "format": fmt,
                               "outtmpl": str(CACHE / f"%(id)s.{kind}.%(ext)s")}) as y:
            info = y.extract_info(f"https://www.youtube.com/watch?v={c.id}", download=True)
            paths.append(Path(y.prepare_filename(info)))
    return paths[0], paths[1]


# -- audio and sync -------------------------------------------------------------------

def decode_audio(path: Path, rate: int = AUDIO_RATE) -> np.ndarray:
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(rate), "-f", "s16le", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0


def onset_envelope(x: np.ndarray, rate: int) -> np.ndarray:
    """Spectral flux at ENV_RATE frames/s (same idea as the live beat tracker)."""
    hop = int(rate / ENV_RATE)
    n_fft = 2 * hop
    if len(x) < n_fft:
        return np.zeros(0)
    frames = np.lib.stride_tricks.sliding_window_view(x, n_fft)[::hop] * np.hanning(n_fft)
    spec = np.log1p(np.abs(np.fft.rfft(frames, axis=1)) * 10)
    flux = np.maximum(np.diff(spec, axis=0), 0).sum(axis=1)
    return np.concatenate([[0.0], flux])


def align(snippet_env: np.ndarray, song_env: np.ndarray) -> tuple[float, float]:
    """(seconds into the song where the snippet ENDS, match strength) by normalized cross-correlation.

    Strength is how far the best match stands above the best match more than a second away: music
    is repetitive, so even a different song correlates well somewhere, but only the right song
    has one position that clearly wins (measured: right song 0.19-0.29, other songs <= 0.06).
    """
    m = len(snippet_env)
    if m < 100 or len(song_env) < m:
        return 0.0, 0.0
    a = (snippet_env - snippet_env.mean()) / (snippet_env.std() + 1e-9)
    windows = np.lib.stride_tricks.sliding_window_view(song_env, m)
    mu, sd = windows.mean(axis=1), windows.std(axis=1) + 1e-9
    r = (windows @ a / m - mu * a.mean()) / sd
    i = int(np.argmax(r))
    far = np.abs(np.arange(len(r)) - i) > ENV_RATE
    runner_up = r[far].max() if far.any() else 0.0
    return (i + m) / ENV_RATE, float(max(0.0, r[i] - runner_up))


# -- video to robot motion ----------------------------------------------------------

def speed_limit(q: np.ndarray, dt: float, max_vel: float) -> np.ndarray:
    """Follow q as closely as joints moving at most max_vel rad/s can (forward pass)."""
    out = q.copy()
    step = max_vel * dt
    for i in range(1, len(q)):
        out[i] = out[i - 1] + np.clip(q[i] - out[i - 1], -step, step)
    return out


def video_to_motion(video: Path, max_vel: float, progress=None) -> tuple[np.ndarray, np.ndarray]:
    """(times s, 14 arm joint angles per time) for the routine, mirrored like a player copying the coach."""
    import cv2
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision

    from moves import ready_pose
    from pose import Smoother, arm_reaches, hand_targets, robot_reaches
    from robot import ArmIK

    ik = ArmIK()
    landmarker = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL)), running_mode=vision.RunningMode.VIDEO, num_poses=1))
    cap = cv2.VideoCapture(str(video))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    every = max(1, round(src_fps / FPS))
    smooth = Smoother(0.6)
    q = ready_pose()
    hands = ik.hand_positions(q)
    times, qs, i = [], [], 0
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            if i % every == 0:
                t = i / src_fps
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                r = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), int(t * 1000))
                if r.pose_world_landmarks:
                    lm = r.pose_world_landmarks[0]
                    pts = np.array([[l.x, l.y, l.z] for l in lm])
                    vis = np.array([l.visibility or 0.0 for l in lm])
                    try:
                        reach = robot_reaches(arm_reaches(pts, vis), mirror=True)
                    except ValueError:
                        reach = {"left": None, "right": None}
                    reach = {s: smooth.update(s, v) for s, v in reach.items()}
                    targets = hand_targets(reach, ik.shoulder, ik.reach)
                    left = targets["left"] if targets["left"] is not None else hands["left"]
                    right = targets["right"] if targets["right"] is not None else hands["right"]
                    q = ik.solve(left, right, q)
                    hands = ik.hand_positions(q)
                times.append(t)
                qs.append(q.copy())  # no person this frame: hold the last pose
                if progress and len(times) % 150 == 0:
                    progress(i / max(total, 1))
            i += 1
    finally:
        cap.release()
        landmarker.close()
    q_arr = speed_limit(np.array(qs), 1.0 / FPS, max_vel)
    return np.array(times), q_arr


def prepare(query: str, max_vel: float = 1.0, progress=None) -> dict:
    cands = search(query)
    if not cands or score(cands[0], query) < 3.0:
        raise RuntimeError(f"no Just Dance video found for {query!r}")
    best = cands[0]
    out = CACHE / f"{best.id}.npz"
    if out.exists():
        return {"path": str(out), "title": best.title, "id": best.id, "cached": True}
    video, audio = download(best)
    times, q = video_to_motion(video, max_vel, progress)
    song = decode_audio(audio)
    np.savez_compressed(out, t=times, q=q, env=onset_envelope(song, AUDIO_RATE), title=best.title, max_vel=max_vel)
    video.unlink(missing_ok=True)  # keep only what the robot needs
    return {"path": str(out), "title": best.title, "id": best.id, "duration": float(times[-1]) if len(times) else 0.0}


# -- worker -------------------------------------------------------------------------

def publish(event: dict) -> None:
    try:
        req = urllib.request.Request(f"{BUS}/events", data=json.dumps(event).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
        urllib.request.urlopen(req, timeout=2).read()
    except Exception:
        pass


def handle(event: dict, max_vel: float) -> None:
    t = event["type"]
    if t == "choreo_request":
        query = event.get("query", "").strip()
        print(f"preparing {query!r}", flush=True)
        try:
            result = prepare(query, max_vel, progress=lambda f: print(f"  {f:.0%}", flush=True))
            print(f"ready: {result['title']}", flush=True)
            publish({"type": "choreo_ready", "query": query, **result})
        except Exception as e:
            print(f"failed: {e}", flush=True)
            publish({"type": "choreo_failed", "query": query, "error": str(e)[:300]})
    elif t == "choreo_align":
        data = np.load(event["path"])
        pcm = np.fromfile(event["audio"], dtype="<i2").astype(np.float32) / 32768.0
        rate = int(event.get("rate", 24000))
        video_t, strength = align(onset_envelope(pcm, rate), data["env"])
        print(f"align: {video_t:.1f}s into the routine, strength {strength:.2f}", flush=True)
        if strength >= float(event.get("min_strength", 0.05)):  # songs repeat; re-sync every 30 s corrects
            publish({"type": "choreo_sync", "path": event["path"], "video_t": video_t,
                     "wall_t": float(event["end_wall_t"]), "strength": round(strength, 2)})
        else:
            publish({"type": "choreo_unsynced", "strength": round(strength, 2)})


def worker(max_vel: float) -> None:
    print("choreo worker ready", flush=True)
    while True:
        try:
            with urllib.request.urlopen(f"{BUS}/stream?types=choreo_request,choreo_align", timeout=60) as resp:
                for raw in resp:
                    if raw.startswith(b"data:"):
                        handle(json.loads(raw[5:]), max_vel)
        except Exception:
            time.sleep(1)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    pr = sub.add_parser("prepare")
    pr.add_argument("query")
    for sp in (pr, sub.add_parser("worker")):
        sp.add_argument("--max-vel", type=float, default=1.0)
    args = p.parse_args()
    if args.cmd == "prepare":
        t0 = time.time()
        print(json.dumps(prepare(args.query, args.max_vel, progress=lambda f: print(f"  {f:.0%}", flush=True))))
        print(f"took {time.time() - t0:.0f} s")
    else:
        worker(args.max_vel)

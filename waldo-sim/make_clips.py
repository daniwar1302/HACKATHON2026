#!/usr/bin/env python3
"""Clean, sped-up clips for video editing (no captions, no cards): one clip per robot action (each tool call that
moves the robot, sped so it runs ~5 s) plus each whole run at 6x. Room view (720p) by default, --head for the robot's
head camera.

  python make_clips.py ~/Videos/waldo-demos/raw ../media/g1-demos/clips          (ffmpeg on PATH)
"""
import glob
import json
import os
import re
import subprocess
import sys
import tempfile

RAW, OUT = sys.argv[1], sys.argv[2]
CAM = "head_camera" if "--head" in sys.argv else "room_hd"
PHYSICAL = ("run_task", "tap", "pick", "place", "guard_status")   # guard_status: the machine runs, the guard hits STOP
MIN_CLIP_S = 2.5
TARGET_S, MIN_SPEED, WHOLE_SPEED, FPS = 5.0, 2.0, 6.0, 30


def frames(d):
    fs = [f for f in glob.glob(f"{RAW}/{d}/*_{CAM}.jpg") if os.path.getsize(f) > 1000]
    return sorted(((float(os.path.basename(f).split("_")[0]), f) for f in fs))


def actions(d):
    """(start, end, label) per physical tool call: from the call until its result arrives."""
    ev, said, armed = [], "", None
    for line in open(f"{RAW}/{d}/agent.log"):
        ts, js = line.split("\t", 1)
        try:
            m = json.loads(js)
        except ValueError:
            continue
        if m.get("type") == "assistant":
            for c in m["message"].get("content", []):
                if c.get("type") == "text" and c["text"].strip():
                    said = c["text"]
                if c.get("type") == "tool_use":
                    name = c["name"].replace("mcp__waldo-arms__", "")
                    arg = c["input"].get("name") or c["input"].get("container") or ""
                    if name == "tap":                  # name the tap after the button the agent said it was pressing
                        hit = re.search(r"[Tt]apping (?:the )?\**\"?([A-Za-z$0-9 ]+?)\"?\**(?: button)?[.,]", said or "")
                        arg = hit.group(1) if hit else ""
                    if name == "guard":
                        armed = float(ts)
                    if name == "guard_status":     # the whole guarded cycle: armed, started, pressure climbs, STOP
                        arg = "guarded_cycle_stop"
                    ev.append((armed if name == "guard_status" and armed else float(ts), "call", name,
                               f"{name}_{arg}" if isinstance(arg, str) and arg else name))
        elif m.get("type") == "user":
            ev.append((float(ts), "result", "", ""))
    out = []
    for i, (t, kind, name, label) in enumerate(ev):
        if kind == "call" and name in PHYSICAL:
            done = next((e[0] for e in ev[i + 1:] if e[1] == "result" and e[0] > t), t + 10)
            out.append((t - 0.5, done + 0.5, re.sub(r"[^a-z0-9_]+", "_", label.lower()).strip("_")))
    return out


def encode(fr, t0, t1, speed, path):
    """Frames in [t0, t1] at `speed`x, each held for its real duration / speed, resampled to a steady FPS."""
    span = [(t, f) for t, f in fr if t0 <= t <= t1]
    if len(span) < 2:
        return 0.0
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as lst:
        for (t, f), (tn, _) in zip(span, span[1:] + [(span[-1][0] + 0.25, None)]):
            lst.write(f"file '{f}'\nduration {max(0.01, (tn - t) / speed):.4f}\n")
        lst.write(f"file '{span[-1][1]}'\n")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst.name, "-vf",
                    f"fps={FPS},format=yuv420p", "-c:v", "libx264", "-crf", "22", "-preset", "slow", "-movflags", "+faststart", path],
                   check=True)
    os.unlink(lst.name)
    return (span[-1][0] - span[0][0]) / speed


def main():
    os.makedirs(OUT, exist_ok=True)
    tag = "" if CAM == "room_hd" else "_head"
    for d in ("it", "rack", "qa", "returns", "lab", "cell"):
        fr = frames(d)
        if not fr:
            continue
        dur = encode(fr, fr[0][0], fr[-1][0], WHOLE_SPEED, f"{OUT}/{d}{tag}_whole_6x.mp4")
        print(f"{d}{tag}_whole_6x.mp4  {dur:5.1f} s")
        for k, (a, b, label) in enumerate([x for x in actions(d) if (x[1] - x[0]) / max(MIN_SPEED, (x[1] - x[0]) / TARGET_S) >= MIN_CLIP_S], 1):
            speed = max(MIN_SPEED, (b - a) / TARGET_S)
            name = f"{d}{tag}_{k:02d}_{label}_{speed:.0f}x.mp4"
            if (b - a) / speed < MIN_CLIP_S:
                continue
            dur = encode(fr, a, b, speed, f"{OUT}/{name}")
            if dur:
                print(f"  {name:48s} {dur:4.1f} s  (real {b - a:4.1f} s)")


if __name__ == "__main__":
    main()

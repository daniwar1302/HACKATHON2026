#!/usr/bin/env python3
"""Cut demo recordings (record_demos.sh) into videos: per demo a full captioned cut (3x) and a highlights cut (the
moments the robot acts, 2x), plus one reel of every highlight. Captions are the agent's own words and tool calls,
at the moment they happened.

  python make_demo_videos.py ~/Videos/waldo-demos/raw media/demos        (needs pillow; ffmpeg on PATH)
"""
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap

from PIL import Image, ImageDraw, ImageFont

RAW, OUT = sys.argv[1], sys.argv[2]
W, H = 1280, 720
FPS_RAW = 4.0                                   # recorder rate (frames per second of real time)
PHYSICAL = ("run_task", "tap", "pick", "place", "guard", "home")
HI_FPS, HI_MAX = 16.0, 72                       # highlights play at 16 fps (4x the recorder), <= 72 frames per action
DEMOS = {   # name: (title, one-line setup)
    "it": ("IT agent: \"the office internet is down\"", "find the router, read its LEDs, power-cycle it, verify"),
    "rack": ("SRE agent: datacenter remote hands", "read the server's error code + drive LEDs, follow the runbook, verify"),
    "qa": ("QA agent: test an app on a physical tablet", "tap through checkout with the robot's fingertip, file what breaks"),
    "returns": ("Logistics agent: warehouse returns", "scan each returned item, read its code, bin it"),
    "lab": ("Coding agent: flash firmware on a real board", "recover a hung board, BOOT+RESET, flash v2, then a guarded machine cycle"),
    "cell": ("Manufacturing agent: machine tending + QC", "load, run, read the QC lamp, unload to PASS or REJECT, 3 parts"),
}
FONT = "/usr/share/fonts/TTF/DejaVuSans.ttf"
BOLD = "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf"
F = lambda n, b=False: ImageFont.truetype(BOLD if b else FONT, n)


def events(name):
    ev, result = [], None
    for line in open(f"{RAW}/{name}/agent.log"):
        ts, js = line.split("\t", 1)
        try:
            d = json.loads(js)
        except ValueError:
            continue
        t = float(ts)
        if d.get("type") == "assistant":
            for c in d["message"].get("content", []):
                if c.get("type") == "tool_use":
                    tool = c["name"].replace("mcp__waldo-arms__", "")
                    args = ", ".join(f"{k}={json.dumps(v)}" for k, v in c["input"].items())
                    ev.append((t, "tool", tool, f"→ {tool}({args})"[:110]))
                elif c.get("type") == "text" and c["text"].strip():
                    ev.append((t, "text", "", c["text"].strip().splitlines()[0]))
        elif d.get("type") == "user":
            ev.append((t, "result", "", ""))
        elif d.get("type") == "result":
            result = d
    return ev, result


def frames(name):
    stamp = lambda p: float(os.path.basename(p).split("_")[0])
    heads = sorted(glob.glob(f"{RAW}/{name}/*_head_camera.jpg"), key=stamp)
    hts = [stamp(h) for h in heads]
    out, j = [], 0
    for room in sorted(glob.glob(f"{RAW}/{name}/*_room_hd.jpg"), key=stamp):
        t = stamp(room)
        while j + 1 < len(hts) and abs(hts[j + 1] - t) <= abs(hts[j] - t):
            j += 1
        out.append((t, room, heads[j] if heads else None))
    return out


def compose(t, room, head, ev, title, speed):
    try:
        im = Image.open(room).convert("RGB").resize((W, H))
    except Exception:
        return None
    if head:
        try:
            hd = Image.open(head).convert("RGB").resize((400, 300))
            im.paste((255, 255, 255), (W - 412, 56, W - 8, 360))
            im.paste(hd, (W - 410, 58))
        except Exception:
            pass
    d = ImageDraw.Draw(im, "RGBA")
    d.rectangle([0, 0, W, 46], fill=(10, 12, 16, 200))
    d.text((14, 9), title, font=F(24, True), fill=(255, 255, 255))
    d.rectangle([W - 412, 362, W - 8, 386], fill=(10, 12, 16, 180))
    d.text((W - 404, 365), "robot head camera", font=F(16), fill=(235, 235, 235))
    d.text((W - 170, 14), f"{speed:g}x · sim", font=F(17), fill=(170, 180, 195))
    txt = next((e[3] for e in reversed(ev) if e[0] <= t and e[1] == "text"), "")
    tool = next((e[3] for e in reversed(ev) if e[0] <= t and e[1] == "tool"), "")
    d.rectangle([0, H - 150, W, H], fill=(10, 12, 16, 215))
    y = H - 140
    for line in textwrap.wrap(re.sub(r"\*\*|`", "", txt), 95)[:3]:
        d.text((20, y), line, font=F(23), fill=(240, 242, 248))
        y += 31
    d.text((20, H - 40), tool, font=F(21), fill=(90, 205, 235))
    return im


def card(title, lines, color=(255, 255, 255)):
    im = Image.new("RGB", (W, H), (16, 18, 24))
    d = ImageDraw.Draw(im)
    d.text((80, 220), title, font=F(46, True), fill=color)
    for i, line in enumerate(lines):
        d.text((80, 310 + 46 * i), line, font=F(28), fill=(195, 205, 220))
    return im


def summary(result):
    if not result:
        return []
    text = re.sub(r"\*\*|`|#+ ", "", result.get("result", ""))
    first = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
    stats = f"{result.get('num_turns')} turns · {result.get('duration_ms', 0) / 1000:.0f} s · ${result.get('total_cost_usd', 0):.2f} · unscripted"
    return textwrap.wrap(first, 70)[:3] + [stats]


def windows(ev, t_end):
    """Highlight spans: each physical action from just before the call to just after its result."""
    spans = []
    for i, (t, kind, tool, _) in enumerate(ev):
        if kind == "tool" and tool in PHYSICAL:
            done = next((e[0] for e in ev[i + 1:] if e[1] == "result"), t_end)
            spans.append([t - 1.5, done + 2.0])
    spans.sort()
    merged = []
    for s in spans:
        if merged and s[0] <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], s[1])
        else:
            merged.append(s)
    return merged


def encode(images, path, fps):
    """images: an iterable of (PIL image or None, repeat). Streamed to disk, never held in memory."""
    tmp = tempfile.mkdtemp()
    k = 0
    for im, n in images:
        if im is None:
            continue
        for _ in range(n):
            im.save(f"{tmp}/{k:06d}.jpg", quality=90)
            k += 1
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", f"{tmp}/%06d.jpg", "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", "25", "-preset", "slow", path], check=True)
    shutil.rmtree(tmp)
    return k


def cut(name):
    """(title card, full-cut frames, highlight frames, outro) generators for one demo."""
    title, setup = DEMOS[name]
    ev, result = events(name)
    fr = frames(name)
    intro, outro = card(title, [setup]), card("Result", summary(result), (120, 230, 140))
    spans = windows(ev, fr[-1][0]) if fr else []
    full = lambda: [(intro, 24)] + [(compose(t, r, h, ev, title, 3), 1) for t, r, h in fr] + [(outro, 36)]
    def hi():
        """Snips: each action at 4x, capped at ~4.5 s of screen time (evenly sampled: start, middle and finish)."""
        yield intro, 20
        for a, b in spans:
            span = [(t, r, h) for t, r, h in fr if a <= t <= b]
            keep = span if len(span) <= HI_MAX else [span[round(i * (len(span) - 1) / (HI_MAX - 1))] for i in range(HI_MAX)]
            for t, r, h in keep:
                yield compose(t, r, h, ev, title, round((b - a) / (len(keep) / HI_FPS), 1)), 1
        yield outro, 40
    return full, hi, len(spans)


def main():
    os.makedirs(OUT, exist_ok=True)
    names = [n for n in DEMOS if os.path.exists(f"{RAW}/{n}/agent.log") and glob.glob(f"{RAW}/{n}/*_room_hd.jpg")]
    for name in names:
        full, hi, nspans = cut(name)
        n = encode(full(), f"{OUT}/{name}_full.mp4", FPS_RAW * 3)
        m = encode(hi(), f"{OUT}/{name}_highlights.mp4", HI_FPS)
        print(f"{name}: full {n / (FPS_RAW * 3):.0f} s, highlights {m / HI_FPS:.0f} s ({nspans} action spans)", flush=True)

    def reel():
        yield card("Agents run real work, through a robot", ["Unitree G1 arms as MCP skills · MuJoCo twin on the robot's real protocol",
                                                             "Claude Code gets a plain-English work order and only the robot tools",
                                                             "Nothing is scripted: the agent decides, acts, senses and verifies"]), 56
        for name in names:
            yield from cut(name)[1]()
        yield card("What agents get", ["verified skills: pick, place, tap, taught presses and transfers",
                                       "fast sensing: LEDs, displays, gauges · background interlocks · receipts",
                                       "teach a new cell by pointing or one demonstration",
                                       "github.com/daniwar1302/HACKATHON2026"]), 64
    k = encode(reel(), f"{OUT}/all_highlights.mp4", HI_FPS)
    print(f"all_highlights: {k / HI_FPS:.0f} s")


if __name__ == "__main__":
    main()

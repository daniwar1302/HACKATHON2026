import glob, json, os, sys, textwrap
# (re)make media/g1_agents_demo.mp4 from timestamped demo recordings: AGENT_LOG=<dir>/agent.log ./demo.sh <which> while recording
# frames as <epoch>_{head_camera,waldo_webcam}.jpg into <S>/vid_<which>/; then: python make_video.py <S> media/g1_agents_demo.mp4
from PIL import Image, ImageDraw, ImageFont
S = sys.argv[1]; OUT = sys.argv[2]
F = lambda n, b=False: ImageFont.truetype("/usr/share/fonts/TTF/DejaVuSans-Bold.ttf" if b else "/usr/share/fonts/TTF/DejaVuSans.ttf", n)
W, H = 1280, 720
frames_dir = f"{S}/vid_frames"; os.system(f"rm -rf {frames_dir}; mkdir -p {frames_dir}")
k = [0]
def emit(im, n=1):
    for _ in range(n):
        im.save(f"{frames_dir}/{k[0]:05d}.jpg", quality=90); k[0] += 1
def card(title, lines, n=30):
    im = Image.new("RGB", (W, H), (18, 20, 26)); d = ImageDraw.Draw(im)
    d.text((80, 220), title, font=F(52, True), fill=(255, 255, 255))
    for i, l in enumerate(lines):
        d.text((80, 320 + 46 * i), l, font=F(30), fill=(190, 200, 215))
    emit(im, n)
def run_stats(name):
    for line in open(f"{S}/vid_{name}/agent.log"):
        d = json.loads(line.split("\t", 1)[1])
        if d.get("type") == "result":
            return f"Matches ground truth · {d.get('num_turns')} turns · {d.get('duration_ms', 0) / 1000:.0f} s · ${d.get('total_cost_usd', 0):.2f}"
    return ""
DEMOS = [
    ("it", "IT agent: \"the office internet is down\"",
     ["Claude Code + robot MCP tools, plain-English ticket", "find router -> read LEDs -> power-cycle -> verify"],
     ["Router hung (internet LED amber) -> power-cycled -> both LEDs green"]),
    ("rack", "SRE agent: datacenter remote hands",
     ["Page: storage on db-07 degraded", "read the panel code + drive LEDs -> follow the runbook -> verify"],
     ["Er42 -> reseated only the failed drive (no needless hard reset) -> all green"]),
    ("lab", "Coding agent: flash firmware on a real board",
     ["Hung board -> robot holds BOOT, taps RESET -> agent flashes v2 from its shell", "then a machine cycle guarded by an overpressure interlock"],
     ["Firmware v2 running, pairing code read off the display", "Faulty machine stopped by the guard at ~8 bar (trips at 9.5)"]),
    ("cell", "Manufacturing agent: machine tending + QC",
     ["Load raw part -> START -> watch stack light -> read QC lamp", "-> unload to PASS or REJECT bin, 3 parts"],
     ["3/3 parts machined and sorted correctly (pass, reject, pass)"]),
]
card("Agents run real work cells", ["Unitree G1 arms as MCP skills · MuJoCo twin on the robot's real protocol",
                                    "Claude Code gets a work order and only the robot tools", "Everything below is unscripted: the agent decides every step"], 45)
for name, title, intro, outro in DEMOS:
    card(title, intro, 30)
    ev = []
    for line in open(f"{S}/vid_{name}/agent.log"):
        ts, js = line.split("\t", 1)
        try: d = json.loads(js)
        except ValueError: continue
        if d.get("type") != "assistant": continue
        for c in d["message"].get("content", []):
            if c.get("type") == "tool_use":
                args = ", ".join(f"{a}={json.dumps(v)}" for a, v in c["input"].items())
                ev.append((float(ts), "tool", f"→ {c['name'].replace('mcp__waldo-arms__', '')}({args})"[:95]))
            elif c.get("type") == "text" and c["text"].strip():
                ev.append((float(ts), "text", c["text"].strip().splitlines()[0]))
    heads = sorted(glob.glob(f"{S}/vid_{name}/*_head_camera.jpg"))
    for h in heads:
        t = float(os.path.basename(h).split("_")[0])
        r = h.replace("head_camera", "waldo_webcam")
        if not os.path.exists(r): continue
        try:
            room, head = Image.open(r).resize((640, 480)), Image.open(h).resize((640, 480))
        except Exception:
            continue          # empty frame while the sim was starting
        im = Image.new("RGB", (W, H), (18, 20, 26))
        im.paste(room, (0, 40)); im.paste(head, (640, 40))
        d = ImageDraw.Draw(im)
        d.text((12, 8), title, font=F(22, True), fill=(255, 255, 255)); d.text((W - 230, 10), "4x speed · sim", font=F(18), fill=(150, 160, 175))
        d.text((12, 500), "room camera", font=F(15), fill=(150, 160, 175)); d.text((652, 500), "robot head camera", font=F(15), fill=(150, 160, 175))
        txt = next((e[2] for e in reversed(ev) if e[0] <= t and e[1] == "text"), "")
        tool = next((e[2] for e in reversed(ev) if e[0] <= t and e[1] == "tool"), "")
        y = 535
        for l in textwrap.wrap(txt, 100)[:3]:
            d.text((20, y), l, font=F(22), fill=(235, 238, 245)); y += 30
        d.text((20, 660), tool, font=F(20), fill=(90, 200, 230))
        emit(im)
    card("Result", outro + [run_stats(name)], 40)
card("What agents get", ["verified skills (pick, place, taught presses & transfers) · fast sensing (LEDs, displays, gauges)",
                         "background interlocks · receipts (before/after frames for every action)",
                         "teach a new cell by pointing or one demonstration · github.com/daniwar1302/HACKATHON2026"], 60)
os.system(f"ffmpeg -y -loglevel error -framerate 10 -i {frames_dir}/%05d.jpg -c:v libx264 -pix_fmt yuv420p -crf 23 {OUT}")
print(k[0], "frames ->", OUT)

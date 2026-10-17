"""Phone camera: an iPhone (or any phone) in Safari becomes one of Waldo's cameras.

The phone opens https://<tunnel>/phone-cam?t=<token>, taps Start, and posts a JPEG a few times
a second. Waldo keeps only the latest frame, in memory, and serves it on 127.0.0.1:8773 as
/frame/phone.jpg (same shape as the camera viewer), never on the public tunnel.

The page and the upload need the token; get the link with `./run.sh bridge phone-link`.
"""
from __future__ import annotations

import hmac
import os
import secrets
import time

from aiohttp import web

from common import ENV_FILE

LOCAL_PORT = 8773
STALE_S = 3.0
MAX_FRAME = 2 * 1024 * 1024

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>Waldo eye</title>
<style>
  :root { --bg:#0b0d10; --fg:#e8eaed; --muted:#9aa0a6; --ok:#34a853; --bad:#ea4335; }
  @media (prefers-color-scheme: light) { :root { --bg:#f8f9fa; --fg:#202124; --muted:#5f6368; } }
  body { margin:0; padding:16px; background:var(--bg); color:var(--fg); font:16px -apple-system, system-ui, sans-serif; }
  video { width:100%; border-radius:12px; background:#000; }
  button { width:100%; padding:14px; margin:12px 0; font-size:18px; border:0; border-radius:12px; background:#1a73e8; color:#fff; }
  #status { color:var(--muted); } .ok { color:var(--ok) !important; } .bad { color:var(--bad) !important; }
</style></head>
<body>
<h2>Waldo's eye</h2>
<p>Point the back camera where the robot looks, keep this page open and the phone awake.</p>
<video id="v" playsinline muted autoplay></video>
<button id="go">Start</button>
<p id="status">Not sending.</p>
<script>
const url = "frame" + location.search, v = document.getElementById("v"), st = document.getElementById("status");
const canvas = document.createElement("canvas"); canvas.width = 640; canvas.height = 480;
let sent = 0, running = false, lock = null;
function say(text, cls) { st.textContent = text; st.className = cls || ""; }
async function loop() {
  while (running) {
    const t = Date.now();
    canvas.getContext("2d").drawImage(v, 0, 0, canvas.width, canvas.height);
    const blob = await new Promise(r => canvas.toBlob(r, "image/jpeg", 0.7));
    try {
      const r = await fetch(url, { method: "POST", body: blob, headers: { "Content-Type": "image/jpeg" } });
      if (!r.ok) throw new Error("HTTP " + r.status);
      sent++; say("Sending to Waldo: " + sent + " frames", "ok");
    } catch (e) { say("Waldo isn't receiving (" + e.message + "). Retrying...", "bad"); }
    await new Promise(r => setTimeout(r, Math.max(0, 250 - (Date.now() - t))));
  }
}
document.getElementById("go").onclick = async () => {
  if (running) { running = false; document.getElementById("go").textContent = "Start"; say("Stopped."); return; }
  try {
    v.srcObject = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment", width: 1280, height: 720 }, audio: false });
    await v.play();
    try { lock = await navigator.wakeLock.request("screen"); } catch (e) {}
    running = true; document.getElementById("go").textContent = "Stop"; loop();
  } catch (e) { say("Camera not available: " + e.message, "bad"); }
};
</script></body></html>
"""


def token() -> str:
    """The phone-cam token from the env file, created on first use."""
    tok = os.environ.get("WALDO_PHONE_CAM_TOKEN")
    if not tok:
        tok = secrets.token_urlsafe(18)
        with ENV_FILE.open("a") as f:
            f.write(f"WALDO_PHONE_CAM_TOKEN={tok}\n")
        os.environ["WALDO_PHONE_CAM_TOKEN"] = tok
    return tok


class PhoneCam:
    def __init__(self, tok: str):
        self.tok = tok
        self.jpeg: bytes | None = None
        self.at = 0.0
        self.frames = 0

    def _ok(self, request: web.Request) -> bool:
        return hmac.compare_digest(request.query.get("t", ""), self.tok)

    async def page(self, request: web.Request) -> web.Response:
        if not self._ok(request):
            return web.Response(status=404)
        return web.Response(text=PAGE, content_type="text/html", headers={"Cache-Control": "no-store"})

    async def upload(self, request: web.Request) -> web.Response:
        if not self._ok(request):
            return web.Response(status=404)
        body = await request.read()
        if not body.startswith(b"\xff\xd8") or len(body) > MAX_FRAME:
            return web.Response(status=400, text="expected one JPEG")
        self.jpeg, self.at, self.frames = body, time.monotonic(), self.frames + 1
        if self.frames == 1 or self.frames % 100 == 0:
            print(f"phone camera: {self.frames} frames received ({len(body) // 1024} KB)", flush=True)
        return web.Response(text="ok")

    async def status(self, request: web.Request) -> web.Response:
        age = round(time.monotonic() - self.at, 1) if self.jpeg else None
        return web.json_response({"frames": self.frames, "last_frame_age_s": age})

    async def latest(self, request: web.Request) -> web.Response:
        if self.jpeg is None or time.monotonic() - self.at > STALE_S:
            return web.Response(status=503, text="No current frame")
        return web.Response(body=self.jpeg, content_type="image/jpeg")

    def add_public_routes(self, app: web.Application) -> None:
        app.router.add_get("/phone-cam", self.page)
        app.router.add_post("/frame", self.upload)  # relative to the page: /frame?t=...

    def local_app(self) -> web.Application:
        app = web.Application()
        app.router.add_get("/frame/phone.jpg", self.latest)
        app.router.add_get("/status", self.status)
        return app

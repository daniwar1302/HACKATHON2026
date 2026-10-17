#!/usr/bin/env python3
"""Web UI for the G1: same tools, same safety checks as g1_mcp.py (it calls them directly).

  uv run --with "mcp>=2.2" ~/HACKATHON2026/waldo-console/g1_webui.py      # http://localhost:8780
  G1_DRY_RUN=1 ...                                                 # rehearse without the robot

Camera panel shows the waldo webcam if the 8766 SSH tunnel is up.
"""
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import g1_mcp  # noqa: E402

TOOLS = {f.__name__: f for f in (g1_mcp.status, g1_mcp.set_mode, g1_mcp.gesture, g1_mcp.wave_hand,
                                  g1_mcp.say, g1_mcp.set_led, g1_mcp.set_volume, g1_mcp.stop)}
HOST, PORT = os.environ.get("G1_UI_HOST", "127.0.0.1"), int(os.environ.get("G1_UI_PORT", "8780"))

PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Waldo Control</title>
<style>
:root{--bg:#f6f7f9;--card:#fff;--ink:#16191d;--mute:#5d6672;--line:#dde1e6;--ok:#14804a;--bad:#c42b1c;--acc:#2457c5}
@media (prefers-color-scheme:dark){:root{--bg:#0f1216;--card:#171b21;--ink:#e8ecf1;--mute:#8d97a3;--line:#2a313a;--ok:#4cc38a;--bad:#ff6b5e;--acc:#7aa2ff}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}
header{position:sticky;top:0;z-index:2;display:flex;gap:12px;align-items:center;justify-content:space-between;padding:12px 16px;background:var(--card);border-bottom:1px solid var(--line)}
h1{font-size:18px;margin:0}h2{font-size:13px;text-transform:uppercase;letter-spacing:.06em;color:var(--mute);margin:0 0 10px}
main{max-width:1100px;margin:0 auto;padding:16px;display:grid;gap:16px;grid-template-columns:repeat(auto-fit,minmax(300px,1fr))}
section{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px}
button{font:inherit;border:1px solid var(--line);background:var(--bg);color:var(--ink);border-radius:7px;padding:8px 12px;cursor:pointer;min-height:40px}
button:hover:not(:disabled){border-color:var(--acc)}button:disabled{opacity:.4;cursor:not-allowed}
#stop{background:var(--bad);color:#fff;border:0;font-weight:700;padding:10px 22px;font-size:16px}
.row{display:flex;flex-wrap:wrap;gap:8px;align-items:center}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:8px}
.pill{display:inline-block;padding:2px 8px;border-radius:99px;font-size:12px;border:1px solid var(--line)}
.ok{color:var(--ok)}.bad{color:var(--bad)}input[type=text]{flex:1;min-width:0;font:inherit;padding:8px;border:1px solid var(--line);border-radius:7px;background:var(--bg);color:var(--ink)}
#log{font:12px/1.5 ui-monospace,monospace;white-space:pre-wrap;max-height:260px;overflow:auto;margin:0}
#cam{width:100%;border-radius:8px;background:#000;aspect-ratio:4/3;object-fit:contain}
dl{display:grid;grid-template-columns:auto 1fr;gap:4px 12px;margin:0}dt{color:var(--mute)}dd{margin:0}
.wide{grid-column:1/-1}
</style></head><body>
<header><h1>Waldo · G1 control <span id="dry" class="pill" hidden>DRY RUN</span> <a href="/controller" style="font-size:14px">virtual controllers →</a></h1><button id="stop" title="Release arms and drop to damp">STOP</button></header>
<main>
<section><h2>Status</h2><dl id="st"><dt>state</dt><dd>loading…</dd></dl><div class="row" style="margin-top:10px"><button onclick="refresh()">Refresh</button></div></section>
<section><h2>Camera</h2><img id="cam" alt="waldo webcam" src="http://localhost:8766/stream/waldo_webcam.mjpg"
 onerror="this.replaceWith(Object.assign(document.createElement('p'),{textContent:'No feed. Start the tunnel: ssh -L 8766:10.2.1.180:8766 waldo@100.120.182.66'}))"></section>
<section><h2>Mode</h2><div class="row">
 <button onclick="act('set_mode',{mode:'stand'},'Stand powers ALL joints, legs included. Continue?')">Stand (fsm 4)</button>
 <button onclick="act('set_mode',{mode:'damp'})">Damp (fsm 1)</button>
 <button onclick="act('set_mode',{mode:'zero_torque'},'Motors off: the robot goes limp. Continue?')">Zero torque (fsm 0)</button></div>
 <p style="color:var(--mute);margin:10px 0 0">Wave and built-in gestures need fsm 801 (balance stand), entered with the robot's remote. Stand (fsm 4) is not enough.</p></section>
<section><h2>Voice &amp; light</h2>
 <div class="row"><input id="say" type="text" maxlength="300" placeholder="Say something…" onkeydown="if(event.key==='Enter')sayIt()"><button onclick="sayIt()">Say</button></div>
 <div class="row" style="margin-top:10px"><label>LED <input id="led" type="color" value="#00ff00"></label><button onclick="led()">Set LED</button>
 <label>Volume <input id="vol" type="range" min="0" max="100" value="60" oninput="volv.textContent=this.value"></label><span id="volv">60</span>
 <button onclick="act('set_volume',{volume:+vol.value})">Set</button></div></section>
<section class="wide"><h2>Arms</h2><div class="row" style="margin-bottom:10px">
 <button id="wave" onclick="act('wave_hand',{turn_around:false})">Wave hand</button></div>
 <div id="gestures" class="grid"></div></section>
<section class="wide"><h2>Log</h2><pre id="log"></pre></section>
</main>
<script>
const $=id=>document.getElementById(id);let busy=false;
function log(m){const t=new Date().toLocaleTimeString();$('log').textContent=`${t}  ${m}\n`+$('log').textContent}
async function call(tool,args={}){const r=await fetch('/api/'+tool,{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify(args)});
 const j=await r.json();if(!r.ok)throw new Error(j.error);return j.result}
async function act(tool,args={},confirmMsg){if(confirmMsg&&!confirm(confirmMsg))return;if(busy&&tool!=='stop'){log('busy: wait for the last command');return}
 busy=tool!=='stop';log(`→ ${tool} ${JSON.stringify(args)}`);
 try{const r=await call(tool,args);log(`← ${typeof r==='string'?r:JSON.stringify(r)}`)}catch(e){log('✗ '+e.message)}finally{busy=false;if(tool!=='status')refresh()}}
function sayIt(){const t=$('say').value.trim();if(t)act('say',{text:t})}
function led(){const h=$('led').value;act('set_led',{r:parseInt(h.slice(1,3),16),g:parseInt(h.slice(3,5),16),b:parseInt(h.slice(5,7),16)})}
$('stop').onclick=()=>act('stop');
async function refresh(){try{const s=await call('status');if(s.dry_run){$('dry').hidden=false;$('st').innerHTML='<dt>mode</dt><dd>dry run: robot not contacted</dd>';return}
 const y=(b)=>b?'<span class="ok">yes</span>':'<span class="bad">no</span>';
 $('st').innerHTML=`<dt>mode</dt><dd>${s.mode}</dd><dt>fsm</dt><dd>${s.fsm} (${s.fsm_name})</dd><dt>wave works</dt><dd>${y(s.wave_works_now)}</dd><dt>gestures work</dt><dd>${y(s.gestures_work_now)}</dd><dt>volume</dt><dd>${s.volume}</dd>`;
 $('wave').disabled=!s.wave_works_now;
 $('gestures').innerHTML='';for(const g of s.gestures){const b=document.createElement('button');b.textContent=g.name.replaceAll('_',' ');
  const fsmOk=!g.needs_fsm||g.needs_fsm.includes(s.fsm);b.disabled=!s.gestures_work_now||!fsmOk;
  b.title=g.needs_fsm?`needs fsm ${g.needs_fsm.join('/')}`:(g.needs_mode_machine?`needs mode_machine ${g.needs_mode_machine.join('/')}`:'');
  b.onclick=()=>act('gesture',{name:g.name});$('gestures').append(b)}}catch(e){log('status failed: '+e.message)}}
refresh();
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        data = body.encode() if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("content-type", ctype)
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/":
            self._send(200, PAGE, "text/html; charset=utf-8")
        elif self.path == "/controller":  # virtual XR controllers that stream straight to teleop's websocket
            with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "controller.html"), "rb") as f:
                self._send(200, f.read(), "text/html; charset=utf-8")
        else:
            self._send(404, json.dumps({"error": "not found"}))

    def do_POST(self):
        name = self.path.removeprefix("/api/")
        if name not in TOOLS:
            return self._send(404, json.dumps({"error": f"unknown tool {name}"}))
        # Same-origin only: a page on another site can't drive the robot through this browser.
        if self.headers.get("origin") not in (None, f"http://{HOST}:{PORT}", f"http://localhost:{PORT}"):
            return self._send(403, json.dumps({"error": "cross-origin request refused"}))
        try:
            args = json.loads(self.rfile.read(int(self.headers.get("content-length") or 0)) or b"{}")
            self._send(200, json.dumps({"result": TOOLS[name](**args)}, default=str))
        except Exception as e:  # surface the tool's own refusal/error text in the UI log
            self._send(500, json.dumps({"error": f"{type(e).__name__}: {e}"}))

    def log_message(self, fmt, *a):
        sys.stderr.write("%s %s\n" % (self.address_string(), fmt % a))


if __name__ == "__main__":
    print(f"G1 web UI on http://{HOST}:{PORT}" + ("  (DRY RUN)" if g1_mcp.DRY else ""), flush=True)
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()

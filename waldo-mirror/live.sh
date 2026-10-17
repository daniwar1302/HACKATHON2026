#!/bin/sh
# Live mirroring control on Waldo, for people and agents alike.
#   live.sh check   robot state + phone camera status (moves nothing)
#   live.sh start   start live mirroring in tmux window waldo:mirror, at 0.8 rad/s,
#                   only once the phone is sending (waits up to 2 minutes)
#   live.sh stop    Ctrl-C the mirror: arms glide back and are handed to the stock controller
set -u
DIR="$(cd "$(dirname "$0")" && pwd)"
LOG=/tmp/mirror_live.log
frame_age() {
  curl -s http://127.0.0.1:8773/status | python3 -c 'import json,sys; a=json.load(sys.stdin)["last_frame_age_s"]; print(a if a is not None else 999)' 2>/dev/null || echo 999
}
running() { tmux list-windows -t waldo -F '#{window_name}' 2>/dev/null | grep -qx mirror; }

case "${1:-}" in
  check)
    cd /tmp && "$HOME/HACKATHON2026/waldo-live/.venv/bin/python" - <<'EOF' 2>/dev/null
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
async def main():
    async with stdio_client(StdioServerParameters(command="/home/waldo/HACKATHON2026/waldo-mcp/run.sh")) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()
            print("robot:", (await s.call_tool("get_state", {})).content[0].text.replace("\n", " "))
asyncio.run(main())
EOF
    echo "phone: last frame $(frame_age) s ago"
    running && echo "mirror: RUNNING (tail: $(tail -1 $LOG 2>/dev/null))" || echo "mirror: not running"
    ;;
  start)
    running && { echo "mirror already running"; exit 1; }
    tmux has-session -t waldo 2>/dev/null || { echo "no waldo tmux session; run waldo-live/start.sh"; exit 1; }
    i=0
    while [ "$(python3 -c "print(1 if float('$(frame_age)') < 1 else 0)")" != 1 ]; do
      i=$((i + 1))
      [ $i -gt 120 ] && { echo "no phone frames in 2 minutes; mirror NOT started"; exit 1; }
      sleep 1
    done
    : > $LOG
    tmux new-window -d -t waldo -n mirror "$DIR/run.sh --live --max-vel 0.8 2>&1 | grep --line-buffered -vE 'W0000|I0000|WARNING: Logging|inference_feedback|XNNPACK|ERROR in convergence' | tee $LOG; echo; read -p '[mirror exited - Enter closes this window] ' x"
    sleep 12
    tail -5 $LOG
    ;;
  stop)
    running || { echo "mirror not running"; exit 0; }
    tmux send-keys -t waldo:mirror C-c
    for i in $(seq 1 20); do grep -q "^released" $LOG && break; sleep 1; done
    tail -3 $LOG
    ;;
  *)
    echo "usage: live.sh check|start|stop"; exit 2 ;;
esac

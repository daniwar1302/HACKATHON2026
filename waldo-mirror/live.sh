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
running() { pgrep -f "/waldo-mirror/.venv/bin/python mirror.py" >/dev/null; }

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
    # No pipes: Ctrl-C in the window (or `live.sh stop`) reaches only the mirror, which then releases the arms.
    tmux new-window -d -t waldo -n mirror "$DIR/run.sh --live --max-vel 0.8 --log $LOG 2>/tmp/mirror_stderr.log; echo; read -p '[mirror exited - Enter closes this window] ' x"
    sleep 12
    tail -5 $LOG
    ;;
  stop)
    running || { echo "mirror not running"; exit 0; }
    pkill -INT -f "/waldo-mirror/.venv/bin/python mirror.py"
    for i in $(seq 1 20); do grep -q "^released" $LOG && break; sleep 1; done
    if grep -q "^released" $LOG; then
      tmux send-keys -t waldo:mirror Enter 2>/dev/null
      tail -2 $LOG
    else
      echo "WARNING: no release confirmed in 20 s; use the hand controller if the arms are not back"
      tail -3 $LOG
    fi
    ;;
  dance|dance-dry)
    # Arm the dance engine. It moves nothing until a dance call's "go Waldo" publishes dance_start.
    pgrep -f "/waldo-mirror/.venv/bin/python dance.py" >/dev/null && { echo "dance engine already armed"; exit 1; }
    flag=""; [ "$1" = dance-dry ] && flag="--dry-run"
    tmux new-window -d -t waldo -n dance "$DIR/dance.sh $flag 2>/tmp/dance_stderr.log | tee /tmp/dance_live.log; echo; read -p '[dance engine exited - Enter closes this window] ' x"
    # The Just Dance worker only downloads and converts routines; it never moves the robot.
    pgrep -f "/waldo-mirror/.venv/bin/python choreo.py worker" >/dev/null || \
      tmux new-window -d -t waldo -n choreo "$DIR/choreo.sh worker 2>/tmp/choreo_stderr.log | tee /tmp/choreo_live.log; echo; read -p '[choreo worker exited - Enter closes this window] ' x"
    sleep 3; tail -2 /tmp/dance_live.log; tail -1 /tmp/choreo_live.log
    ;;
  dance-off)
    pkill -INT -f "/waldo-mirror/.venv/bin/python dance.py" && echo "dance engine disarmed (arms released if it was dancing)" || echo "dance engine was not armed"
    pkill -INT -f "/waldo-mirror/.venv/bin/python choreo.py worker" && echo "choreo worker stopped"
    sleep 3; tmux send-keys -t waldo:dance Enter 2>/dev/null; tmux send-keys -t waldo:choreo Enter 2>/dev/null; true
    ;;
  *)
    echo "usage: live.sh check|start|stop|dance|dance-dry|dance-off"; exit 2 ;;
esac

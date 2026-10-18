#!/usr/bin/env bash
# Record every Claude demo: HD room view + head camera (~4 fps, epoch-stamped) + the agent's timestamped events.
#   ./record_demos.sh [out_dir] [demo ...]        then: python make_demo_videos.py <out_dir>
set -uo pipefail
cd "$(dirname "$0")"
out=${1:-$HOME/Videos/waldo-demos/raw}; shift || true
demos=("$@"); [ ${#demos[@]} -eq 0 ] && demos=(it rack qa returns lab cell)
declare -A off=([cell]=1000 [lab]=1100 [it]=1200 [rack]=1400 [qa]=1600 [returns]=1800)
for d in "${demos[@]}"; do
  dir="$out/$d"; rm -rf "$dir"; mkdir -p "$dir"
  port=$((18766 + ${off[$d]}))
  python3 - "$dir" "$port" <<'PY' &
import sys, time, urllib.request
out, port = sys.argv[1], sys.argv[2]
while True:
    t = time.time()
    for cam in ("room_hd", "head_camera"):
        try:
            data = urllib.request.urlopen(f"http://127.0.0.1:{port}/frame/{cam}.jpg", timeout=3).read()
            if data:
                open(f"{out}/{t:.2f}_{cam}.jpg", "wb").write(data)
        except Exception:
            pass
    time.sleep(max(0, 0.25 - (time.time() - t)))
PY
  rec=$!
  AGENT_LOG="$dir/agent.log" DEMO_NO_VIEWER=1 timeout 1200 ./demo.sh "$d" > "$dir/demo.out" 2>&1
  kill $rec
  echo "$d done: $(ls "$dir"/*room_hd.jpg 2>/dev/null | wc -l) frames"
done

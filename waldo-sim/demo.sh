#!/usr/bin/env bash
# Live demo: Claude runs a cell through the waldo-arms MCP tools, in a real-time MuJoCo sim with the viewer open.
#   ./demo.sh cell    machine tending + QC (3 parts -> pass / reject bins)
#   ./demo.sh it      the office internet is down: find the router, read its LEDs, power-cycle, verify
#   ./demo.sh qa      app QA on a real tablet: tap through the purchase flow, report what breaks
#   ./demo.sh rack    SRE page: read the server's error code, follow the runbook (hard reset / reseat a drive), verify
#   ./demo.sh lab     hung dev board -> bootloader -> flash v2 -> read code; guarded machine cycle (faulty machine)
# Needs: assets fetched (./fetch_assets.sh), a desktop session (MuJoCo viewer), the claude CLI logged in.
set -euo pipefail
cd "$(dirname "$0")"
which=${1:-cell}
case "$which" in
  cell) off=1000; seed=1; mcp=mcp_cell.json; order=work_order.txt; tools=(mcp__waldo-arms) ;;
  lab)  off=1100; seed=3; mcp=mcp_lab.json;  order=work_order_lab.txt; tools=(mcp__waldo-arms "Bash(uv run --no-project --with pyzmq usb_flash.py:*)") ;;
  qa)   off=1600; seed=0; mcp=mcp_qa.json;   order=work_order_qa.txt; tools=(mcp__waldo-arms) ;;
  rack) off=1400; seed=1; mcp=mcp_rack.json; order=work_order_rack.txt; tools=(mcp__waldo-arms) ;;
  it)   off=1200; seed=1; mcp=mcp_it.json;   order=work_order_it.txt; tools=(mcp__waldo-arms) ;;
  *) echo "usage: $0 cell|lab|it|rack|qa"; exit 1 ;;
esac
port=$((15810 + off))
viewer=--viewer; [ -n "${DEMO_NO_VIEWER:-}" ] && viewer=""      # DEMO_NO_VIEWER=1: headless (testing)
uvx="uv run -q --no-project --with mujoco --with pyzmq --with pillow --with numpy"
if ! ss -ltn | grep -q ":$port "; then
  MUJOCO_GL=glfw $uvx python sim_server.py --speed 1 --scene "$(case $which in it) echo devices;; qa) echo tablet;; *) echo "$which";; esac)" --port-offset "$off" $viewer > "/tmp/waldo-demo-$which.log" 2>&1 &
  sim=$!
  trap 'kill $sim 2>/dev/null' EXIT          # only the sim this script started
  for _ in $(seq 90); do ss -ltn | grep -q ":$port " && break; sleep 1; done
fi
$uvx python -I -c "
import sys; sys.path.insert(0, '../waldo-arms')
from planner import Arm
a = Arm('tcp://127.0.0.1:$port', timeout_s=30); a('sim_reset', seed=$seed); a('enable'); print('sim ready:', a('sim_devices')['state'])"
echo; echo "== work order =="; cat "$order"; echo
claude -p "$(cat "$order")" --mcp-config "$mcp" --strict-mcp-config --allowedTools "${tools[@]}" --max-turns 80 \
  --output-format stream-json --verbose < /dev/null | python3 agent_view.py
echo; echo "== ground truth (from the sim) =="
$uvx python -I -c "
import sys; sys.path.insert(0, '../waldo-arms')
from planner import Arm
a = Arm('tcp://127.0.0.1:$port', timeout_s=30); st = a('sim_devices')['state']; print(st)
if '$which' == 'cell': print({k: v['in_container'] for k, v in a('sim_objects')['objects'].items()})
print([e for _, e in a('sim_devices')['history'] if 'FAULT' in e] or 'no faults')"

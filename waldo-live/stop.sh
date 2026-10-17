#!/bin/sh
# Stop everything started by ./start.sh.
tmux kill-session -t waldo 2>/dev/null && echo "stopped" || echo "nothing running"

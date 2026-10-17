#!/bin/sh
# Start bus, watcher, bridge and brain in one tmux session called "waldo", one pane each.
# Ctrl-b d detaches (everything keeps running); ./start.sh again reattaches; ./stop.sh stops it all.
DIR="$(cd "$(dirname "$0")" && pwd)"
S=waldo
if ! tmux has-session -t "$S" 2>/dev/null; then
  pane() { echo "cd '$DIR' && ./run.sh $1; echo; read -p '[$2 exited - Enter closes this pane] ' x"; }
  pkill -f "python (bus|watch|bridge|brain).py" 2>/dev/null  # leftovers from a pane killed mid-pipe
  bus=$(tmux new-session -d -P -F '#{pane_id}' -s "$S" -x 220 -y 50 "$(pane bus bus)")
  sleep 1  # the others subscribe to the bus
  watcher=$(tmux split-window -P -F '#{pane_id}' -t "$bus" -h "$(pane 'watch run --task tasks/lift_cup.json --verbose' watcher)")
  bridge=$(tmux split-window -P -F '#{pane_id}' -t "$bus" -v "$(pane 'bridge serve 2>&1 | grep --line-buffered -v intake-idle-cap' bridge)")
  brain=$(tmux split-window -P -F '#{pane_id}' -t "$watcher" -v "$(pane 'brain serve' brain)")
  tmux set -t "$S" mouse on
  tmux set -t "$S" pane-border-status top
  tmux select-pane -t "$bus" -T bus
  tmux select-pane -t "$watcher" -T watcher
  tmux select-pane -t "$bridge" -T bridge
  tmux select-pane -t "$brain" -T brain
fi
[ -t 1 ] && exec tmux attach -t "$S"
echo "started tmux session '$S' (attach with: tmux attach -t $S)"

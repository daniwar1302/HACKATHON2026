#!/bin/sh
# Start bus, watcher, bridge and brain in one tmux session called "waldo", one pane each.
# Ctrl-b d detaches (everything keeps running); ./start.sh again reattaches; ./stop.sh stops it all.
DIR="$(cd "$(dirname "$0")" && pwd)"
S=waldo
if ! tmux has-session -t "$S" 2>/dev/null; then
  pane() { echo "cd '$DIR' && ./run.sh $1; echo; read -p '[$2 exited - Enter closes this pane] ' x"; }
  tmux new-session -d -s "$S" -x 220 -y 50 "$(pane bus bus)"
  sleep 1  # the others subscribe to the bus
  tmux split-window -t "$S" -h "$(pane 'watch run --task tasks/lift_cup.json --verbose' watcher)"
  tmux split-window -t "$S:0.0" -v "$(pane 'bridge serve' bridge)"
  tmux split-window -t "$S:0.1" -v "$(pane 'brain serve' brain)"
  tmux set -t "$S" mouse on
  tmux set -t "$S" pane-border-status top
  tmux select-pane -t "$S:0.0" -T bus
  tmux select-pane -t "$S:0.1" -T bridge
  tmux select-pane -t "$S:0.2" -T watcher
  tmux select-pane -t "$S:0.3" -T brain
fi
[ -t 1 ] && exec tmux attach -t "$S"
echo "started tmux session '$S' (attach with: tmux attach -t $S)"

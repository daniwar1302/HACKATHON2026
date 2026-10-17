#!/bin/sh
# Restart one service pane (bus | watcher | bridge | brain) in the running "waldo" tmux session.
DIR="$(cd "$(dirname "$0")" && pwd)"
case "$1" in
  bus) cmd="bus"; py="bus.py" ;;
  watcher) cmd="watch run --task tasks/lift_cup.json --verbose"; py="watch.py" ;;
  bridge) cmd="bridge serve 2>&1 | grep --line-buffered -v intake-idle-cap"; py="bridge.py" ;;
  brain) cmd="brain serve"; py="brain.py" ;;
  *) echo "usage: $0 bus|watcher|bridge|brain"; exit 1 ;;
esac
pane=$(tmux list-panes -t waldo -F "#{pane_id} #{pane_title}" | awk -v t="$1" '$2==t{print $1}')
[ -n "$pane" ] || { echo "no '$1' pane; run ./start.sh"; exit 1; }
pkill -f "python $py" 2>/dev/null  # a pane piped through grep can leave python running
sleep 1
tmux respawn-pane -k -t "$pane" "cd '$DIR' && ./run.sh $cmd; echo; read -p '[$1 exited - Enter closes this pane] ' x"
tmux select-pane -t "$pane" -T "$1"
echo "restarted $1"

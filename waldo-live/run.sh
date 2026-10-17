#!/bin/sh
# ./run.sh bus | watch ... | bridge ... | brain ...   (settings from ~/.waldo-ask/env)
DIR="$(cd "$(dirname "$0")" && pwd)"
[ -f "$HOME/.waldo-ask/env" ] && { set -a; . "$HOME/.waldo-ask/env"; set +a; }
cmd="$1"; shift
cd "$DIR" && exec "$DIR/.venv/bin/python" "$cmd.py" "$@"

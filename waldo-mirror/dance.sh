#!/bin/sh
# ./dance.sh [--dry-run] [--max-vel 1.0] [--offset 0.35]   (armed; moves only after dance_start)
DIR="$(cd "$(dirname "$0")" && pwd)"
export CYCLONEDDS_HOME="$HOME"/cyclonedds/install
export LD_LIBRARY_PATH="$CYCLONEDDS_HOME/lib:$LD_LIBRARY_PATH"
export PYTHONUNBUFFERED=1
cd "$DIR" && exec "$DIR/.venv/bin/python" dance.py "$@"

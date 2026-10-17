#!/bin/sh
# Start the Waldo MCP server on stdio. Run on Waldo; setup.sh first.
DIR="$(cd "$(dirname "$0")" && pwd)"
export CYCLONEDDS_HOME=/home/waldo/cyclonedds/install
export LD_LIBRARY_PATH="$CYCLONEDDS_HOME/lib:$LD_LIBRARY_PATH"
exec "$DIR/.venv/bin/python" "$DIR/server.py"

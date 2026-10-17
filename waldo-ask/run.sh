#!/bin/sh
# Start the waldo-ask MCP server on stdio. Settings come from ~/.waldo-ask/env (not in git).
DIR="$(cd "$(dirname "$0")" && pwd)"
[ -f "$HOME/.waldo-ask/env" ] && . "$HOME/.waldo-ask/env"
export WALDO_ASK_CHANNEL WALDO_ASK_TO WALDO_ASK_IDENTITY INKBOX_API_KEY
cd "$DIR" && exec "$DIR/.venv/bin/python" server.py

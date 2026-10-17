#!/bin/sh
# Place one test call to WALDO_OPERATOR_PHONE (settings in ~/.waldo-ask/env).
DIR="$(cd "$(dirname "$0")" && pwd)"
set -a; . "$HOME/.waldo-ask/env"; set +a
exec "$DIR/../waldo-ask/.venv/bin/python" "$DIR/test_call.py"

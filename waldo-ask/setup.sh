#!/bin/sh
# One-time setup on Waldo. Inkbox needs Python >= 3.11, so this venv is separate from the robot SDK's.
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
~/.local/bin/uv venv --clear --python 3.12 "$DIR/.venv"
~/.local/bin/uv pip install --python "$DIR/.venv/bin/python" "mcp>=1.2,<2" "inkbox==0.7.19" pytest

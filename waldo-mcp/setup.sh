#!/bin/sh
# One-time setup on Waldo: a venv on the Python that already has unitree_sdk2py, plus mcp.
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
~/.local/bin/uv venv --clear --python "$HOME"/miniconda3/envs/tv/bin/python --system-site-packages "$DIR/.venv"
~/.local/bin/uv pip install --python "$DIR/.venv/bin/python" "mcp>=1.2,<2"

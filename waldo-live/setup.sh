#!/bin/sh
# One-time setup on Waldo (Python 3.12: audioop is gone in 3.13, Inkbox needs >= 3.11).
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
~/.local/bin/uv venv --clear --python 3.12 "$DIR/.venv"
~/.local/bin/uv pip install --python "$DIR/.venv/bin/python" \
  aiohttp "inkbox==0.7.19" anthropic "mcp>=1.2,<2" pytest pytest-asyncio

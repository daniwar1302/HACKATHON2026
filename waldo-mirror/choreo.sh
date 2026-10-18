#!/bin/sh
# ./choreo.sh prepare "<song>" | worker      (runs in the mirror venv: MediaPipe + Unitree IK + yt-dlp)
DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONUNBUFFERED=1 GLOG_minloglevel=2 TF_CPP_MIN_LOG_LEVEL=2
cd "$DIR" && exec "$DIR/.venv/bin/python" choreo.py "$@"

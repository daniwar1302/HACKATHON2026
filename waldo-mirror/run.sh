#!/bin/sh
# ./run.sh [--live] [--copy] [--max-vel 1.5]   (dry run unless --live)
DIR="$(cd "$(dirname "$0")" && pwd)"
export CYCLONEDDS_HOME=/home/waldo/cyclonedds/install
export LD_LIBRARY_PATH="$CYCLONEDDS_HOME/lib:$LD_LIBRARY_PATH"
export PYTHONUNBUFFERED=1 GLOG_minloglevel=2 TF_CPP_MIN_LOG_LEVEL=2
cd "$DIR" && exec "$DIR/.venv/bin/python" mirror.py "$@"

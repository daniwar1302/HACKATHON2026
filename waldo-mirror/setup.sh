#!/bin/sh
# One-time setup on Waldo: a venv on the robot SDK's Python (tv env: unitree_sdk2py, pinocchio, casadi),
# plus MediaPipe for pose tracking. numpy stays at the tv env's version so the IK keeps working.
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
TV=/home/waldo/miniconda3/envs/tv/bin/python
~/.local/bin/uv venv --clear --python "$TV" --system-site-packages "$DIR/.venv"
NUMPY=$("$TV" -c "import numpy; print(numpy.__version__)")
~/.local/bin/uv pip install --python "$DIR/.venv/bin/python" "numpy==$NUMPY" mediapipe pytest
mkdir -p "$DIR/models"
[ -f "$DIR/models/pose_landmarker_full.task" ] || curl -sSL -o "$DIR/models/pose_landmarker_full.task" \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task
"$DIR/.venv/bin/python" -c "import mediapipe, numpy, pinocchio, casadi; print('mediapipe', mediapipe.__version__, 'numpy', numpy.__version__)"

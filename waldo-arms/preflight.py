#!/usr/bin/env python3
"""Before an agent touches the real cell: everything the waldo-arms skills depend on, read-only (no motion).
Same env vars as server.py. Exit code 0 only if every check passes.

  G1_ARM_ADDR=tcp://localhost:5810 G1_CAM_BASE=http://localhost:8766 G1_CAM_HOST=localhost \
    uv run --with "mcp>=2.2" --with mujoco --with pyzmq --with numpy --with pillow waldo-arms/preflight.py
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import server as S  # noqa: E402
import vision  # noqa: E402

results = []


def check(name, fn, fix):
    t0 = time.monotonic()
    try:
        detail = fn()
        ok = True
    except Exception as e:                      # noqa: BLE001 (report every failure, keep going)
        ok, detail = False, f"{type(e).__name__}: {str(e)[:160]}"
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name:28s} {detail}  ({(time.monotonic() - t0) * 1000:.0f} ms)")
    if not ok:
        print(f"      fix: {fix}")


def arm_state():
    a = S.arm("state", timeout_s=5)["arm"]
    fsm = a["fsm"][0] if isinstance(a["fsm"], (list, tuple)) else a["fsm"]
    assert fsm in (500, 501, 801), f"FSM {a['fsm']}: arms only take commands in 500/501/801"
    assert not a.get("frozen"), f"frozen: {a.get('freeze_reason')}"
    return f"fsm {a['fsm']}, enabled {a['enabled']}, moving {a['moving']}"


def camera():
    im = S._frame()
    assert im.shape[:2] == (480, 640), f"frame {im.shape}"
    assert im.std() > 0.02, "frame is flat (camera covered, or a stalled feed)"
    return f"head frame {im.shape[1]}x{im.shape[0]}, mean {im.mean():.2f}"


def cam_config():
    cfg = S._head_config()
    assert cfg["source"] == "live", "using fallback intrinsics: teleimager config port unreachable (tunnel 60000?)"
    return f"fx {cfg['color_intrinsics']['fx']:.0f}, depth scale {cfg['depth_scale']}"


def depth():
    d = S._depth_m()
    valid = float(((d > vision.DEPTH_RANGE[0]) & (d < vision.DEPTH_RANGE[1])).mean())
    assert valid > 0.5, f"only {valid:.0%} valid depth"
    return f"{valid:.0%} valid"


def desk_height():
    cfg = S._head_config()
    d = S._depth_m()
    T, how = S._pelvis_from_cam()
    h, w = d.shape
    zs = []
    for u in range(w // 4, 3 * w // 4, w // 8):
        z, _ = vision.depth_at(d, u, int(h * 0.62))
        if z:
            zs.append(vision.pixel_to_pelvis(u, int(h * 0.62), z, cfg["color_intrinsics"], T)[2])
    assert zs, "no depth on the lower-middle band of the image"
    z = float(np.median(zs))
    assert -0.20 < z < 0.10, f"desk at z {z:.3f} m in the pelvis frame: wrong height, or the camera model is off"
    return f"desk top ~{z:.3f} m (pelvis frame, {how}); set_table({z:.3f}) if it differs from -0.06"


def twin():
    from motion import Twin
    Twin(-0.06, kind="bare")
    return "MuJoCo twin builds (URDF + meshes present)"


def files():
    out = []
    for name, p in (("taught", S.TAUGHT_DIR), ("stations", S.STATIONS), ("gauges", S.GAUGES)):
        out.append(f"{name} {'ok' if p.exists() else 'none yet'}")
    return ", ".join(out)


check("arm_server + FSM", arm_state, "start arm_server on the Jetson; FSM 0 -> 1 -> 4 -> 500 (quickstart.md 3); ssh -L 5810")
check("head camera", camera, "open the 8766 tunnel; if stalled: systemctl --user restart g1-camera-viewer on waldo")
check("camera config", cam_config, "ssh -L 60000:robot.example:60000 (teleimager config)")
check("depth stream", depth, "ssh -L 55556:robot.example:55556; teleimager running with depth on the Jetson")
check("desk height via depth", desk_height, "check the camera mount / waist angles; look's L/R dots must sit on the grippers")
check("planner twin", twin, "./waldo-sim/fetch_assets.sh")
check("taught / registered files", files, "teach and register the cell (docs/real-cell-bringup.md)")
print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)

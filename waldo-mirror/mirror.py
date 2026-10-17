"""Mirror a person's arms onto the G1, from the phone camera.

    ./run.sh                     dry run: track the person, solve the robot's arm angles, print them. Sends nothing.
    ./run.sh --live              drive the real arms (robot must be balancing, on the gantry, with a spotter)

Stops (gliding the arms back and handing them to the stock controller) on Ctrl-C, SIGTERM,
or a `stop_requested` event on the waldo-live bus. If the person is lost for 0.5 s the arms
hold still; after 5 s they glide back to where they started.
"""
from __future__ import annotations

import argparse
import json
import signal
import threading
import time
import urllib.request
from pathlib import Path

import numpy as np

from pose import Smoother, arm_reaches, hand_targets, robot_reaches

HERE = Path(__file__).resolve().parent
MODEL = HERE / "models" / "pose_landmarker_full.task"
BUS = "http://127.0.0.1:8770"
HOLD_AFTER_S, RETURN_AFTER_S = 0.5, 5.0


class Stop(Exception):
    pass


def watch_bus(stop: threading.Event) -> None:
    """Set `stop` on any stop_requested bus event (operator said stop on a call, watcher reflex...)."""
    while not stop.is_set():
        try:
            with urllib.request.urlopen(f"{BUS}/stream?types=stop_requested", timeout=30) as resp:
                for raw in resp:
                    if raw.startswith(b"data:"):
                        print(f"bus: {json.loads(raw[5:]).get('by', '?')} requested stop")
                        stop.set()
                        return
        except Exception:
            time.sleep(1)


def frames(url: str):
    """Yield (timestamp_ms, RGB array) from the camera's latest-frame URL; None when there's no frame."""
    import cv2

    while True:
        try:
            jpeg = urllib.request.urlopen(url, timeout=1).read()
            bgr = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
            yield int(time.monotonic() * 1000), cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        except Exception:
            yield int(time.monotonic() * 1000), None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--live", action="store_true", help="drive the real arms")
    p.add_argument("--copy", action="store_true", help="your left arm drives its left arm (default: mirror)")
    p.add_argument("--max-vel", type=float, default=1.5, help="joint speed cap, rad/s (max 3)")
    p.add_argument("--camera", default="http://127.0.0.1:8773/frame/phone.jpg")
    p.add_argument("--iface", default="enp5s0")
    p.add_argument("--fps", type=float, default=10.0)
    args = p.parse_args()
    if not 0.2 <= args.max_vel <= 3.0:
        raise SystemExit("--max-vel must be between 0.2 and 3 rad/s")

    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision

    from robot import ArmIK, ArmSdk, fsm_id

    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    threading.Thread(target=watch_bus, args=(stop,), daemon=True).start()

    ik = ArmIK()
    print(f"robot reach L/R {ik.reach['left']:.2f}/{ik.reach['right']:.2f} m; mode {'copy' if args.copy else 'mirror'}; "
          f"{'LIVE' if args.live else 'DRY RUN (nothing sent to the robot)'}")

    arms = None
    q = np.zeros(14)
    if args.live:
        state = fsm_id(args.iface)
        if state not in (200, 500):
            raise SystemExit(f"robot is not balancing (FSM {state}); stand it up first. Nothing sent.")
        print("LIVE: robot on the gantry, spotter holding the controller (L1+A), nobody within reach. Starting in 5 s, Ctrl-C to abort.")
        for n in range(5, 0, -1):
            if stop.wait(1):
                return
            print(f"  {n}...")
        arms = ArmSdk(args.iface, max_vel=args.max_vel)
        arms.engage()
        q = arms.measured()
        print("arms engaged")

    landmarker = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL)), running_mode=vision.RunningMode.VIDEO, num_poses=1))
    smooth = Smoother(0.5)
    last_seen, last_print, period = time.monotonic(), 0.0, 1.0 / args.fps
    hands = ik.hand_positions(q)
    try:
        for ts, rgb in frames(args.camera):
            t0 = time.monotonic()
            if stop.is_set():
                break
            seen = False
            if rgb is not None:
                result = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)
                if result.pose_world_landmarks:
                    lm = result.pose_world_landmarks[0]
                    pts = np.array([[l.x, l.y, l.z] for l in lm])
                    vis = np.array([l.visibility or 0.0 for l in lm])
                    try:
                        reach = robot_reaches(arm_reaches(pts, vis), mirror=not args.copy)
                    except ValueError:
                        reach = {"left": None, "right": None}
                    reach = {s: smooth.update(s, r) for s, r in reach.items()}
                    seen = any(r is not None for r in reach.values())
                    if seen:
                        last_seen = time.monotonic()
                        targets = hand_targets(reach, ik.shoulder, ik.reach)
                        left = targets["left"] if targets["left"] is not None else hands["left"]
                        right = targets["right"] if targets["right"] is not None else hands["right"]
                        q = ik.solve(left, right, q)
                        hands = ik.hand_positions(q)
                        if arms:
                            arms.set_target(q)
            lost = time.monotonic() - last_seen
            if arms and lost > RETURN_AFTER_S:
                arms.set_target(arms.start)
            if time.monotonic() - last_print > 0.5:
                last_print = time.monotonic()
                deg = np.degrees(q).round().astype(int)
                status = "tracking" if lost < HOLD_AFTER_S else ("holding" if lost < RETURN_AFTER_S else "no person, arms home")
                print(f"{status:>22} | L sh p/r/y {deg[0]:+4d} {deg[1]:+4d} {deg[2]:+4d} elbow {deg[3]:+4d} | "
                      f"R sh p/r/y {deg[7]:+4d} {deg[8]:+4d} {deg[9]:+4d} elbow {deg[10]:+4d}")
            time.sleep(max(0.0, period - (time.monotonic() - t0)))
    except KeyboardInterrupt:
        pass
    finally:
        landmarker.close()
        if arms:
            print("releasing arms: back to start, then handing back to the stock controller")
            arms.release()
            print("released")


if __name__ == "__main__":
    main()

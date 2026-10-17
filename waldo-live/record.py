"""Record labeled frames from the watcher's camera, for `watch.py bench`.

    ./run.sh record --step grasp --label grip=slipping --label person=no --seconds 4

Counts down 3 s, then saves frames at 5 fps into ~/clips/<step>/ and adds them to
~/clips/<step>/labels.json, so bench can score Jev against them right away:

    ./run.sh watch bench --task tasks/lift_cup.json --step grasp \
        --frames ~/clips/grasp --labels ~/clips/grasp/labels.json
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path

from common import frame_url, load_env
from watch import Task

CLIPS = Path.home() / "clips"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", default="tasks/lift_cup.json")
    parser.add_argument("--step", required=True)
    parser.add_argument("--label", action="append", required=True, help="check=answer, repeatable")
    parser.add_argument("--seconds", type=float, default=4.0)
    parser.add_argument("--fps", type=float, default=5.0)
    args = parser.parse_args()

    task = Task.load(args.task)
    checks = {c.id: c for c in task.steps[args.step]}
    labels = dict(l.split("=", 1) for l in args.label)
    for check, answer in labels.items():
        if check not in checks or answer not in checks[check].answers:
            raise SystemExit(f"{check}={answer} isn't in step {args.step}: "
                             + "; ".join(f"{c.id}={'|'.join(c.answers)}" for c in checks.values()))

    out = CLIPS / args.step
    out.mkdir(parents=True, exist_ok=True)
    labels_file = out / "labels.json"
    all_labels = json.loads(labels_file.read_text()) if labels_file.exists() else {}

    for n in (3, 2, 1):
        print(f"{n}...", flush=True)
        time.sleep(1)
    print("recording", flush=True)
    take = time.strftime("%H%M%S")
    tag = "-".join(f"{c}_{a}" for c, a in labels.items())
    saved, deadline = 0, time.monotonic() + args.seconds
    while time.monotonic() < deadline:
        tick = time.monotonic()
        try:
            jpeg = urllib.request.urlopen(frame_url(task.camera), timeout=2).read()
        except Exception as e:
            print(f"camera: {e}")
            continue
        name = f"{tag}-{take}-{saved:03d}.jpg"
        (out / name).write_bytes(jpeg)
        all_labels[name] = labels
        saved += 1
        time.sleep(max(0.0, 1 / args.fps - (time.monotonic() - tick)))
    labels_file.write_text(json.dumps(all_labels, indent=1))
    print(f"saved {saved} frames to {out} ({len(all_labels)} labeled in total)")


if __name__ == "__main__":
    load_env()
    main()

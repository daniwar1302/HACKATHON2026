#!/usr/bin/env python3
"""The USB side of a firmware update, for the lab sim: what esptool / dfu-util is on a real board. Only works while
the board is in its bootloader (hold BOOT, tap RESET: run_task("enter_bootloader")).
  uv run --with pyzmq usb_flash.py [--offset 1100]
"""
import argparse
import json

import zmq

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--offset", type=int, default=1100, help="the lab sim's --port-offset")
a = ap.parse_args()
s = zmq.Context.instance().socket(zmq.REQ)
s.setsockopt(zmq.RCVTIMEO, 5000)
s.connect(f"tcp://127.0.0.1:{15810 + a.offset}")
s.send(json.dumps({"cmd": "sim_flash"}).encode())
r = json.loads(s.recv())
print(f"flashed: firmware v{r['firmware']} written, verify OK" if r.get("ok") else f"flash FAILED: {r.get('error')}")

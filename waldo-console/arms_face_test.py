#!/usr/bin/env python3
"""Bring both G1 hands up in front of the face, hold, hand back. Client for Michael's arm_server.py.

The server (on the Jetson, port 5810) does IK, collision/table path checks, workspace guards and
watchdogs. This client only picks targets and sequences the server's own commands:

  1. state     -> refuse unless the robot is in an arm-capable FSM
  2. ik + check -> try face-height targets (highest first); keep the first pair that solves AND passes
                   the both-arms collision check. Nothing moves in this step.
  3. prepare_enable / enable -> server takes the arms from Unitree's hold (2 s ramp)
  4. move_q    -> one synchronized move of both arms
  5. hold, move_q back to the pose captured at enable, then release -> 2 s ramp to Unitree's own hold.
     Ctrl+C or any error skips to: move back (if enabled) and release.

A heartbeat thread pings `state` every 0.5 s: the server freezes the arms if a client goes quiet ~2 s.

Reach the server from the laptop with a tunnel through waldo:
  ssh -i ~/.ssh/id_ed25519_waldo -N -L 5810:192.168.123.164:5810 waldo@100.120.182.66
Then:
  uv run --with pyzmq ~/HACKATHON2026/waldo-console/arms_face_test.py --dry   # solve + check only, no motion
  uv run --with pyzmq ~/HACKATHON2026/waldo-console/arms_face_test.py         # move
"""
import argparse
import json
import threading
import time

import zmq

ARM_FSM = {500, 501, 801}
# Pelvis frame, metres (server workspace: x 0.10..0.60, y +/-0.55, z -0.35..0.45).
# Highest/closest first: "in front of the face" is as high as the arms can reach in front of the head.
X = 0.28
HEIGHTS = (0.42, 0.38, 0.34, 0.30, 0.26)
HALF_WIDTHS = (0.14, 0.18, 0.22)
MOVE_S, HOLD_S, RAMP_S = 4.0, 3.0, 2.0


class Arm:
    def __init__(self, addr, timeout_s=30):
        self.addr, self.ctx, self.timeout_ms = addr, zmq.Context.instance(), int(timeout_s * 1000)
        self.sock = self._sock()

    def _sock(self):
        s = self.ctx.socket(zmq.REQ)
        s.setsockopt(zmq.LINGER, 0)
        s.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        s.connect(self.addr)
        return s

    def __call__(self, cmd, **kw):
        self.sock.send(json.dumps(dict(cmd=cmd, **kw)).encode())
        try:
            r = json.loads(self.sock.recv())
        except zmq.Again:
            self.sock.close()
            self.sock = self._sock()  # a REQ socket is stuck after a timeout
            raise TimeoutError(f"{cmd}: no reply from {self.addr}")
        return r


def heartbeat(addr, stop):
    hb = Arm(addr, timeout_s=2)
    while not stop.wait(0.5):
        try:
            hb("state")
        except Exception as e:  # keep trying; the server's own watchdog freezes if we really are gone
            print("  heartbeat:", e)


def solve(arm, side, xyz):
    r = arm("ik", side=side, xyz=list(xyz))
    return r.get("q7") if r.get("ok") else None


def pick_pose(arm):
    """Highest face-height pose where both arms solve and the combined pose passes the server's check."""
    for z in HEIGHTS:
        for w in HALF_WIDTHS:
            left, right = solve(arm, "left", (X, w, z)), solve(arm, "right", (X, -w, z))
            if left is None or right is None:
                print(f"  z={z:.2f} w={w:.2f}: IK did not converge")
                continue
            r = arm("check", q=left + right)
            verdict = r.get("check") or {}  # reply ok = request succeeded; check.ok = path is clear
            if not (r.get("ok") and verdict.get("ok")):
                print(f"  z={z:.2f} w={w:.2f}: rejected: {r.get('error') or verdict.get('reason')}")
                continue
            print(f"  z={z:.2f} w={w:.2f}: OK")
            return dict(z=z, w=w, q=left + right)
    return None


def fsm_of(state):
    a = state.get("arm") or {}
    f = a.get("fsm")
    return f[0] if isinstance(f, (list, tuple)) else f


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--addr", default="tcp://localhost:5810")
    ap.add_argument("--dry", action="store_true", help="solve and check only; never enable or move")
    a = ap.parse_args()

    arm = Arm(a.addr)
    st = arm("state")
    if not st.get("ok"):
        raise SystemExit(f"state failed: {st.get('error')}")
    fsm = fsm_of(st)
    print(f"server up. fsm={fsm}  left hand at {st.get('fk', {}) and st['fk'].get('grasp_xyz')}  "
          f"right hand at {st.get('fk_right', {}) and st['fk_right'].get('grasp_xyz')}")

    print("choosing a face-height pose (no motion):")
    pose = pick_pose(arm)
    if pose is None:
        raise SystemExit("no reachable, collision-free face-height pose found; nothing moved")
    print(f"target: hands at x={X} y=+/-{pose['w']} z={pose['z']} (pelvis frame)")
    if a.dry:
        print("dry run: done, nothing moved")
        return
    if fsm not in ARM_FSM:
        raise SystemExit(f"robot is in fsm {fsm}; arms need one of {sorted(ARM_FSM)}. Nothing moved.")

    stop = threading.Event()
    threading.Thread(target=heartbeat, args=(a.addr, stop), daemon=True).start()
    enabled = False
    try:
        tok = arm("prepare_enable")
        if not tok.get("ok"):
            raise RuntimeError(f"prepare_enable: {tok.get('error')}")
        r = arm("enable", prepared_token=tok["token"], ramp_s=RAMP_S)
        if not r.get("ok"):
            raise RuntimeError(f"enable: {r.get('error')}")
        enabled = True
        home = arm("state")["arm"]["q_cmd"]  # the hold pose we just took over; we return here before release
        print("enabled; moving hands up in front of the face")
        r = arm("move_q", q=pose["q"], duration=MOVE_S)
        if not r.get("ok"):
            raise RuntimeError(f"move_q: {r.get('error')}")
        print(f"holding {HOLD_S}s")
        time.sleep(HOLD_S)
    except KeyboardInterrupt:
        print("\nCtrl+C")
    except Exception as e:
        print("ERROR:", e)
    finally:
        if enabled:
            print("moving arms back down")
            try:
                r = arm("move_q", q=home, duration=MOVE_S)
                print("  back" if r.get("ok") else f"  return move refused: {r.get('error')}")
            except KeyboardInterrupt:
                print("  return move interrupted")
            print("releasing arms back to Unitree's hold")
            r = arm("release", ramp_s=RAMP_S)
            print("released" if r.get("ok") else f"release failed: {r.get('error')} -- use the remote / stop()")
        stop.set()


if __name__ == "__main__":
    main()

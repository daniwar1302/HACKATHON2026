#!/usr/bin/env python3
"""Pick benchmark: the unchanged waldo-arms MCP against waldo-sim, N randomized scenes, scored on ground truth.

Each trial: look -> find the target in the head image -> point_to_xyz -> approach from above -> grasp -> lift 15 cm.
Success = the object is held and lifted > 10 cm. Reports success rate, pointing error and refusals, so a change
to the pipeline (pointing method, approach heights, hand choice, a routine) can be measured before the real robot.

The target finder here is a colour threshold, a stand-in for the VLM; swap `find_target` for vlm_point.ground.

  (sim running)  uv run --with "mcp>=2.2" --with pyzmq --with numpy --with pillow bench_pick.py --trials 20
"""
import argparse
import asyncio
import base64
import io
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import zmq
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from PIL import Image

HERE = Path(__file__).resolve().parent
SIM_ENV = {"G1_ARM_ADDR": "tcp://127.0.0.1:15810", "G1_CAM_BASE": "http://127.0.0.1:18766", "G1_CAM_HOST": "127.0.0.1",
           "G1_DEPTH_PORT": "15556", "G1_CAMCFG_PORT": "16000", "G1_ROUTINES": str(HERE / "routines")}
COLOURS = {"red_cube": lambda im: (im[..., 0] > 150) & (im[..., 1] < 90) & (im[..., 2] < 90),
           "green_cube": lambda im: (im[..., 1] > 150) & (im[..., 0] < 90) & (im[..., 2] < 110),
           "blue_cube": lambda im: (im[..., 2] > 150) & (im[..., 0] < 90) & (im[..., 1] < 120)}


def find_target(rgb: np.ndarray, target: str):
    """Pixel centroid of the target colour, or None. Stand-in for VLM grounding."""
    m = COLOURS[target](rgb.astype(int))
    return None if m.sum() < 20 else tuple(np.argwhere(m).mean(0)[::-1])


async def trial(s, sim, seed, target):
    async def tool(n, a={}):
        r = await s.call_tool(n, a)
        if r.is_error:
            raise RuntimeError(r.content[0].text.replace("Error executing tool ", ""))
        return r
    J = lambda r: json.loads(r.content[0].text)
    sim("sim_reset", seed=seed)
    rgb = np.asarray(Image.open(io.BytesIO(base64.b64decode((await tool("look", {"annotate": False})).content[0].data))).convert("RGB"))
    px = find_target(rgb, target)
    if px is None:
        return {"seed": seed, "ok": False, "why": "target not visible"}
    p = J(await tool("point_to_xyz", {"points": [list(px)]}))["points"][0]
    truth = np.array(sim("sim_objects")["objects"][target]["xyz"])
    if p["xyz"] is None:
        return {"seed": seed, "ok": False, "why": "no depth at target"}
    x, y, z = p["xyz"]
    err_xy = float(np.linalg.norm(np.array([x, y]) - truth[:2]))
    side = "left" if y > 0 else "right"
    try:
        await tool("enable")
        await tool("move_hand", {"side": side, "x": x, "y": y, "z": z + 0.10, "seconds": 2})
        await tool("move_hand", {"side": side, "x": x, "y": y, "z": z - 0.005, "seconds": 2})
        await tool("gripper", {"side": side, "action": "close"})
        await tool("move_hand", {"side": side, "x": x, "y": y, "z": z + 0.15, "seconds": 2})
        o = sim("sim_objects")["objects"][target]
        ok = o["held_by"] == side and o["lifted_m"] > 0.10
        return {"seed": seed, "ok": ok, "side": side, "err_xy_mm": round(err_xy * 1000, 1), "lifted_cm": round(o["lifted_m"] * 100, 1),
                "why": "" if ok else "not lifted"}
    except RuntimeError as e:
        return {"seed": seed, "ok": False, "side": side, "err_xy_mm": round(err_xy * 1000, 1), "why": str(e)[:90]}
    finally:
        for t in (("gripper", {"side": side, "action": "open"}), ("release", {})):
            try:
                await tool(*t)
            except RuntimeError:
                pass


async def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--target", default="red_cube", choices=sorted(COLOURS))
    a = ap.parse_args()
    z = zmq.Context().socket(zmq.REQ)
    z.connect(SIM_ENV["G1_ARM_ADDR"])
    def sim(cmd, **kw):
        z.send(json.dumps(dict(cmd=cmd, **kw)).encode())
        return json.loads(z.recv())
    server = str(HERE.parent / "waldo-arms" / "server.py")
    t0, rows = time.time(), []
    async with stdio_client(StdioServerParameters(command=sys.executable, args=[server], env={**os.environ, **SIM_ENV})) as (rd, wr), ClientSession(rd, wr) as s:
        await s.initialize()
        await s.call_tool("set_table", {"z": -0.12})
        for seed in range(a.trials):
            r = await trial(s, sim, seed, a.target)
            rows.append(r)
            print(f"seed {seed:3d}  {'OK ' if r['ok'] else 'FAIL'}  {r.get('side', ''):5s}  err_xy {r.get('err_xy_mm', '-'):>5} mm  {r.get('why', '')}", flush=True)
    ok = sum(r["ok"] for r in rows)
    errs = [r["err_xy_mm"] for r in rows if "err_xy_mm" in r]
    print(f"\n{a.target}: {ok}/{len(rows)} picked ({100 * ok / len(rows):.0f}%), pointing error xy median {np.median(errs):.1f} mm, "
          f"max {max(errs):.1f} mm, {time.time() - t0:.1f} s total")


if __name__ == "__main__":
    asyncio.run(main())

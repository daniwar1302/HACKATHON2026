#!/usr/bin/env python3
"""Local VLM grounding for hand targets: "where is the red cup?" -> box + point on the head image (0-1000 scale).

Uses NVIDIA Cosmos-Reason2-2B (Qwen3-VL based, cached on Waldo) with its 2D grounding prompt. Output coordinates
are normalized 0-1000, so they go straight into waldo-arms `point_to_xyz(points, scale="norm1000")`.

Measured on Waldo's RTX 5060 (2026-10-17): load 2.5 s, 4-4.5 GiB VRAM, ~5 s per query. Whole objects ground well
("the robot" -> the right box); small parts ("the robot's left hand") came back unusable. Speed-up plan in
docs/vlm-hand-tasks.md. Keep the model loaded (import this module once) rather than calling the CLI per query.

On Waldo, with the Isaac-GR00T venv (torch 2.9 cu128, transformers 4.57, flash-attn):
  ~/Isaac-GR00T/.venv/bin/python vlm_point.py "the red cup"            # head camera frame
  ~/Isaac-GR00T/.venv/bin/python vlm_point.py "the red cup" frame.jpg  # any image
"""
from __future__ import annotations

import glob
import io
import json
import os
import re
import sys
import time
import urllib.request

CAM = os.environ.get("G1_CAM_BASE", "http://10.2.1.180:8766")
MODEL = os.environ.get("G1_VLM", "") or (sorted(glob.glob(os.path.expanduser(
    "~/.cache/huggingface/hub/models--nvidia--Cosmos-Reason2-2B/snapshots/*/")))[-1:] or ["nvidia/Cosmos-Reason2-2B"])[0]

_model = _proc = None


def load():
    """Load once and keep resident on the GPU (the expensive part)."""
    global _model, _proc
    if _model is None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        _proc = AutoProcessor.from_pretrained(MODEL)
        _model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
    return _model, _proc


def head_image():
    from PIL import Image
    with urllib.request.urlopen(f"{CAM}/frame/head_camera.jpg", timeout=5) as r:
        return Image.open(io.BytesIO(r.read())).convert("RGB")


def parse(text: str) -> list[dict]:
    """Model output (often wrapped in ```json fences) -> [{"label", "bbox_2d"[x1,y1,x2,y2], "point"[x,y]}], 0-1000."""
    m = re.search(r"\[[\s\S]*\]", text)
    try:
        items = json.loads(m.group(0)) if m else []
    except ValueError:
        items = []
    out = []
    for it in items if isinstance(items, list) else []:
        b = it.get("bbox_2d") if isinstance(it, dict) else None
        if isinstance(b, list) and len(b) == 4 and all(isinstance(v, (int, float)) for v in b):
            out.append({"label": it.get("label"), "bbox_2d": b, "point": [(b[0] + b[2]) / 2, (b[1] + b[3]) / 2]})
    return out


def ground(query: str, image=None) -> dict:
    """Find `query` in the image (head camera by default). Returns boxes and center points on the 0-1000 scale.
    For a top-down grasp the box center is a fair point; for a handle or rim, ask for that part explicitly."""
    import torch
    model, proc = load()
    image = image or head_image()
    msgs = [{"role": "user", "content": [{"type": "image", "image": image},
            {"type": "text", "text": f"Locate the bounding box of {query}. Return a json."}]}]
    inputs = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True,
                                      return_tensors="pt").to(model.device)
    t0 = time.time()
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=96, do_sample=False)
    text = proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    return {"query": query, "found": parse(text), "raw": text.strip(), "ms": round((time.time() - t0) * 1000),
            "scale": "norm1000", "image_size": list(image.size)}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    if sys.argv[1] == "--self-check":   # parser only, no GPU
        assert parse('```json\n[{"bbox_2d": [215, 105, 415, 992], "label": "robot"}]\n```')[0]["point"] == [315.0, 548.5]
        assert parse('[{"image": "0.34", "label": "x"}]') == [] and parse("no json") == []
        print("parse ok")
        sys.exit(0)
    img = None
    if len(sys.argv) > 2:
        from PIL import Image
        img = Image.open(sys.argv[2]).convert("RGB")
    print(json.dumps(ground(sys.argv[1], img), indent=2))

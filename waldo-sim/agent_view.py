#!/usr/bin/env python3
"""Pretty-print `claude -p --output-format stream-json` live: the agent's words and every robot tool call, for a
demo audience. Reads stdin."""
import json
import os
import sys
import time

t0 = time.time()
log = open(os.environ["AGENT_LOG"], "a") if os.environ.get("AGENT_LOG") else None   # timestamped raw events (video captions)
for line in sys.stdin:
    if log:
        log.write(f"{time.time():.2f}\t{line}")
        log.flush()
    try:
        d = json.loads(line)
    except ValueError:
        continue
    ts = f"\033[2m{time.time() - t0:5.0f}s\033[0m"
    if d.get("type") == "assistant":
        for c in d["message"].get("content", []):
            if c.get("type") == "text" and c["text"].strip():
                print(f"{ts} \033[1m{c['text'].strip()}\033[0m", flush=True)
            elif c.get("type") == "tool_use":
                name = c["name"].replace("mcp__waldo-arms__", "")
                args = ", ".join(f"{k}={json.dumps(v)}" for k, v in c["input"].items())
                print(f"{ts} \033[36m→ {name}({args[:110]})\033[0m", flush=True)
    elif d.get("type") == "user":
        for c in d["message"].get("content", []):
            if c.get("type") == "tool_result":
                body = c.get("content")
                text = body if isinstance(body, str) else " ".join(x.get("text", "[image]") if x.get("type") == "text" else "[image]" for x in (body or []))
                print(f"{ts}   \033[2m{text.strip()[:140]}\033[0m", flush=True)
    elif d.get("type") == "result":
        print(f"\n\033[1;32m== done: {d.get('num_turns')} turns, {d.get('duration_ms', 0) / 1000:.0f} s, ${d.get('total_cost_usd', 0):.2f}\033[0m\n")
        print(d.get("result", ""))

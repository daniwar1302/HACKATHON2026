# waldo-live

The live loop from [`docs/live-call-framework.md`](../docs/live-call-framework.md): a Jev camera watcher, a Claude
brain that runs the robot task, and a phone-call bridge (Inkbox ⇄ OpenAI Realtime) that reacts to what the cameras
see while you're on the call. Everything talks through a small event bus on Waldo's localhost.

| File | What it is | Port |
|------|------------|------|
| `bus.py` | Event bus. `POST /events`, `GET /stream` (SSE), `GET /recent`; logs to `~/.waldo/events.jsonl` | 127.0.0.1:8770 |
| `watch.py` | Camera → Jev checks → debounced `check_failed` / `check_cleared` events; reflex stop on critical checks | — |
| `deciders.py` | `JevDecider` (~0.08 s/frame), `ClaudeDecider` fallback (Haiku 5.5) | — |
| `bridge.py` | Places the call, relays audio to OpenAI Realtime, injects bus events mid-call, passes instructions to the brain | 127.0.0.1:8771, public via Inkbox tunnel |
| `brain.py` | Claude agent loop (Opus 5.5 plans, Haiku 5.5 answers the voice agent) over the robot's MCP tools | 127.0.0.1:8772 |
| `tasks/lift_cup.json` | Watcher checks per step: approach / grasp / lift | — |

## Setup (on Waldo)

```sh
cd ~/HACKATHON2026/waldo-live && ./setup.sh && .venv/bin/python -m pytest -q
./run.sh bridge setup-signing-key     # once: Inkbox signs call WebSockets, the bridge verifies them
```

Settings live in `~/.waldo-ask/env` (mode 600, never in git):

| Variable | Used by | |
|----------|---------|---|
| `INKBOX_API_KEY`, `WALDO_ASK_IDENTITY`, `WALDO_OPERATOR_PHONE`, `WALDO_INKBOX_SIGNING_KEY` | bridge | ✅ set |
| `TYPESAFE_API_KEY` | watcher | ✅ set |
| `OPENAI_API_KEY` | bridge (voice) | ✅ set, **account needs credits** |
| `ANTHROPIC_API_KEY` | brain | ❌ needed |
| `WALDO_STOP_CMD` | watcher reflex + `stop_robot` on calls | ❌ needs the robot's stop command |

The brain's robot tools come from `~/.waldo/brain.json`:

```json
{"servers": {"g1": {"command": "/home/waldo/HACKATHON2026/waldo-mcp/run.sh"}}, "stop_tool": "g1__stop"}
```

## Run

Four terminals (or tmux panes):

```sh
./run.sh bus
./run.sh watch run --task tasks/lift_cup.json --verbose
./run.sh bridge serve
./run.sh brain serve
```

Then start a task: `./run.sh brain task "lift the red cup" --steps approach,grasp,lift`.
Test the call path on its own: `./run.sh bridge call "the cup slipped twice" "should I grab it by the handle or the rim?"`.

## Phase 0: is Jev good enough?

Record frames of good and failed grasps into a folder, label a few, and measure:

```sh
./run.sh watch bench --task tasks/lift_cup.json --step grasp --frames ~/clips/grasp --labels ~/clips/grasp/labels.json
# labels.json: {"0001.jpg": {"grip": "slipping"}, "0002.jpg": {"grip": "secure"}}
```

If accuracy is poor, run the watcher with `--decider claude --fps 1`.

## Measured so far (2026-10-17)

- Jev on the room webcam: **0.07–0.12 s per frame**, steady at 5 fps, nothing firing on an idle bench.
  The grip check still needs tuning on real clips (it reads "secure ~0.7" with nothing happening).
- Inkbox tunnel `waldo-agent.inkboxwire.com` is active; signing key created.
- OpenAI Realtime: key valid, `gpt-realtime-2.1` available, but the account has **no credits**.

## Safety

- The watcher reflex and `stop_robot` are conveniences. The hand controller, firmware limits and the
  robot servers' own checks are the real safety layer.
- Only the call WebSocket is public. It needs a one-time token from the call the bridge placed **and** a
  valid Inkbox signature. Calls are only placed from the local bus, and only to `WALDO_OPERATOR_PHONE`.
- The voice agent can stop the robot and pass instructions. It cannot move the robot; only the brain can,
  through the robot servers.

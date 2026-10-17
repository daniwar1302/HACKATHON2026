# Live call + Jev watcher: framework

Goal: when a robot task goes wrong, Waldo notices within about 1 s, stops, and calls the operator. During
the call the voice agent keeps reacting to what the cameras see ("the cup just slipped again") and passes
the operator's instructions to the robot's agent.

## Components

```
                ┌──────────────────────────── Waldo ────────────────────────────┐
 room webcam ──►│ 1. WATCHER (waldo-watch)                                     │
 head camera    │    frames @4–5 fps ──► Jev checks ──► debounce ──► events     │
                │                                     │                        │
                │                         critical? ──┴──► REFLEX: stop robot  │
                │                                     ▼                        │
                │ 2. EVENT BUS  (localhost only: POST /events, GET /stream)     │
                │        ▲                    │                    │           │
                │        │ progress           ▼                    ▼           │
                │ 3. BRAIN (waldo-brain)              4. CALL BRIDGE (waldo-call)
                │    Claude agent loop                   Inkbox call WS ⇄ OpenAI Realtime
                │    robot MCP tools ──► WidowX / G1     tools: consult_robot, robot_status,
                │    waldo-ask hints                     give_instruction, stop_robot, hang_up
                └────────────────────────────────────────────────┬─────────────┘
                                                                 │ wss via Inkbox tunnel
                                                         Inkbox phone ──► operator's phone
```

| # | Component | Runs | Job |
|---|-----------|------|-----|
| 1 | **Watcher** | Waldo, always on during a task | Turns video into a few typed, debounced events. Never writes text, never plans |
| 2 | **Event bus** | Waldo, `127.0.0.1` only | Tiny pub/sub that connects everything; also the log |
| 3 | **Brain** | Waldo | Claude agent loop that runs the task with the robot tools, decides when to escalate, and takes operator instructions |
| 4 | **Call bridge** | Waldo, reached through the Inkbox tunnel | Phone audio ⇄ OpenAI Realtime voice agent. Puts watcher and brain events into the call, and passes the operator's words to the brain |

Split of models: **Jev** decides fast (is this frame OK?), **Claude** thinks (what to do about it),
**Realtime** talks (keeps the conversation natural while Claude thinks).

## 1. Watcher

Each task step comes with a few **checks**: a Jev decision question with a fixed answer set and a "bad" answer.
The brain sets the current step; the watcher only runs that step's checks.

```yaml
task: lift the red cup
steps:
  approach: [{id: progress, q: "Is the gripper moving toward the red cup?", answers: [yes, no, unclear], bad: no}]
  grasp:    [{id: grip, q: "Is the red cup held firmly between the gripper fingers?", answers: [secure, slipping, missed], bad: [slipping, missed]}]
  lift:     [{id: lifted, q: "Is the red cup off the table and still in the gripper?", answers: [yes, falling, no], bad: [falling, no]},
             {id: person, q: "Is a person's hand or body within reach of the arm?", answers: [no, yes], bad: yes, severity: critical}]
```

Loop:
1. Grab the latest frame (`/frame/waldo_webcam.jpg`), downscale to about 320 px, JPEG.
2. Send all of the current step's checks to Jev in parallel. At most 2 frames in flight. **Drop frames, never queue them**, so results always describe the present.
3. Smooth P(bad) per check with an exponential moving average.
4. **Fire** when the smoothed P(bad) is ≥ 0.7 for 3 frames in a row (about 0.6 s at 5 fps). **Clear** when it falls back below 0.4. Hysteresis means a single noisy frame doesn't trigger anything.
5. Publish an event with a snapshot.

```json
{"type": "check_failed", "check": "grip", "answer": "slipping", "p": 0.86, "severity": "warn",
 "task": "lift the red cup", "step": "grasp", "t": 1760731200.42, "frame": "events/0142.jpg"}
```

**Reflex:** `severity: critical` makes the watcher call the robot's `stop` itself, with no LLM in the loop
(about 50 ms). This is a convenience, **not the safety system**. Hard safety stays local and deterministic:
the hand controller, firmware limits, and depth sensing.

Jev's image input is experimental. If it isn't accurate or fast enough, use the same interface with
Claude Haiku 5.5 at about 1 fps (slower, but smarter).

## 2. Event bus

A small HTTP server on `127.0.0.1`: `POST /events` publishes, `GET /stream` is Server-Sent Events.
Every event is also appended to `~/.waldo/events.jsonl`, which is the record of each run.

Event types: `check_failed`, `check_cleared`, `step_changed`, `robot_stopped`, `brain_progress`,
`operator_instruction`, `call_started`, `call_ended`.

## 3. Brain

Our own Claude agent loop using the Anthropic API, not Claude Code, so we control interrupts and model choice:

- Runs the task through the robot MCP tools (`widowx`, `waldo-mcp`). Publishes `step_changed` so the watcher knows which checks to run.
- **Between tool calls** it reads new events. A `check_failed` interrupts the plan: stop, then decide. It can retry with a trusted hint from `waldo-ask` memory, or escalate.
- **Escalation:** if a call is already live, publish the problem (the bridge brings it up in the call). If not, place an Inkbox call with a purpose: "Lifting the red cup. The grip slipped twice. Ask where to grasp it."
- `operator_instruction` events from the call become a user turn ("Operator says: use the handle, left hand"). Claude answers with a plan, and the bridge reads it back for confirmation.
- Model split: Haiku 5.5 for routine steps and call consults, the bigger model only for replanning.
- Saves each instruction and its outcome as a `waldo-ask` hint, so the same problem is solved from memory next time.

## 4. Call bridge

Port the patterns from `inkbox-ai/hermes-agent-plugin` `realtime.py`, swapping the Hermes parts for our brain.

**Setup:** `identity.place_call(to_number=OPERATOR, client_websocket_url="wss://<inkbox-tunnel>/call/<id>")`.
Inkbox connects its media WebSocket. The bridge opens OpenAI Realtime (`gpt-realtime-2`), sends `session.update`
(instructions, tools, PCM audio), and relays audio both ways, resampling 16 kHz ⇄ 24 kHz. Plain text
or iMessage carries the snapshot photo at call start.

**Tools the voice agent gets:**

| Tool | Does | Latency |
|------|------|---------|
| `robot_status()` | Current task, step, last events, from the bus | instant |
| `consult_robot(query)` | Asks the brain (Claude) and speaks the answer; filler speech while it waits | 2–5 s |
| `give_instruction(text)` | Publishes `operator_instruction` for the brain | instant |
| `stop_robot()` | Calls the robot `stop` directly; "stop" on the call never waits on Claude | about 100 ms |
| `hang_up()` | Says goodbye and ends the call | |

**Live injection** (the reactive part): the bridge subscribes to the bus while the call is up.

| Event | Injection |
|-------|-----------|
| `check_failed` warn | `conversation.item.create` with a system note `"[camera] grip slipping (p=0.86)"`, then `response.create` once the agent stops speaking |
| `check_failed` critical / `robot_stopped` | `response.cancel` to stop mid-sentence, then inject and speak immediately |
| `brain_progress` | Note only; the agent mentions it when relevant ("trying the handle now") |

Rate limit: one spoken injection every 3 s, and the same check isn't repeated while it's still failing.

**Security:** only call the configured operator number, and reject or ignore other inbound calls. Verify
Inkbox's HMAC signatures on webhooks and the WebSocket. The public endpoint exposes only the bridge;
the robot tools and the event bus stay on localhost.

## Timeline: cup slips mid-call

| t | What happens |
|---|---|
| 0.0 s | Cup starts slipping |
| ~0.2 s | Next frame captured |
| ~0.5 s | First Jev result: slipping 0.81 *(latency to be measured)* |
| ~0.9 s | Third bad frame in a row → `check_failed` |
| ~1.0 s | Brain stops the arm; bridge injects the event |
| ~1.5 s | Voice agent: "Hold on, the cup's slipping again, I've stopped the arm." |
| ~4 s | Operator: "Grab it by the handle." → `give_instruction` |
| ~6 s | Brain (Claude) plans; voice agent reads it back: "Handle, left side, slower. Go?" "Yes." |
| ~7 s | Retry; the watcher now runs the grasp checks against the new plan |

If no call is live, add 5–15 s for the phone to ring.

## Build order

| Phase | What | Rough time |
|-------|------|-----------|
| 0 | **Jev benchmark**: real latency and accuracy on recorded WidowX cup clips. Go/no-go for Jev vs Haiku | 2 h |
| 1 | Watcher + event bus + reflex stop, tested on recordings | 3 h |
| 2 | Call bridge: outbound call, audio relay, `consult_robot`, `stop_robot` | 4–6 h |
| 3 | Live injection + brain loop + instruction flow | 3 h |
| 4 | End-to-end rehearsal on the real arm, thresholds tuned | 2 h |

Phases 0–1 and 2 are independent and can be built in parallel.

## Needs

- TypeSafe / Jev API key, OpenAI API key (Realtime), Inkbox API key with a phone number
- Recorded clips of the task (good and failed grasps) for phase 0
- Bay owner's OK for the robot and Claude setup on Waldo

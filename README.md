# HACKATHON2026: Waldo

**Waldo gives an AI agent eyes that react in a tenth of a second, a phone number to call you when it's unsure,
and a memory so it needs you less each time.**

Waldo is a lab workstation driving robot hardware: WidowX AI arms (Trossen) and a Unitree G1 humanoid. This repo
lets a Claude agent run physical tasks like "lift the red cup", notice within about a second when something goes
wrong, stop, phone the operator, talk it through live while the cameras keep watching, and remember the answer.

## The flow

```
                                 ┌──────────────────────── WALDO (the lab PC) ────────────────────────┐
                                 │                                                                    │
 you: "lift the red cup" ───────►│  BRAIN  (Claude Opus 5.5)                                          │
                                 │   plans the task, step by step ───────► robot tools (MCP) ──► 🦾 arm
                                 │   │  ▲              ▲                                              │
                                 │   │  │ alerts       │ your instructions                            │
                                 │   │  │              │                                              │
                                 │   ▼  │       ┌──────┴──────┐                                       │
                                 │  set_step    │  EVENT BUS  │◄──────────────┐                       │
                                 │   │          └──────▲──────┘               │                       │
                                 │   ▼                 │ "grip: slipping 0.86"│                       │
  📷 room camera ───5 fps───────►│  WATCHER  ──► Jev (0.08 s/frame) ──► debounce (~0.6 s)             │
                                 │                     │                      │                       │
                                 │       critical? ────┴──► REFLEX STOP (no AI, ~50 ms)               │
                                 │                                            │                       │
                                 │  CALL BRIDGE ◄─────── call_requested ──────┘                       │
                                 │   OpenAI Realtime voice  ◄── live camera alerts injected mid-call   │
                                 │   tools: status · consult brain · give instruction · stop · hang up│
                                 └──────┬─────────────────────────────────────────────────────────────┘
                                        │ Inkbox (tunnel + phone line)
                                        ▼
                              📱 your phone: "The cup slipped twice. Handle or rim?"
                                    you: "Handle, left hand."  ──► saved as a HINT
                                                                    (next time: no call needed)
```

| t | What happens |
|---|---|
| 0 s | The cup slips |
| 0.6 s | Jev flags it 3 frames in a row → alert |
| 1 s | Arm stopped |
| ~5 s | Your phone rings |
| — | You answer; the call keeps reacting to the camera |
| ~10 s after you answer | Claude has a new plan and reads it back; you say "go" |
| next time | The hint solves it with no call |

## Why this is built for agents

All of this exists so **an AI agent can do physical work without a person watching it**. A human operator wouldn't
need most of it.

1. **Agents are slow and can't watch.** Claude takes seconds per decision and can't look at video 5 times a second.
   Jev is a fast, cheap "eye" that turns video into a few typed events, so the agent only wakes up when something
   actually changes.
2. **The agent can tell when it's unsure.** Jev returns probabilities ("slipping, 0.86"), which gives the agent a
   measurable "I'm confused" signal instead of finding out after failing several times.
3. **Asking a human is a tool the agent calls.** `call_operator` / `ask_human` replace constant supervision. Inkbox gives
   the agent its own identity (phone line, email, iMessage), so it reaches you the way a coworker would.
4. **The conversation is live on both ends.** The voice agent is connected to the robot's sensors and tells you what's
   happening as it happens. Your words go straight back to the planning agent, while the robot stays stopped.
5. **Every human answer makes the agent more independent.** Answers become hints with a success score; advice that
   worked is reused automatically. The metric that matters is how rarely it needs you.
6. **It plugs into any agent.** Robot, camera checks and the human channel are tools (MCP servers and a local event
   bus). Our brain uses them today; Claude Code, Hermes or another agent could use the same tools.
7. **The safety layers don't depend on the AI.** The reflex stop, the robot tools' limits and the hand controller don't
   rely on any model being right.

## What's in the repo

| Folder | What it does |
|--------|--------------|
| [`waldo-live/`](waldo-live/) | The live loop: event bus, Jev watcher, Inkbox ⇄ OpenAI Realtime call bridge, Claude brain, clip recorder |
| [`waldo-ask/`](waldo-ask/) | MCP server for asking the operator by email, iMessage, SMS or phone, with the hint memory |
| [`waldo-mcp/`](waldo-mcp/) | MCP server for the G1 humanoid over the Unitree SDK: stand, gestures, stop, cameras (no walking) |
| [`waldo-call/`](waldo-call/) | One-shot test call through Inkbox's hosted voice agent |
| [`docs/live-call-framework.md`](docs/live-call-framework.md) | Detailed design: components, event schema, timings, security |
| [`docs/recording-guide.md`](docs/recording-guide.md) | Shot list for recording labeled clips to tune Jev |
| [`docs/robot-survey.md`](docs/robot-survey.md) | What's on Waldo: hardware, network, existing control code, safety rules |

## Running it

Everything runs on Waldo. From your Mac:

```sh
ssh -t waldo@100.120.182.66 '~/HACKATHON2026/waldo-live/start.sh'      # bus, watcher, bridge, brain in one tmux window
ssh waldo@100.120.182.66 'cd ~/HACKATHON2026/waldo-live && ./run.sh brain task "lift the red cup"'
ssh waldo@100.120.182.66 '~/HACKATHON2026/waldo-live/stop.sh'
```

In tmux: click a pane to focus it, **Ctrl-b then d** to leave everything running. Details are in [`waldo-live/README.md`](waldo-live/README.md).

## Status (2026-10-17)

| Piece | State |
|-------|-------|
| Jev watcher | ✅ Live on the room webcam: 5 fps, 0.07–0.12 s per frame. Grip check needs tuning on recorded clips |
| Event bus, start/stop scripts | ✅ Running on Waldo |
| Call bridge | ✅ Built, tested with fakes; Inkbox tunnel `waldo-agent.inkboxwire.com` active, signing key set |
| Phone calls | ✅ Test call to the operator worked (Inkbox shared iMessage line, transcript read back) |
| **Phone commands (MVP)** | ✅ **Text "call me" → Waldo calls → "wave" → "Go ahead?" → "Yes" → the G1 waves** (~1 s from yes to motion) |
| Brain | ✅ Built, tested with a fake model; runs on Waldo |
| Tests | ✅ 12 (`waldo-ask`) + 14 (`waldo-live`) passing on Waldo |

**Still needed:**

- [x] OpenAI credits for the phone voice
- [ ] Anthropic API key with credits ($10–20) for the brain
- [ ] From the bay owner: the `widowx` MCP server command, a safe arm-stop command, and an OK to use the robot
- [ ] Recorded clips of good and failed grasps ([guide](docs/recording-guide.md)) to tune the watcher

Keys live only on Waldo in `~/.waldo-ask/env` (mode 600). Never commit them: this repo is public.

## Connecting to Waldo

1. Install [Tailscale](https://tailscale.com/download), sign in, and accept the invite for the machine `waldo-waldos`.
2. Open an SSH session that also forwards the camera stream:

   ```sh
   ssh -L 8766:10.2.1.180:8766 waldo@100.120.182.66
   ```

   Ask the team for the password.
3. While that session is open, go to <http://localhost:8766/> to see the head camera and the room webcam.

Closing the SSH session also cuts off the camera stream. If you can't connect, check that Tailscale says
**Connected** and that `waldo-waldos` is in its machine list.

## Safety

- Someone holds the hand controller or stays within reach of the arms' stop during every run.
- No walking on the G1 (site rule). The G1 server has no walking tool.
- "Person within reach" shots and tests happen only with the arm parked or its motors off.
- Waldo is shared: check with the bay owner before using the robot.

# waldo-ask

An MCP server that lets Waldo's Claude ask the operator for help by email, iMessage or SMS
(through [Inkbox](https://inkbox.ai)), then **remembers the answers** so it stops needing to ask.
It works next to the robot servers (`widowx`, `waldo-mcp`) and never moves a robot itself.

## Why it makes Claude faster

| Mechanism | Effect |
|-----------|--------|
| Hint memory: every answer is saved with its task and situation | The same problem next time is answered from memory in milliseconds, with no message and no wait |
| `report_outcome` after acting | Advice that worked becomes trusted and is reused automatically; advice that failed drops out |
| Best hints in the server instructions | Claude knows past advice before its first tool call |
| `recall_hints` at the start of a task | Claude plans with past advice instead of discovering the problem again |
| Reply checked every 2 s | Your answer reaches Claude about 2 s after you send it |
| Specific questions ("handle or rim?") | One short reply instead of a back-and-forth |

Advice is reused from memory only when it is **trusted** (it worked at least once, and more often than it
failed) and matches the situation closely enough. Otherwise Claude asks a person.

## Tools

| Tool | What it does |
|------|--------------|
| `recall_hints(task)` | Past advice for similar tasks |
| `ask_human(task, situation, question, camera, wait_seconds, force_human)` | Answers from trusted memory, or messages you with a camera photo and waits up to 120 s |
| `wait_for_reply(ask_id)` | Keeps waiting if `ask_human` returned `waiting` |
| `report_outcome(hint_id, worked, note)` | Marks advice as working or not |
| `notify_human(message)` | One-way update |

A reply that starts with **STOP / abort / cancel** returns `status: abort`, which tells Claude to stop and end the task.

## Safety

- Only replies from the configured operator address or number count. They must also arrive after the question,
  in the same thread or carrying its code (e.g. `[A7F2]`).
- The robot stays stopped while waiting. No reply means no action.
- Replies are instructions *to Claude*, so they still go through the robot servers' own limits and approvals.

## Setup on Waldo

```sh
cd ~/HACKATHON2026/waldo-ask && ./setup.sh && .venv/bin/python -m pytest -q
mkdir -p ~/.waldo-ask && chmod 700 ~/.waldo-ask
cat > ~/.waldo-ask/env <<'EOF'
INKBOX_API_KEY=ApiKey_...
WALDO_ASK_IDENTITY=waldo        # Inkbox agent handle
WALDO_ASK_CHANNEL=email         # email | imessage | sms
WALDO_ASK_TO=you@example.com    # or +15551234567 for imessage/sms
EOF
chmod 600 ~/.waldo-ask/env
```

The Inkbox agent identity (`waldo`) must exist with a mailbox, phone number or iMessage line, depending on the channel.
Create it in the Inkbox Console, or with `Inkbox().create_identity("waldo")`.

Register it with Claude Code on Waldo:

```sh
claude mcp add waldo-ask -- /home/waldo/HACKATHON2026/waldo-ask/run.sh
```

Hints live in `~/.waldo-ask/hints.json` on Waldo, not in git.

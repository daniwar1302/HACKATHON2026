# waldo-mcp

An MCP server that lets Claude operate Waldo's Unitree G1. It runs on Waldo and talks to the robot
directly through `unitree_sdk2py` over DDS (`enp5s0`). Frames come from the camera viewer on port 8766.

## Tools

| Tool | What it does |
|------|--------------|
| `get_state` | Locomotion state (FSM id/name, whether it's balancing) |
| `look(camera)` | One JPEG from `head_camera` or `waldo_webcam` |
| `stand_up(supported)` | Limp → damp → locked stand → balance, verifying each step. `supported=true` means the operator confirmed the gantry or feet on the floor |
| `gesture(name)` | Built-in arm gesture (wave, clap, high five, heart, …). Only works while balancing |
| `stop` | StopMove and hold a balanced stand. Always safe |
| `damp` | Emergency only: motors go limp |

There is no walking tool (site rule).

## Setup

On Waldo, once:

```sh
git clone https://github.com/daniwar1302/HACKATHON2026.git ~/HACKATHON2026
~/HACKATHON2026/waldo-mcp/setup.sh
```

To update later: `git -C ~/HACKATHON2026 pull`.

On your Mac (needs SSH key login to Waldo, e.g. `ssh-copy-id operator@lab-pc.example`):

```sh
claude mcp add waldo -- ssh operator@lab-pc.example /home/operator/HACKATHON2026/waldo-mcp/run.sh
```

Then ask Claude Code something like "check Waldo's state and show me the room camera." Claude Code asks you
to approve each tool call, and that approval is the operator confirmation. Don't auto-allow
`stand_up`, `gesture` or `damp`.

## Safety

- Someone holds the hand controller the whole time. **L1+A** damps the robot no matter what this server does.
- Use the gantry for the first runs.
- Check that nobody else is using the robot. Waldo is shared.

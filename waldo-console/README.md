# waldo-console

Laptop-side controls for the G1. Nothing gets installed on Waldo: every file here runs on **your** machine and
reaches the robot over SSH, Tailscale and the camera tunnel.

| File | What it is |
|------|------------|
| `controller.html` | **Virtual XR controllers for teleop.** Stands in for a Quest/Pico: streams `CONTROLLER_MOVE` to `xr_teleoperate`'s websocket. Hand position and wrist sliders, trigger and squeeze (grippers), buttons, DAMP, and keyboard driving, with the head camera as the main view. Pose changes are rate-limited (0.25 m/s, 60°/s) and the page resumes from the last pose it sent. **No thumbstick walking** (site rule). |
| `g1_webui.py` | Browser console at `http://localhost:8780`: status, FSM modes, built-in gestures, wave, speech, LED, volume, STOP. Also serves `controller.html` at `/controller`. |
| `g1_mcp.py` | MCP server with the same tools for agents: `status set_mode gesture wave_hand say set_led set_volume stop`. Each tool is one allow-listed `unitree_sdk2py` call over SSH, with argument caps and a `G1_DRY_RUN=1` rehearsal mode. Robot commands queue on a lock on Waldo so several sessions can share the robot; `stop` skips the queue. Overlaps [`waldo-mcp`](../waldo-mcp/) on state/gestures/stop; adds speech, LED, volume and the queue. |
| `arms_face_test.py` | Client for Michael's `arm_server.py` (Jetson :5810): picks the highest face-height pose that both solves IK and passes the collision check, enables, moves both hands up, holds, returns, releases. `--dry` only solves and checks. |

Needs: `uv`, an SSH key on Waldo (`ssh-copy-id -i ~/.ssh/id_ed25519_lab operator@lab-pc.example`), Tailscale on
Michael's tailnet. Full startup steps: [`docs/quickstart.md`](../docs/quickstart.md). Every arm command and the
teleop controller map: [`docs/g1-arm-commands.md`](../docs/g1-arm-commands.md).

```sh
uv run --with "mcp>=2.2" waldo-console/g1_webui.py                       # console + /controller
claude mcp add g1-console -- uv run --with "mcp>=2.2" ~/HACKATHON2026/waldo-console/g1_mcp.py
uv run --with pyzmq waldo-console/arms_face_test.py --dry                 # needs arm_server + tunnel
```

Env overrides: `G1_HOST` (operator@lab-pc.example), `G1_KEY` (~/.ssh/id_ed25519_lab), `G1_PY` (Waldo's unitree
conda python), `G1_IFACE` (enp5s0), `G1_UI_PORT` (8780), `G1_DRY_RUN=1`.

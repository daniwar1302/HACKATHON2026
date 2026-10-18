"""The real arm_server's `trajectory` returns at once and runs in the background; skills must wait on `state`, keep
its heartbeat alive and notice a freeze. Checked against a fake server that behaves that way (no hardware needed).

  uv run --with "mcp>=2.2" --with pyzmq --with numpy --with pillow test_real_protocol.py
"""
import time

import server as S
from mcp.server.mcpserver.exceptions import ToolError
from planner import PlanError


class FakeRealArm:
    def __init__(self, busy_polls=3, freeze=False):
        self.busy, self.freeze, self.calls = busy_polls, freeze, []

    def __call__(self, cmd, timeout_s=30, **kw):
        self.calls.append((cmd, kw))
        if cmd.startswith("sim_"):
            raise ToolError(f"unknown cmd {cmd}")
        if cmd == "state":
            self.busy -= 1
            frozen = self.freeze and self.busy < 1
            return {"ok": True, "arm": {"moving": self.busy > 0 and not frozen, "frozen": frozen,
                                        "freeze_reason": "tracking error" if frozen else None}}
        return {"ok": True}


def run(fake):
    S.arm, S._sim = fake, None
    S._parm("trajectory", q=[[0.0] * 14] * 2, times=[0.0, 0.5], side="left", contact=True)


fake = FakeRealArm()
t0 = time.monotonic()
run(fake)
assert [c for c, _ in fake.calls].count("state") >= 3, fake.calls            # waited, polling state
assert "contact" not in dict(fake.calls)["trajectory"], "sim-only flag sent to the real server"
assert time.monotonic() - t0 >= 0.4

fake = FakeRealArm(busy_polls=2, freeze=True)
try:
    run(fake)
    raise AssertionError("a freeze mid-trajectory must fail the skill")
except PlanError as e:
    assert "froze" in str(e)

S.arm, S._sim = FakeRealArm(), None
t0 = time.monotonic()
S._parm("sim_wait", seconds=0.3)                     # real robot: sim_wait is a plain sleep
assert time.monotonic() - t0 >= 0.3
print("real-protocol checks ok")

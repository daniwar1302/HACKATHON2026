"""Deciders turn one camera frame plus the current checks into P(answer) per check.

JevDecider is the fast path (~0.1 s per frame). ClaudeDecider is the fallback
when Jev's image input isn't good enough: smarter, slower, run it at ~1 fps.
"""
from __future__ import annotations

import base64
import json
import os
import uuid
from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class Check:
    """One decision question for the current task step.

    answers maps each option to a short description of what it looks like.
    A yes/no check uses answers {"yes": ..., "no": ...}.
    """
    id: str
    question: str
    answers: dict[str, str]
    bad: tuple[str, ...]
    severity: str = "warn"  # warn | critical

    def p_bad(self, probabilities: dict[str, float]) -> float:
        return sum(probabilities.get(a, 0.0) for a in self.bad)

    @property
    def is_yes_no(self) -> bool:
        return set(self.answers) == {"yes", "no"}

    @classmethod
    def from_dict(cls, d: dict) -> "Check":
        bad = d["bad"] if isinstance(d["bad"], list) else [d["bad"]]
        return cls(d["id"], d["question"], dict(d["answers"]), tuple(bad), d.get("severity", "warn"))


class Decider(Protocol):
    name: str

    async def decide(self, session, jpeg: bytes, checks: list[Check]) -> dict[str, dict[str, float]]: ...


@dataclass
class JevDecider:
    api_key: str = field(default_factory=lambda: os.environ.get("TYPESAFE_API_KEY", ""))
    model: str = os.environ.get("WALDO_JEV_MODEL", "jev-1.14.0")
    base_url: str = os.environ.get("TYPESAFE_API_BASE", "https://api.typesafe.ai")
    name: str = "jev"

    def request(self, jpeg: bytes, checks: list[Check]) -> dict:
        ref = str(uuid.uuid4())  # TypeSafe recommends a fresh reference per image per request
        questions = {}
        for c in checks:
            if c.is_yes_no:
                questions[c.id] = {"type": "noul", "instructions": f"{c.question} Look at `camera`. Yes means: {c.answers['yes']}"}
            else:
                questions[c.id] = {"type": "choice", "instructions": f"{c.question} Look at `camera`.", "criteria": c.answers}
        return {
            "model": self.model,
            "state": {"camera": ref},
            "assets": {ref: {"type": "image", "format": "jpg", "data_base64": base64.b64encode(jpeg).decode()}},
            "questions": questions,
        }

    @staticmethod
    def parse(answers: dict, checks: list[Check]) -> dict[str, dict[str, float]]:
        out = {}
        for c in checks:
            a = answers.get(c.id)
            if not a:
                continue  # Jev drops questions it can't answer
            if a.get("type") == "noul":
                yes = float(a.get("noul", 0.0))
                out[c.id] = {"yes": yes, "no": 1.0 - yes}
            elif isinstance(a.get("probabilities"), dict):
                out[c.id] = {k: float(v) for k, v in a["probabilities"].items()}
            elif "choice" in a:
                out[c.id] = {a["choice"]: float(a.get("confidence", 1.0))}
        return out

    async def decide(self, session, jpeg, checks):
        headers = {"Authorization": f"Bearer {self.api_key}", "TypeSafe-Danger-Zone-Images": "true"}
        async with session.post(f"{self.base_url}/v1/systemone", json=self.request(jpeg, checks), headers=headers, timeout=5) as resp:
            resp.raise_for_status()
            return self.parse((await resp.json()).get("answers", {}), checks)


@dataclass
class ClaudeDecider:
    model: str = os.environ.get("WALDO_CLAUDE_DECIDER_MODEL", "claude-haiku-5-5")
    name: str = "claude"

    def __post_init__(self):
        from anthropic import AsyncAnthropic

        self.client = AsyncAnthropic()

    async def decide(self, session, jpeg, checks):
        schema = {
            "type": "object",
            "properties": {
                c.id: {
                    "type": "object",
                    "properties": {a: {"type": "number"} for a in c.answers},
                    "required": list(c.answers),
                    "additionalProperties": False,
                }
                for c in checks
            },
            "required": [c.id for c in checks],
            "additionalProperties": False,
        }
        lines = [f"- {c.id}: {c.question} Options: " + "; ".join(f"{a} = {d}" for a, d in c.answers.items()) for c in checks]
        resp = await self.client.messages.create(
            model=self.model,
            max_tokens=1000,
            output_config={"effort": "low", "format": {"type": "json_schema", "schema": schema}},
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": base64.b64encode(jpeg).decode()}},
                    {"type": "text", "text": "Answer each check about this robot camera frame with a probability per option (summing to 1).\n" + "\n".join(lines)},
                ],
            }],
        )
        if resp.stop_reason == "refusal":
            return {}
        text = next(b.text for b in resp.content if b.type == "text")
        return json.loads(text)

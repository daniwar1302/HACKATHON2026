"""Advice the operator has given before, so Claude can reuse it instead of asking again.

Every answered question becomes a hint: the task, what went wrong, the question,
and the operator's answer. Outcomes reported afterwards raise or lower its score.
Matching is plain word overlap; no model call, so a lookup takes milliseconds.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

STOPWORDS = set(
    "a an and are at be by can do for from have how i im i'm in is it its it's me my of on or "
    "should so that the this to up what where which why will with would you your".split()
)


def words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS and len(w) > 1}


@dataclass
class Hint:
    task: str
    situation: str
    question: str
    answer: str
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    created_at: float = field(default_factory=time.time)
    worked: int = 0
    failed: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def trusted(self) -> bool:
        """Confirmed to work at least once and more often than it failed."""
        return self.worked > 0 and self.worked > self.failed

    def summary(self) -> dict:
        return {
            "hint_id": self.id,
            "task": self.task,
            "situation": self.situation,
            "question": self.question,
            "answer": self.answer,
            "worked": self.worked,
            "failed": self.failed,
            "notes": self.notes[-3:],
        }


class HintStore:
    def __init__(self, path: Path):
        self.path = path
        self.hints: list[Hint] = []
        if path.exists():
            self.hints = [Hint(**h) for h in json.loads(path.read_text())]

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
        with os.fdopen(fd, "w") as f:
            json.dump([asdict(h) for h in self.hints], f, indent=1)
        os.replace(tmp, self.path)

    def add(self, task: str, situation: str, question: str, answer: str) -> Hint:
        hint = Hint(task=task, situation=situation, question=question, answer=answer)
        self.hints.append(hint)
        self._save()
        return hint

    def get(self, hint_id: str) -> Hint | None:
        return next((h for h in self.hints if h.id == hint_id), None)

    def report(self, hint_id: str, worked: bool, note: str = "") -> Hint | None:
        hint = self.get(hint_id)
        if hint is None:
            return None
        if worked:
            hint.worked += 1
        else:
            hint.failed += 1
        if note:
            hint.notes.append(note)
        self._save()
        return hint

    def search(self, query: str, limit: int = 3) -> list[tuple[float, Hint]]:
        """Hints ranked by word overlap with the query, ignoring ones that mostly failed."""
        q = words(query)
        if not q:
            return []
        scored = []
        for h in self.hints:
            if h.failed > h.worked + 1:
                continue
            hw = words(f"{h.task} {h.situation} {h.question}")
            if not hw:
                continue
            score = len(q & hw) / len(q | hw)
            if score > 0:
                scored.append((round(score, 2), h))
        scored.sort(key=lambda s: (s[0], s[1].worked - s[1].failed), reverse=True)
        return scored[:limit]

    def best(self, limit: int = 8) -> list[Hint]:
        """Most useful hints overall, for the server's startup instructions."""
        good = [h for h in self.hints if h.worked >= h.failed]
        good.sort(key=lambda h: (h.worked - h.failed, h.created_at), reverse=True)
        return good[:limit]

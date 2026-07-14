"""Skill library — the store the controller selects from and edits.

Each skill is a JSON file in skills/ holding the bespoke Python the model wrote (the body
of a control function), plus metadata used for retrieval. Selection here is intentionally
simple (keyword overlap on description/goal); swap in embeddings later without changing
callers. This file is domain-agnostic — only the `code` payload changed from JS to Python.
"""
from __future__ import annotations

import dataclasses
import json
import re
import time
from pathlib import Path


@dataclasses.dataclass
class Skill:
    name: str
    description: str          # one-line "use this when ..." summary, for retrieval
    code: str                 # the Python function body (runs in bridge primitive scope)
    task: str = ""            # task name it was created for
    created_at: float = dataclasses.field(default_factory=time.time)
    uses: int = 0
    successes: int = 0

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


class SkillLibrary:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, name: str) -> Path:
        return self.root / f"{name}.json"

    def all(self) -> list[Skill]:
        out = []
        for p in sorted(self.root.glob("*.json")):
            out.append(Skill(**json.loads(p.read_text())))
        return out

    def get(self, name: str) -> Skill | None:
        p = self._path(name)
        return Skill(**json.loads(p.read_text())) if p.exists() else None

    def save(self, skill: Skill) -> None:
        self._path(skill.name).write_text(json.dumps(skill.to_dict(), indent=2))

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return set(re.findall(r"[a-z0-9_]+", text.lower()))

    def retrieve(self, query: str, k: int = 3) -> list[Skill]:
        """Return up to k skills most relevant to the query by token overlap."""
        q = self._tokens(query)
        scored = []
        for s in self.all():
            overlap = len(q & self._tokens(f"{s.name} {s.description} {s.task}"))
            if overlap:
                scored.append((overlap, s))
        scored.sort(key=lambda t: t[0], reverse=True)
        return [s for _, s in scored[:k]]

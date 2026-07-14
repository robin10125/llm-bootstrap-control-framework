"""Task specs: goal description + success check + episode setup.

A task is a YAML file under tasks/. The same `SuccessCheck` shape is used two ways:
top-level tasks author it by hand in YAML, and runtime-generated subgoals emit it as JSON
alongside each subgoal (see Controller.run_node). A check is one or more clauses:

    name: lift_cube
    goal: "Pick up the red cube and lift it at least 10 cm off the table."
    setup:                  # passed to Bridge.reset / HandSim.reset
      cube_pos: [0.0, 0.0]
    max_iterations: 5
    success:
      expr: "obs['cube']['z'] > 0.12 and obs['grasped']"   # python bool over `obs`
      llm_judge: null        # NL criterion scored by an LLM over before/after obs

On a robot there are no clean inventory predicates, so `expr` over the observation (joint
/ object / contact state) and `llm_judge` are the two clauses. Prefer `expr` when the goal
is a crisp world-state condition; `llm_judge` carries the cases that aren't (the clause the
design memo flagged as primary for robot envs).
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, Callable

import yaml

# A judge turns (criterion, obs_before, obs_after) into (passed, reason).
Judge = Callable[[str, dict, dict], tuple[bool, str]]


@dataclasses.dataclass
class SuccessCheck:
    expr: str | None = None
    llm_judge: str | None = None

    @classmethod
    def from_dict(cls, data: dict | None) -> "SuccessCheck":
        """Build from a (possibly None) mapping, ignoring unknown keys defensively."""
        data = data or {}
        fields = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in fields})

    def is_empty(self) -> bool:
        return not (self.expr or self.llm_judge)

    def evaluate(self, obs: dict, obs_before: dict | None = None,
                 judge: Judge | None = None) -> tuple[bool, list[str]]:
        """Return (passed, list of unmet-reason strings).

        The `expr` clause is evaluated against `obs`. The `llm_judge` clause needs `judge`;
        without one it is reported as unmet rather than skipped, so an episode never
        silently "passes" on an unevaluated judge.
        """
        reasons: list[str] = []
        if self.expr:
            try:
                if not eval(self.expr, {"__builtins__": {}}, {"obs": obs}):  # noqa: S307 (trusted task files)
                    reasons.append(f"expr failed: {self.expr}")
            except Exception as e:  # noqa: BLE001
                reasons.append(f"expr error ({self.expr}): {e}")
        if self.llm_judge:
            if judge is None:
                reasons.append(f"llm_judge unevaluated (no judge available): {self.llm_judge}")
            else:
                ok, why = judge(self.llm_judge, obs_before or {}, obs)
                if not ok:
                    reasons.append(f"judge: {why}")
        return (len(reasons) == 0, reasons)


@dataclasses.dataclass
class Task:
    name: str
    goal: str
    setup: dict[str, Any] = dataclasses.field(default_factory=dict)
    max_iterations: int = 5
    success: SuccessCheck = dataclasses.field(default_factory=SuccessCheck)

    @classmethod
    def load(cls, path: str | Path) -> "Task":
        data = yaml.safe_load(Path(path).read_text())
        success = SuccessCheck.from_dict(data.get("success"))
        return cls(
            name=data["name"],
            goal=data["goal"],
            setup=data.get("setup", {}),
            max_iterations=data.get("max_iterations", 5),
            success=success,
        )

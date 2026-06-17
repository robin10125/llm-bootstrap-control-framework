from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any

import numpy as np


@dataclass(frozen=True)
class WorldState:
    """Privileged simulator state normalized for LLM prompts and compilers."""

    time_s: float
    object_pos: np.ndarray
    object_vel: np.ndarray
    base_q: np.ndarray
    hand_q: np.ndarray
    ctrl: np.ndarray
    ctrl_lo: np.ndarray
    ctrl_hi: np.ndarray
    actuator_names: list[str]
    appendages: dict[str, list[str]] = field(default_factory=dict)
    joint_schema: dict[str, Any] = field(default_factory=dict)
    derived: dict[str, Any] = field(default_factory=dict)
    fingertip_pos: dict[str, np.ndarray] = field(default_factory=dict)
    contacts: list[dict[str, Any]] = field(default_factory=list)
    history: list[dict[str, Any]] = field(default_factory=list)

    def compact(self) -> dict[str, Any]:
        return {
            "time_s": round(self.time_s, 3),
            "object": {
                "pos": self.object_pos.round(4).tolist(),
                "vel": self.object_vel.round(4).tolist(),
            },
            "base_q": self.base_q.round(4).tolist(),
            "hand_q": self.hand_q.round(4).tolist(),
            "ctrl": {
                name: round(float(value), 4)
                for name, value in zip(self.actuator_names, self.ctrl, strict=False)
            },
            "appendages": self.appendages,
            "joint_schema": self.joint_schema,
            "derived": self.derived,
            "fingertips": {
                name: pos.round(4).tolist() for name, pos in self.fingertip_pos.items()
            },
            "contacts": self.contacts,
        }


@dataclass(frozen=True)
class CandidateProgram:
    interface: str
    source: dict[str, Any]
    raw_text: str = ""


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class SafetyLimits:
    max_episode_seconds: float = 3.0
    max_blocks: int = 32
    max_loop_iterations: int = 16
    max_force_n: float = 8.0
    max_impulse: float = 3.0
    forbidden_symbols: tuple[str, ...] = (
        "grasp",
        "throw",
        "fold",
        "pick",
        "pick_up",
        "use_chopsticks",
        "place",
        "lift",
    )


@dataclass(frozen=True)
class CompiledPolicy:
    interface: str
    action_stream: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def horizon(self) -> int:
        return int(self.action_stream.shape[0])


@dataclass(frozen=True)
class RolloutResult:
    interface: str
    task: str
    seed: int
    success: bool
    score: float
    total_return: float
    final_object_pos: tuple[float, float, float]
    max_object_z: float
    errors: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def row(self) -> dict[str, Any]:
        return {
            "interface": self.interface,
            "task": self.task,
            "seed": self.seed,
            "success": self.success,
            "score": self.score,
            "total_return": self.total_return,
            "final_object_x": self.final_object_pos[0],
            "final_object_y": self.final_object_pos[1],
            "final_object_z": self.final_object_pos[2],
            "max_object_z": self.max_object_z,
            "errors": "; ".join(self.errors),
            "task_metrics": json.dumps(self.metadata.get("task_metrics", {}), sort_keys=True),
        }

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from llm_framework.core.state import RolloutResult, WorldState


ScoreFn = Callable[["TaskContext", RolloutResult], tuple[bool, float]]


@dataclass(frozen=True)
class TaskContext:
    name: str
    goal: str
    seed: int
    episode_seconds: float
    object_start: tuple[float, float, float]
    target_xy: tuple[float, float] | None = None
    metadata: dict = field(default_factory=dict)

    def compact(self) -> dict:
        return {
            "name": self.name,
            "goal": self.goal,
            "seed": self.seed,
            "episode_seconds": self.episode_seconds,
            "object_start": [round(v, 4) for v in self.object_start],
            "target_xy": None if self.target_xy is None else [round(v, 4) for v in self.target_xy],
            "metadata": self.metadata,
        }


def build_task_context(name: str, seed: int, world: WorldState, episode_seconds: float) -> TaskContext:
    obj = tuple(float(v) for v in world.object_pos[:3])
    if name == "lift":
        return TaskContext(
            name=name,
            goal="raise the object above the table and keep it controlled",
            seed=seed,
            episode_seconds=episode_seconds,
            object_start=obj,
            metadata={"height_threshold": obj[2] + 0.05},
        )
    if name == "push":
        target = _target_from_seed(seed, obj, distance=0.08)
        return TaskContext(
            name=name,
            goal="move the object along the table to the target xy",
            seed=seed,
            episode_seconds=episode_seconds,
            object_start=obj,
            target_xy=target,
            metadata={"xy_tolerance": 0.045},
        )
    if name == "place":
        target = _target_from_seed(seed + 17, obj, distance=0.09)
        return TaskContext(
            name=name,
            goal="move the object to the target xy and leave it stable there",
            seed=seed,
            episode_seconds=episode_seconds,
            object_start=obj,
            target_xy=target,
            metadata={"xy_tolerance": 0.045, "height_threshold": obj[2] + 0.04},
        )
    if name == "stabilize":
        return TaskContext(
            name=name,
            goal="keep the object inside a small radius while maintaining hand control",
            seed=seed,
            episode_seconds=episode_seconds,
            object_start=obj,
            metadata={"xy_tolerance": 0.035},
        )
    raise KeyError(f"unknown task {name!r}")


def score_task(ctx: TaskContext, result: RolloutResult) -> tuple[bool, float]:
    final = np.asarray(result.final_object_pos)
    start = np.asarray(ctx.object_start)
    if ctx.name == "lift":
        threshold = float(ctx.metadata["height_threshold"])
        success = bool(result.max_object_z >= threshold)
        score = max(0.0, result.max_object_z - start[2]) * 10.0 + (1.0 if success else 0.0)
        return success, round(float(score), 4)
    if ctx.name in {"push", "place"}:
        assert ctx.target_xy is not None
        target = np.asarray(ctx.target_xy)
        start_d = float(np.linalg.norm(target - start[:2]))
        final_d = float(np.linalg.norm(target - final[:2]))
        progress = (start_d - final_d) / max(start_d, 1e-6)
        success = final_d <= float(ctx.metadata["xy_tolerance"])
        score = progress + (1.0 if success else 0.0)
        if ctx.name == "place":
            score += 0.25 if result.max_object_z >= float(ctx.metadata["height_threshold"]) else 0.0
        return success, round(float(score), 4)
    if ctx.name == "stabilize":
        drift = float(np.linalg.norm(final[:2] - start[:2]))
        success = drift <= float(ctx.metadata["xy_tolerance"])
        return success, round(float(1.0 - drift / 0.15 + (1.0 if success else 0.0)), 4)
    raise KeyError(ctx.name)


def task_metrics(ctx: TaskContext, result: RolloutResult) -> dict:
    final = np.asarray(result.final_object_pos)
    start = np.asarray(ctx.object_start)
    if ctx.name == "lift":
        threshold = float(ctx.metadata["height_threshold"])
        return {
            "metric": "height",
            "height_threshold": round(threshold, 4),
            "max_object_z": round(float(result.max_object_z), 4),
            "height_margin": round(float(result.max_object_z - threshold), 4),
            "lift_delta": round(float(result.max_object_z - start[2]), 4),
            "score_note": "score is shaped progress, not absolute object height",
        }
    if ctx.name in {"push", "place"} and ctx.target_xy is not None:
        target = np.asarray(ctx.target_xy)
        return {
            "metric": "xy_distance",
            "target_xy": [round(float(v), 4) for v in target],
            "final_xy": [round(float(v), 4) for v in final[:2]],
            "final_distance": round(float(np.linalg.norm(target - final[:2])), 4),
            "xy_tolerance": round(float(ctx.metadata["xy_tolerance"]), 4),
        }
    if ctx.name == "stabilize":
        return {
            "metric": "xy_drift",
            "drift": round(float(np.linalg.norm(final[:2] - start[:2])), 4),
            "xy_tolerance": round(float(ctx.metadata["xy_tolerance"]), 4),
        }
    return {}


def _target_from_seed(seed: int, obj: tuple[float, float, float], distance: float) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    angle = float(rng.uniform(0.0, 2.0 * np.pi))
    return (
        round(float(obj[0] + distance * np.cos(angle)), 4),
        round(float(obj[1] + distance * np.sin(angle)), 4),
    )

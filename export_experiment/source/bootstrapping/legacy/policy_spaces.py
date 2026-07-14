#!/usr/bin/env python3
"""Policy parameterizations for search and behavior cloning.

Two spaces:

- `template`: the original fixed schedule template (open, approach, descend, pre-grasp,
  grasp, wait, lift, hold) with 12 tunable continuous parameters. Structure is baked in;
  only parameters are searched. Lift tasks only. Kept as the easy-mode comparison point.
- `sequence`: variable-length primitive sequences (see sequence_policy.py), flat latent of
  MAX_STEPS x TOKEN_DIM values. Search and the neural policy must discover structure, and
  LLM-authored schedules keep their structure when encoded.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from sequence_policy import FLAT_DIM, flat_to_schedule, schedule_to_flat
from tasks import Task


POSE_OPEN = "open hand"
POSE_PRE = "pre grasp"
POSE_GRASP = "grasp hard"

TEMPLATE_DIM = 12


@dataclass(frozen=True)
class PolicySpace:
    name: str
    dim: int
    decode: Callable[[np.ndarray, dict[str, Any], str, Task], dict[str, Any]]
    encode: Callable[[dict[str, Any]], np.ndarray]


def clip(value: float, lo: float, hi: float) -> float:
    return min(max(float(value), lo), hi)


def inv(value: float) -> float:
    return float(np.arctanh(clip(value, -0.98, 0.98)))


def make_policy(
    *,
    name: str,
    setup: dict[str, Any],
    approach: tuple[float, float, float, float],
    descend: tuple[float, float, float, float],
    pre_d: float,
    grasp_d: float,
    lift: tuple[float, float, float, float] | None,
    success_lift: bool,
) -> dict[str, Any]:
    steps: list[dict[str, Any]] = [
        {"primitive": "hand_pose", "params": {"name": POSE_OPEN}, "duration_s": 0.25},
        {"primitive": "set_base", "params": xyz(approach), "duration_s": round(approach[3], 3)},
        {"primitive": "set_base", "params": xyz(descend), "duration_s": round(descend[3], 3)},
        {"primitive": "hand_pose", "params": {"name": POSE_PRE}, "duration_s": round(pre_d, 3)},
        {
            "primitive": "hand_pose",
            "params": {"name": POSE_GRASP},
            "duration_s": round(grasp_d, 3),
            "assert": [{"metric": "contacts.hand_object_count", "op": ">=", "value": 2}],
        },
        {"primitive": "wait", "params": {}, "duration_s": 0.25},
    ]
    if lift is not None:
        steps.append({"primitive": "set_base", "params": xyz(lift), "duration_s": round(lift[3], 3)})
        # Hold after the lift so the final lifted check measures a held lift, not a
        # transient bounce. encode ignores wait steps, so the latent round-trip is
        # unaffected.
        steps.append({"primitive": "wait", "params": {}, "duration_s": 0.75})
    success = [{"metric": "contacts.hand_object_count", "op": ">=", "value": 2}]
    if success_lift:
        success = [{"metric": "lifted", "op": "==", "value": True}]
    return {
        "name": name,
        "goal": "grasp the cube with the Shadow Hand and lift it without dropping if possible",
        "setup": setup,
        "steps": steps,
        "success": success,
    }


def xyz(values: tuple[float, float, float, float]) -> dict[str, float]:
    return {"x": round(values[0], 4), "y": round(values[1], 4), "z": round(values[2], 4)}


def decode_template(vec: np.ndarray, setup: dict[str, Any], name: str, task: Task | None = None) -> dict[str, Any]:
    if task is not None and task.name not in {"lift", "lift_perturbed"}:
        raise ValueError(f"template policy space only supports lift tasks, not {task.name!r}")
    # Squash unconstrained outputs into safe primitive ranges.
    t = np.tanh(vec)
    obj = setup.get("object_pos", [0.0, 0.0, 0.025])
    approach_x = clip(obj[0] + 0.08 * t[0] - 0.035, -0.12, 0.08)
    approach_y = clip(obj[1] + 0.08 * t[1], -0.10, 0.10)
    approach_z = clip(0.08 + 0.05 * t[2], 0.03, 0.14)
    approach_d = clip(0.8 + 0.45 * t[3], 0.25, 1.5)
    descend_x = clip(obj[0] + 0.08 * t[4] - 0.035, -0.12, 0.08)
    descend_y = clip(obj[1] + 0.08 * t[5], -0.10, 0.10)
    descend_z = clip(0.13 + 0.06 * t[6], 0.07, 0.19)
    descend_d = clip(0.8 + 0.5 * t[7], 0.25, 1.7)
    pre_d = clip(0.45 + 0.35 * t[8], 0.15, 1.0)
    grasp_d = clip(0.9 + 0.6 * t[9], 0.3, 1.8)
    lift_z = clip(0.05 + 0.08 * t[10], -0.02, 0.13)
    lift_d = clip(0.9 + 0.7 * t[11], 0.25, 2.0)

    return make_policy(
        name=name,
        setup=setup,
        approach=(approach_x, approach_y, approach_z, approach_d),
        descend=(descend_x, descend_y, descend_z, descend_d),
        pre_d=pre_d,
        grasp_d=grasp_d,
        lift=(descend_x, descend_y, lift_z, lift_d),
        success_lift=True,
    )


def encode_template(policy: dict[str, Any]) -> np.ndarray:
    steps = policy["steps"]
    set_base_steps = [s for s in steps if s["primitive"] == "set_base"]
    obj = policy.get("setup", {}).get("object_pos", [0.0, 0.0, 0.025])
    approach = set_base_steps[0]
    descend = set_base_steps[1] if len(set_base_steps) > 1 else set_base_steps[0]
    lift = set_base_steps[-1] if len(set_base_steps) > 2 else {"params": {"z": 0.05}, "duration_s": 0.9}
    pre = next((s for s in steps if s["primitive"] == "hand_pose" and s["params"]["name"] == POSE_PRE), {"duration_s": 0.45})
    grasp = next((s for s in steps if s["primitive"] == "hand_pose" and s["params"]["name"] == POSE_GRASP), {"duration_s": 0.9})
    return np.array([
        inv((approach["params"]["x"] - obj[0] + 0.035) / 0.08),
        inv((approach["params"]["y"] - obj[1]) / 0.08),
        inv((approach["params"]["z"] - 0.08) / 0.05),
        inv((approach["duration_s"] - 0.8) / 0.45),
        inv((descend["params"]["x"] - obj[0] + 0.035) / 0.08),
        inv((descend["params"]["y"] - obj[1]) / 0.08),
        inv((descend["params"]["z"] - 0.13) / 0.06),
        inv((descend["duration_s"] - 0.8) / 0.5),
        inv((pre["duration_s"] - 0.45) / 0.35),
        inv((grasp["duration_s"] - 0.9) / 0.6),
        inv((lift["params"]["z"] - 0.05) / 0.08),
        inv((lift["duration_s"] - 0.9) / 0.7),
    ], dtype=float)


def decode_sequence(vec: np.ndarray, setup: dict[str, Any], name: str, task: Task) -> dict[str, Any]:
    return flat_to_schedule(vec, setup, name=name, goal=task.goal, success=task.success_checks(setup))


SPACES = {
    "template": PolicySpace("template", TEMPLATE_DIM, decode_template, encode_template),
    "sequence": PolicySpace("sequence", FLAT_DIM, decode_sequence, schedule_to_flat),
}


def get_space(name: str) -> PolicySpace:
    if name not in SPACES:
        raise KeyError(f"unknown policy space {name!r}; available: {sorted(SPACES)}")
    return SPACES[name]


def template_expert(name: str, setup: dict[str, Any]) -> dict[str, Any]:
    """Scripted lift expert expressed in the template's own step structure, so it can be
    encoded into the 12-dim template latent (the task-suite experts use primitives the
    template codec cannot represent)."""
    obj = setup.get("object_pos", [0.0, 0.0, 0.025])
    x = obj[0] - 0.035
    y = obj[1]
    return make_policy(
        name=name,
        setup=setup,
        approach=(x, y, 0.08, 0.8),
        descend=(x, y, 0.14, 0.7),
        pre_d=0.4,
        grasp_d=1.0,
        lift=(x, y, 0.055, 1.0),
        success_lift=True,
    )


def expert_for(space: PolicySpace, task: Task, name: str, setup: dict[str, Any]) -> dict[str, Any]:
    if space.name == "template":
        return template_expert(name, setup)
    return task.scripted_expert(name, setup)


def jitter_residual(
    space: PolicySpace,
    reference_policy: dict[str, Any],
    setup: dict[str, Any],
    task: Task,
    *,
    name: str,
    attempt: int,
) -> dict[str, Any]:
    """Deterministic offline stand-in for an LLM residual edit: re-encode the reference
    schedule and apply a small structured perturbation that grows with the attempt."""
    vec = space.encode(reference_policy)
    rng = np.random.default_rng(hash((name, attempt)) % (2**32))
    vec = vec + rng.normal(0.0, 0.12 * attempt, vec.shape)
    return space.decode(vec, setup, f"{name}_jitter", task)

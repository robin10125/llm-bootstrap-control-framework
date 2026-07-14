#!/usr/bin/env python3
"""Structured eval vector — the shared task-progress vocabulary.

This is the ground-truth, sim-derived measurement vector that everything in the
reward/curriculum stack reads against: the LLM authors rewards and milestones over these
fields, PPO's shaped reward is a function of them, and the supervisor judges whether an LLM
demo is *more helpful* than the policy by comparing their summaries. It is computed from
`mjx.Data` (never from the policy's self-report) so it can't be gamed by the policy. See
`llm_reward_curriculum_plan.md`.

A field has a name, an episode reduction (how a per-step trajectory collapses to a single
summary number — e.g. closest approach is a `min`, peak lift is a `max`), a human description
(used verbatim in LLM prompts), and a nominal range (used for prompt grounding + spec
validation). The order here defines the flat-array layout used in the jitted env.
"""
from __future__ import annotations

import jax.numpy as jp

# name, reduction over the episode, description (for prompts), nominal (lo, hi)
EVAL_FIELDS: list[tuple[str, str, str, tuple[float, float]]] = [
    ("palm_obj_dist", "min", "distance from the hand/gripper grasp-site to the object (m); "
                             "smaller is closer", (0.0, 0.4)),
    ("min_finger_dist", "min", "distance from the nearest fingertip to the object (m)", (0.0, 0.4)),
    ("n_contacts", "max", "number of fingertips touching the object (0..n_fingers)", (0.0, 5.0)),
    ("closure", "max", "mean finger-closure fraction (0 = open hand, 1 = fully closed)", (0.0, 1.0)),
    ("lift", "max", "height of the object above its start (m); task success is > 0.05", (-0.1, 0.2)),
    ("obj_xy_disp", "max", "how far the object slid horizontally from its start (m); "
                           "large means it was knocked away", (0.0, 0.4)),
]

FIELD_NAMES: list[str] = [f[0] for f in EVAL_FIELDS]
FIELD_INDEX: dict[str, int] = {n: i for i, n in enumerate(FIELD_NAMES)}
N_FIELDS = len(EVAL_FIELDS)

# boolean mask: True where the episode reduction is a min (else max). Used to collapse a
# per-step [T, F] eval trajectory into an [F] summary inside jit.
_IS_MIN = jp.asarray([f[1] == "min" for f in EVAL_FIELDS])


def reduce_summary(eval_seq: jp.ndarray) -> jp.ndarray:
    """[T, F] per-step eval trajectory -> [F] episode summary (per-field min/max)."""
    mins = eval_seq.min(axis=0)
    maxs = eval_seq.max(axis=0)
    return jp.where(_IS_MIN, mins, maxs)


def predicate(op: str, value, thr: float):
    """Comparison usable on numpy or jax scalars/arrays. Returns a bool-ish of `value`."""
    if op == "lt":
        return value < thr
    if op == "le":
        return value <= thr
    if op == "gt":
        return value > thr
    if op == "ge":
        return value >= thr
    raise ValueError(f"unknown predicate op {op!r} (use lt/le/gt/ge)")


def fields_doc() -> str:
    """Human-readable field table for LLM prompts."""
    lines = ["The task eval vector has these fields (computed from the simulator):"]
    for name, reduce, desc, (lo, hi) in EVAL_FIELDS:
        lines.append(f"- {name} [{lo:g}..{hi:g}], episode-{reduce}: {desc}")
    return "\n".join(lines)

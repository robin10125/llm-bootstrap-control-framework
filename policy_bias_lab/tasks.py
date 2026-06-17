from __future__ import annotations

from typing import Any

import jax.numpy as jp

from policy_bias_lab.schema import FIELD_INDEX


def task_fitness(task: str, eval_summary: jp.ndarray) -> jp.ndarray:
    lift = eval_summary[..., FIELD_INDEX["lift"]]
    reach = eval_summary[..., FIELD_INDEX["palm_obj_dist"]]
    finger = eval_summary[..., FIELD_INDEX["min_finger_dist"]]
    contacts = eval_summary[..., FIELD_INDEX["n_contacts"]]
    closure = eval_summary[..., FIELD_INDEX["closure"]]
    xy = eval_summary[..., FIELD_INDEX["obj_xy_disp"]]
    if task == "lift":
        return 12.0 * lift - 1.0 * reach - 0.5 * finger + 0.3 * contacts + 0.2 * closure
    if task == "push":
        return 8.0 * xy - 0.4 * reach + 0.2 * contacts
    if task == "stabilize":
        return -6.0 * xy - 0.5 * reach + 0.2 * contacts
    raise KeyError(f"unknown task {task!r}")


def task_success(task: str, eval_summary: jp.ndarray) -> jp.ndarray:
    if task == "lift":
        return eval_summary[..., FIELD_INDEX["lift"]] > 0.05
    if task == "push":
        return eval_summary[..., FIELD_INDEX["obj_xy_disp"]] > 0.06
    if task == "stabilize":
        return eval_summary[..., FIELD_INDEX["obj_xy_disp"]] < 0.035
    raise KeyError(f"unknown task {task!r}")


def task_metadata(task: str) -> dict[str, Any]:
    if task == "lift":
        return {"objective": "raise object above starting height", "success": "lift > 0.05m"}
    if task == "push":
        return {"objective": "move object horizontally from start", "success": "obj_xy_disp > 0.06m"}
    if task == "stabilize":
        return {"objective": "avoid object drift while maintaining control", "success": "obj_xy_disp < 0.035m"}
    raise KeyError(f"unknown task {task!r}")

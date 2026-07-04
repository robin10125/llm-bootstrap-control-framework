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


# ------------------------------------------------------------------------------------------------
# Per-task ARBITER definitions. These live here -- the injected TASK side -- not in the framework:
# the framework (ppo_arbiter / agentic_orchestrator) must stay task-agnostic, so anything that
# names task-specific eval fields or encodes how progress on a task is graded is task data.
# ------------------------------------------------------------------------------------------------

def task_graded_objective(task: str, ev: dict[str, Any]) -> float:
    """Dense selection objective over the trained-policy eval dict for `task`.

    Ranking on final success alone is blind when no candidate ever succeeds at short-PPO budgets
    (observed: the marginal-value run flatlined at a tie-breaker floor), so each task defines a
    graded objective over its own eval fields, with anti-exploit gating where the task needs it.
    """
    if task == "lift":
        # reach -> grasp -> lift -> sustain, later (harder) stages weighted more; lift credited
        # only as far as there is a grasp, so lifting WITHOUT a secure grip (throwing/knocking the
        # object up) scores ~0 on the lift terms.
        lift_thresh = 0.05  # m; matches this task's success definition above
        reach = float(ev.get("eval_reach_rate", 0.0))
        grasp = float(ev.get("eval_grasp_rate", 0.0))
        lift_reached = float(ev.get("eval_lift_reached_rate", 0.0))
        lift_max_norm = min(float(ev.get("eval_lift_max", 0.0)) / lift_thresh, 1.0)
        succ = float(ev.get("eval_success_rate", 0.0))
        return (1.0 * succ + 0.5 * grasp + 0.2 * reach
                + 0.3 * min(lift_reached, grasp) + 0.1 * min(lift_max_norm, grasp))
    if task in ("push", "stabilize"):
        return float(ev.get("eval_success_rate", 0.0))
    raise KeyError(f"unknown task {task!r}")


def task_progress_metrics(task: str) -> tuple[str, ...]:
    """Training-telemetry row keys expected to RISE while `task` is being learned (used by the
    framework's generic still-improving-vs-converged trend probe)."""
    if task == "lift":
        return ("base_return", "reach_rate", "grasp_rate", "lift_reached_rate", "success")
    if task in ("push", "stabilize"):
        return ("base_return", "success")
    raise KeyError(f"unknown task {task!r}")

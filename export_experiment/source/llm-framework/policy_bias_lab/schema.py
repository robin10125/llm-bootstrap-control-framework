from __future__ import annotations

from dataclasses import dataclass
from typing import Any


EVAL_FIELDS: tuple[str, ...] = (
    "palm_obj_dist",
    "min_finger_dist",
    "n_contacts",
    "closure",
    "lift",
    "obj_xy_disp",
)
FIELD_INDEX = {name: i for i, name in enumerate(EVAL_FIELDS)}

ACTION_GROUPS: tuple[str, ...] = (
    "base_xy",
    "base_z",
    "hand",
    "thumb",
    "index",
    "middle",
    "ring",
    "little",
    "all",
)
PRIOR_DIRECTIONS: tuple[str, ...] = (
    "toward_object_xy",
    "lower_base",
    "raise_base",
    "close_hand",
    "open_hand",
    "stabilize",
)


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    errors: tuple[str, ...] = ()


def default_bias_spec(task: str = "multi") -> dict[str, Any]:
    """Deterministic fixture used by tests and explicit fixture runs."""
    return {
        "name": f"default_{task}_shadow_bias",
        "description": "Bias gross pose before contact, contact before transport, and stabilize after progress.",
        "reward_terms": [
            {"name": "finger_near_progress", "observable": "min_finger_dist", "direction": "minimize", "weight": 0.35, "scale": 0.08, "tasks": ["lift"]},
            {"name": "contact_progress", "observable": "n_contacts", "direction": "maximize", "weight": 0.35, "scale": 3.0, "tasks": ["lift"]},
            {"name": "avoid_knockaway_progress", "observable": "obj_xy_disp", "direction": "minimize", "weight": 0.20, "scale": 0.10, "tasks": ["lift"]},
        ],
        "action_priors": [
            {"name": "center_grasp_frame", "group": "base_xy", "direction": "toward_object_xy", "weight": 0.7},
            {"name": "descend_before_contact", "group": "base_z", "direction": "lower_base", "weight": 0.35},
            {"name": "close_after_approach", "group": "hand", "direction": "close_hand", "weight": 0.45},
        ],
        "exploration_groups": [
            {"group": "base_xy", "scale": 1.25},
            {"group": "base_z", "scale": 0.9},
            {"group": "hand", "scale": 0.85},
        ],
        "supervised_targets": [
            {"name": "approach_target", "group": "base_xy", "direction": "toward_object_xy", "weight": 0.8},
            {"name": "lower_target", "group": "base_z", "direction": "lower_base", "weight": 0.25},
            {"name": "close_target", "group": "hand", "direction": "close_hand", "weight": 0.55},
        ],
        "curriculum": [
            {"stage": "approach", "focus": ["palm_obj_dist", "min_finger_dist"]},
            {"stage": "contact", "focus": ["n_contacts", "closure"]},
            {"stage": "transport_or_stabilize", "focus": ["lift", "obj_xy_disp"]},
        ],
    }


def validate_bias_spec(spec: dict[str, Any]) -> ValidationResult:
    errors: list[str] = []
    if not isinstance(spec, dict):
        return ValidationResult(False, ("bias spec must be a JSON object",))
    if not isinstance(spec.get("name", ""), str) or not spec.get("name"):
        errors.append("spec.name must be a non-empty string")

    for section in ("reward_terms", "action_priors", "exploration_groups", "supervised_targets"):
        if section in spec and not isinstance(spec[section], list):
            errors.append(f"{section} must be a list")

    for i, term in enumerate(spec.get("reward_terms", [])):
        if not isinstance(term, dict):
            errors.append(f"reward_terms[{i}] must be an object")
            continue
        observable = term.get("observable")
        if observable not in EVAL_FIELDS:
            errors.append(f"reward_terms[{i}] has unknown observable {observable!r}")
        if term.get("direction") not in {"minimize", "maximize"}:
            errors.append(f"reward_terms[{i}] direction must be minimize|maximize")
        _numeric(term, "weight", f"reward_terms[{i}]", errors, default_ok=False)
        _numeric(term, "scale", f"reward_terms[{i}]", errors, default_ok=True)
        _numeric(term, "max_step", f"reward_terms[{i}]", errors, default_ok=True)
        tasks = term.get("tasks")
        if tasks is not None and (
            not isinstance(tasks, list) or not all(isinstance(item, str) for item in tasks)
        ):
            errors.append(f"reward_terms[{i}].tasks must be a list of strings")

    for section in ("action_priors", "supervised_targets"):
        for i, prior in enumerate(spec.get(section, [])):
            if not isinstance(prior, dict):
                errors.append(f"{section}[{i}] must be an object")
                continue
            if prior.get("group") not in ACTION_GROUPS:
                errors.append(f"{section}[{i}] has unknown group {prior.get('group')!r}")
            if prior.get("direction") not in PRIOR_DIRECTIONS:
                errors.append(f"{section}[{i}] has unknown direction {prior.get('direction')!r}")
            _numeric(prior, "weight", f"{section}[{i}]", errors, default_ok=False)

    for i, group in enumerate(spec.get("exploration_groups", [])):
        if not isinstance(group, dict):
            errors.append(f"exploration_groups[{i}] must be an object")
            continue
        if group.get("group") not in ACTION_GROUPS:
            errors.append(f"exploration_groups[{i}] has unknown group {group.get('group')!r}")
        _numeric(group, "scale", f"exploration_groups[{i}]", errors, default_ok=False)

    return ValidationResult(not errors, tuple(errors))


def _numeric(obj: dict[str, Any], key: str, label: str, errors: list[str], *, default_ok: bool) -> None:
    if key not in obj and default_ok:
        return
    try:
        float(obj[key])
    except (KeyError, TypeError, ValueError):
        errors.append(f"{label}.{key} must be numeric")

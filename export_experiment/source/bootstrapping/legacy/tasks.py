#!/usr/bin/env python3
"""Task suite for primitive-policy bootstrapping.

Each task defines: a setup sampler (object pose, object variation, optional target and
perturbation), success checks for the policy JSON, a shaped score for search, a goal
string for LLM prompts, and a scripted expert schedule that serves both as the offline
mock LLM and as the scripted-expert baseline.

Tasks:

- `lift`: grasp the object and hold it above the lift threshold.
- `lift_perturbed`: lift, with a horizontal velocity kick to the object mid-episode.
- `place`: lift the object, carry it to a target xy, and release it there.
- `push`: slide the object to a target xy without needing to lift.

Object sets (orthogonal to task): `cube` (fixed cube) or `varied` (box/sphere/cylinder
with randomized size, mass, and friction).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


OBS_DIM = 5  # object xyz + target xy (zeros when the task has no target)

LIFT_CLEARANCE = 0.035
PLACE_TOLERANCE = 0.035
PUSH_TOLERANCE = 0.04


def obs_from_setup(setup: dict[str, Any]) -> np.ndarray:
    obj = setup.get("object_pos", [0.0, 0.0, 0.025])
    target = setup.get("target", [0.0, 0.0])
    return np.array([obj[0], obj[1], obj[2], target[0], target[1]], dtype=float)


def sample_object_spec(rng: np.random.Generator, object_set: str) -> tuple[dict[str, Any] | None, float]:
    """Return (object spec or None, half height)."""
    if object_set == "cube":
        return None, 0.025
    shape = rng.choice(["box", "sphere", "cylinder"])
    mass = round(float(rng.uniform(0.03, 0.12)), 4)
    friction = round(float(rng.uniform(0.6, 1.4)), 3)
    if shape == "box":
        half = [round(float(rng.uniform(0.018, 0.030)), 4) for _ in range(3)]
        spec = {"shape": "box", "size": half}
        half_height = half[2]
    elif shape == "sphere":
        r = round(float(rng.uniform(0.018, 0.028)), 4)
        spec = {"shape": "sphere", "size": [r]}
        half_height = r
    else:
        r = round(float(rng.uniform(0.016, 0.026)), 4)
        h = round(float(rng.uniform(0.018, 0.032)), 4)
        spec = {"shape": "cylinder", "size": [r, h]}
        half_height = h
    spec["mass"] = mass
    spec["friction"] = friction
    return spec, half_height


@dataclass(frozen=True)
class Task:
    name: str
    goal: str
    has_target: bool
    sample_setup: Callable[[np.random.Generator, float, str], dict[str, Any]]
    success_checks: Callable[[dict[str, Any]], list[dict[str, Any]]]
    score: Callable[[dict[str, Any], dict[str, Any]], float]
    scripted_expert: Callable[[str, dict[str, Any]], dict[str, Any]]


def _base_setup(rng: np.random.Generator, radius: float, object_set: str) -> dict[str, Any]:
    xy = rng.uniform(-radius, radius, size=2)
    spec, half_height = sample_object_spec(rng, object_set)
    setup: dict[str, Any] = {"object_pos": [round(float(xy[0]), 4), round(float(xy[1]), 4), round(half_height, 4)]}
    if spec is not None:
        setup["object"] = spec
    return setup


def _sample_target(rng: np.random.Generator, setup: dict[str, Any], min_dist: float, max_dist: float) -> None:
    angle = rng.uniform(0.0, 2.0 * np.pi)
    dist = rng.uniform(min_dist, max_dist)
    obj = setup["object_pos"]
    setup["target"] = [round(float(obj[0] + dist * np.cos(angle)), 4), round(float(obj[1] + dist * np.sin(angle)), 4)]


def _half_height(setup: dict[str, Any]) -> float:
    return float(setup.get("object_pos", [0, 0, 0.025])[2])


def _lift_threshold(setup: dict[str, Any]) -> float:
    return _half_height(setup) + LIFT_CLEARANCE


def _max_contacts_and_z(result: dict[str, Any]) -> tuple[int, float]:
    final = result.get("final_state", {})
    max_contacts = 0
    max_z = float(final.get("object", {}).get("z", 0.0) or 0.0)
    for step in result.get("trace", []):
        after = step.get("after", {})
        max_contacts = max(max_contacts, int(after.get("contacts", {}).get("hand_object_count", 0) or 0))
        max_z = max(max_z, float(after.get("object", {}).get("z", 0.0) or 0.0))
    return max_contacts, max_z


def _final_xy(result: dict[str, Any]) -> tuple[float, float]:
    obj = result.get("final_state", {}).get("object", {})
    return float(obj.get("x", 0.0) or 0.0), float(obj.get("y", 0.0) or 0.0)


# --- lift ---------------------------------------------------------------------------

def _lift_setup(rng: np.random.Generator, radius: float, object_set: str) -> dict[str, Any]:
    return _base_setup(rng, radius, object_set)


def _lift_success(setup: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"metric": "lifted", "op": "==", "value": True}]


def _lift_score(result: dict[str, Any], setup: dict[str, Any]) -> float:
    max_contacts, max_z = _max_contacts_and_z(result)
    final = result.get("final_state", {})
    score = 0.15 * max_contacts
    if final.get("grasped"):
        score += 0.4
    if final.get("lifted"):
        score += 1.0
    score += max(0.0, max_z - _half_height(setup)) * 4.0
    if result.get("success"):
        score += 0.25
    if result.get("errors"):
        score -= 0.5
    return round(float(score), 4)


def _grasp_steps(setup: dict[str, Any]) -> list[dict[str, Any]]:
    if setup.get("embodiment") == "gripper":
        # Centered straddle grasp: open, position above, descend around the object, close.
        # At base z 0.135 the finger pads span the full height of a 5 cm object.
        dz_descend = 0.085 + (0.025 - _half_height(setup))
        return [
            {"primitive": "hand_pose", "params": {"name": "open hand"}, "duration_s": 0.4},
            {"primitive": "approach_object", "params": {"ox": 0.0, "oy": 0.0, "z": 0.05}, "duration_s": 0.8},
            {"primitive": "move_delta", "params": {"dz": round(dz_descend, 4)}, "duration_s": 0.8},
            {
                "primitive": "grasp",
                "params": {"closure": 0.85},
                "duration_s": 1.4,
                "until": [{"metric": "contacts.hand_object_count", "op": ">=", "value": 2}],
            },
            {"primitive": "wait", "params": {}, "duration_s": 0.3},
        ]
    # Shadow Hand: object-centric grasp tuned after the Codex demonstration. Approach
    # high, descend shallow with a lateral bias so the object seats between the
    # fingertips, pre-shape, close hard, then settle reactively until the grasp holds.
    dz_descend = 0.0449 + (0.025 - _half_height(setup))  # descend lower for shorter objects
    return [
        {"primitive": "hand_pose", "params": {"name": "open hand"}, "duration_s": 0.25},
        # Object-centric first approach, while the hand is clear of the object.
        {"primitive": "approach_object", "params": {"ox": -0.0466, "oy": 0.0047, "z": 0.0587}, "duration_s": 0.704},
        # Fixed deltas from here on: re-centering on a possibly-shoved object mid-descend
        # chases its own disturbance, which is exactly how the Codex demo failed less.
        {"primitive": "move_delta", "params": {"dx": 0.0156, "dy": -0.0259, "dz": round(dz_descend, 4)}, "duration_s": 1.001},
        {"primitive": "hand_pose", "params": {"name": "pre grasp"}, "duration_s": 0.381},
        {"primitive": "hand_pose", "params": {"name": "grasp hard"}, "duration_s": 1.458},
        # Reactive settle: ends once the grasp is multi-finger, timeout as backstop.
        {
            "primitive": "wait",
            "params": {},
            "duration_s": 0.6,
            "until": [{"metric": "contacts.hand_object_count", "op": ">=", "value": 3}],
        },
    ]


def _lift_expert(name: str, setup: dict[str, Any]) -> dict[str, Any]:
    steps = _grasp_steps(setup) + [
        {"primitive": "move_delta", "params": {"dz": -0.12}, "duration_s": 0.862},
        {"primitive": "wait", "params": {}, "duration_s": 0.75},
    ]
    return {
        "name": name,
        "goal": TASKS["lift"].goal,
        "setup": setup,
        "steps": steps,
        "success": _lift_success(setup),
    }


# --- lift_perturbed -----------------------------------------------------------------

def _lift_perturbed_setup(rng: np.random.Generator, radius: float, object_set: str) -> dict[str, Any]:
    setup = _base_setup(rng, radius, object_set)
    angle = rng.uniform(0.0, 2.0 * np.pi)
    speed = rng.uniform(0.10, 0.25)
    setup["perturb"] = {
        "time_s": round(float(rng.uniform(0.8, 1.8)), 3),
        "velocity": [round(float(speed * np.cos(angle)), 4), round(float(speed * np.sin(angle)), 4), 0.0],
    }
    return setup


# --- place --------------------------------------------------------------------------

def _place_setup(rng: np.random.Generator, radius: float, object_set: str) -> dict[str, Any]:
    setup = _base_setup(rng, radius, object_set)
    _sample_target(rng, setup, 0.06, 0.12)
    return setup


def _place_success(setup: dict[str, Any]) -> list[dict[str, Any]]:
    tx, ty = setup["target"]
    return [
        {"metric": "object.x", "op": ">=", "value": round(tx - PLACE_TOLERANCE, 4)},
        {"metric": "object.x", "op": "<=", "value": round(tx + PLACE_TOLERANCE, 4)},
        {"metric": "object.y", "op": ">=", "value": round(ty - PLACE_TOLERANCE, 4)},
        {"metric": "object.y", "op": "<=", "value": round(ty + PLACE_TOLERANCE, 4)},
        {"metric": "lifted", "op": "==", "value": False},
    ]


def _place_score(result: dict[str, Any], setup: dict[str, Any]) -> float:
    max_contacts, max_z = _max_contacts_and_z(result)
    tx, ty = setup["target"]
    ox, oy = _final_xy(result)
    start = setup["object_pos"]
    start_dist = float(np.hypot(tx - start[0], ty - start[1]))
    final_dist = float(np.hypot(tx - ox, ty - oy))
    progress = max((start_dist - final_dist) / max(start_dist, 1e-6), -1.5)
    score = 0.1 * max_contacts + 1.2 * progress
    if max_z > _lift_threshold(setup):
        score += 0.3  # carried, not dragged
    if final_dist <= PLACE_TOLERANCE and not result.get("final_state", {}).get("lifted"):
        score += 1.0
    if result.get("success"):
        score += 0.25
    if result.get("errors"):
        score -= 0.5
    return round(float(score), 4)


def _place_expert(name: str, setup: dict[str, Any]) -> dict[str, Any]:
    tx, ty = setup["target"]
    # Grasp-point offset relative to the base, measured from each embodiment's expert grasp.
    gx, gy = (0.0, 0.0) if setup.get("embodiment") == "gripper" else (-0.031, -0.0212)
    z_set_down = 0.085 + (0.025 - _half_height(setup))
    steps = _grasp_steps(setup) + [
        {"primitive": "move_delta", "params": {"dz": -0.12}, "duration_s": 1.0},
        {"primitive": "set_base", "params": {"x": round(tx + gx, 4), "y": round(ty + gy, 4), "z": -0.0164}, "duration_s": 1.6},
        {"primitive": "wait", "params": {}, "duration_s": 0.4},
        {"primitive": "set_base", "params": {"x": round(tx + gx, 4), "y": round(ty + gy, 4), "z": round(z_set_down, 4)}, "duration_s": 1.0},
        {"primitive": "hand_pose", "params": {"name": "open hand"}, "duration_s": 0.6},
        {"primitive": "move_delta", "params": {"dz": -0.07}, "duration_s": 0.6},
        {"primitive": "wait", "params": {}, "duration_s": 0.5},
    ]
    return {
        "name": name,
        "goal": TASKS["place"].goal,
        "setup": setup,
        "steps": steps,
        "success": _place_success(setup),
    }


# --- push ---------------------------------------------------------------------------

def _push_setup(rng: np.random.Generator, radius: float, object_set: str) -> dict[str, Any]:
    setup = _base_setup(rng, radius, object_set)
    _sample_target(rng, setup, 0.05, 0.10)
    return setup


def _push_success(setup: dict[str, Any]) -> list[dict[str, Any]]:
    tx, ty = setup["target"]
    return [
        {"metric": "object.x", "op": ">=", "value": round(tx - PUSH_TOLERANCE, 4)},
        {"metric": "object.x", "op": "<=", "value": round(tx + PUSH_TOLERANCE, 4)},
        {"metric": "object.y", "op": ">=", "value": round(ty - PUSH_TOLERANCE, 4)},
        {"metric": "object.y", "op": "<=", "value": round(ty + PUSH_TOLERANCE, 4)},
    ]


def _push_score(result: dict[str, Any], setup: dict[str, Any]) -> float:
    tx, ty = setup["target"]
    ox, oy = _final_xy(result)
    start = setup["object_pos"]
    start_dist = float(np.hypot(tx - start[0], ty - start[1]))
    final_dist = float(np.hypot(tx - ox, ty - oy))
    progress = max((start_dist - final_dist) / max(start_dist, 1e-6), -1.5)
    score = 1.5 * progress
    if final_dist <= PUSH_TOLERANCE:
        score += 1.0
    if result.get("success"):
        score += 0.25
    if result.get("errors"):
        score -= 0.5
    return round(float(score), 4)


def _push_expert(name: str, setup: dict[str, Any]) -> dict[str, Any]:
    obj = setup["object_pos"]
    tx, ty = setup["target"]
    direction = np.array([tx - obj[0], ty - obj[1]])
    norm = float(np.linalg.norm(direction))
    unit = direction / max(norm, 1e-6)
    gripper = setup.get("embodiment") == "gripper"
    behind = 0.06
    gx = 0.0 if gripper else -0.035
    z_push = (0.10 if gripper else 0.13) + (0.025 - _half_height(setup))
    # The base travels `behind` plus the target distance, minus the standoff at which the
    # fist face meets the object face (object half-width + fist half-width), so the fist
    # stops with the object on target rather than shoved past it.
    standoff = _half_height(setup) + (0.015 if gripper else 0.045)
    travel = max(behind + norm - standoff, 0.01)
    steps = [
        {"primitive": "hand_pose", "params": {"name": "close hand"}, "duration_s": 0.4},
        {
            "primitive": "approach_object",
            "params": {"ox": round(float(gx - behind * unit[0]), 4), "oy": round(float(-behind * unit[1]), 4), "z": 0.06},
            "duration_s": 0.8,
        },
        {"primitive": "move_delta", "params": {"dz": round(z_push - 0.06, 4)}, "duration_s": 0.6},
        {
            "primitive": "move_delta",
            "params": {"dx": round(float(travel * unit[0]), 4), "dy": round(float(travel * unit[1]), 4)},
            "duration_s": 3.0,
        },
        {"primitive": "move_delta", "params": {"dz": -0.06}, "duration_s": 0.5},
        {"primitive": "wait", "params": {}, "duration_s": 0.4},
    ]
    return {
        "name": name,
        "goal": TASKS["push"].goal,
        "setup": setup,
        "steps": steps,
        "success": _push_success(setup),
    }


TASKS: dict[str, Task] = {}


def _register(task: Task) -> None:
    TASKS[task.name] = task


_register(Task(
    name="lift",
    goal="grasp the object and lift it above the lift threshold, holding it there",
    has_target=False,
    sample_setup=_lift_setup,
    success_checks=_lift_success,
    score=_lift_score,
    scripted_expert=_lift_expert,
))
_register(Task(
    name="lift_perturbed",
    goal="grasp the object and lift it above the lift threshold, holding it there; the object may be knocked mid-episode, so prefer reactive, robust schedules",
    has_target=False,
    sample_setup=_lift_perturbed_setup,
    success_checks=_lift_success,
    score=_lift_score,
    scripted_expert=_lift_expert,
))
_register(Task(
    name="place",
    goal="grasp the object, carry it to the target xy position, and set it down there",
    has_target=True,
    sample_setup=_place_setup,
    success_checks=_place_success,
    score=_place_score,
    scripted_expert=_place_expert,
))
_register(Task(
    name="push",
    goal="push the object along the table to the target xy position without needing to lift it",
    has_target=True,
    sample_setup=_push_setup,
    success_checks=_push_success,
    score=_push_score,
    scripted_expert=_push_expert,
))


def get_task(name: str) -> Task:
    if name not in TASKS:
        raise KeyError(f"unknown task {name!r}; available: {sorted(TASKS)}")
    return TASKS[name]

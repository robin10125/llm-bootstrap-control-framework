#!/usr/bin/env python3
"""Run a robot primitive schedule in a deterministic bootstrapping harness.

This is a representation experiment, not a physics simulator. The goal is to make the
policy surface explicit enough that an LLM can author or tune the same primitive sequence
that a conventional policy uses: ordered primitive calls with timing and parameters.
"""
from __future__ import annotations

import argparse
import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any


TABLE_Z = 0.025
CUBE_SIZE = 0.05
CUBE_Z_REST = TABLE_Z + CUBE_SIZE / 2
GRASP_Z_MIN = 0.055
GRASP_Z_MAX = 0.095
GRASP_XY_TOL = 0.025
MIN_GRASP_FORCE = 0.55


DEFAULT_STATE = {
    "time_s": 0.0,
    "tool": {"x": 0.0, "y": 0.0, "z": 0.18},
    "gripper": {"width": 0.06, "force": 0.0},
    "cube": {"x": 0.0, "y": 0.0, "z": CUBE_Z_REST},
    "grasped": False,
}


PRIMITIVES = {
    "move_tool": {
        "params": {"x": (-0.30, 0.30), "y": (-0.30, 0.30), "z": (0.04, 0.35)},
        "duration": (0.05, 5.0),
    },
    "open_gripper": {
        "params": {"width": (0.02, 0.08)},
        "duration": (0.02, 2.0),
    },
    "close_gripper": {
        "params": {"force": (0.0, 1.0)},
        "duration": (0.02, 2.0),
    },
    "lift": {
        "params": {"z": (0.06, 0.35)},
        "duration": (0.05, 5.0),
    },
    "wait": {
        "params": {},
        "duration": (0.01, 5.0),
    },
}


class PolicyError(ValueError):
    """Raised for invalid primitive schedules."""


def initial_state(setup: dict[str, Any] | None = None) -> dict[str, Any]:
    state = deepcopy(DEFAULT_STATE)
    setup = setup or {}
    cube_xy = setup.get("cube_xy", [0.0, 0.0])
    state["cube"]["x"] = float(cube_xy[0])
    state["cube"]["y"] = float(cube_xy[1])
    state["cube"]["z"] = CUBE_Z_REST
    return state


def validate_step(step: dict[str, Any], index: int) -> None:
    primitive = step.get("primitive")
    if primitive not in PRIMITIVES:
        raise PolicyError(f"step {index}: unknown primitive {primitive!r}")

    spec = PRIMITIVES[primitive]
    params = step.get("params", {})
    expected = set(spec["params"])
    actual = set(params)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing:
        raise PolicyError(f"step {index}: {primitive} missing params {missing}")
    if extra:
        raise PolicyError(f"step {index}: {primitive} has unknown params {extra}")

    for name, value in params.items():
        lo, hi = spec["params"][name]
        if not isinstance(value, (int, float)) or not lo <= float(value) <= hi:
            raise PolicyError(f"step {index}: {primitive}.{name}={value!r} outside [{lo}, {hi}]")

    duration = step.get("duration_s")
    lo, hi = spec["duration"]
    if not isinstance(duration, (int, float)) or not lo <= float(duration) <= hi:
        raise PolicyError(f"step {index}: duration_s={duration!r} outside [{lo}, {hi}]")


def run_policy(policy: dict[str, Any]) -> dict[str, Any]:
    state = initial_state(policy.get("setup"))
    trace: list[dict[str, Any]] = []
    errors: list[str] = []

    for idx, step in enumerate(policy.get("steps", []), start=1):
        try:
            validate_step(step, idx)
            before = deepcopy(state)
            apply_step(state, step)
            assertions = evaluate_checks(state, step.get("assert", []))
            trace.append({
                "step": idx,
                "primitive": step["primitive"],
                "params": step.get("params", {}),
                "duration_s": step["duration_s"],
                "before": before,
                "after": deepcopy(state),
                "assertions": assertions,
            })
            if assertions["unmet"]:
                errors.extend(f"step {idx}: {msg}" for msg in assertions["unmet"])
                break
        except PolicyError as exc:
            errors.append(str(exc))
            break

    success_checks = evaluate_checks(state, policy.get("success", []))
    success = not errors and not success_checks["unmet"]
    return {
        "policy": policy.get("name", "unnamed"),
        "goal": policy.get("goal", ""),
        "success": success,
        "errors": errors,
        "unmet": success_checks["unmet"],
        "final_state": state,
        "trace": trace,
    }


def apply_step(state: dict[str, Any], step: dict[str, Any]) -> None:
    primitive = step["primitive"]
    params = step.get("params", {})
    state["time_s"] = round(state["time_s"] + float(step["duration_s"]), 4)

    if primitive == "move_tool":
        state["tool"].update({k: float(params[k]) for k in ("x", "y", "z")})
        carry_cube_if_grasped(state)
        return

    if primitive == "open_gripper":
        state["gripper"]["width"] = float(params["width"])
        state["gripper"]["force"] = 0.0
        state["grasped"] = False
        settle_cube(state)
        return

    if primitive == "close_gripper":
        state["gripper"]["force"] = float(params["force"])
        state["gripper"]["width"] = 0.025
        state["grasped"] = can_grasp(state)
        carry_cube_if_grasped(state)
        return

    if primitive == "lift":
        state["tool"]["z"] = float(params["z"])
        carry_cube_if_grasped(state)
        return

    if primitive == "wait":
        carry_cube_if_grasped(state)
        settle_cube(state)
        return

    raise PolicyError(f"unhandled primitive {primitive!r}")


def can_grasp(state: dict[str, Any]) -> bool:
    tool = state["tool"]
    cube = state["cube"]
    xy_error = math.hypot(tool["x"] - cube["x"], tool["y"] - cube["y"])
    return (
        xy_error <= GRASP_XY_TOL
        and GRASP_Z_MIN <= tool["z"] <= GRASP_Z_MAX
        and state["gripper"]["force"] >= MIN_GRASP_FORCE
    )


def carry_cube_if_grasped(state: dict[str, Any]) -> None:
    if not state["grasped"]:
        settle_cube(state)
        return
    state["cube"]["x"] = state["tool"]["x"]
    state["cube"]["y"] = state["tool"]["y"]
    state["cube"]["z"] = max(CUBE_Z_REST, state["tool"]["z"] - CUBE_SIZE / 2)


def settle_cube(state: dict[str, Any]) -> None:
    if not state["grasped"]:
        state["cube"]["z"] = CUBE_Z_REST


def evaluate_checks(state: dict[str, Any], checks: list[dict[str, Any]]) -> dict[str, Any]:
    unmet = []
    for check in checks:
        metric = check["metric"]
        actual = metric_value(state, metric)
        op = check["op"]
        expected = check["value"]
        if not compare(actual, op, expected):
            unmet.append(f"{metric} {op} {expected!r} failed; actual={actual!r}")
    return {"ok": not unmet, "unmet": unmet}


def metric_value(state: dict[str, Any], metric: str) -> Any:
    value: Any = state
    for part in metric.split("."):
        value = value[part]
    return value


def compare(actual: Any, op: str, expected: Any) -> bool:
    if op == "==":
        return actual == expected
    if op == "!=":
        return actual != expected
    if op == ">=":
        return actual >= expected
    if op == ">":
        return actual > expected
    if op == "<=":
        return actual <= expected
    if op == "<":
        return actual < expected
    raise PolicyError(f"unknown check operator {op!r}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    policy = json.loads(args.policy.read_text())
    result = run_policy(policy)
    text = json.dumps(result, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
    print(text)
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

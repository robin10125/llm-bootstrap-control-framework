#!/usr/bin/env python3
"""Convert primitive-policy run traces into JSONL transitions for BC/RL bootstrap."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PRIMITIVE_IDS = {
    "move_tool": 0,
    "open_gripper": 1,
    "close_gripper": 2,
    "lift": 3,
    "wait": 4,
}


def obs_vector(state: dict[str, Any]) -> list[float]:
    return [
        float(state["tool"]["x"]),
        float(state["tool"]["y"]),
        float(state["tool"]["z"]),
        float(state["gripper"]["width"]),
        float(state["gripper"]["force"]),
        float(state["cube"]["x"]),
        float(state["cube"]["y"]),
        float(state["cube"]["z"]),
        1.0 if state["grasped"] else 0.0,
    ]


def action_vector(step: dict[str, Any]) -> list[float]:
    primitive = step["primitive"]
    params = step.get("params", {})
    return [
        float(PRIMITIVE_IDS[primitive]),
        float(params.get("x", 0.0)),
        float(params.get("y", 0.0)),
        float(params.get("z", 0.0)),
        float(params.get("width", 0.0)),
        float(params.get("force", 0.0)),
        float(step["duration_s"]),
    ]


def transition_rows(run: dict[str, Any], source: str, include_failures: bool) -> list[dict[str, Any]]:
    if not run.get("success") and not include_failures:
        return []

    rows = []
    trace = run.get("trace", [])
    for i, step in enumerate(trace):
        done = i == len(trace) - 1
        row = {
            "source": source,
            "policy": run.get("policy", "unnamed"),
            "goal": run.get("goal", ""),
            "success": bool(run.get("success")),
            "t": i,
            "obs": obs_vector(step["before"]),
            "action": action_vector(step),
            "next_obs": obs_vector(step["after"]),
            "reward": reward(run, step, done),
            "done": done,
            "primitive": step["primitive"],
            "params": step.get("params", {}),
            "duration_s": step["duration_s"],
        }
        rows.append(row)
    return rows


def reward(run: dict[str, Any], step: dict[str, Any], done: bool) -> float:
    after = step["after"]
    shaped = 0.0
    if after["grasped"]:
        shaped += 0.25
    shaped += max(0.0, float(after["cube"]["z"]) - 0.05)
    if done and run.get("success"):
        shaped += 1.0
    if done and not run.get("success"):
        shaped -= 1.0
    return round(shaped, 4)


def iter_run_files(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(sorted(path.glob("*.json")))
        else:
            files.append(path)
    return files


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--include-failures", action="store_true")
    args = parser.parse_args()

    count = 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for run_file in iter_run_files(args.runs):
            run = json.loads(run_file.read_text())
            for row in transition_rows(run, str(run_file), args.include_failures):
                f.write(json.dumps(row, separators=(",", ":")) + "\n")
                count += 1

    print(json.dumps({"out": str(args.out), "transitions": count}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

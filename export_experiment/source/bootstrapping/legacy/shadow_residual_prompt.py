#!/usr/bin/env python3
"""Build an LLM prompt for residual edits to a Shadow Hand primitive policy."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_PRIMITIVES = Path("shadow_primitives.json")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--primitives", type=Path, default=DEFAULT_PRIMITIVES)
    parser.add_argument("--goal", default="Improve the primitive schedule so the Shadow Hand grasps the cube more reliably.")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    policy = json.loads(args.policy.read_text())
    result = json.loads(args.result.read_text())
    primitives = json.loads(args.primitives.read_text())
    prompt = build_residual_prompt(policy, result, primitives, args.goal)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(prompt)
    print(prompt)
    return 0


def build_residual_prompt(
    policy: dict[str, Any],
    result: dict[str, Any],
    primitives: dict[str, Any],
    residual_goal: str,
) -> str:
    final_state = result.get("final_state", {})
    compact_trace = compact_trajectory(result.get("trace", []))
    scene = scene_context(final_state)

    return f"""You are acting as a residual policy editor for a Shadow Dexterous Hand in MuJoCo.

Your job is to edit an existing primitive schedule after seeing its robot trajectory. The
edited schedule is not just for this one rollout: it will be used as a demonstration sample
for a neural policy network to imitate. Make the policy simple, smooth, conservative, and
learnable. Prefer small parameter/timing edits over clever brittle sequences.

Return exactly one JSON object and no prose. The JSON must have the same schema as the
input policy: name, goal, setup, steps, and success. Use only the primitive vocabulary
listed below.

# Residual Editing Objective
{residual_goal}

# Scene And Orientation Facts
{scene}

# Primitive Vocabulary
```json
{json.dumps(primitives, indent=2)}
```

# Important Control Notes
- `set_base` commands the floating base position actuators.
- In this palm-down scene, positive `set_base.z` lowers the hand toward the table/object;
  decreasing `set_base.z` lifts the hand away from the table.
- `set_base.x` and `set_base.y` are actuator targets, not guaranteed instantaneous state.
  Use durations long enough for the hand to settle.
- `hand_pose` commands all Shadow Hand finger/wrist actuators toward a named keyframe pose.
- `hand_pose` preserves the current base target.
- `wait` preserves all current targets while stepping physics.
- A grasp demo should generally: open hand, approach above object, descend, pre-shape,
  close/grasp, optionally wait to settle, then lift only if contact is stable.

# Observation And Check Metrics
- `contacts.hand_object_count`: number of distinct Shadow Hand bodies touching the cube.
- `contacts.hand_object_bodies`: names of Shadow Hand bodies touching the cube.
- `grasped`: true when at least two distinct hand bodies contact the cube.
- `lifted`: true when object z exceeds 0.06m.
- `object.z`: cube center height in metres.

# Previous Policy JSON
```json
{json.dumps(policy, indent=2)}
```

# Previous Rollout Summary
```json
{json.dumps({
        "success": result.get("success"),
        "errors": result.get("errors", []),
        "unmet": result.get("unmet", []),
        "final_state": final_state,
        "compact_trajectory": compact_trace,
    }, indent=2)}
```

# Editing Requirements
- Output one complete replacement policy JSON object.
- Keep the action space primitive-level; do not emit Python or low-level actuator commands.
- Make each step explicit with `primitive`, `params`, and `duration_s`.
- Include assertions after important contact/lift phases when helpful.
- Preserve useful successful behavior from the previous rollout.
- If the previous rollout failed, alter sequence, timing, or primitive parameters to fix
  the failure.
- If the previous rollout succeeded, make it more robust and easier for a neural policy to
  imitate: fewer unnecessary steps, stable contacts, clear subgoals, and moderate durations.
- The resulting JSON will be executed by the primitive runner and successful traces will be
  added to behavior-cloning/RL bootstrap data.
"""


def scene_context(final_state: dict[str, Any]) -> str:
    object_pos = final_state.get("object", {})
    base = final_state.get("base", {})
    grasp_site = final_state.get("grasp_site")
    fingertips = final_state.get("fingertips", {})
    contacts = final_state.get("contacts", {})
    lines = [
        "- Robot: right Shadow Dexterous Hand, palm-down over a table, mounted on a floating translational base.",
        "- Object: cube on table in `scene_cube.xml`; table is near z=0 and cube center starts near z=0.025m.",
        "- Base qpos uses slide joints `slide_x`, `slide_y`, `slide_z`; positive base z lowers the palm in world coordinates.",
    ]
    if object_pos:
        lines.append(f"- Last observed object position: {json.dumps(object_pos, separators=(',', ':'))}.")
    if base:
        lines.append(f"- Last observed base qpos: {json.dumps(base, separators=(',', ':'))}.")
    if grasp_site is not None:
        lines.append(f"- Last observed palm grasp site position: {json.dumps(grasp_site, separators=(',', ':'))}.")
    if fingertips:
        lines.append(f"- Last observed fingertip positions: {json.dumps(fingertips, separators=(',', ':'))}.")
    if contacts:
        lines.append(f"- Last observed hand-object contacts: {json.dumps(contacts, separators=(',', ':'))}.")
    return "\n".join(lines)


def compact_trajectory(trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compact = []
    for step in trace:
        before = step.get("before", {})
        after = step.get("after", {})
        compact.append({
            "step": step.get("step"),
            "primitive": step.get("primitive"),
            "params": step.get("params", {}),
            "duration_s": step.get("duration_s"),
            "before": compact_obs(before),
            "after": compact_obs(after),
            "assertions": step.get("assertions", {}),
        })
    return compact


def compact_obs(obs: dict[str, Any]) -> dict[str, Any]:
    return {
        "time_s": obs.get("time_s"),
        "base": obs.get("base"),
        "commands": obs.get("commands"),
        "object": obs.get("object"),
        "grasp_site": obs.get("grasp_site"),
        "contacts": obs.get("contacts"),
        "grasped": obs.get("grasped"),
        "lifted": obs.get("lifted"),
    }


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Execute primitive schedules on the parallel-jaw gripper scene.

Same primitive language as the Shadow Hand runner; the hand-specific primitives map onto
the two-finger gripper:

- `hand_pose` names map to finger separations (open hand -> fully open, grasp/close
  poses -> closed; pinches are mid-separations).
- `grasp` closure in [0, 1] interpolates fully open -> fully closed.
- `set_base`, `move_delta`, `approach_object`, `wait`, `until`, object variation, and
  perturbation behave identically.

This exists to test that LLM bootstrapping is not Shadow-Hand-specific: select it with
`"embodiment": "gripper"` in a policy's setup.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from shadow_policy_runner import ShadowPolicyRunner

GRIPPER_MODEL = Path(__file__).resolve().parent / "models" / "gripper_scene.xml"

# Per-finger slide ctrl (metres) for each named pose. The finger joints' positive
# direction moves the fingers apart, so 0.06 = fully open and 0 = fully closed
# (a residual ~14 mm gap remains between the pads at 0).
FULLY_OPEN = 0.06
FULLY_CLOSED = 0.0
POSE_SEPARATIONS = {
    "open hand": 0.06,
    "pre grasp": 0.045,
    "grasp soft": 0.015,
    "grasp hard": 0.0,
    "grasp sphere": 0.01,
    "two finger pinch": 0.005,
    "three finger pinch": 0.01,
    "close hand": 0.0,
}


class GripperPolicyRunner(ShadowPolicyRunner):
    HAND_PREFIX = "g_"

    def __init__(self, model_path: Path = GRIPPER_MODEL, object_spec: dict[str, Any] | None = None):
        # keyframes_path is unused (the pose-target loader is overridden) but must be a Path.
        super().__init__(model_path=model_path, keyframes_path=model_path, object_spec=object_spec)

    def _load_pose_targets(self, keyframes_path) -> dict[str, np.ndarray]:
        targets = {}
        for name, separation in POSE_SEPARATIONS.items():
            ctrl = np.zeros(self.model.nu)
            for i in range(self.model.nu):
                if self.model.actuator(i).name in {"g_left", "g_right"}:
                    ctrl[i] = separation
            targets[name] = ctrl
        return targets

    def _palm_pos(self) -> list[float]:
        return self.data.body("g_palm").xpos.round(4).tolist()

    def _fingertips(self) -> dict[str, list[float]]:
        return {
            "left": self.data.body("g_left_finger").xpos.round(4).tolist(),
            "right": self.data.body("g_right_finger").xpos.round(4).tolist(),
        }


def main() -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    policy = json.loads(args.policy.read_text())
    result = GripperPolicyRunner(object_spec=policy.get("setup", {}).get("object")).run(policy)
    text = json.dumps(result, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
    print(text)
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

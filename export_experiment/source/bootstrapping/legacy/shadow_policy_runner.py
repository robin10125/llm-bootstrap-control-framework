#!/usr/bin/env python3
"""Execute primitive schedules against MuJoCo manipulation scenes.

Primitive language (v2):

- `set_base`: absolute floating-base position targets.
- `move_delta`: base targets relative to the current commanded targets.
- `approach_object`: base x/y targets relative to the object's current position
  (sensor-conditioned at command time), absolute base z.
- `hand_pose`: named keyframe pose macros.
- `grasp`: continuous closure in [0, 1] interpolating open hand -> grasp hard.
- `wait`: step physics without changing targets.

Any step may carry an `until` list of checks (same schema as `assert`); the step then
ends as soon as the checks hold, with `duration_s` acting as a timeout. The trace records
the elapsed time and whether the condition was met.

Setups may specify object variation and mid-episode perturbations:

```json
"setup": {
  "object_pos": [x, y, z],
  "object": {"shape": "box|sphere|cylinder", "size": [...], "mass": 0.05, "friction": 1.0},
  "perturb": {"time_s": 2.0, "velocity": [vx, vy, vz]}
}
```
"""
from __future__ import annotations

import argparse
import json
import math
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "hand-manipulation" / "env" / "models" / "shadow_hand" / "scene_cube.xml"
DEFAULT_KEYFRAMES = ROOT / "hand-manipulation" / "env" / "models" / "shadow_hand" / "keyframes.xml"

HAND_JOINT_ORDER = [
    "rh_WRJ2", "rh_WRJ1",
    "rh_FFJ4", "rh_FFJ3", "rh_FFJ2", "rh_FFJ1",
    "rh_MFJ4", "rh_MFJ3", "rh_MFJ2", "rh_MFJ1",
    "rh_RFJ4", "rh_RFJ3", "rh_RFJ2", "rh_RFJ1",
    "rh_LFJ5", "rh_LFJ4", "rh_LFJ3", "rh_LFJ2", "rh_LFJ1",
    "rh_THJ5", "rh_THJ4", "rh_THJ3", "rh_THJ2", "rh_THJ1",
]

BASE_RANGES = {
    "x": (-0.30, 0.30),
    "y": (-0.30, 0.30),
    "z": (-0.15, 0.20),
}

DELTA_RANGE = 0.12

POSES = {
    "open hand",
    "pre grasp",
    "grasp soft",
    "grasp hard",
    "grasp sphere",
    "two finger pinch",
    "three finger pinch",
    "close hand",
}

OBJECT_SHAPES = {"box", "sphere", "cylinder"}
UNTIL_CHECK_INTERVAL_S = 0.02
LIFT_CLEARANCE = 0.035


class PolicyError(ValueError):
    """Raised for invalid primitive schedules."""


class ShadowPolicyRunner:
    HAND_PREFIX = "rh_"
    BASE_ACTUATORS = {"base_x", "base_y", "base_z"}
    OPEN_POSE = "open hand"
    CLOSED_POSE = "grasp hard"

    def __init__(
        self,
        model_path: Path = DEFAULT_MODEL,
        keyframes_path: Path = DEFAULT_KEYFRAMES,
        object_spec: dict[str, Any] | None = None,
    ):
        self.object_spec = normalize_object_spec(object_spec)
        self.model = self._load_model(Path(model_path), self.object_spec)
        self.data = mujoco.MjData(self.model)
        self.object_half_height = object_half_height(self.object_spec)
        self.pose_targets = self._load_pose_targets(Path(keyframes_path))
        self.last_pose = None
        self._pending_perturb: dict[str, Any] | None = None
        self.reset({})

    def _load_model(self, model_path: Path, object_spec: dict[str, Any] | None) -> mujoco.MjModel:
        if object_spec is None:
            return mujoco.MjModel.from_xml_path(str(model_path))
        xml = rewrite_object_xml(model_path, object_spec)
        # Write next to the original so relative meshdir/asset paths resolve.
        with tempfile.NamedTemporaryFile("w", dir=model_path.parent, suffix=".xml", delete=False) as f:
            f.write(xml)
            tmp_path = Path(f.name)
        try:
            return mujoco.MjModel.from_xml_path(str(tmp_path))
        finally:
            tmp_path.unlink(missing_ok=True)

    def _load_pose_targets(self, keyframes_path: Path) -> dict[str, np.ndarray]:
        keyframes = self._load_keyframes(keyframes_path)
        return {
            name: self._pose_to_ctrl(joints)
            for name, joints in keyframes.items()
            if name in POSES
        }

    def reset(self, setup: dict[str, Any]) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.data.ctrl[:] = 0.0
        self.last_pose = None
        object_pos = setup.get("object_pos")
        if object_pos is not None:
            adr = self._object_qposadr()
            self.data.qpos[adr:adr + 3] = [float(v) for v in object_pos]
            self.data.qpos[adr + 3:adr + 7] = [1.0, 0.0, 0.0, 0.0]
        perturb = setup.get("perturb")
        self._pending_perturb = dict(perturb) if perturb else None
        mujoco.mj_forward(self.model, self.data)
        self._step_seconds(float(setup.get("settle_s", 0.25)))

    def run(self, policy: dict[str, Any]) -> dict[str, Any]:
        self.reset(policy.get("setup", {}))
        trace = []
        errors = []

        for idx, step in enumerate(policy.get("steps", []), start=1):
            try:
                self._validate_step(step, idx)
                before = self.observe()
                elapsed, until_met = self._apply_step(step)
                after = self.observe()
                assertions = evaluate_checks(after, step.get("assert", []))
                trace.append({
                    "step": idx,
                    "primitive": step["primitive"],
                    "params": step.get("params", {}),
                    "duration_s": step["duration_s"],
                    "elapsed_s": round(elapsed, 4),
                    "until_met": until_met,
                    "before": before,
                    "after": after,
                    "assertions": assertions,
                })
                if assertions["unmet"]:
                    errors.extend(f"step {idx}: {msg}" for msg in assertions["unmet"])
                    break
            except PolicyError as exc:
                errors.append(str(exc))
                break

        final_state = self.observe()
        success_checks = evaluate_checks(final_state, policy.get("success", []))
        success = not errors and not success_checks["unmet"]
        return {
            "policy": policy.get("name", "unnamed"),
            "goal": policy.get("goal", ""),
            "success": success,
            "errors": errors,
            "unmet": success_checks["unmet"],
            "final_state": final_state,
            "trace": trace,
        }

    def observe(self) -> dict[str, Any]:
        contacts = self._object_contacts()
        object_pos = self.data.body("object").xpos
        grasp_site = self.data.site(self.model.site("grasp_site").id).xpos
        return {
            "time_s": round(float(self.data.time), 4),
            "base": {
                "x": round(float(self.data.qpos[self.model.joint("slide_x").qposadr[0]]), 4),
                "y": round(float(self.data.qpos[self.model.joint("slide_y").qposadr[0]]), 4),
                "z": round(float(self.data.qpos[self.model.joint("slide_z").qposadr[0]]), 4),
            },
            "commands": {
                "base_x": round(float(self.data.ctrl[self.model.actuator("base_x").id]), 4),
                "base_y": round(float(self.data.ctrl[self.model.actuator("base_y").id]), 4),
                "base_z": round(float(self.data.ctrl[self.model.actuator("base_z").id]), 4),
                "hand_pose": self.last_pose,
            },
            "palm": self._palm_pos(),
            "grasp_site": grasp_site.round(4).tolist(),
            "object": {
                "x": round(float(object_pos[0]), 4),
                "y": round(float(object_pos[1]), 4),
                "z": round(float(object_pos[2]), 4),
            },
            "fingertips": self._fingertips(),
            "contacts": {
                "object_pairs": contacts["pairs"],
                "hand_object_bodies": contacts["hand_bodies"],
                "hand_object_count": len(contacts["hand_bodies"]),
            },
            "grasped": bool(len(contacts["hand_bodies"]) >= 2),
            "lifted": bool(object_pos[2] > self.object_half_height + LIFT_CLEARANCE),
        }

    def _palm_pos(self) -> list[float]:
        return self.data.body("rh_palm").xpos.round(4).tolist()

    def _fingertips(self) -> dict[str, list[float]]:
        return {
            short: self.data.body(body).xpos.round(4).tolist()
            for short, body in {
                "ff": "rh_ffdistal",
                "mf": "rh_mfdistal",
                "rf": "rh_rfdistal",
                "lf": "rh_lfdistal",
                "th": "rh_thdistal",
            }.items()
        }

    def _apply_step(self, step: dict[str, Any]) -> tuple[float, bool | None]:
        self._command_step(step)
        return self._step_seconds(float(step["duration_s"]), until=step.get("until"))

    def _base_ctrl(self, axis: str) -> float:
        return float(self.data.ctrl[self.model.actuator(f"base_{axis}").id])

    def _set_base_ctrl(self, x: float, y: float, z: float) -> None:
        self.data.ctrl[self.model.actuator("base_x").id] = clip_range(x, BASE_RANGES["x"])
        self.data.ctrl[self.model.actuator("base_y").id] = clip_range(y, BASE_RANGES["y"])
        self.data.ctrl[self.model.actuator("base_z").id] = clip_range(z, BASE_RANGES["z"])

    def _set_hand_ctrl(self, targets: np.ndarray) -> None:
        for i in range(self.model.nu):
            if self.model.actuator(i).name not in self.BASE_ACTUATORS:
                self.data.ctrl[i] = targets[i]

    def _command_step(self, step: dict[str, Any]) -> None:
        primitive = step["primitive"]
        params = step.get("params", {})
        if primitive == "set_base":
            self._set_base_ctrl(float(params["x"]), float(params["y"]), float(params["z"]))
        elif primitive == "move_delta":
            self._set_base_ctrl(
                self._base_ctrl("x") + float(params.get("dx", 0.0)),
                self._base_ctrl("y") + float(params.get("dy", 0.0)),
                self._base_ctrl("z") + float(params.get("dz", 0.0)),
            )
        elif primitive == "approach_object":
            obj = self.data.body("object").xpos
            self._set_base_ctrl(
                float(obj[0]) + float(params.get("ox", 0.0)),
                float(obj[1]) + float(params.get("oy", 0.0)),
                float(params["z"]),
            )
        elif primitive == "hand_pose":
            pose_name = params["name"]
            self._set_hand_ctrl(self.pose_targets[pose_name])
            self.last_pose = pose_name
        elif primitive == "grasp":
            closure = float(params["closure"])
            open_ctrl = self.pose_targets[self.OPEN_POSE]
            closed_ctrl = self.pose_targets[self.CLOSED_POSE]
            self._set_hand_ctrl(open_ctrl + closure * (closed_ctrl - open_ctrl))
            self.last_pose = f"closure:{closure:.2f}"
        elif primitive == "wait":
            pass
        else:
            raise PolicyError(f"unknown primitive {primitive!r}")

    def _step_seconds(
        self,
        duration_s: float,
        frame_callback=None,
        fps: int = 30,
        until: list[dict[str, Any]] | None = None,
    ) -> tuple[float, bool | None]:
        n = max(1, int(math.ceil(duration_s / self.model.opt.timestep)))
        start_t = float(self.data.time)
        next_frame_t = start_t
        frame_dt = 1.0 / fps if fps > 0 else duration_s + 1.0
        next_check_t = start_t
        until_met: bool | None = None if not until else False
        for _ in range(n):
            self._apply_pending_perturb()
            mujoco.mj_step(self.model, self.data)
            now = float(self.data.time)
            if frame_callback is not None and now + 1e-12 >= next_frame_t:
                frame_callback()
                next_frame_t += frame_dt
            if until and now + 1e-12 >= next_check_t:
                next_check_t += UNTIL_CHECK_INTERVAL_S
                if not evaluate_checks(self.observe(), until)["unmet"]:
                    until_met = True
                    break
        return float(self.data.time) - start_t, until_met

    def _apply_pending_perturb(self) -> None:
        if self._pending_perturb is None:
            return
        if float(self.data.time) < float(self._pending_perturb.get("time_s", 0.0)):
            return
        velocity = [float(v) for v in self._pending_perturb.get("velocity", [0.0, 0.0, 0.0])]
        adr = self._object_qveladr()
        self.data.qvel[adr:adr + 3] += velocity
        self._pending_perturb = None

    def _validate_step(self, step: dict[str, Any], idx: int) -> None:
        primitive = step.get("primitive")
        params = step.get("params", {})
        duration = step.get("duration_s")
        if not isinstance(duration, (int, float)) or not 0.01 <= float(duration) <= 10.0:
            raise PolicyError(f"step {idx}: invalid duration_s={duration!r}")
        until = step.get("until")
        if until is not None:
            if not isinstance(until, list) or not all(
                isinstance(c, dict) and {"metric", "op", "value"} <= set(c) for c in until
            ):
                raise PolicyError(f"step {idx}: until must be a list of metric/op/value checks")

        if primitive == "set_base":
            expected = set(BASE_RANGES)
            if set(params) != expected:
                raise PolicyError(f"step {idx}: set_base params must be {sorted(expected)}")
            for key, (lo, hi) in BASE_RANGES.items():
                value = params[key]
                if not isinstance(value, (int, float)) or not lo <= float(value) <= hi:
                    raise PolicyError(f"step {idx}: set_base.{key}={value!r} outside [{lo}, {hi}]")
            return

        if primitive == "move_delta":
            if not params or not set(params) <= {"dx", "dy", "dz"}:
                raise PolicyError(f"step {idx}: move_delta params must be a subset of [dx, dy, dz]")
            for key, value in params.items():
                if not isinstance(value, (int, float)) or not -DELTA_RANGE <= float(value) <= DELTA_RANGE:
                    raise PolicyError(f"step {idx}: move_delta.{key}={value!r} outside [-{DELTA_RANGE}, {DELTA_RANGE}]")
            return

        if primitive == "approach_object":
            if "z" not in params or not set(params) <= {"ox", "oy", "z"}:
                raise PolicyError(f"step {idx}: approach_object params must be [ox?, oy?, z]")
            for key in ("ox", "oy"):
                value = params.get(key, 0.0)
                if not isinstance(value, (int, float)) or not -DELTA_RANGE <= float(value) <= DELTA_RANGE:
                    raise PolicyError(f"step {idx}: approach_object.{key}={value!r} outside [-{DELTA_RANGE}, {DELTA_RANGE}]")
            z = params["z"]
            lo, hi = BASE_RANGES["z"]
            if not isinstance(z, (int, float)) or not lo <= float(z) <= hi:
                raise PolicyError(f"step {idx}: approach_object.z={z!r} outside [{lo}, {hi}]")
            return

        if primitive == "hand_pose":
            pose_name = params.get("name")
            if set(params) != {"name"} or pose_name not in self.pose_targets:
                raise PolicyError(f"step {idx}: unknown hand_pose {pose_name!r}")
            return

        if primitive == "grasp":
            closure = params.get("closure")
            if set(params) != {"closure"} or not isinstance(closure, (int, float)) or not 0.0 <= float(closure) <= 1.0:
                raise PolicyError(f"step {idx}: grasp.closure={closure!r} outside [0, 1]")
            return

        if primitive == "wait":
            if params:
                raise PolicyError(f"step {idx}: wait takes no params")
            return

        raise PolicyError(f"step {idx}: unknown primitive {primitive!r}")

    def _load_keyframes(self, path: Path) -> dict[str, dict[str, float]]:
        root = ET.parse(path).getroot()
        out: dict[str, dict[str, float]] = {}
        keyframe = root.find("keyframe")
        if keyframe is None:
            raise PolicyError(f"no <keyframe> element found in {path}")
        for key in keyframe:
            values = [float(v) for v in key.attrib["qpos"].split()]
            out[key.attrib["name"]] = dict(zip(HAND_JOINT_ORDER, values))
        return out

    def _pose_to_ctrl(self, pose: dict[str, float]) -> np.ndarray:
        ctrl = np.zeros(self.model.nu)
        for i in range(self.model.nu):
            name = self.model.actuator(i).name
            if name in self.BASE_ACTUATORS:
                continue
            target = self._actuator_pose_target(name, pose)
            lo, hi = self.model.actuator(i).ctrlrange
            ctrl[i] = float(np.clip(target, lo, hi))
        return ctrl

    @staticmethod
    def _actuator_pose_target(actuator_name: str, pose: dict[str, float]) -> float:
        suffix = actuator_name.removeprefix("rh_A_")
        if suffix.endswith("J0"):
            prefix = "rh_" + suffix[:2]
            return pose[prefix + "J2"] + pose[prefix + "J1"]
        return pose["rh_" + suffix]

    def _object_qposadr(self) -> int:
        for i in range(self.model.njnt):
            joint = self.model.joint(i)
            if self.model.body(self.model.jnt_bodyid[i]).name == "object":
                return int(joint.qposadr[0])
        raise PolicyError("object freejoint not found")

    def _object_qveladr(self) -> int:
        for i in range(self.model.njnt):
            if self.model.body(self.model.jnt_bodyid[i]).name == "object":
                return int(self.model.jnt_dofadr[i])
        raise PolicyError("object freejoint not found")

    def _object_contacts(self) -> dict[str, Any]:
        pairs = []
        hand_bodies = set()
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1 = self.model.geom(c.geom1)
            g2 = self.model.geom(c.geom2)
            b1 = self.model.body(self.model.geom_bodyid[c.geom1]).name
            b2 = self.model.body(self.model.geom_bodyid[c.geom2]).name
            if "object" not in {b1, b2}:
                continue
            n1 = g1.name or b1
            n2 = g2.name or b2
            pairs.append([n1, n2])
            other_body = b2 if b1 == "object" else b1
            if other_body.startswith(self.HAND_PREFIX):
                hand_bodies.add(other_body)
        return {"pairs": pairs, "hand_bodies": sorted(hand_bodies)}


def runner_for_policy(policy: dict[str, Any]) -> ShadowPolicyRunner:
    """Build the right runner for a policy's setup (object variation, embodiment)."""
    setup = policy.get("setup", {})
    if setup.get("embodiment") == "gripper":
        from gripper_policy_runner import GripperPolicyRunner
        return GripperPolicyRunner(object_spec=setup.get("object"))
    return ShadowPolicyRunner(object_spec=setup.get("object"))


def normalize_object_spec(spec: dict[str, Any] | None) -> dict[str, Any] | None:
    if spec is None:
        return None
    shape = spec.get("shape", "box")
    if shape not in OBJECT_SHAPES:
        raise PolicyError(f"unknown object shape {shape!r}")
    size = [float(v) for v in spec.get("size", [0.025, 0.025, 0.025])]
    expected = {"box": 3, "sphere": 1, "cylinder": 2}[shape]
    if len(size) != expected:
        raise PolicyError(f"object shape {shape!r} needs {expected} size values, got {size}")
    out: dict[str, Any] = {"shape": shape, "size": size}
    if "mass" in spec:
        out["mass"] = float(spec["mass"])
    if "friction" in spec:
        friction = spec["friction"]
        if isinstance(friction, (int, float)):
            out["friction"] = [float(friction), 0.05, 0.001]
        else:
            out["friction"] = [float(v) for v in friction]
    return out


def object_half_height(spec: dict[str, Any] | None) -> float:
    if spec is None:
        return 0.025
    if spec["shape"] == "box":
        return spec["size"][2]
    if spec["shape"] == "sphere":
        return spec["size"][0]
    return spec["size"][1]  # cylinder half-height


def rewrite_object_xml(model_path: Path, spec: dict[str, Any]) -> str:
    tree = ET.parse(model_path)
    geom = tree.getroot().find(".//body[@name='object']/geom")
    if geom is None:
        raise PolicyError(f"no object geom found in {model_path}")
    geom.set("type", spec["shape"])
    geom.set("size", " ".join(f"{v:g}" for v in spec["size"]))
    if "mass" in spec:
        geom.set("mass", f"{spec['mass']:g}")
    if "friction" in spec:
        geom.set("friction", " ".join(f"{v:g}" for v in spec["friction"]))
    return ET.tostring(tree.getroot(), encoding="unicode")


def clip_range(value: float, bounds: tuple[float, float]) -> float:
    return min(max(float(value), bounds[0]), bounds[1])


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
    result = runner_for_policy(policy).run(policy)
    text = json.dumps(result, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
    print(text)
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

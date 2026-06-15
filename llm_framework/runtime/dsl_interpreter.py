from __future__ import annotations

from typing import Any

import numpy as np

from llm_framework.core.state import CandidateProgram, CompiledPolicy, WorldState
from llm_framework.runtime.appendages import appendage_joint_map
from llm_framework.runtime.controllers import (
    actuator_index,
    ctrl_limits,
    ctrl_open,
    duration_to_steps,
    hand_shape_target,
    latent_decode_target,
    repeat_target,
    targets_to_actions,
)
from llm_framework.runtime.dsl_macros import expand_generated_primitives


def compile_dsl(program: CandidateProgram, env: Any, world: WorldState) -> CompiledPolicy:
    source = expand_generated_primitives(program.source)
    idx = actuator_index(env)
    appendages = appendage_joint_map(env)
    lo, hi = ctrl_limits(env)
    current = ctrl_open(env).copy()
    targets: list[np.ndarray] = []
    trace: list[dict[str, Any]] = []

    def append(duration_s: float, next_target: np.ndarray, op: str) -> None:
        nonlocal current
        n = duration_to_steps(env, duration_s)
        next_target = np.clip(next_target, lo, hi).astype(np.float32)
        targets.append(repeat_target(next_target, n))
        trace.append({"op": op, "duration_s": duration_s, "steps": n})
        current = next_target

    for block in source.get("blocks", []):
        op = block["op"]
        duration = float(block.get("duration_s", block.get("timeout_s", env.cfg.control_dt)))
        next_target = current.copy()
        if op == "set_base_target":
            target = block.get("target", {})
            for axis in ("x", "y", "z"):
                name = f"base_{axis}"
                if axis in block and name in idx:
                    next_target[idx[name]] = float(block[axis])
                elif isinstance(target, dict) and name in target and name in idx:
                    next_target[idx[name]] = float(target[name])
                elif isinstance(target, dict) and axis in target and name in idx:
                    next_target[idx[name]] = float(target[axis])
            append(duration, next_target, op)
        elif op == "move_frame":
            target = block.get("target", {})
            offset = np.asarray(target.get("offset", [0.0, 0.0, 0.0]), dtype=float)
            if target.get("object") == "object" and {"base_x", "base_y"} <= idx.keys():
                next_target[idx["base_x"]] = float(world.object_pos[0] + offset[0])
                next_target[idx["base_y"]] = float(world.object_pos[1] + offset[1])
                if "base_z" in idx and len(offset) > 2:
                    next_target[idx["base_z"]] = float(offset[2])
            append(duration, next_target, op)
        elif op == "track_relative_pose":
            offset = np.asarray(block.get("offset", [0.0, 0.0, 0.0]), dtype=float)
            if {"base_x", "base_y"} <= idx.keys():
                next_target[idx["base_x"]] = float(world.object_pos[0] + offset[0])
                next_target[idx["base_y"]] = float(world.object_pos[1] + offset[1])
            if "base_z" in idx and len(offset) > 2:
                next_target[idx["base_z"]] = float(offset[2])
            append(duration, next_target, op)
        elif op in {"set_joint_targets", "set_appendage_joints"}:
            joints = _joint_targets_for_block(block, appendages)
            for name, value in joints.items():
                if name in idx:
                    next_target[idx[name]] = float(value)
            append(duration, next_target, op)
        elif op == "set_joint_target":
            name = str(block.get("joint", ""))
            if name in idx:
                next_target[idx[name]] = float(block.get("target", block.get("value", current[idx[name]])))
            append(duration, next_target, op)
        elif op == "move_joint_delta":
            name = str(block.get("joint", ""))
            if name in idx:
                next_target[idx[name]] += float(block.get("delta", 0.0))
            append(duration, next_target, op)
        elif op in {"set_hand_shape", "seek_contact", "maintain_contact"}:
            fraction = float(block.get("fraction_closed", block.get("closure", 0.65)))
            if op == "seek_contact":
                fraction = max(fraction, 0.55)
            if op == "maintain_contact":
                fraction = max(fraction, 0.75)
            shaped = hand_shape_target(env, fraction)
            for i in getattr(env, "hand_act_ids", range(env.nu)):
                next_target[i] = shaped[i]
            append(duration, next_target, op)
        elif op == "set_impedance":
            append(duration, next_target, op)
        elif op == "apply_wrench_or_impulse":
            direction = np.asarray(block.get("direction", [0.0, 0.0, 0.0]), dtype=float)
            magnitude = float(block.get("magnitude", 0.03))
            if np.linalg.norm(direction) > 1e-6 and {"base_x", "base_y"} <= idx.keys():
                unit = direction / np.linalg.norm(direction)
                next_target[idx["base_x"]] += float(unit[0] * magnitude * 0.04)
                next_target[idx["base_y"]] += float(unit[1] * magnitude * 0.04)
            append(duration, next_target, op)
        elif op in {"call_controller", "latent_decode"}:
            token = str(block.get("token") or block.get("controller") or "oppose_and_stabilize")
            gain = float(block.get("gain", 0.6))
            next_target = latent_decode_target(
                env, token, gain, current, (float(world.object_pos[0]), float(world.object_pos[1]))
            )
            append(duration, next_target, op)
        elif op == "call_appendage_agent":
            joints = _appendage_agent_targets(block, appendages, current)
            for name, value in joints.items():
                if name in idx:
                    next_target[idx[name]] = float(value)
            append(duration, next_target, op)
        elif op in {"wait", "monitor", "call_subscript"}:
            append(duration, next_target, op)
        elif op == "return":
            break

    if targets:
        target_stream = np.concatenate(targets, axis=0)
    else:
        target_stream = repeat_target(current, env.horizon)
    if target_stream.shape[0] < env.horizon:
        pad = repeat_target(target_stream[-1], env.horizon - target_stream.shape[0])
        target_stream = np.concatenate([target_stream, pad], axis=0)
    target_stream = target_stream[: env.horizon]
    return CompiledPolicy(
        interface=program.interface,
        action_stream=targets_to_actions(env, target_stream),
        metadata={"trace": trace, "target_steps": int(target_stream.shape[0])},
    )


def _joint_targets_for_block(block: dict[str, Any], appendages: dict[str, list[str]]) -> dict[str, float]:
    targets = dict(block.get("targets", {}))
    appendage = block.get("appendage")
    if appendage and "values" in block:
        names = appendages.get(str(appendage), [])
        values = block["values"]
        if isinstance(values, dict):
            targets.update(values)
        elif isinstance(values, list):
            targets.update({name: value for name, value in zip(names, values, strict=False)})
        else:
            targets.update({name: float(values) for name in names})
    return {str(k): float(v) for k, v in targets.items()}


def _appendage_agent_targets(
    block: dict[str, Any],
    appendages: dict[str, list[str]],
    current: np.ndarray,
) -> dict[str, float]:
    appendage = str(block.get("appendage", ""))
    allowed = set(appendages.get(appendage, []))
    if not allowed:
        return {}
    if isinstance(block.get("targets"), dict):
        return {str(k): float(v) for k, v in block["targets"].items() if str(k) in allowed}
    program = block.get("program", {})
    targets: dict[str, float] = {}
    for nested in program.get("blocks", []) if isinstance(program, dict) else []:
        op = nested.get("op")
        if op == "set_joint_target":
            name = str(nested.get("joint", ""))
            if name in allowed:
                targets[name] = float(nested.get("target", nested.get("value", 0.0)))
        elif op == "move_joint_delta":
            name = str(nested.get("joint", ""))
            if name in allowed:
                # Current values are unavailable by name here; keep deltas explicit by
                # requiring the nested program to include a target unless this is a no-op.
                targets.setdefault(name, float(nested.get("target", 0.0)))
        elif op in {"set_joint_targets", "set_appendage_joints"}:
            for name, value in _joint_targets_for_block(nested, {appendage: list(allowed)}).items():
                if name in allowed:
                    targets[name] = value
    return targets

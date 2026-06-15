from __future__ import annotations

from typing import Any

from llm_framework.core.state import CandidateProgram, SafetyLimits, ValidationResult
from llm_framework.runtime.dsl_macros import expand_generated_primitives


COMMON_OPS = {
    "set_base_target",
    "move_frame",
    "track_relative_pose",
    "set_joint_targets",
    "set_joint_target",
    "move_joint_delta",
    "set_appendage_joints",
    "call_appendage_agent",
    "set_hand_shape",
    "set_impedance",
    "seek_contact",
    "maintain_contact",
    "apply_wrench_or_impulse",
    "wait",
    "monitor",
    "call_subscript",
    "call_controller",
    "latent_decode",
    "return",
}


def validate_dsl(program: CandidateProgram, limits: SafetyLimits, *, allow_calls: bool = True) -> ValidationResult:
    errors: list[str] = []
    source = expand_generated_primitives(program.source)
    blocks = source.get("blocks")
    if not isinstance(blocks, list) or not blocks:
        errors.append("program must contain a non-empty blocks list")
        return ValidationResult(False, tuple(errors))
    if len(blocks) > limits.max_blocks:
        errors.append(f"too many blocks: {len(blocks)} > {limits.max_blocks}")

    for i, block in enumerate(blocks):
        op = block.get("op")
        if not isinstance(op, str):
            errors.append(f"block {i} missing string op")
            continue
        lowered = op.lower()
        generated_names = set((source.get("generated_primitives") or {}).keys()) if isinstance(source.get("generated_primitives"), dict) else set()
        if lowered in limits.forbidden_symbols and op not in generated_names:
            errors.append(f"block {i} uses forbidden task primitive {op!r}")
        if lowered not in COMMON_OPS:
            errors.append(f"block {i} has unknown op {op!r}")
        if lowered in {"call_subscript", "call_controller", "call_appendage_agent"} and not allow_calls:
            errors.append(f"block {i} uses calls but this interface forbids them")
        duration = block.get("duration_s", block.get("timeout_s", 0.0))
        if duration is not None:
            try:
                if float(duration) > limits.max_episode_seconds:
                    errors.append(f"block {i} duration exceeds episode limit")
            except (TypeError, ValueError):
                errors.append(f"block {i} duration is not numeric")
        for field in ("name", "controller", "subscript", "token"):
            value = block.get(field)
            if isinstance(value, str) and value.lower() in limits.forbidden_symbols:
                errors.append(f"block {i} field {field} uses forbidden symbol {value!r}")
        if "max_force" in block and float(block["max_force"]) > limits.max_force_n:
            errors.append(f"block {i} max_force exceeds safety limit")
        if "magnitude" in block and float(block["magnitude"]) > limits.max_impulse:
            errors.append(f"block {i} magnitude exceeds impulse limit")
        if lowered == "call_appendage_agent":
            if not isinstance(block.get("appendage"), str):
                errors.append(f"block {i} call_appendage_agent needs appendage")
            if not isinstance(block.get("program", block.get("targets", {})), (dict, list)):
                errors.append(f"block {i} call_appendage_agent needs program or targets")
            program = block.get("program")
            if isinstance(program, dict):
                nested = program.get("blocks")
                if not isinstance(nested, list):
                    errors.append(f"block {i} appendage program needs blocks")
                elif len(nested) > limits.max_blocks:
                    errors.append(f"block {i} appendage program has too many blocks")
                else:
                    for j, nested_block in enumerate(nested):
                        nested_op = nested_block.get("op") if isinstance(nested_block, dict) else None
                        if not isinstance(nested_op, str):
                            errors.append(f"block {i}.{j} missing string op")
                            continue
                        if nested_op.lower() in limits.forbidden_symbols:
                            errors.append(f"block {i}.{j} uses forbidden task primitive {nested_op!r}")
                        if nested_op.lower() not in COMMON_OPS:
                            errors.append(f"block {i}.{j} has unknown op {nested_op!r}")

    return ValidationResult(not errors, tuple(errors))

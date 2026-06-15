from __future__ import annotations

import json
from typing import Any

from llm_framework.core.state import CandidateProgram, CompiledPolicy, SafetyLimits, ValidationResult, WorldState
from llm_framework.core.tasks import TaskContext
from llm_framework.interfaces.script_dsl import ScriptDSLInterface
from llm_framework.runtime.dsl_interpreter import compile_dsl
from llm_framework.runtime.dsl_validator import validate_dsl


class HybridInterface(ScriptDSLInterface):
    name = "hybrid"
    allow_calls = True

    def build_prompt(self, ctx: TaskContext, state: WorldState) -> str:
        return (
            "Write a hybrid JSON control program for a dexterous robotic hand.\n"
            "The program may use generic script blocks plus call_controller/call_subscript/"
            "call_appendage_agent. "
            "Do not use task-named primitives such as grasp, lift, throw, fold, pick_up, "
            "place, or use_chopsticks.\n"
            f"Task: {ctx.goal}\n"
            f"Context: {json.dumps({'task': ctx.compact(), 'state': state.compact()}, indent=2)}\n"
            "Generic controllers available: track_frame_to_pose, track_object_relative_offset, "
            "close_hand_shape_fraction, hold_contact_count, apply_base_velocity_profile, "
            "stabilize_object_height, latent_decode.\n"
            "Appendage subagents: emit call_appendage_agent with appendage=index|middle|ring|little|"
            "thumb|wrist and an inline program limited to that appendage's joints. Example: "
            '{"op":"call_appendage_agent","appendage":"index","program":{"blocks":['
            '{"op":"set_joint_target","joint":"rh_A_FFJ4","target":0.4}]}}.\n'
            'Schema: {"constants": {}, "blocks": [{"op": "...", "...": "..."}]}\n'
        )

    def mock_response(self, ctx: TaskContext, state: WorldState) -> str:
        x, y, _z = ctx.object_start
        blocks: list[dict[str, Any]] = [
            {"op": "call_controller", "controller": "track_object_relative_offset", "offset": [0.0, 0.0, 0.04], "duration_s": 0.45},
            {
                "op": "call_appendage_agent",
                "appendage": "index",
                "duration_s": 0.2,
                "program": {"blocks": [{"op": "set_appendage_joints", "appendage": "index", "values": 0.58}]},
            },
            {
                "op": "call_appendage_agent",
                "appendage": "thumb",
                "duration_s": 0.2,
                "program": {"blocks": [{"op": "set_appendage_joints", "appendage": "thumb", "values": 0.68}]},
            },
            {"op": "call_controller", "controller": "latent_decode", "token": "oppose_and_stabilize", "gain": 0.65, "duration_s": 0.45},
        ]
        if ctx.name in {"push", "place"} and ctx.target_xy:
            tx, ty = ctx.target_xy
            blocks += [
                {"op": "call_controller", "controller": "apply_base_velocity_profile", "token": "center_over_object", "gain": 0.5, "duration_s": 0.2},
                {"op": "set_base_target", "x": tx, "y": ty, "z": 0.12, "duration_s": 1.1},
                {"op": "call_controller", "controller": "latent_decode", "token": "release", "gain": 0.9, "duration_s": 0.3},
            ]
        else:
            blocks += [
                {"op": "call_controller", "controller": "hold_contact_count", "token": "close_around_object", "gain": 0.85, "duration_s": 0.4},
                {"op": "set_base_target", "x": x, "y": y, "z": 0.0, "duration_s": 0.8},
                {"op": "call_controller", "controller": "stabilize_object_height", "token": "stabilize_height", "gain": 0.5, "duration_s": 0.3},
            ]
        blocks.append({"op": "return", "status": "done"})
        return json.dumps({"constants": {}, "blocks": blocks})

    def validate(self, program: CandidateProgram, limits: SafetyLimits) -> ValidationResult:
        return validate_dsl(program, limits, allow_calls=True)

    def compile(self, program: CandidateProgram, ctx: TaskContext, state: WorldState, env: Any) -> CompiledPolicy:
        compiled = compile_dsl(program, env, state)
        return CompiledPolicy(
            interface=self.name,
            action_stream=compiled.action_stream,
            metadata=compiled.metadata | {"hybrid": True},
        )

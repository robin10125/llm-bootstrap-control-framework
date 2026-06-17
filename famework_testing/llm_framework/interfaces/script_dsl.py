from __future__ import annotations

import json
from typing import Any

from llm_framework.core.state import CandidateProgram, CompiledPolicy, SafetyLimits, ValidationResult, WorldState
from llm_framework.core.tasks import TaskContext
from llm_framework.runtime.dsl_interpreter import compile_dsl
from llm_framework.runtime.dsl_parser import parse_json_program
from llm_framework.runtime.dsl_validator import validate_dsl


class ScriptDSLInterface:
    name = "script_dsl"
    allow_calls = False

    def build_prompt(self, ctx: TaskContext, state: WorldState) -> str:
        return (
            "Write a constrained JSON control program for a dexterous robotic hand.\n"
            "Return ONLY JSON. Do not use task-named action primitives such as grasp, lift, "
            "throw, fold, pick_up, place, or use_chopsticks.\n"
            f"Task: {ctx.goal}\n"
            f"Context: {json.dumps({'task': ctx.compact(), 'state': state.compact()}, indent=2)}\n"
            "Allowed block ops: set_frame_target, set_base_target, move_frame, track_relative_pose, set_joint_targets, "
            "set_joint_target, move_joint_delta, set_appendage_joints, set_hand_shape, set_impedance, "
            "seek_contact, maintain_contact, apply_wrench_or_impulse, wait, monitor, return.\n"
            "Use set_joint_target for individual actuator/joint control. Use set_appendage_joints "
            "when reasoning about one finger/thumb/wrist as an isolated appendage. Work in actual "
            "world coordinates for base placement: prefer set_frame_target with frame='grasp_site' "
            "or frame='palm'; set_base_target is a raw slide-actuator command.\n"
            'Schema: {"constants": {}, "blocks": [{"op": "...", "...": "..."}]}\n'
        )

    def mock_response(self, ctx: TaskContext, state: WorldState) -> str:
        x, y, _z = ctx.object_start
        blocks: list[dict[str, Any]] = [
            {"op": "set_impedance", "fingers": ["thumb", "index", "middle"], "stiffness": 0.35, "damping": 0.7, "duration_s": 0.05},
            {"op": "set_frame_target", "frame": "grasp_site", "target": {"object": "object", "offset": [0.0, 0.0, 0.03]}, "duration_s": 0.5},
        ]
        if ctx.name in {"push", "place"} and ctx.target_xy:
            tx, ty = ctx.target_xy
            direction = [tx - x, ty - y, 0.0]
            blocks += [
                {"op": "set_hand_shape", "fraction_closed": 0.85, "duration_s": 0.25},
                {"op": "apply_wrench_or_impulse", "object": "object", "direction": direction, "magnitude": 0.8, "duration_s": 0.9},
                {"op": "set_frame_target", "frame": "grasp_site", "target": {"x": tx, "y": ty, "z": 0.10}, "duration_s": 0.7},
                {"op": "set_hand_shape", "fraction_closed": 0.15, "duration_s": 0.25},
            ]
        else:
            blocks += [
                {"op": "set_appendage_joints", "appendage": "index", "values": 0.65, "duration_s": 0.12},
                {"op": "set_appendage_joints", "appendage": "thumb", "values": 0.72, "duration_s": 0.12},
                {"op": "seek_contact", "link": "rh_ffdistal", "object": "object", "max_force": 1.5, "fraction_closed": 0.7, "duration_s": 0.45},
                {"op": "maintain_contact", "force_range": [0.4, 2.5], "slip_limit": 0.05, "fraction_closed": 0.9, "duration_s": 0.35},
                {"op": "set_frame_target", "frame": "grasp_site", "target": {"x": x, "y": y, "z": 0.10}, "duration_s": 0.8},
            ]
        blocks.append({"op": "return", "status": "done"})
        return json.dumps({"constants": {}, "blocks": blocks})

    def parse(self, text: str) -> CandidateProgram:
        return parse_json_program(text, interface=self.name)

    def validate(self, program: CandidateProgram, limits: SafetyLimits) -> ValidationResult:
        return validate_dsl(program, limits, allow_calls=self.allow_calls)

    def compile(self, program: CandidateProgram, ctx: TaskContext, state: WorldState, env: Any) -> CompiledPolicy:
        return compile_dsl(program, env, state)

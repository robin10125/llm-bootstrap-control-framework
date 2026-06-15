from __future__ import annotations

import json
from typing import Any

from llm_framework.core.state import CandidateProgram, CompiledPolicy, SafetyLimits, ValidationResult, WorldState
from llm_framework.core.tasks import TaskContext
from llm_framework.runtime.dsl_interpreter import compile_dsl
from llm_framework.runtime.dsl_parser import parse_json_program
from llm_framework.runtime.dsl_validator import validate_dsl


class LatentStubInterface:
    name = "latent_stub"

    def build_prompt(self, ctx: TaskContext, state: WorldState) -> str:
        return (
            "Return ONLY JSON containing generic latent action decoder blocks for a dexterous hand.\n"
            "Tokens are low-level motion/contact modes, not task names. Do not use task-named "
            "primitives such as grasp, lift, throw, fold, pick_up, place, or use_chopsticks.\n"
            f"Task: {ctx.goal}\n"
            f"Context: {json.dumps({'task': ctx.compact(), 'state': state.compact()}, indent=2)}\n"
            'Schema: {"blocks": [{"op": "latent_decode", "token": "...", "duration_s": 0.5, "gain": 0.5}]}\n'
            "Useful tokens: approach_object, center_over_object, oppose_and_stabilize, "
            "close_around_object, raise, stabilize_height, release, open_and_clear.\n"
        )

    def mock_response(self, ctx: TaskContext, state: WorldState) -> str:
        blocks = [
            {"op": "latent_decode", "token": "approach_object", "duration_s": 0.4, "gain": 0.8},
            {"op": "latent_decode", "token": "oppose_and_stabilize", "duration_s": 0.5, "gain": 0.8},
        ]
        if ctx.name in {"push", "place"}:
            blocks += [
                {"op": "latent_decode", "token": "center_over_object", "duration_s": 0.4, "gain": 0.5},
                {"op": "latent_decode", "token": "release", "duration_s": 0.3, "gain": 0.7},
            ]
        else:
            blocks += [
                {"op": "latent_decode", "token": "raise", "duration_s": 0.7, "gain": 0.9},
                {"op": "latent_decode", "token": "stabilize_height", "duration_s": 0.5, "gain": 0.5},
            ]
        return json.dumps({"blocks": blocks})

    def parse(self, text: str) -> CandidateProgram:
        return parse_json_program(text, interface=self.name)

    def validate(self, program: CandidateProgram, limits: SafetyLimits) -> ValidationResult:
        return validate_dsl(program, limits, allow_calls=True)

    def compile(self, program: CandidateProgram, ctx: TaskContext, state: WorldState, env: Any) -> CompiledPolicy:
        compiled = compile_dsl(program, env, state)
        return CompiledPolicy(
            interface=self.name,
            action_stream=compiled.action_stream,
            metadata=compiled.metadata | {"latent_stub": True},
        )


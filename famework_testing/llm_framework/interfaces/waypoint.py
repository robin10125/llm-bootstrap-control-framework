from __future__ import annotations

import json
from typing import Any

from llm_framework.adapters.experiment_runtime import waypoint_compiler
from llm_framework.core.state import CandidateProgram, CompiledPolicy, SafetyLimits, ValidationResult, WorldState
from llm_framework.core.tasks import TaskContext
from llm_framework.runtime.controllers import targets_to_actions
from llm_framework.runtime.dsl_parser import parse_json_program


class WaypointInterface:
    name = "waypoint"

    def build_prompt(self, ctx: TaskContext, state: WorldState) -> str:
        return (
            "You control a dexterous hand mounted on a movable base.\n"
            "Return ONLY JSON. Do not use task-named primitives or prose.\n"
            f"Task: {ctx.goal}\n"
            f"Context: {json.dumps({'task': ctx.compact(), 'state': state.compact()}, indent=2)}\n"
            "Schema:\n"
            '{"waypoints": [{"t": <seconds>, "pos": [base_x, base_y, base_z], "open": <0..1>}, ...]}\n'
            "Use pos only as actuator targets. open=1 means open hand, open=0 means closed hand shape.\n"
        )

    def mock_response(self, ctx: TaskContext, state: WorldState) -> str:
        x, y, _z = ctx.object_start
        if ctx.name in {"push", "place"} and ctx.target_xy:
            tx, ty = ctx.target_xy
            points = [
                {"t": 0.0, "pos": [x, y, 0.02], "open": 0.2},
                {"t": 0.8, "pos": [x, y, 0.13], "open": 0.2},
                {"t": 2.0, "pos": [tx, ty, 0.13], "open": 0.2},
                {"t": 2.5, "pos": [tx, ty, 0.02], "open": 0.8},
            ]
        else:
            points = [
                {"t": 0.0, "pos": [x, y, 0.00], "open": 1.0},
                {"t": 0.8, "pos": [x, y, 0.15], "open": 1.0},
                {"t": 1.3, "pos": [x, y, 0.15], "open": 0.0},
                {"t": 2.2, "pos": [x, y, 0.00], "open": 0.0},
            ]
        return json.dumps({"waypoints": points})

    def parse(self, text: str) -> CandidateProgram:
        return parse_json_program(text, interface=self.name)

    def validate(self, program: CandidateProgram, limits: SafetyLimits) -> ValidationResult:
        waypoints = program.source.get("waypoints")
        errors: list[str] = []
        if not isinstance(waypoints, list) or not waypoints:
            errors.append("waypoint program must contain non-empty waypoints")
        for i, point in enumerate(waypoints or []):
            if not isinstance(point, dict):
                errors.append(f"waypoint {i} must be an object")
                continue
            if not {"t", "pos", "open"} <= set(point):
                errors.append(f"waypoint {i} needs t, pos, open")
            if float(point.get("t", 0.0)) > limits.max_episode_seconds:
                errors.append(f"waypoint {i} exceeds episode limit")
        return ValidationResult(not errors, tuple(errors))

    def compile(self, program: CandidateProgram, ctx: TaskContext, state: WorldState, env: Any) -> CompiledPolicy:
        comp = waypoint_compiler(env)
        targets = comp.compile(program.source, env)
        return CompiledPolicy(
            interface=self.name,
            action_stream=targets_to_actions(env, targets),
            metadata={"source": "bootstrapping.WaypointCompiler", "task": ctx.name},
        )

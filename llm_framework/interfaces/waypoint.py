from __future__ import annotations

import json
from typing import Any

from llm_framework.adapters.bootstrapping_env import waypoint_compiler
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
        if getattr(env, "is_fake_env", False):
            targets = _compile_waypoints_locally(program.source, env)
            source = "local waypoint compiler"
        else:
            comp = waypoint_compiler(env)
            targets = comp.compile(program.source, env)
            source = "bootstrapping.WaypointCompiler"
        return CompiledPolicy(
            interface=self.name,
            action_stream=targets_to_actions(env, targets),
            metadata={"source": source, "task": ctx.name},
        )


def _compile_waypoints_locally(plan: dict[str, Any], env: Any):
    import numpy as np

    waypoints = plan["waypoints"]
    idx = {env.model.actuator(i).name: i for i in range(env.nu)}
    targets = np.repeat(np.asarray(env.ctrl_open, dtype=np.float32)[None, :], env.horizon, axis=0)
    times = np.arange(env.horizon) * float(env.cfg.control_dt)
    ts = np.asarray([float(w["t"]) for w in waypoints], dtype=float)
    vecs = []
    for point in waypoints:
        vec = np.asarray(env.ctrl_open, dtype=np.float32).copy()
        pos = point["pos"]
        for axis, value in zip(("x", "y", "z"), pos, strict=True):
            name = f"base_{axis}"
            if name in idx:
                vec[idx[name]] = float(value)
        frac_open = float(np.clip(point["open"], 0.0, 1.0))
        shaped = frac_open * np.asarray(env.ctrl_open) + (1.0 - frac_open) * np.asarray(env.ctrl_close)
        for i in getattr(env, "hand_act_ids", []):
            vec[i] = shaped[i]
        vecs.append(vec)
    order = np.argsort(ts)
    ts = ts[order]
    vecs = np.asarray(vecs, dtype=np.float32)[order]
    for d in range(env.nu):
        targets[:, d] = np.interp(times, ts, vecs[:, d])
    return targets

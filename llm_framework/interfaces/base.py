from __future__ import annotations

from typing import Any, Protocol

from llm_framework.core.state import CandidateProgram, CompiledPolicy, SafetyLimits, ValidationResult, WorldState
from llm_framework.core.tasks import TaskContext


class ControlInterface(Protocol):
    name: str

    def build_prompt(self, ctx: TaskContext, state: WorldState) -> str:
        ...

    def mock_response(self, ctx: TaskContext, state: WorldState) -> str:
        ...

    def parse(self, text: str) -> CandidateProgram:
        ...

    def validate(self, program: CandidateProgram, limits: SafetyLimits) -> ValidationResult:
        ...

    def compile(self, program: CandidateProgram, ctx: TaskContext, state: WorldState, env: Any) -> CompiledPolicy:
        ...


def interface_by_name(name: str) -> ControlInterface:
    if name == "waypoint":
        from llm_framework.interfaces.waypoint import WaypointInterface

        return WaypointInterface()
    if name == "script_dsl":
        from llm_framework.interfaces.script_dsl import ScriptDSLInterface

        return ScriptDSLInterface()
    if name == "hybrid":
        from llm_framework.interfaces.hybrid import HybridInterface

        return HybridInterface()
    if name in {"latent", "latent_stub"}:
        from llm_framework.interfaces.latent import LatentStubInterface

        return LatentStubInterface()
    if name in {"recursive", "recursive_units"}:
        from llm_framework.interfaces.recursive_units import RecursiveUnitsInterface

        return RecursiveUnitsInterface()
    raise KeyError(f"unknown interface {name!r}")

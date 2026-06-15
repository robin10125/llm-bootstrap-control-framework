from __future__ import annotations

from llm_framework.core.state import CandidateProgram, SafetyLimits
from llm_framework.interfaces.script_dsl import ScriptDSLInterface
from llm_framework.runtime.dsl_parser import parse_json_program


def test_forbidden_task_primitive_is_rejected() -> None:
    program = CandidateProgram(
        interface="script_dsl",
        source={"blocks": [{"op": "grasp", "duration_s": 0.1}]},
    )

    result = ScriptDSLInterface().validate(program, SafetyLimits())

    assert not result.ok
    assert "forbidden" in result.errors[0]


def test_parser_accepts_fenced_json() -> None:
    program = parse_json_program(
        '```json\n{"blocks": [{"op": "wait", "duration_s": 0.1}]}\n```',
        interface="script_dsl",
    )

    assert program.source["blocks"][0]["op"] == "wait"


def test_forbidden_nested_appendage_primitive_is_rejected() -> None:
    program = CandidateProgram(
        interface="hybrid",
        source={
            "blocks": [{
                "op": "call_appendage_agent",
                "appendage": "index",
                "program": {"blocks": [{"op": "grasp", "duration_s": 0.1}]},
            }]
        },
    )

    result = ScriptDSLInterface().validate(program, SafetyLimits())

    assert not result.ok
    assert any("forbidden" in err for err in result.errors)

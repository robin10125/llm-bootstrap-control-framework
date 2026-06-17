from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from llm_framework.adapters.bootstrapping_env import call_bootstrapping_llm
from llm_framework.core.state import CandidateProgram, CompiledPolicy, SafetyLimits, ValidationResult, WorldState
from llm_framework.core.tasks import TaskContext
from llm_framework.interfaces.script_dsl import ScriptDSLInterface
from llm_framework.llm.types import LLMCallResult
from llm_framework.runtime.dsl_interpreter import compile_dsl
from llm_framework.runtime.dsl_macros import expand_generated_primitives
from llm_framework.runtime.dsl_parser import ParseError, parse_json_program
from llm_framework.runtime.dsl_validator import validate_dsl
from llm_framework.runtime.unit_schema import fundamental_units, schema_for_unit


PHASE_ORDER = (
    "approach",
    "descend_or_precontact",
    "close_until_touch_or_settle",
    "verify_contact_or_settle",
    "lift_or_transport",
    "stabilize_or_release",
)


SCHEDULE_DIRECTIVE = (
    "Use an explicit timed schedule for coordination. Put base/frame placement before "
    "appendage closure, include short monitor or wait blocks after contact-seeking phases, "
    "and only schedule transport after the planned contact/settle dwell. Do not use loops, "
    "runtime conditionals, or task-named high-level primitives."
)


class RecursiveUnitsInterface(ScriptDSLInterface):
    """Recursive LLM planner: task -> unit commands -> joint/base commands -> final policy."""

    name = "recursive_units"
    allow_calls = True

    def build_prompt(self, ctx: TaskContext, state: WorldState) -> str:
        return _top_prompt(ctx, state)

    def parse(self, text: str) -> CandidateProgram:
        return _normalize_candidate(parse_json_program(text, interface=self.name))

    def mock_response(self, ctx: TaskContext, state: WorldState, reasoning_bias: dict[str, Any] | None = None) -> str:
        units = fundamental_units(state)
        unit_policies = [_mock_unit_policy(unit, ctx, state) for unit in units]
        blocks = _scheduled_blocks([block for policy in unit_policies for block in policy["blocks"]])
        recursive_trace: dict[str, Any] = {
            "recursive_trace": {
                "top": {"command": ctx.goal, "units": units},
                "phase_schedule": _default_phase_schedule(ctx),
                "unit_policies": unit_policies,
                "compiler": "mock concatenation of lowest-level unit blocks",
            },
        }
        if reasoning_bias:
            recursive_trace["recursive_trace"]["reasoning_bias"] = reasoning_bias
        return json.dumps({
            **recursive_trace,
            "blocks": blocks + [{"op": "return", "status": "done"}],
        })

    def complete_with_llm(
        self,
        ctx: TaskContext,
        state: WorldState,
        *,
        backend: str,
        model: str | None,
        log_dir: Path | None,
        tag: str,
    ) -> LLMCallResult:
        return self._complete_with_llm(ctx, state, backend=backend, model=model, log_dir=log_dir, tag=tag)

    def _complete_with_llm(
        self,
        ctx: TaskContext,
        state: WorldState,
        *,
        backend: str,
        model: str | None,
        log_dir: Path | None,
        tag: str,
        reasoning_bias: dict[str, Any] | None = None,
        initial_calls: list[dict[str, Any]] | None = None,
    ) -> LLMCallResult:
        if backend == "mock":
            return LLMCallResult(text=self.mock_response(ctx, state, reasoning_bias=reasoning_bias), ok=True, error=None, source="mock")
        units = fundamental_units(state)
        calls: list[dict[str, Any]] = list(initial_calls or [])

        top = _call_json(
            backend,
            _top_prompt(ctx, state, reasoning_bias=reasoning_bias),
            model=model,
            log_dir=log_dir,
            tag=f"{tag}_top",
        )
        calls.append({"stage": "top", "ok": top.ok, "error": top.error})
        if not top.ok:
            return top
        try:
            top_obj = _json(top.text)
        except ParseError as exc:
            return LLMCallResult(text=top.text, ok=False, error=str(exc), source=backend, metadata={"calls": calls})

        unit_commands = _unit_commands(top_obj, units, ctx.goal)
        unit_policies = []
        for unit in units:
            prompt = _unit_prompt(ctx, state, unit, unit_commands[unit], reasoning_bias=reasoning_bias)
            response = _call_json(
                backend,
                prompt,
                model=model,
                log_dir=log_dir,
                tag=f"{tag}_unit_{unit}",
            )
            calls.append({"stage": f"unit:{unit}", "ok": response.ok, "error": response.error})
            if not response.ok:
                return LLMCallResult(text=response.text, ok=False, error=response.error, source=backend, metadata={"calls": calls})
            try:
                policy = _json(response.text)
            except ParseError as exc:
                return LLMCallResult(text=response.text, ok=False, error=f"{unit}: {exc}", source=backend, metadata={"calls": calls})
            policy.setdefault("unit", unit)
            unit_policies.append(policy)

        compiler_input = {
            "top_trace": top_obj.get("trace", []),
            "phase_schedule": top_obj.get("phase_schedule", _default_phase_schedule(ctx)),
            "unit_commands": [{"unit": unit, "command": unit_commands[unit]} for unit in units],
            "unit_policies": unit_policies,
        }
        if reasoning_bias:
            compiler_input["reasoning_bias"] = reasoning_bias
        final = _call_json(
            backend,
            _compiler_prompt(ctx, state, compiler_input, reasoning_bias=reasoning_bias),
            model=model,
            log_dir=log_dir,
            tag=f"{tag}_compiler",
        )
        calls.append({"stage": "compiler", "ok": final.ok, "error": final.error})
        if not final.ok:
            return final
        try:
            final_obj = _json(final.text)
        except ParseError as exc:
            return LLMCallResult(text=final.text, ok=False, error=str(exc), source=backend, metadata={"calls": calls})
        if not isinstance(final_obj.get("recursive_trace"), dict):
            final_obj["recursive_trace"] = compiler_input
        if reasoning_bias:
            final_obj["recursive_trace"]["reasoning_bias"] = reasoning_bias
        final_obj["blocks"] = _scheduled_blocks(final_obj.get("blocks", []))
        return LLMCallResult(
            text=json.dumps(final_obj, indent=2),
            ok=True,
            error=None,
            source=backend,
            metadata={"calls": calls},
        )

    def validate(self, program: CandidateProgram, limits: SafetyLimits) -> ValidationResult:
        return validate_dsl(_normalize_candidate(program), limits, allow_calls=True)

    def compile(self, program: CandidateProgram, ctx: TaskContext, state: WorldState, env: Any) -> CompiledPolicy:
        program = _normalize_candidate(program)
        compiled = compile_dsl(program, env, state)
        return CompiledPolicy(
            interface=self.name,
            action_stream=compiled.action_stream,
            metadata=compiled.metadata | {
                "recursive_units": True,
                "recursive_trace": program.source.get("recursive_trace", {}),
            },
        )

    def repair_with_llm(
        self,
        ctx: TaskContext,
        state: WorldState,
        previous_program: CandidateProgram,
        previous_result: Any,
        *,
        backend: str,
        model: str | None,
        log_dir: Path | None,
        tag: str,
    ) -> LLMCallResult:
        if backend == "mock":
            repaired = dict(previous_program.source)
            repaired["reflection"] = {
                "failure": "mock repair: keep scheduled phase order and increase contact/lift dwell",
                "changes": ["longer descend", "longer close/settle", "longer lift"],
            }
            return LLMCallResult(text=json.dumps(_normalize_source(repaired), indent=2), ok=True, error=None, source="mock")
        response = _call_json(
            backend,
            _repair_prompt(ctx, state, previous_program.source, previous_result),
            model=model,
            log_dir=log_dir,
            tag=tag,
        )
        if not response.ok:
            return response
        try:
            repaired = _json(response.text)
        except ParseError as exc:
            return LLMCallResult(text=response.text, ok=False, error=str(exc), source=backend)
        repaired = _normalize_source(repaired)
        repaired.setdefault("recursive_trace", previous_program.source.get("recursive_trace", {}))
        return LLMCallResult(text=json.dumps(repaired, indent=2), ok=True, error=None, source=backend)


def _top_prompt(ctx: TaskContext, state: WorldState, reasoning_bias: dict[str, Any] | None = None) -> str:
    units = fundamental_units(state)
    return (
        "You are the top-level policy decomposition agent for a Shadow-style dexterous hand.\n"
        "Work in actual world coordinates. The context includes object, palm, grasp_site, "
        "and fingertip positions. For base motion, prefer `set_frame_target` with "
        "`frame: \"grasp_site\"` or `frame: \"palm\"` and a world-coordinate target; "
        "`set_base_target` is a raw slide-actuator command and should only be used when "
        "you intentionally want raw base coordinates.\n"
        "Decompose the task into one command for EVERY fundamental unit listed below. "
        "Do not emit robot actions yet. The base command should describe the desired palm/base "
        "end state; finger/wrist commands should be relative to that base/palm end state.\n"
        "Also produce a sequential phase schedule. For coordinated manipulation, prefer this "
        "general order: approach over/near the work area, descend or pre-contact, move "
        "appendages toward constraint/contact/alignment, verify the coordination condition "
        "or settle, then transport or continue. This prevents units from moving away "
        "prematurely when exact checks are not obvious. "
        f"{SCHEDULE_DIRECTIVE}\n"
        "Return ONLY JSON with this schema:\n"
        '{"trace":[{"level":"task","command":"..."}],'
        '"phase_schedule":[{"phase":"approach","intent":"...","min_duration_s":0.2}],'
        '"unit_commands":[{"unit":"base","command":"..."},{"unit":"index","command":"..."}]}\n\n'
        f"Task: {ctx.goal}\n"
        f"Task context: {json.dumps(ctx.compact(), indent=2)}\n"
        f"Fundamental units: {json.dumps(units)}\n"
        f"World and derived context: {json.dumps(state.compact(), indent=2)}\n"
        f"{_reasoning_bias_section(reasoning_bias)}"
    )


def _unit_prompt(ctx: TaskContext, state: WorldState, unit: str, command: str, reasoning_bias: dict[str, Any] | None = None) -> str:
    return (
        f"You are the subagent for the `{unit}` fundamental unit only.\n"
        "Use the actual world-coordinate frames in the context. If this unit is `base`, "
        "prefer `set_frame_target` for palm/grasp_site placement so the command accounts "
        "for the physical offset between the slide-base origin and the hand frame.\n"
        "Translate the supplied high-level unit command into lower-level commands until every "
        "command is a lowest-level actuator/base command. You may only command joints listed "
        "in this unit schema. Return the recursive trace and lowest-level blocks.\n"
        "Allowed lowest-level ops: set_frame_target, set_base_target, set_joint_target, set_joint_targets, "
        "set_appendage_joints, wait, monitor, return. Do not use task-named "
        "primitives.\n"
        "Every block must include a `phase` chosen from the phase schedule. "
        f"{SCHEDULE_DIRECTIVE}\n"
        "Return ONLY JSON with schema:\n"
        '{"unit":"...","trace":[{"level":"unit","command":"..."},{"level":"joint","command":"..."}],'
        '"blocks":[{"phase":"close_until_touch_or_settle","op":"set_joint_target",'
        '"joint":"...","target":0.0,"duration_s":0.1},'
        '{"phase":"verify_contact_or_settle","op":"monitor","duration_s":0.1}]}\n\n'
        f"Task: {ctx.goal}\n"
        f"Unit command: {command}\n"
        f"Phase schedule: {json.dumps(_default_phase_schedule(ctx), indent=2)}\n"
        f"Relevant unit schema: {json.dumps(schema_for_unit(state, unit), indent=2)}\n"
        f"Derived context: {json.dumps(state.derived, indent=2)}\n"
        f"Object/base context: {json.dumps({'object_pos': state.object_pos.round(4).tolist(), 'base_q': state.base_q.round(4).tolist()}, indent=2)}\n"
        f"{_reasoning_bias_section(reasoning_bias)}"
    )


def _compiler_prompt(
    ctx: TaskContext,
    state: WorldState,
    compiler_input: dict[str, Any],
    reasoning_bias: dict[str, Any] | None = None,
) -> str:
    return (
        "You are the final compiler agent. Combine all unit policies into one executable JSON "
        "policy for the robot. Preserve recursive_trace. Use only lowest-level blocks. "
        "Use actual world coordinates for base placement. Prefer `set_frame_target` with "
        "`frame` equal to `grasp_site` or `palm` when moving the hand relative to the object; "
        "this compiler will convert that desired frame pose into raw base actuator targets. "
        "Use `set_base_target` only for deliberate raw slide coordinates.\n"
        "You may freely define useful generated primitives/macros in `generated_primitives` "
        "with any names you find useful, including contact-oriented names such as "
        "`close_until_touching`. If you invent a primitive, include its implementation as "
        "phase-tagged lower-level blocks and call it from `blocks`; it will be expanded before "
        "validation/execution.\n"
        "Respect phase order strictly: approach -> descend_or_precontact -> "
        "close_until_touch_or_settle -> verify_contact_or_settle -> lift_or_transport -> "
        "stabilize_or_release. This is more important than grouping by unit. Do not raise or "
        "transport the base before the close/touch and verify/settle phases have occurred. "
        "Do not invent unsupported ops. "
        f"{SCHEDULE_DIRECTIVE}\n"
        "Allowed ops: set_frame_target, set_base_target, set_joint_target, set_joint_targets, set_appendage_joints, "
        "wait, monitor, return.\n"
        'Return ONLY JSON: {"recursive_trace": {...}, "generated_primitives": {...}, "blocks": [...]}\n\n'
        f"Task: {ctx.goal}\n"
        f"Required phase schedule: {json.dumps(_default_phase_schedule(ctx), indent=2)}\n"
        f"Actuator schema: {json.dumps(state.joint_schema, indent=2)}\n"
        f"Input traces and unit policies: {json.dumps(compiler_input, indent=2)}\n"
        f"{_reasoning_bias_section(reasoning_bias)}"
    )


def _repair_prompt(ctx: TaskContext, state: WorldState, previous_program: dict[str, Any], previous_result: Any) -> str:
    return (
        "You are a reflection and error-correction agent for a failed robot policy.\n"
        "Analyze why the previous scheduled policy failed, then output a corrected executable "
        "JSON policy. Preserve or update recursive_trace, include a short `reflection`, and "
        "use only supported lowest-level ops.\n\n"
        "Important execution constraints:\n"
        "- Keep strict phase order: approach -> descend_or_precontact -> "
        "close_until_touch_or_settle -> verify_contact_or_settle -> lift_or_transport -> "
        "stabilize_or_release.\n"
        "- If task metrics or rollout observations did not improve toward the requested "
        "state, assume a required coordination condition was missed or lost. Adjust the "
        "earlier scheduled base, appendage, contact, alignment, clearance, or stability "
        "phase before later transport phases.\n"
        "- There is no runtime conditional loop. Tune fixed phase durations, frame targets, "
        "joint targets, and monitor/wait dwell times based on the previous rollout metrics.\n"
        "- Use actual world coordinates for base placement. Prefer `set_frame_target` with "
        "`frame` equal to `grasp_site` or `palm`; use raw `set_base_target` only if you "
        "are intentionally commanding slide actuator coordinates.\n"
        "- Use the joint schema for actuator direction. For base_z in the Shadow scene, "
        "lower/negative values lower the palm toward the table/object and higher values "
        "raise/retract it.\n"
        "- For finger actuators, use closed_target/open_target from the schema; do not "
        "guess actuator direction.\n"
        f"- {SCHEDULE_DIRECTIVE}\n\n"
        "Allowed ops: set_frame_target, set_base_target, set_joint_target, set_joint_targets, "
        "set_appendage_joints, wait, monitor, return, or any generated "
        "primitive you define in `generated_primitives`.\n"
        'Return ONLY JSON: {"reflection": {...}, "recursive_trace": {...}, "generated_primitives": {...}, "blocks": [...]}\n\n'
        f"Task: {ctx.goal}\n"
        f"Task context: {json.dumps(ctx.compact(), indent=2)}\n"
        f"Actuator schema: {json.dumps(state.joint_schema, indent=2)}\n"
        f"Derived context: {json.dumps(state.derived, indent=2)}\n"
        f"Previous rollout result: {json.dumps(_previous_result_payload(previous_result), indent=2)}\n"
        f"Previous program: {json.dumps(previous_program, indent=2)}\n"
    )


def _previous_result_payload(previous_result: Any) -> dict[str, Any]:
    row = previous_result.row() if hasattr(previous_result, "row") else {}
    payload = {"row": row}
    return payload


def _reasoning_bias_section(reasoning_bias: dict[str, Any] | None) -> str:
    if not reasoning_bias:
        return ""
    return (
        "\nReasoning-bias context from the strategy agent. Treat this as an inductive "
        "bias for policy exploration, not as an executable primitive list:\n"
        f"{json.dumps(reasoning_bias, indent=2)}\n"
    )


def _unit_commands(top_obj: dict[str, Any], units: list[str], goal: str) -> dict[str, str]:
    found = {
        str(item.get("unit")): str(item.get("command", ""))
        for item in top_obj.get("unit_commands", [])
        if isinstance(item, dict)
    }
    return {unit: found.get(unit) or f"hold or adjust {unit} only as needed to support: {goal}" for unit in units}


def _call_json(backend: str, prompt: str, *, model: str | None, log_dir: Path | None, tag: str) -> LLMCallResult:
    response = call_bootstrapping_llm(backend, prompt, model=model, log_dir=log_dir, tag=tag)
    return LLMCallResult(text=response.text, ok=response.ok, error=response.error, source=response.source)


def _json(text: str) -> dict[str, Any]:
    return parse_json_program(text, interface="recursive_units").source


def _default_phase_schedule(ctx: TaskContext) -> list[dict[str, Any]]:
    return [
        {"phase": "approach", "intent": "move relevant base/frame near the work area while appendages stay clear", "min_duration_s": 0.2},
        {"phase": "descend_or_precontact", "intent": "move the base/palm or appendages toward the object/contact region", "min_duration_s": 0.2},
        {"phase": "close_until_touch_or_settle", "intent": "move the affected appendages toward contact; use touch checks if obvious", "min_duration_s": 0.2},
        {"phase": "verify_contact_or_settle", "intent": "confirm contact or wait long enough for contact/force to settle", "min_duration_s": 0.2},
        {"phase": "lift_or_transport", "intent": "move the base or object only after the preceding phases", "min_duration_s": 0.2},
        {"phase": "stabilize_or_release", "intent": "hold, stabilize, or release according to the task", "min_duration_s": 0.2},
    ]


def _normalize_candidate(program: CandidateProgram) -> CandidateProgram:
    source = _normalize_source(program.source)
    return CandidateProgram(interface=program.interface, source=source, raw_text=program.raw_text)


def _normalize_source(source: dict[str, Any]) -> dict[str, Any]:
    source = expand_generated_primitives(source)
    source = dict(source)
    if "blocks" in source:
        source["blocks"] = _scheduled_blocks(source["blocks"])
    return source


def _scheduled_blocks(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    indexed = list(enumerate(blocks))
    returns = [(i, b) for i, b in indexed if b.get("op") == "return"]
    active = [(i, b) for i, b in indexed if b.get("op") != "return"]

    def key(item: tuple[int, dict[str, Any]]) -> tuple[int, int]:
        idx, block = item
        phase = block.get("phase")
        try:
            phase_idx = PHASE_ORDER.index(str(phase))
        except ValueError:
            # Unphased blocks preserve their relative position after phased blocks. This
            # avoids guessing semantics while still enforcing explicit schedules.
            phase_idx = len(PHASE_ORDER)
        return phase_idx, idx

    return [b for _, b in sorted(active, key=key)] + [b for _, b in returns]


def _mock_unit_policy(unit: str, ctx: TaskContext, state: WorldState) -> dict[str, Any]:
    if unit == "base":
        x, y, _z = ctx.object_start
        return {
            "unit": unit,
            "trace": [
                {"level": "unit", "command": "move grasp_site over object, descend to contact height, then raise after contact settle"},
                {"level": "base", "command": "set actual world-frame grasp_site targets in scheduled phases"},
            ],
            "blocks": [
                {"phase": "approach", "op": "set_frame_target", "frame": "grasp_site", "target": {"x": x, "y": y, "z": 0.10}, "duration_s": 0.3},
                {"phase": "descend_or_precontact", "op": "set_frame_target", "frame": "grasp_site", "target": {"x": x, "y": y, "z": 0.04}, "duration_s": 0.4},
                {"phase": "lift_or_transport", "op": "set_frame_target", "frame": "grasp_site", "target": {"x": x, "y": y, "z": 0.10}, "duration_s": 0.5},
            ],
        }
    joints = state.appendages.get(unit, [])
    blocks = [
        {"phase": "close_until_touch_or_settle", "op": "set_joint_target", "joint": joint, "target": _closed_target(joint, state), "duration_s": 0.25}
        for joint in joints
    ]
    blocks.append({"phase": "verify_contact_or_settle", "op": "monitor", "duration_s": 0.25})
    return {
        "unit": unit,
        "trace": [{"level": "unit", "command": f"shape {unit} relative to palm"}, {"level": "joint", "command": "set each actuator target"}],
        "blocks": blocks,
    }


def _closed_target(joint: str, state: WorldState) -> float:
    for unit in state.joint_schema.get("units", {}).values():
        for entry in unit.get("joints", []):
            if entry.get("name") == joint:
                meaning = entry.get("meaning", {})
                if "closed_target" in meaning:
                    return float(meaning["closed_target"])
    return 0.0

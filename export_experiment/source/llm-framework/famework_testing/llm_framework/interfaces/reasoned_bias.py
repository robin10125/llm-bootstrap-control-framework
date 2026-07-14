from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from llm_framework.core.state import WorldState
from llm_framework.core.tasks import TaskContext
from llm_framework.interfaces.recursive_units import RecursiveUnitsInterface, _call_json
from llm_framework.llm.types import LLMCallResult
from llm_framework.runtime.dsl_parser import ParseError, parse_json_program


class OptimalBiasPlainInterface(RecursiveUnitsInterface):
    """Recursive policy interface with a plain optimal-approach reasoning prepass."""

    name = "optimal_bias_plain"
    guidance_style = "plain"

    def build_prompt(self, ctx: TaskContext, state: WorldState) -> str:
        return _reasoning_prompt(ctx, state, style=self.guidance_style)

    def mock_response(self, ctx: TaskContext, state: WorldState, reasoning_bias: dict[str, Any] | None = None) -> str:
        return super().mock_response(ctx, state, reasoning_bias=reasoning_bias or _mock_reasoning_bias(ctx, state, self.guidance_style))

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
        if backend == "mock":
            bias = _mock_reasoning_bias(ctx, state, self.guidance_style)
            return LLMCallResult(text=self.mock_response(ctx, state, reasoning_bias=bias), ok=True, error=None, source="mock")

        response = _call_json(
            backend,
            _reasoning_prompt(ctx, state, style=self.guidance_style),
            model=model,
            log_dir=log_dir,
            tag=f"{tag}_reasoning_bias",
        )
        calls = [{"stage": f"reasoning_bias:{self.guidance_style}", "ok": response.ok, "error": response.error}]
        if not response.ok:
            return response
        try:
            bias = parse_json_program(response.text, interface=self.name).source
        except ParseError as exc:
            return LLMCallResult(text=response.text, ok=False, error=str(exc), source=backend, metadata={"calls": calls})

        return self._complete_with_llm(
            ctx,
            state,
            backend=backend,
            model=model,
            log_dir=log_dir,
            tag=tag,
            reasoning_bias=bias,
            initial_calls=calls,
        )


class OptimalBiasGuidedInterface(OptimalBiasPlainInterface):
    """Recursive policy interface with a guided pedagogical strategy prepass."""

    name = "optimal_bias_guided"
    guidance_style = "guided"


def _reasoning_prompt(ctx: TaskContext, state: WorldState, *, style: str) -> str:
    if style == "guided":
        directive = (
            "Reason about the optimal approach at the relevant abstraction levels before "
            "any policy is generated. Identify which level currently matters most: for "
            "example, do not emphasize fine finger tuning if gross approach/orientation is "
            "the bottleneck. Explain the approach pedagogically to a policy with no prior "
            "task knowledge: include useful physical intuitions, reusable abstractions, "
            "and simple analogies only when they sharpen the control strategy. Search the "
            "provided strategy memory for similar or identical tasks; adapt any effective "
            "explanations without copying irrelevant details. Propose reward-shaping terms "
            "and direct policy-initialization constraints that would bias exploration toward "
            "good action basins. Include failure modes, observables that diagnose them, and "
            "how the explanation should be revised after downstream rollout performance."
        )
    else:
        directive = (
            "Reason about the optimal approach to completing the task before any policy is "
            "generated. Given the robot, environment, task, actuator schema, and derived "
            "state, produce inductive biases that should steer downstream policy exploration "
            "toward promising action basins. Do not emit robot actions."
        )

    return (
        "You are a strategy and inductive-bias agent for dexterous robot policy search.\n"
        f"{directive}\n\n"
        "The output is consumed by downstream policy-generation agents. It should bias "
        "their choices, not hardcode a high-level task primitive. Keep it task-agnostic in "
        "form: describe reusable principles, bottlenecks, rewards, supervision hints, and "
        "repair rules that can be applied to unknown manipulation tasks. Do not emit robot "
        "actions, executable blocks, or task-named control primitives.\n\n"
        "Return ONLY JSON with this schema:\n"
        "{\n"
        '  "style": "plain|guided",\n'
        '  "task_summary": "...",\n'
        '  "optimal_approach": ["..."],\n'
        '  "abstraction_priorities": [\n'
        '    {"level": "gross_pose|contact_geometry|force_timing|fine_actuation", "priority": 1, "reason": "..."}\n'
        "  ],\n"
        '  "pedagogical_explanation": "...",\n'
        '  "similar_task_adaptations": [{"source": "...", "adaptation": "..."}],\n'
        '  "reward_biases": [{"name": "...", "signal": "...", "why": "..."}],\n'
        '  "policy_initialization_hints": ["..."],\n'
        '  "failure_modes": [{"mode": "...", "diagnostic": "...", "correction": "..."}],\n'
        '  "iteration_rules": ["..."]\n'
        "}\n\n"
        f"Requested style: {style}\n"
        f"Task: {ctx.goal}\n"
        f"Task context: {json.dumps(ctx.compact(), indent=2)}\n"
        f"Robot and environment context: {json.dumps(state.compact(), indent=2)}\n"
        f"Strategy memory: {json.dumps(_strategy_memory(), indent=2)}\n"
    )


def _strategy_memory() -> list[dict[str, str]]:
    return [
        {
            "task_family": "lift small object from table",
            "explanation": (
                "First make the grasp frame coincide with the object region; then establish "
                "opposing contacts before vertical transport. Height reward is only useful "
                "after approach and contact geometry are roughly correct."
            ),
        },
        {
            "task_family": "push object on table",
            "explanation": (
                "Keep the contact point behind the object relative to the target direction "
                "and reward reducing target xy distance before adding fine stability terms."
            ),
        },
        {
            "task_family": "place or stabilize object",
            "explanation": (
                "Bias toward low acceleration and sustained support contacts before release; "
                "late-stage precision matters only after the object is already near the target."
            ),
        },
    ]


def _mock_reasoning_bias(ctx: TaskContext, state: WorldState, style: str) -> dict[str, Any]:
    return {
        "style": style,
        "task_summary": ctx.goal,
        "optimal_approach": [
            "Prioritize gross frame placement relative to the object before appendage closure.",
            "Create stable opposing contacts before transport or release.",
            "Only tune fine joints after object-frame alignment and contact timing are plausible.",
        ],
        "abstraction_priorities": [
            {"level": "gross_pose", "priority": 1, "reason": "object-frame distance dominates early failure"},
            {"level": "contact_geometry", "priority": 2, "reason": "transport needs opposing support"},
            {"level": "fine_actuation", "priority": 3, "reason": "joint tuning is useful after alignment"},
        ],
        "pedagogical_explanation": (
            "Treat the hand like a fixture that must first put its working frame at the "
            "object, then close support surfaces around it, then move the fixture."
        ),
        "similar_task_adaptations": [
            {"source": "lift small object from table", "adaptation": "use approach-contact-transport ordering"}
        ],
        "reward_biases": [
            {"name": "frame_alignment", "signal": "reduce object_to_grasp_site_xy_distance", "why": "moves exploration toward reachable contact"},
            {"name": "contact_before_transport", "signal": "contacts or settle dwell before lift/transport", "why": "discourages moving away prematurely"},
        ],
        "policy_initialization_hints": [
            "Initialize base/frame targets from actual object coordinates.",
            "Schedule appendage closure after frame placement and before transport.",
        ],
        "failure_modes": [
            {"mode": "base offset", "diagnostic": "large object_to_grasp_site_xy_distance", "correction": "retarget grasp_site in world coordinates"},
            {"mode": "premature transport", "diagnostic": "object stays low or slips", "correction": "increase contact/settle phase before transport"},
        ],
        "iteration_rules": [
            "If no object progress occurs, revise gross pose before finger targets.",
            "If contact exists but task progress is poor, revise force/timing or transport direction.",
        ],
    }

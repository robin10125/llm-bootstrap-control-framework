from __future__ import annotations

import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from policy_bias_lab.bias import (
    ADAPTIVE_REWARD_TEMPLATE_NAMES,
    CORE_REWARD_TEMPLATE_NAMES,
    REWARD_TEMPLATE_NAMES,
    default_reward_template_weights,
    reward_template_metadata,
)
from policy_bias_lab.schema import EVAL_FIELDS

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class DynamicRewardConfig:
    llm_backend: str
    llm_model: str | None
    arm: str
    task: str
    log_dir: Path
    cheap_checkup_steps: int = 50
    deep_checkup_seconds: float = 7200.0
    max_template_weight: float = 1.25
    min_base_reward_weight: float = 0.0
    max_base_reward_weight: float = 1.0
    post_new_reward_checkup_steps: int = 10
    post_new_reward_fast_checkups: int = 3
    allow_reward_rewrite: bool = False
    # Fixed-baseline mode: compile the initial program once, then hold reward weights and
    # base_reward_weight constant (no deep checkups, no mid-run rewrite) for a reproducible,
    # maximally-shaped baseline to compare action priors against.
    freeze_reward_shaping: bool = False
    previous_run_context: dict[str, Any] | None = None
    pre_run_reward_analysis: dict[str, Any] | None = None
    # Dense-to-sparse schedule: linearly anneal adaptive shaping toward zero (and restore
    # base_reward_weight toward 1.0) over the final fraction of the per-arm wall-clock budget.
    anneal_shaping: bool = False
    target_train_seconds: float | None = None
    shaping_anneal_start_fraction: float = 0.7


def load_pre_run_reward_analysis(
    *,
    llm_backend: str,
    llm_model: str | None,
    task: str,
    tasks: list[str],
    env_summary: dict[str, Any],
    bias_spec: dict[str, Any],
    previous_run_context: dict[str, Any] | None,
    log_dir: Path,
) -> dict[str, Any]:
    log_dir.mkdir(parents=True, exist_ok=True)
    prompt = build_pre_run_reward_analysis_prompt(
        task=task,
        tasks=tasks,
        env_summary=env_summary,
        bias_spec=bias_spec,
        previous_run_context=previous_run_context,
    )
    (log_dir / "pre_run_reward_analysis_prompt.md").write_text(prompt)
    if llm_backend in {"none", "mock", "fixture", "fake"}:
        analysis = _fallback_pre_run_reward_analysis(task, env_summary, previous_run_context)
        (log_dir / "pre_run_reward_analysis_fixture.json").write_text(json.dumps(analysis, indent=2) + "\n")
        return analysis
    from experiment_runtime.llm_backend import call_llm

    response = call_llm(
        llm_backend,
        prompt,
        model=llm_model,
        timeout_s=900.0,
        log_dir=log_dir,
        tag="pre_run_reward_analysis",
    )
    parsed = _parse_json(response.text) if response.ok else None
    analysis = _sanitize_pre_run_reward_analysis(parsed, task, env_summary, previous_run_context)
    (log_dir / "pre_run_reward_analysis.json").write_text(json.dumps(analysis, indent=2) + "\n")
    return analysis


def build_pre_run_reward_analysis_prompt(
    *,
    task: str,
    tasks: list[str],
    env_summary: dict[str, Any],
    bias_spec: dict[str, Any],
    previous_run_context: dict[str, Any] | None,
) -> str:
    return (
        "You are a task-agnostic pre-run failure-mode analyst for reward shaping in robot policy learning.\n"
        "The policy will be trained by PPO from observations to actuator actions. Before training starts, "
        "identify likely ways the policy could enter suboptimal or reward-hacking basins, and propose "
        "simple diagnostics and conservative reward-shaping priorities. Do not use hidden reward values, "
        "PPO losses, or task-specific hacks. Use only the task description, environment observables, robot "
        "action groups, adaptive reward slots, and any previous-run summaries provided.\n\n"
        "Reason in a general form that transfers to unseen tasks. First describe plausible failure modes "
        "and reward-hacking paths from the task/environment interface. Then describe diagnostics and ways "
        "to avoid those problems. Then describe the optimal kinematic forms and approaches that would make "
        "success causally reliable. Prefer evals over rewards when uncertain.\n\n"
        "Return JSON only with fields: priority_summary, likely_failure_modes, diagnostic_eval_plan, "
        "initial_reward_strategy, reward_curriculum, activation_criteria, base_reward_weight_strategy, "
        "reward_hacking_risks, action_prior_implications, optimal_kinematic_approach, prompt_improvements.\n"
        "likely_failure_modes should be a list of objects with name, mechanism, observables, why_it_matters, "
        "and suggested_countermeasures. diagnostic_eval_plan should include task-independent eval ideas. "
        "initial_reward_strategy and reward_curriculum should describe reward ideas in observable terms "
        "and mark any new reward ideas as requiring compilation. base_reward_weight_strategy may propose "
        "temporarily lowering the base reward weight if a shaped reward is needed to prevent a stronger "
        "base objective from overpowering task-valid behavior; justify this cautiously.\n\n"
        f"Task: {task}\n"
        f"Task suite: {json.dumps(tasks)}\n"
        f"Eval fields: {json.dumps(EVAL_FIELDS)}\n"
        f"Environment summary: {json.dumps(env_summary, indent=2)}\n"
        f"Initial bias spec summary: {json.dumps(_bias_spec_summary(bias_spec), indent=2)}\n"
        f"Previous run context: {json.dumps(previous_run_context or {}, indent=2)}\n"
    )


class DynamicRewardCoach:
    def __init__(self, cfg: DynamicRewardConfig):
        self.cfg = cfg
        self.log_dir = cfg.log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.template_names = tuple(REWARD_TEMPLATE_NAMES)
        self.current_weights = np.asarray(default_reward_template_weights(cfg.task), dtype=np.float32).copy()
        self.base_reward_weight = float(np.clip(1.0, cfg.min_base_reward_weight, cfg.max_base_reward_weight))
        self.last_deep_seconds = 0.0
        self.checkup_index = 0
        self.fast_checkups_remaining = 0
        self.active_adaptive_reward_terms: list[dict[str, Any]] = []
        self.rejected: list[dict[str, Any]] = []
        self._wall_start: float | None = None
        self._adaptive_slot_indices = [
            self.template_names.index(name) for name in ADAPTIVE_REWARD_TEMPLATE_NAMES
        ]
        self.last_anneal_factor = 1.0
        (self.log_dir / "reward_templates.json").write_text(json.dumps(reward_template_metadata(), indent=2) + "\n")
        if cfg.pre_run_reward_analysis:
            (self.log_dir / "pre_run_reward_analysis.json").write_text(
                json.dumps(cfg.pre_run_reward_analysis, indent=2) + "\n"
            )

    def _anneal_factor(self) -> float:
        """Linear dense-to-sparse factor in [0, 1] based on per-arm wall-clock progress."""
        if not self.cfg.anneal_shaping or not self.cfg.target_train_seconds:
            return 1.0
        if self._wall_start is None:
            return 1.0
        frac = (time.monotonic() - self._wall_start) / float(self.cfg.target_train_seconds)
        start = float(self.cfg.shaping_anneal_start_fraction)
        if frac <= start:
            return 1.0
        if frac >= 1.0:
            return 0.0
        return max(0.0, 1.0 - (frac - start) / (1.0 - start))

    def applied_weights(self) -> np.ndarray:
        """Intended weights with adaptive shaping scaled by the current anneal factor."""
        factor = self._anneal_factor()
        self.last_anneal_factor = factor
        weights = self.current_weights.copy()
        if factor < 1.0:
            for idx in self._adaptive_slot_indices:
                weights[idx] = weights[idx] * factor
        return weights

    def applied_base_reward_weight(self) -> float:
        """Base weight restored toward 1.0 as adaptive shaping anneals out."""
        factor = self._anneal_factor()
        return float(self.base_reward_weight + (1.0 - self.base_reward_weight) * (1.0 - factor))

    def __call__(self, ctx: dict[str, Any]) -> dict[str, Any]:
        self.checkup_index += 1
        if self._wall_start is None:
            self._wall_start = time.monotonic()
        rows = list(ctx.get("rows", []))
        elapsed = float(ctx.get("elapsed_seconds", 0.0))
        latest = rows[-1] if rows else {}
        previous = self.current_weights.copy()
        previous_base = self.base_reward_weight
        record = {
            "index": self.checkup_index,
            "kind": "cheap",
            "iter": int(ctx.get("iter", -1)),
            "elapsed_seconds": round(elapsed, 3),
            "arm": self.cfg.arm,
            "task": self.cfg.task,
            "previous_weights": _weights_dict(previous),
            "updated_weights": _weights_dict(self.current_weights),
            "previous_base_reward_weight": round(previous_base, 6),
            "base_reward_weight": round(self.base_reward_weight, 6),
            "diagnostics": _coach_safe_diagnostics(latest),
            "decision": "observe_only",
        }
        self._write_record(record)
        if not self.cfg.freeze_reward_shaping and (self.fast_checkups_remaining > 0 or self._should_deep_check(elapsed)):
            self.last_deep_seconds = elapsed
            self._deep_checkup(rows, latest, elapsed)
            if self.fast_checkups_remaining > 0:
                self.fast_checkups_remaining -= 1
        applied = self.applied_weights()
        record["anneal_factor"] = round(self.last_anneal_factor, 6)
        if self.last_anneal_factor < 1.0:
            record["applied_weights"] = _weights_dict(applied)
            record["applied_base_reward_weight"] = round(self.applied_base_reward_weight(), 6)
            self._write_record(record)
        out: dict[str, Any] = {
            "reward_weights": applied.tolist(),
            "base_reward_weight": self.applied_base_reward_weight(),
        }
        if self.fast_checkups_remaining > 0:
            out["next_checkup_interval"] = max(1, int(self.cfg.post_new_reward_checkup_steps))
        elif int(self.cfg.cheap_checkup_steps) > 0:
            out["next_checkup_interval"] = int(self.cfg.cheap_checkup_steps)
        return out

    def rewrite_reward_code(self, ctx: dict[str, Any], bias_spec: dict[str, Any]) -> tuple[dict[str, Any], list[float], dict[str, Any]]:
        rows = list(ctx.get("rows", []))
        latest = rows[-1] if rows else {}
        elapsed = float(ctx.get("elapsed_seconds", 0.0))
        previous = self.current_weights.copy()
        if not self.cfg.allow_reward_rewrite:
            record = {
                "kind": "reward_rewrite",
                "status": "disabled",
                "elapsed_seconds": round(elapsed, 3),
                "arm": self.cfg.arm,
                "task": self.cfg.task,
            }
            self._write_rewrite_record(record)
            return bias_spec, self.current_weights.tolist(), record

        prompt = self._build_rewrite_prompt(rows, latest, bias_spec)
        parsed: dict[str, Any] | None = None
        backend_record: dict[str, Any] = {"backend": self.cfg.llm_backend}
        if self.cfg.llm_backend not in {"none", "mock", "fixture", "fake"}:
            from experiment_runtime.llm_backend import call_llm

            response = call_llm(
                self.cfg.llm_backend,
                prompt,
                model=self.cfg.llm_model,
                timeout_s=900.0,
                log_dir=self.log_dir / "llm",
                tag="reward_rewrite",
            )
            backend_record |= {"response_ok": response.ok, "response_error": response.error}
            parsed = _parse_json(response.text) if response.ok else None
        else:
            backend_record |= {"response_ok": False, "response_error": "llm backend disabled; using deterministic fallback"}

        decision = self._compile_reward_rewrite(parsed, latest, rows)
        adapted_spec = dict(bias_spec)
        adapted_spec["adaptive_reward_terms"] = decision["adaptive_reward_terms"]
        self.active_adaptive_reward_terms = list(decision["adaptive_reward_terms"])
        self.current_weights = self._weights_with_rewrite(previous, decision)
        previous_base = self.base_reward_weight
        self._apply_base_reward_weight(decision.get("base_reward_weight"))
        if decision.get("adaptive_reward_terms"):
            self.fast_checkups_remaining = max(
                self.fast_checkups_remaining,
                int(self.cfg.post_new_reward_fast_checkups),
            )
        record = {
            "kind": "reward_rewrite",
            "status": "compiled",
            "elapsed_seconds": round(elapsed, 3),
            "arm": self.cfg.arm,
            "task": self.cfg.task,
            "backend": backend_record,
            "prompt": prompt,
            "parsed": parsed,
            "previous_weights": _weights_dict(previous),
            "updated_weights": _weights_dict(self.current_weights),
            "previous_base_reward_weight": round(previous_base, 6),
            "base_reward_weight": round(self.base_reward_weight, 6),
            "fast_checkups_remaining": self.fast_checkups_remaining,
            "decision": decision,
        }
        self._write_rewrite_record(record)
        return adapted_spec, self.current_weights.tolist(), record

    def apply_program(self, weights: Any, base_reward_weight: float, adaptive_terms: list[dict[str, Any]]) -> None:
        """Adopt a pre-generated reward program (no LLM call) so every arm/seed trains on the
        IDENTICAL frozen shaping — the prior's marginal effect is then unconfounded by
        per-arm reward-program variance."""
        self.current_weights = np.asarray(weights, dtype=np.float32).copy()
        self.base_reward_weight = float(np.clip(base_reward_weight, self.cfg.min_base_reward_weight, self.cfg.max_base_reward_weight))
        self.active_adaptive_reward_terms = list(adaptive_terms)

    def initial_reward_program(self, bias_spec: dict[str, Any]) -> tuple[dict[str, Any], list[float], dict[str, Any]]:
        """Compile the pre-run failure-mode analysis into a concrete shaped-reward program
        *before* training starts, so shaping guides exploration from iteration 0 instead of
        only appearing at the mid-run rewrite. Logged to ``initial_reward_program.json``."""
        # Gated by the runner's --initial-reward-program flag, so compile whenever called
        # (independent of allow_reward_rewrite, so it works in --freeze-reward-shaping mode).
        previous = self.current_weights.copy()

        prompt = self._build_initial_program_prompt(bias_spec)
        parsed: dict[str, Any] | None = None
        backend_record: dict[str, Any] = {"backend": self.cfg.llm_backend}
        if self.cfg.llm_backend not in {"none", "mock", "fixture", "fake"}:
            from experiment_runtime.llm_backend import call_llm

            response = call_llm(
                self.cfg.llm_backend,
                prompt,
                model=self.cfg.llm_model,
                timeout_s=900.0,
                log_dir=self.log_dir / "llm",
                tag="initial_reward_program",
            )
            backend_record |= {
                "response_ok": response.ok,
                "response_error": response.error,
                "notes": getattr(response, "notes", None),
            }
            parsed = _parse_json(response.text) if response.ok else None
        else:
            backend_record |= {"response_ok": False, "response_error": "llm backend disabled; using deterministic fallback"}

        decision = self._compile_reward_rewrite(parsed, {}, [])
        adapted_spec = dict(bias_spec)
        adapted_spec["adaptive_reward_terms"] = decision["adaptive_reward_terms"]
        self.active_adaptive_reward_terms = list(decision["adaptive_reward_terms"])
        self.current_weights = self._weights_with_rewrite(previous, decision)
        previous_base = self.base_reward_weight
        self._apply_base_reward_weight(decision.get("base_reward_weight"))
        record = {
            "kind": "initial_reward_program",
            "status": "compiled",
            "arm": self.cfg.arm,
            "task": self.cfg.task,
            "backend": backend_record,
            "prompt": prompt,
            "parsed": parsed,
            "previous_weights": _weights_dict(previous),
            "updated_weights": _weights_dict(self.current_weights),
            "previous_base_reward_weight": round(previous_base, 6),
            "base_reward_weight": round(self.base_reward_weight, 6),
            "decision": decision,
        }
        self._write_initial_record(record)
        return adapted_spec, self.current_weights.tolist(), record

    def _build_initial_program_prompt(self, bias_spec: dict[str, Any]) -> str:
        return (
            "You are a task-agnostic reward-code author for direct robot policy learning.\n"
            "Training has NOT started yet. Compile the pre-run failure-mode analysis into a concrete, "
            "conservative shaped-reward program that will guide PPO exploration from the very first "
            "iteration. Reward shaping is most valuable early, before the policy commits to a basin, so "
            "front-load prerequisite-structure shaping now rather than waiting for failures to appear.\n\n"
            "You may not use hidden env reward, base return, train return, or PPO losses. You may use the "
            "task description, eval observables, robot action groups, and the pre-run analysis. Prefer "
            "bounded potential-progress terms gated on prerequisites so they are policy-invariant and "
            "cannot be hacked. Encode the basin a competent strategy should enter; do not reward raw "
            "scalar magnitude.\n\n"
            "Before selecting rewards, describe likely failure modes, reward-hacking paths, ways to avoid "
            "them, and the optimal kinematic forms implied by the task and robot. You may set "
            "base_reward_weight in [0, 1] only if a needed shaped reward would otherwise be overpowered by "
            "the base objective; keep the base objective dominant when possible since shaping will be "
            "annealed out later.\n\n"
            "Return JSON only with fields: diagnosis, priority_summary, failure_modes, reward_hacking_risks, "
            "avoidance_plan, optimal_kinematic_approach, selected_templates, base_reward_weight, "
            "reward_curriculum, adaptive_reward_terms, proposed_evals, rejected_risks, prompt_improvements.\n"
            "adaptive_reward_terms is a list of bounded reward-code clauses. Each clause supports: name, "
            "kind, observable, direction, weight, scale, threshold, tasks, gate, max_step. kind is one of "
            "absolute_bound_penalty, progress_penalty, progress_reward, absolute_good_reward. observable is "
            "one eval field. direction is minimize|maximize. gate is a list of conditions with field, op in "
            "< <= > >=, and value. Use no more than 8 adaptive clauses.\n"
            "Reward-form guidance: for prerequisite approach/reach toward a target (e.g. shrinking a "
            "distance), prefer absolute_good_reward as a continuous potential (rewards being inside a "
            "band every step) over change-based progress_reward, which has ZERO gradient when the "
            "observable plateaus and so cannot pull a stalled policy across the last gap. Reserve "
            "progress_* for quantities that move steadily.\n\n"
            f"Task: {self.cfg.task}\n"
            f"Arm: {self.cfg.arm}\n"
            f"Eval fields: {json.dumps(EVAL_FIELDS)}\n"
            f"Available adaptive reward slots: {json.dumps(list(ADAPTIVE_REWARD_TEMPLATE_NAMES))}\n"
            f"Pre-run failure-mode analysis: {json.dumps(self.cfg.pre_run_reward_analysis or {}, indent=2)}\n"
            f"Previous run context: {json.dumps(self.cfg.previous_run_context or {}, indent=2)}\n"
            f"Current bias spec summary: {json.dumps(_bias_spec_summary(bias_spec), indent=2)}\n"
        )

    def _write_initial_record(self, record: dict[str, Any]) -> None:
        path = self.log_dir / "initial_reward_program.json"
        path.write_text(json.dumps(record, indent=2) + "\n")

    def _should_deep_check(self, elapsed: float) -> bool:
        return (
            self.cfg.deep_checkup_seconds > 0
            and elapsed - self.last_deep_seconds >= self.cfg.deep_checkup_seconds
        )

    def _deep_checkup(self, rows: list[dict[str, Any]], latest: dict[str, Any], elapsed: float) -> None:
        tag = f"deep_{self.checkup_index:03d}"
        prompt = self._build_prompt(rows, latest)
        llm_dir = self.log_dir / "llm"
        if self.cfg.llm_backend in {"none", "mock", "fixture", "fake"}:
            record = {
                "index": self.checkup_index,
                "kind": "deep",
                "elapsed_seconds": round(elapsed, 3),
                "backend": self.cfg.llm_backend,
                "decision": "no_llm_call",
                "prompt": prompt,
            }
            self._write_record(record)
            return
        from experiment_runtime.llm_backend import call_llm

        response = call_llm(
            self.cfg.llm_backend,
            prompt,
            model=self.cfg.llm_model,
            timeout_s=600.0,
            log_dir=llm_dir,
            tag=tag,
        )
        parsed = _parse_json(response.text) if response.ok else None
        decision = self._apply_deep_decision(parsed)
        record = {
            "index": self.checkup_index,
            "kind": "deep",
            "elapsed_seconds": round(elapsed, 3),
            "backend": self.cfg.llm_backend,
            "response_ok": response.ok,
            "response_error": response.error,
            "parsed": parsed,
            "decision": decision,
            "updated_weights": _weights_dict(self.current_weights),
        }
        self._write_record(record)

    def _build_prompt(self, rows: list[dict[str, Any]], latest: dict[str, Any]) -> str:
        recent = [_coach_safe_diagnostics(row) for row in rows[-8:]]
        return (
            "You are a conservative reward-shaping coach for a robot PPO policy.\n"
            "Diagnose clear current-stage policy failures, reward-hacking risks, and missing diagnostics. "
            "Prefer evaluations over rewards when uncertain. Only activate simple, well-principled shaped "
            "rewards that guide the policy toward robust task completion.\n\n"
            "You must not request direct use of the hidden environment reward, base return, train return, "
            "or PPO losses. You may reason over task description, eval summaries, saturation metrics, "
            "adaptive reward-slot returns, and active/rejected reward history.\n\n"
            "Before choosing weights, describe likely failure modes, reward-hacking paths, ways to avoid "
            "those problems, and the optimal kinematic forms or approaches implied by the task and robot.\n"
            "You may propose a reward curriculum. You may also select a base_reward_weight in [0, 1] if "
            "the base objective appears to overpower a shaped reward needed to avoid invalid dynamics; "
            "lowering it should be temporary and explicitly justified.\n\n"
            "Return JSON only with fields: diagnosis, failure_modes, reward_hacking_risks, avoidance_plan, "
            "optimal_kinematic_approach, selected_templates, base_reward_weight, reward_curriculum, "
            "proposed_evals, proposed_new_rewards, rejected_risks, prompt_improvements.\n"
            "selected_templates may map active adaptive template slot names to nonnegative weights. "
            "Unknown rewards/evals may be proposed, but they will be logged as requiring recompilation.\n\n"
            f"Task: {self.cfg.task}\n"
            f"Arm: {self.cfg.arm}\n"
            f"Eval fields: {json.dumps(EVAL_FIELDS)}\n"
            f"Active adaptive reward slots: {json.dumps(_adaptive_weight_dict(self.current_weights), indent=2)}\n"
            f"Active adaptive reward terms: {json.dumps(self.active_adaptive_reward_terms, indent=2)}\n"
            f"Current base reward weight: {self.base_reward_weight}\n"
            f"Latest diagnostics: {json.dumps(_coach_safe_diagnostics(latest), indent=2)}\n"
            f"Recent diagnostics: {json.dumps(recent, indent=2)}\n"
            f"Pre-run failure-mode analysis: {json.dumps(self.cfg.pre_run_reward_analysis or {}, indent=2)}\n"
            f"Rejected history: {json.dumps(self.rejected[-10:], indent=2)}\n"
        )

    def _build_rewrite_prompt(self, rows: list[dict[str, Any]], latest: dict[str, Any], bias_spec: dict[str, Any]) -> str:
        recent = [_coach_safe_diagnostics(row) for row in rows[-10:]]
        return (
            "You are a task-agnostic reward-code troubleshooting agent for direct robot policy learning.\n"
            "Your job is to inspect current policy dynamics, summarize priority ordering, and if needed "
            "rewrite the shaped reward program once so PPO is guided away from reward hacks and toward "
            "robust task completion. Be conservative but creative: prefer simple diagnostics and rewards "
            "that encode prerequisite structure, causal coordination, stability, and task-valid progress.\n\n"
            "You may not use hidden env reward, base return, train return, or PPO losses. You may use task "
            "description, eval observables, action saturation, active reward contributions, and previous "
            "run failure summaries. Avoid quick hacks that merely increase scalar reward; encode the basin "
            "that a competent strategy should enter.\n\n"
            "Before selecting rewards, describe likely failure modes, reward-hacking paths, ways to avoid "
            "those problems, and the optimal kinematic forms or approaches implied by the task and robot.\n"
            "You may define a curriculum and may set base_reward_weight in [0, 1] when an important shaped "
            "reward would otherwise be overpowered by the base objective. Lowering the base objective must "
            "be justified as preserving task-valid learning, not as hiding poor performance.\n\n"
            "Return JSON only with fields: diagnosis, priority_summary, failure_modes, reward_hacking_risks, "
            "avoidance_plan, optimal_kinematic_approach, selected_templates, base_reward_weight, "
            "reward_curriculum, adaptive_reward_terms, proposed_evals, rejected_risks, prompt_improvements.\n"
            "selected_templates maps active adaptive template slot names to weights. adaptive_reward_terms is a list of "
            "bounded reward-code clauses. Each clause supports: name, kind, observable, direction, weight, "
            "scale, threshold, tasks, gate, max_step. kind is one of absolute_bound_penalty, "
            "progress_penalty, progress_reward, absolute_good_reward. observable is one eval field. "
            "direction is minimize|maximize. gate is a list of conditions with field, op in < <= > >=, "
            "and value. Use no more than 8 adaptive clauses.\n"
            "Reward-form guidance: for prerequisite approach/reach toward a target (e.g. shrinking a "
            "distance), prefer absolute_good_reward as a continuous potential (rewards being inside a "
            "band every step) over change-based progress_reward, which has ZERO gradient when the "
            "observable plateaus and so cannot pull a stalled policy across the last gap. Reserve "
            "progress_* for quantities that move steadily.\n\n"
            "General priorities to reason over for any task:\n"
            "1. Preserve task validity before optimizing task magnitude.\n"
            "2. Identify prerequisites implied by the task before rewarding later-stage progress.\n"
            "3. Penalize behavior only when diagnostics show it undermines task-valid success.\n"
            "4. Keep reward code bounded and observable-only; when uncertain, prefer evals over rewards.\n"
            "5. Treat saturation, unstable state changes, loss of required interaction, or incidental progress as possible failure modes, not universal failures.\n\n"
            f"Task: {self.cfg.task}\n"
            f"Arm: {self.cfg.arm}\n"
            f"Eval fields: {json.dumps(EVAL_FIELDS)}\n"
            f"Available adaptive reward slots: {json.dumps(list(ADAPTIVE_REWARD_TEMPLATE_NAMES))}\n"
            f"Active adaptive reward slots: {json.dumps(_adaptive_weight_dict(self.current_weights), indent=2)}\n"
            f"Active adaptive reward terms: {json.dumps(self.active_adaptive_reward_terms, indent=2)}\n"
            f"Current base reward weight: {self.base_reward_weight}\n"
            f"Latest diagnostics: {json.dumps(_coach_safe_diagnostics(latest), indent=2)}\n"
            f"Recent diagnostics: {json.dumps(recent, indent=2)}\n"
            f"Pre-run failure-mode analysis: {json.dumps(self.cfg.pre_run_reward_analysis or {}, indent=2)}\n"
            f"Previous run context: {json.dumps(self.cfg.previous_run_context or {}, indent=2)}\n"
            f"Current bias spec summary: {json.dumps(_bias_spec_summary(bias_spec), indent=2)}\n"
        )

    def _compile_reward_rewrite(
        self,
        parsed: dict[str, Any] | None,
        latest: dict[str, Any],
        rows: list[dict[str, Any]],
    ) -> dict[str, Any]:
        fallback = _fallback_reward_rewrite(self.cfg.task, latest, rows, self.cfg.previous_run_context)
        if not isinstance(parsed, dict):
            return fallback | {"source": "deterministic_fallback_no_valid_json"}

        selected = parsed.get("selected_templates") if isinstance(parsed.get("selected_templates"), dict) else {}
        adaptive = parsed.get("adaptive_reward_terms") if isinstance(parsed.get("adaptive_reward_terms"), list) else []
        return {
            "source": "llm_reward_rewrite",
            "diagnosis": parsed.get("diagnosis"),
            "priority_summary": parsed.get("priority_summary"),
            "failure_modes": parsed.get("failure_modes"),
            "reward_hacking_risks": parsed.get("reward_hacking_risks"),
            "avoidance_plan": parsed.get("avoidance_plan"),
            "optimal_kinematic_approach": parsed.get("optimal_kinematic_approach"),
            "selected_templates": selected,
            "base_reward_weight": parsed.get("base_reward_weight"),
            "reward_curriculum": parsed.get("reward_curriculum"),
            "adaptive_reward_terms": adaptive[:len(ADAPTIVE_REWARD_TEMPLATE_NAMES)],
            "proposed_evals": parsed.get("proposed_evals"),
            "rejected_risks": parsed.get("rejected_risks"),
            "prompt_improvements": parsed.get("prompt_improvements"),
        }

    def _weights_with_rewrite(self, previous: np.ndarray, decision: dict[str, Any]) -> np.ndarray:
        weights = previous.copy()
        selected = decision.get("selected_templates") or {}
        if isinstance(selected, dict):
            for name, value in selected.items():
                if name not in self.template_names:
                    continue
                try:
                    weight = float(value)
                except (TypeError, ValueError):
                    continue
                if 0.0 <= weight <= self.cfg.max_template_weight:
                    weights[self.template_names.index(name)] = weight
        for name in ADAPTIVE_REWARD_TEMPLATE_NAMES[:len(decision.get("adaptive_reward_terms") or [])]:
            weights[self.template_names.index(name)] = 1.0
        return weights

    def _apply_deep_decision(self, parsed: dict[str, Any] | None) -> dict[str, Any]:
        if not isinstance(parsed, dict):
            return {"status": "ignored", "reason": "no valid JSON"}
        selected = parsed.get("selected_templates") or {}
        if not isinstance(selected, dict):
            return {"status": "ignored", "reason": "selected_templates must be object"}
        previous = self.current_weights.copy()
        accepted: dict[str, float] = {}
        rejected: list[dict[str, Any]] = []
        for name, value in selected.items():
            if name not in self.template_names:
                rejected.append({"name": name, "reason": "unknown_template_requires_recompile"})
                continue
            try:
                weight = float(value)
            except (TypeError, ValueError):
                rejected.append({"name": name, "reason": "weight_not_numeric"})
                continue
            if weight < 0.0 or weight > self.cfg.max_template_weight:
                rejected.append({"name": name, "reason": "weight_out_of_bounds", "value": weight})
                continue
            self.current_weights[self.template_names.index(name)] = weight
            accepted[name] = weight
        for reward in parsed.get("proposed_new_rewards") or []:
            rejected.append({"proposal": reward, "reason": "new_reward_requires_recompile_not_applied_mid_segment"})
        for item in rejected:
            self.rejected.append(item)
        base_update = self._apply_base_reward_weight(parsed.get("base_reward_weight"))
        return {
            "status": "applied" if accepted else "no_template_change",
            "previous_weights": _weights_dict(previous),
            "accepted": accepted,
            "base_reward_update": base_update,
            "reward_curriculum": parsed.get("reward_curriculum"),
            "rejected": rejected,
        }

    def _apply_base_reward_weight(self, raw: Any) -> dict[str, Any]:
        if raw is None:
            return {"status": "unchanged", "value": round(self.base_reward_weight, 6)}
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return {"status": "rejected", "reason": "not_numeric", "value": raw}
        if value < self.cfg.min_base_reward_weight or value > self.cfg.max_base_reward_weight:
            return {
                "status": "rejected",
                "reason": "out_of_bounds",
                "value": value,
                "bounds": [self.cfg.min_base_reward_weight, self.cfg.max_base_reward_weight],
            }
        previous = self.base_reward_weight
        self.base_reward_weight = value
        return {"status": "applied", "previous": round(previous, 6), "value": round(value, 6)}

    def _write_record(self, record: dict[str, Any]) -> None:
        path = self.log_dir / f"checkup_{self.checkup_index:03d}_{record['kind']}.json"
        path.write_text(json.dumps(record, indent=2) + "\n")

    def _write_rewrite_record(self, record: dict[str, Any]) -> None:
        path = self.log_dir / "reward_rewrite.json"
        path.write_text(json.dumps(record, indent=2) + "\n")


def _coach_safe_diagnostics(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "iter": row.get("iter"),
        "elapsed_seconds": row.get("elapsed_seconds"),
        "eval_summary": row.get("eval_summary"),
        "success": row.get("success"),
        "instant_success": row.get("instant_success"),
        "lift_max": row.get("lift_max"),
        "reach_rate": row.get("reach_rate"),
        "grasp_rate": row.get("grasp_rate"),
        "lift_reached_rate": row.get("lift_reached_rate"),
        "hard_clip_frac": row.get("hard_clip_frac"),
        "saturation_frac": row.get("saturation_frac"),
        "action_abs_mean": row.get("action_abs_mean"),
        "adaptive_reward_slot_returns": _adaptive_template_returns_dict(row.get("reward_template_returns")),
    }


def _template_returns_dict(values: Any) -> dict[str, float] | None:
    if values is None:
        return None
    out: dict[str, float] = {}
    for name, value in zip(REWARD_TEMPLATE_NAMES, values):
        try:
            out[name] = round(float(value), 6)
        except (TypeError, ValueError):
            out[name] = 0.0
    return out


def _weights_dict(values: Any) -> dict[str, float]:
    return {name: round(float(value), 6) for name, value in zip(REWARD_TEMPLATE_NAMES, values)}


def _adaptive_weight_dict(values: Any) -> dict[str, float]:
    offset = len(CORE_REWARD_TEMPLATE_NAMES)
    return {
        name: round(float(value), 6)
        for name, value in zip(ADAPTIVE_REWARD_TEMPLATE_NAMES, list(values)[offset:])
    }


def _adaptive_template_returns_dict(values: Any) -> dict[str, float] | None:
    if values is None:
        return None
    raw = list(values)
    offset = len(CORE_REWARD_TEMPLATE_NAMES)
    return {
        name: round(float(value), 6)
        for name, value in zip(ADAPTIVE_REWARD_TEMPLATE_NAMES, raw[offset:])
    }


def _bias_spec_summary(spec: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": spec.get("name"),
        "reward_term_count": len(spec.get("reward_terms", []) or []),
        "adaptive_reward_term_count": len(spec.get("adaptive_reward_terms", []) or []),
        "action_prior_count": len(spec.get("action_priors", []) or []),
        "supervised_target_count": len(spec.get("supervised_targets", []) or []),
        "exploration_group_count": len(spec.get("exploration_groups", []) or []),
    }


def _sanitize_pre_run_reward_analysis(
    parsed: dict[str, Any] | None,
    task: str,
    env_summary: dict[str, Any],
    previous_run_context: dict[str, Any] | None,
) -> dict[str, Any]:
    fallback = _fallback_pre_run_reward_analysis(task, env_summary, previous_run_context)
    if not isinstance(parsed, dict):
        return fallback | {"source": "fallback_no_valid_json"}
    out = {"source": "llm_pre_run_reward_analysis"}
    for key in (
        "priority_summary",
        "likely_failure_modes",
        "diagnostic_eval_plan",
        "initial_reward_strategy",
        "reward_curriculum",
        "activation_criteria",
        "base_reward_weight_strategy",
        "reward_hacking_risks",
        "action_prior_implications",
        "optimal_kinematic_approach",
        "prompt_improvements",
    ):
        value = parsed.get(key)
        out[key] = value if value not in (None, "", []) else fallback.get(key)
    return out


def _fallback_pre_run_reward_analysis(
    task: str,
    env_summary: dict[str, Any],
    previous_run_context: dict[str, Any] | None,
) -> dict[str, Any]:
    _ = (task, env_summary, previous_run_context)
    failure_modes: list[dict[str, Any]] = [
        {
            "name": "incidental_progress",
            "mechanism": "The policy improves a task-relevant observable through a side effect rather than through the intended controlled mechanism.",
            "observables": "Use environment-provided eval fields that distinguish progress from prerequisite control.",
            "why_it_matters": "Apparent progress may not transfer to robust task completion.",
            "suggested_countermeasures": [
                "Track prerequisite-conditioned task progress.",
                "Introduce bounded penalties only after diagnostics show the side effect contradicts task validity.",
            ],
        },
        {
            "name": "noncausal_actuation",
            "mechanism": "The robot produces large or repeated actions that do not establish the task prerequisites.",
            "observables": "Use action magnitude, saturation, and prerequisite-specific evals.",
            "why_it_matters": "The policy can enter an active but non-causal basin.",
            "suggested_countermeasures": [
                "Track action saturation and prerequisite progress together.",
                "Reward prerequisites before later-stage progress when the task implies ordering.",
            ],
        },
        {
            "name": "overpowering_prior_or_shaping",
            "mechanism": "Auxiliary biases dominate exploration and push the policy into a narrow suboptimal basin.",
            "observables": "Use action statistics, shaped reward contribution magnitudes, and task-valid progress diagnostics.",
            "why_it_matters": "A useful prior can become a hard-coded strategy if not relaxed or corrected.",
            "suggested_countermeasures": [
                "Log auxiliary contribution magnitudes.",
                "Allow early scalar prior/reward/base-weight adjustment.",
            ],
        },
    ]
    return {
        "source": "deterministic_fallback",
        "priority_summary": [
            "Define task-valid progress before optimizing progress magnitude.",
            "Track whether progress is causally coupled to task prerequisites.",
            "Prefer conservative, bounded shaping and diagnostics before adding new rewards.",
        ],
        "likely_failure_modes": failure_modes,
        "diagnostic_eval_plan": [
            "prerequisite_conditioned_task_progress",
            "bounded_task_validity",
            "noncausal_actuation_rate",
            "incidental_progress_before_prerequisite",
            "sustained_success_duration",
        ],
        "initial_reward_strategy": [
            "Start with prerequisite and stability templates at low weights when they are task-relevant.",
            "Avoid ungated task-progress shaping when the base environment already rewards task magnitude.",
            "Escalate penalties only after diagnostics confirm a specific failure mode.",
        ],
        "reward_curriculum": [
            "Stage rewards by prerequisite, controlled progress, then stability or completion.",
            "Retain earlier-stage rewards only while they remain unsaturated or prevent regression.",
        ],
        "activation_criteria": [
            "Increase prerequisite shaping when task progress rises without prerequisite satisfaction.",
            "Increase validity shaping when a side effect contradicts the task description.",
            "Reduce or relax priors if side-effect metrics worsen while prerequisites remain unsatisfied.",
        ],
        "base_reward_weight_strategy": [
            "Keep the base reward at full weight unless diagnostics show it overpowers necessary task-valid shaping.",
            "Temporarily lower the base reward only with explicit restoration criteria.",
        ],
        "reward_hacking_risks": [
            "Rewarding a proxy that can improve while violating task validity.",
            "Rewarding a later-stage metric before prerequisites are satisfied.",
            "Rewarding action magnitude or posture without causal task progress.",
        ],
        "action_prior_implications": [
            "Use runtime-relative priors only when the observation semantics support them.",
            "Avoid strong unconditional priors before the relevant geometry or prerequisites are reliable.",
            "Make prior weights adjustable early in training.",
        ],
        "optimal_kinematic_approach": [
            "Identify the body frames, contacts, support relations, and motion directions required by the task.",
            "Prefer smooth, stable, low-saturation motions that preserve future controllability.",
        ],
        "prompt_improvements": [
            "Ask checkup agents to compare task-valid progress against raw proxy progress.",
            "Report whether auxiliary rewards are active, gated off, or too small relative to observed behavior.",
        ],
    }


def _fallback_reward_rewrite(
    task: str,
    latest: dict[str, Any],
    rows: list[dict[str, Any]],
    previous_run_context: dict[str, Any] | None,
) -> dict[str, Any]:
    _ = (task, latest, rows, previous_run_context)
    return {
        "source": "deterministic_fallback",
        "diagnosis": (
            "No valid LLM reward rewrite was available. Leaving reward code unchanged so the "
            "experiment does not receive task-specific fallback knowledge."
        ),
        "priority_summary": [
            "A model-generated analysis is required before adding adaptive rewards.",
            "Keep current weights until diagnostics justify a task-valid shaped reward.",
        ],
        "selected_templates": {},
        "base_reward_weight": None,
        "reward_curriculum": [],
        "adaptive_reward_terms": [],
        "proposed_evals": [],
        "rejected_risks": [
            "Deterministic fallback rewards would contaminate zero-shot failure discovery.",
        ],
    }


def _parse_json(text: str) -> dict[str, Any] | None:
    candidates = [m.group(1) for m in re.finditer(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S)]
    m = re.search(r"\{.*\}", text, flags=re.S)
    if m:
        candidates.append(m.group(0))
    for raw in candidates:
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jp
import numpy as np

from policy_bias_lab.schema import ACTION_GROUPS, PRIOR_DIRECTIONS
from policy_bias_lab.symbolic_control import encode_rules, make_rule_action_fn, sustained_bool

ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAPPING = ROOT.parent / "bootstrapping"


@dataclass
class ActionPriorConfig:
    llm_backend: str
    llm_model: str | None
    task: str
    tasks: list[str]
    arm: str
    env_summary: dict[str, Any]
    log_dir: Path
    max_weight: float = 0.6
    max_checkups: int = 3
    previous_run_context: dict[str, Any] | None = None
    pre_run_reward_analysis: dict[str, Any] | None = None
    candidate_count: int = 5
    selection_envs: int = 128
    selection_steps: int | None = None
    selection_seed: int = 0
    success_hold_seconds: float = 0.5
    success_lift_threshold: float = 0.05
    # Contact-gated selection thresholds. These use only signals a real robot can
    # observe (fingertip contact count, object height, object lateral drift), so the
    # whole selection procedure is realizable on hardware with a few real rollouts.
    min_contacts: float = 1.0
    max_xy_disp: float = 0.08


def load_action_prior_rules(cfg: ActionPriorConfig, current_bias_spec: dict[str, Any]) -> list[dict[str, Any]]:
    cfg.log_dir.mkdir(parents=True, exist_ok=True)
    if cfg.llm_backend in {"fixture", "mock", "fake", "none"}:
        rules = _fallback_action_priors(cfg.task)
        (cfg.log_dir / "action_prior_fixture.json").write_text(json.dumps(rules, indent=2) + "\n")
        return rules
    prompt = build_action_prior_prompt(cfg, current_bias_spec)
    (cfg.log_dir / "action_prior_prompt.md").write_text(prompt)
    response_text = _call_llm(cfg.llm_backend, prompt, model=cfg.llm_model, log_dir=cfg.log_dir, tag="action_prior")
    (cfg.log_dir / "action_prior_completion.txt").write_text(response_text)
    parsed = _parse_json(response_text)
    rules = _sanitize_action_prior_rules(parsed.get("action_priors") if isinstance(parsed, dict) else None, cfg.max_weight)
    if not rules:
        rules = _fallback_action_priors(cfg.task)
    (cfg.log_dir / "action_prior_rules.json").write_text(json.dumps(rules, indent=2) + "\n")
    return rules


def load_pareto_action_prior_rules(
    cfg: ActionPriorConfig,
    current_bias_spec: dict[str, Any],
    env: Any,
) -> list[dict[str, Any]]:
    return _load_pareto_rules(
        cfg=cfg,
        current_bias_spec=current_bias_spec,
        env=env,
        section="action_priors",
        label="action_prior",
    )


def load_pareto_supervised_target_rules(
    cfg: ActionPriorConfig,
    current_bias_spec: dict[str, Any],
    env: Any,
) -> list[dict[str, Any]]:
    return _load_pareto_rules(
        cfg=cfg,
        current_bias_spec=current_bias_spec,
        env=env,
        section="supervised_targets",
        label="supervised_target",
    )


def build_action_prior_prompt(cfg: ActionPriorConfig, current_bias_spec: dict[str, Any]) -> str:
    direction_semantics = {
        "toward_object_xy": "runtime-relative lateral approach operator when the injected environment exposes the needed relative state",
        "lower_base": "negative motion along the base vertical control direction",
        "raise_base": "positive motion along the base vertical control direction",
        "close_hand": "move selected appendage actuators toward their closed-side normalized controls",
        "open_hand": "move selected appendage actuators toward their open-side normalized controls",
        "stabilize": "weak damping-like bias toward neutral controls for the selected group",
    }
    return (
        "You are designing runtime action priors for a direct robot policy model.\n"
        "The policy is trained by PPO and outputs actuator actions. Action priors are not "
        "hard commands; they are weak pre-tanh mean shifts that bias early exploration.\n\n"
        "Critical generalization rule: do not assume fixed starting positions, object locations, "
        "coordinates, waypoints, scripts, or phase-specific constants tied to one reset. Use only "
        "symbolic action groups and directions. If a direction has runtime analytic semantics, rely "
        "on that operator rather than inventing coordinates.\n\n"
        "Use action priors to bias broad exploration basins, not to solve the whole task. Prefer "
        "conservative weights. Before proposing priors, describe likely failure modes, reward-hacking "
        "routes, ways to avoid those problems, and the optimal kinematic forms or approaches implied "
        "by the injected task/environment data. If uncertain, use lower weights or omit a prior.\n\n"
        f"Allowed action groups: {json.dumps(ACTION_GROUPS)}\n"
        f"Allowed directions: {json.dumps(PRIOR_DIRECTIONS)}\n"
        f"Direction semantics: {json.dumps(direction_semantics, indent=2)}\n"
        "Return JSON only with fields: rationale, potential_failure_modes, reward_hacking_risks, "
        "avoidance_plan, optimal_kinematic_approach, action_priors, rejected_risks.\n"
        "Each action prior: name, group, direction, weight. Weights must be nonnegative and should "
        f"usually be <= {cfg.max_weight}. The runtime system can later adjust these scalar weights "
        "without recompilation, but cannot change the rule structure without recompilation.\n\n"
        f"Task: {cfg.task}\n"
        f"Task suite: {json.dumps(cfg.tasks)}\n"
        f"Arm: {cfg.arm}\n"
        f"Environment summary: {json.dumps(cfg.env_summary, indent=2)}\n"
        f"Current bias spec action prior count, for reference only: {len(current_bias_spec.get('action_priors', []) or [])}\n"
        f"Pre-run reward failure-mode analysis: {json.dumps(cfg.pre_run_reward_analysis or {}, indent=2)}\n"
        f"Previous run context: {json.dumps(cfg.previous_run_context or {}, indent=2)}\n"
    )


def build_candidate_prompt(cfg: ActionPriorConfig, current_bias_spec: dict[str, Any], *, section: str) -> str:
    mechanism = (
        "runtime action priors, which are fixed weak mean shifts used during PPO exploration"
        if section == "action_priors"
        else "supervised initialization targets, which are fixed behavioral targets used only before PPO"
    )
    output_key = "action_priors" if section == "action_priors" else "supervised_targets"
    return (
        f"Design several candidate {mechanism} for a direct robot policy model.\n"
        "Each candidate is a small set of weak rules. A rule names a symbolic action group and a "
        "direction; its weight is a gentle pre-tanh mean shift that nudges early behavior. Rules do "
        "not command trajectories, they only bias which basin exploration starts in.\n\n"
        "OBJECTIVE: enter a contact-rich grasp basin. The hand should approach the object, make and "
        "keep fingertip contact, then lift it. After you propose candidates, the runner AUTOMATICALLY "
        "rolls out each one and scores it by a CONTACT-GATED metric: lift only counts while at least "
        f"{int(cfg.min_contacts)} fingertip(s) are in contact AND the object's lateral drift stays small. "
        "A candidate that 'lifts' by shoving or flinging the object sideways with no contact scores ~0. "
        "So design every candidate to win that metric: prioritize approach + finger contact over base "
        "translation, and never rely on lateral pushing.\n\n"
        "Hard constraints: no fixed positions, object coordinates, waypoints, hand-coded phases, or "
        "one-reset constants. Symbolic groups and directions only. Runtime-relative operators may use "
        "injected state such as the exposed object displacement. Keep schemas simple and weights "
        "conservative; when uncertain, use a smaller weight or omit the rule.\n\n"
        "Make candidates genuinely different from each other (e.g. whole-hand enclosure vs. "
        "finger-first opposition vs. minimal hint) so the contact-gated rollout can discriminate.\n\n"
        f"Allowed action groups: {json.dumps(ACTION_GROUPS)}\n"
        f"Allowed directions: {json.dumps(PRIOR_DIRECTIONS)}\n"
        "Return JSON only with fields: rationale, potential_failure_modes, reward_hacking_risks, "
        "avoidance_plan, optimal_kinematic_approach, candidates.\n"
        f"Each candidate must contain: name, principle, expected_strengths, risks, {output_key}.\n"
        f"Each item in {output_key}: name, group, direction, weight. Weights are nonnegative and "
        f"must be <= {cfg.max_weight}; use lower weights when uncertain.\n\n"
        f"Candidate count requested: {cfg.candidate_count}\n"
        f"Task: {cfg.task}\n"
        f"Task suite: {json.dumps(cfg.tasks)}\n"
        f"Arm: {cfg.arm}\n"
        f"Environment summary: {json.dumps(cfg.env_summary, indent=2)}\n"
        f"Pre-run reward failure-mode analysis: {json.dumps(cfg.pre_run_reward_analysis or {}, indent=2)}\n"
        f"Previous run context: {json.dumps(cfg.previous_run_context or {}, indent=2)}\n"
        f"Current bias spec {section} count, for reference only: {len(current_bias_spec.get(section, []) or [])}\n"
    )


def _load_pareto_rules(
    *,
    cfg: ActionPriorConfig,
    current_bias_spec: dict[str, Any],
    env: Any,
    section: str,
    label: str,
) -> list[dict[str, Any]]:
    cfg.log_dir.mkdir(parents=True, exist_ok=True)
    output_key = "action_priors" if section == "action_priors" else "supervised_targets"
    if cfg.llm_backend in {"fixture", "mock", "fake", "none"}:
        candidates = _fallback_candidates(section)
        response_text = ""
    else:
        prompt = build_candidate_prompt(cfg, current_bias_spec, section=section)
        (cfg.log_dir / f"{label}_candidates_prompt.md").write_text(prompt)
        response_text = _call_llm(cfg.llm_backend, prompt, model=cfg.llm_model, log_dir=cfg.log_dir, tag=f"{label}_candidates")
        (cfg.log_dir / f"{label}_candidates_completion.txt").write_text(response_text)
        parsed = _parse_json(response_text)
        candidates = _sanitize_candidates(
            parsed.get("candidates") if isinstance(parsed, dict) else None,
            output_key=output_key,
            max_weight=cfg.max_weight,
            limit=cfg.candidate_count,
        )
        if not candidates:
            candidates = _fallback_candidates(section)
    candidates = candidates[:max(1, int(cfg.candidate_count))]
    (cfg.log_dir / f"{label}_candidates.json").write_text(json.dumps(candidates, indent=2) + "\n")
    score_candidate = _make_candidate_scorer(
        env=env,
        task=cfg.task,
        envs=cfg.selection_envs,
        steps=cfg.selection_steps,
        hold_seconds=cfg.success_hold_seconds,
        lift_threshold=cfg.success_lift_threshold,
        min_contacts=cfg.min_contacts,
        max_xy_disp=cfg.max_xy_disp,
    )
    scores = [
        score_candidate(list(candidate.get(output_key, [])), seed=cfg.selection_seed + idx * 997)
        | {"candidate_name": str(candidate.get("name", f"candidate_{idx}"))}
        for idx, candidate in enumerate(candidates)
    ]
    (cfg.log_dir / f"{label}_candidate_scores.json").write_text(json.dumps(scores, indent=2) + "\n")

    # Objective, deterministic selection. The metric is contact-gated (a fling with no
    # fingertip contact scores ~0) and uses only signals a real robot can measure, so the
    # same selection runs on hardware with a handful of real rollouts -- no simulator, no
    # qualitative LLM Pareto pick.
    ranking = sorted(
        ({"candidate_name": s.get("candidate_name"), "objective_score": _candidate_score(s)} for s in scores),
        key=lambda r: r["objective_score"],
        reverse=True,
    )
    selected_name = _select_by_objective(candidates, scores)
    selection_record: dict[str, Any] = {
        "selected_candidate_name": selected_name,
        "source": "objective_contact_gated",
        "metric": (
            "primary=contact_gated_success (sustained lift with >=min_contacts fingertip contacts "
            "and bounded object xy drift); anti-fling; real-world-observable signals only"
        ),
        "min_contacts": cfg.min_contacts,
        "max_xy_disp": cfg.max_xy_disp,
        "ranking": ranking,
    }
    selected = next((candidate for candidate in candidates if str(candidate.get("name")) == selected_name), candidates[0])
    rules = list(selected.get(output_key, []))
    selection_record["selected_rules"] = rules
    (cfg.log_dir / f"{label}_pareto_selection.json").write_text(json.dumps(selection_record, indent=2) + "\n")
    (cfg.log_dir / f"{label}_rules.json").write_text(json.dumps(rules, indent=2) + "\n")
    return rules


class ActionPriorCoach:
    def __init__(self, cfg: ActionPriorConfig, rules: list[dict[str, Any]]):
        self.cfg = cfg
        self.rules = rules
        self.log_dir = cfg.log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkup_index = 0
        self.current_weights = np.asarray([float(rule.get("weight", 0.0)) for rule in rules], dtype=np.float32)
        (self.log_dir / "action_prior_rules.json").write_text(json.dumps(rules, indent=2) + "\n")
        (self.log_dir / "action_prior_weights_initial.json").write_text(json.dumps(self.weights_dict(), indent=2) + "\n")

    def __call__(self, ctx: dict[str, Any]) -> list[float]:
        self.checkup_index += 1
        previous = self.current_weights.copy()
        rows = list(ctx.get("rows", []))
        latest = rows[-1] if rows else {}
        if self.checkup_index <= self.cfg.max_checkups and self.cfg.llm_backend not in {"fixture", "mock", "fake", "none"}:
            prompt = self._build_checkup_prompt(rows, latest)
            response_text = _call_llm(
                self.cfg.llm_backend,
                prompt,
                model=self.cfg.llm_model,
                log_dir=self.log_dir / "llm",
                tag=f"action_prior_checkup_{self.checkup_index:03d}",
            )
            parsed = _parse_json(response_text)
            self.current_weights = self._apply_weight_decision(parsed)
            source = "llm"
            decision = parsed
        else:
            self.current_weights = self._fallback_update(latest)
            source = "fallback"
            decision = {"reason": "backend disabled or max_checkups reached"}
        record = {
            "index": self.checkup_index,
            "iter": ctx.get("iter"),
            "elapsed_seconds": ctx.get("elapsed_seconds"),
            "source": source,
            "previous_weights": _weights_by_rule(self.rules, previous),
            "updated_weights": self.weights_dict(),
            "diagnostics": _safe_diagnostics(latest),
            "decision": decision,
        }
        (self.log_dir / f"action_prior_checkup_{self.checkup_index:03d}.json").write_text(json.dumps(record, indent=2) + "\n")
        return self.current_weights.tolist()

    def weights_dict(self) -> dict[str, float]:
        return _weights_by_rule(self.rules, self.current_weights)

    def _build_checkup_prompt(self, rows: list[dict[str, Any]], latest: dict[str, Any]) -> str:
        return (
            "You are adjusting runtime scalar weights for an already-compiled action-prior rule set.\n"
            "Changing these weights does not recompile. You cannot add/remove rules in this checkup. "
            "The goal is to improve exploration while avoiding brittle start-state assumptions and "
            "task-invalid shortcuts.\n\n"
            "Do not assume fixed positions or hidden task details. Use diagnostics to identify whether "
            "a prior appears too weak, too strong, or correlated with invalid progress. Reduce weights "
            "that correlate with failure-mode diagnostics; increase weights only when diagnostics show "
            "under-exploration of a task-valid basin. Return JSON only with fields: diagnosis, "
            "failure_modes, reward_hacking_risks, selected_weights, rejected_risks.\n"
            "selected_weights maps action prior names to nonnegative scalar weights.\n\n"
            f"Task: {self.cfg.task}\n"
            f"Arm: {self.cfg.arm}\n"
            f"Rules: {json.dumps(self.rules, indent=2)}\n"
            f"Current weights: {json.dumps(self.weights_dict(), indent=2)}\n"
            f"Pre-run reward failure-mode analysis: {json.dumps(self.cfg.pre_run_reward_analysis or {}, indent=2)}\n"
            f"Latest diagnostics: {json.dumps(_safe_diagnostics(latest), indent=2)}\n"
            f"Recent diagnostics: {json.dumps([_safe_diagnostics(row) for row in rows[-8:]], indent=2)}\n"
        )

    def _apply_weight_decision(self, parsed: dict[str, Any] | None) -> np.ndarray:
        if not isinstance(parsed, dict) or not isinstance(parsed.get("selected_weights"), dict):
            return self._fallback_update({})
        out = self.current_weights.copy()
        names = [str(rule.get("name", f"prior_{idx}")) for idx, rule in enumerate(self.rules)]
        for name, value in parsed["selected_weights"].items():
            if name not in names:
                continue
            try:
                weight = float(value)
            except (TypeError, ValueError):
                continue
            if 0.0 <= weight <= self.cfg.max_weight:
                out[names.index(name)] = weight
        return out

    def _fallback_update(self, latest: dict[str, Any]) -> np.ndarray:
        out = self.current_weights.copy()
        saturation = float(latest.get("saturation_frac") or 0.0)
        action_abs = float(latest.get("action_abs_mean") or 0.0)
        if saturation > 0.20 or action_abs > 0.85:
            out *= 0.85
        return np.clip(out, 0.0, self.cfg.max_weight).astype(np.float32)


def _fallback_action_priors(task: str) -> list[dict[str, Any]]:
    _ = task
    return [
        {"name": "weak_global_stabilize", "group": "all", "direction": "stabilize", "weight": 0.02},
    ]


def _sanitize_action_prior_rules(raw_rules: Any, max_weight: float) -> list[dict[str, Any]]:
    if not isinstance(raw_rules, list):
        return []
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for idx, raw in enumerate(raw_rules[:12]):
        if not isinstance(raw, dict):
            continue
        group = str(raw.get("group", ""))
        direction = str(raw.get("direction", ""))
        if group not in ACTION_GROUPS or direction not in PRIOR_DIRECTIONS:
            continue
        try:
            weight = float(raw.get("weight", 0.0))
        except (TypeError, ValueError):
            continue
        if weight < 0.0:
            continue
        name = str(raw.get("name") or f"action_prior_{idx}")
        if name in seen:
            name = f"{name}_{idx}"
        seen.add(name)
        out.append({
            "name": name,
            "group": group,
            "direction": direction,
            "weight": min(weight, max_weight),
        })
    return out


def _sanitize_candidates(
    raw_candidates: Any,
    *,
    output_key: str,
    max_weight: float,
    limit: int,
) -> list[dict[str, Any]]:
    if not isinstance(raw_candidates, list):
        return []
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for idx, raw in enumerate(raw_candidates[:max(1, limit) * 2]):
        if not isinstance(raw, dict):
            continue
        rules = _sanitize_action_prior_rules(raw.get(output_key), max_weight)
        if not rules:
            continue
        name = str(raw.get("name") or f"candidate_{idx}")
        if name in seen:
            name = f"{name}_{idx}"
        seen.add(name)
        out.append({
            "name": name,
            "principle": str(raw.get("principle", ""))[:4000],
            "expected_strengths": raw.get("expected_strengths", []),
            "risks": raw.get("risks", []),
            output_key: rules,
        })
        if len(out) >= max(1, limit):
            break
    return out


def _fallback_candidates(section: str) -> list[dict[str, Any]]:
    output_key = "action_priors" if section == "action_priors" else "supervised_targets"
    base_weight = 0.55 if section == "supervised_targets" else 0.38
    vertical_weight = 0.24 if section == "supervised_targets" else 0.20
    hand_weight = 0.46 if section == "supervised_targets" else 0.34
    candidates = [
        {
            "name": "balanced_relative_enclosure",
            "principle": "Use runtime-relative lateral approach, modest vertical approach, and broad hand enclosure.",
            "expected_strengths": ["simple", "runtime-relative", "strong early basin"],
            "risks": ["can push if approach dominates contact"],
            output_key: [
                {"name": "relative_lateral_approach", "group": "base_xy", "direction": "toward_object_xy", "weight": base_weight},
                {"name": "modest_descent", "group": "base_z", "direction": "lower_base", "weight": vertical_weight},
                {"name": "broad_enclosure", "group": "hand", "direction": "close_hand", "weight": hand_weight},
            ],
        },
        {
            "name": "finger_first_enclosure",
            "principle": "Bias individual digits toward contact while keeping palm/base motion weaker.",
            "expected_strengths": ["less base-driven pushing", "more fingertip-oriented"],
            "risks": ["may under-approach if relative base motion is too weak"],
            output_key: [
                {"name": "weak_relative_lateral_approach", "group": "base_xy", "direction": "toward_object_xy", "weight": base_weight * 0.70},
                {"name": "index_enclosure", "group": "index", "direction": "close_hand", "weight": hand_weight * 0.90},
                {"name": "middle_enclosure", "group": "middle", "direction": "close_hand", "weight": hand_weight * 0.90},
                {"name": "thumb_counter_enclosure", "group": "thumb", "direction": "close_hand", "weight": hand_weight * 0.75},
                {"name": "weak_descent", "group": "base_z", "direction": "lower_base", "weight": vertical_weight * 0.65},
            ],
        },
        {
            "name": "conservative_contact_basin",
            "principle": "Keep all biases weak to preserve policy expressivity while still indicating approach/contact direction.",
            "expected_strengths": ["low saturation risk", "easy to override"],
            "risks": ["may not move exploration enough early"],
            output_key: [
                {"name": "conservative_lateral_approach", "group": "base_xy", "direction": "toward_object_xy", "weight": base_weight * 0.55},
                {"name": "conservative_descent", "group": "base_z", "direction": "lower_base", "weight": vertical_weight * 0.55},
                {"name": "conservative_enclosure", "group": "hand", "direction": "close_hand", "weight": hand_weight * 0.55},
            ],
        },
        {
            "name": "opposition_grasp_basin",
            "principle": "Emphasize thumb-finger opposition more than the support fingers.",
            "expected_strengths": ["better form for precision grasp", "less whole-hand collapse"],
            "risks": ["may be too sparse for large objects"],
            output_key: [
                {"name": "relative_lateral_approach", "group": "base_xy", "direction": "toward_object_xy", "weight": base_weight * 0.85},
                {"name": "thumb_opposition", "group": "thumb", "direction": "close_hand", "weight": hand_weight * 0.95},
                {"name": "index_opposition", "group": "index", "direction": "close_hand", "weight": hand_weight * 0.85},
                {"name": "middle_support", "group": "middle", "direction": "close_hand", "weight": hand_weight * 0.65},
                {"name": "light_descent", "group": "base_z", "direction": "lower_base", "weight": vertical_weight * 0.65},
            ],
        },
        {
            "name": "minimal_prior",
            "principle": "Use only weak stabilizing contact-form hints and leave most exploration to PPO.",
            "expected_strengths": ["least intrusive", "low clipping risk"],
            "risks": ["weakest initial guidance"],
            output_key: [
                {"name": "minimal_lateral_approach", "group": "base_xy", "direction": "toward_object_xy", "weight": base_weight * 0.35},
                {"name": "minimal_hand_enclosure", "group": "hand", "direction": "close_hand", "weight": hand_weight * 0.35},
            ],
        },
    ]
    return candidates


def _make_candidate_scorer(
    *,
    env: Any,
    task: str,
    envs: int,
    steps: int | None,
    hold_seconds: float,
    lift_threshold: float,
    min_contacts: float,
    max_xy_disp: float,
) -> Any:
    _ = task
    rollout_steps = max(1, min(int(steps or env.horizon), int(env.horizon)))
    reset = jax.jit(lambda keys: jax.vmap(env.reset)(keys))
    step_fn = jax.vmap(env.step)
    action_from_obs, _info = make_rule_action_fn(env)

    def rollout(key, group_ids, direction_ids, weights):
        state = reset(jax.random.split(key, int(envs)))

        def body(carry, _):
            state = carry
            action = jax.vmap(lambda obs: action_from_obs(obs, group_ids, direction_ids, weights))(state.obs)
            nstate = step_fn(state, action)
            return nstate, (
                nstate.reward,
                nstate.metrics["success"],
                nstate.metrics["lift"],
                nstate.metrics["eval"],
                action,
            )

        _state, traj = jax.lax.scan(body, state, None, length=rollout_steps)
        return traj

    rollout_jit = jax.jit(rollout)

    hold_steps = max(1, int(round(float(hold_seconds) / max(float(env.cfg.control_dt), 1e-9))))

    def score(rules: list[dict[str, Any]], *, seed: int) -> dict[str, Any]:
        encoded = encode_rules(rules)
        reward, instant_success, lift, eval_traj, action = rollout_jit(
            jax.random.PRNGKey(seed),
            jp.asarray(encoded["group_ids"], dtype=jp.int32),
            jp.asarray(encoded["direction_ids"], dtype=jp.int32),
            jp.asarray(encoded["weights"], dtype=jp.float32),
        )
        reward.block_until_ready()
        # Per-step, per-env observable signals (shape [T, E]). All are quantities a real
        # robot can measure: fingertip contact count, object height, lateral drift, palm gap.
        palm_t = eval_traj[:, :, 0]
        contacts_t = eval_traj[:, :, 2]
        lift_t = eval_traj[:, :, 4]
        xy_t = eval_traj[:, :, 5]

        in_contact = contacts_t >= float(min_contacts)
        lifted = lift_t > float(lift_threshold)
        not_flung = xy_t <= float(max_xy_disp)
        # Contact-gated grasp-lift: lift only counts WITH contact and WITHOUT being flung.
        grasp_lift = in_contact & lifted & not_flung
        # Fling: object went up but with no fingertip contact -- the failure mode to suppress.
        fling_only = lifted & jp.logical_not(in_contact)

        contact_gated_success = sustained_bool(grasp_lift, hold_steps)
        fling_fraction = sustained_bool(fling_only, hold_steps)
        contact_engagement = jp.mean(in_contact.astype(jp.float32))
        contact_conditioned_lift = jp.mean(jp.where(in_contact & not_flung, lift_t, 0.0))
        contacts_mean = jp.mean(contacts_t)
        palm_dist_min = jp.mean(jp.min(palm_t, axis=0))

        action_abs = jp.mean(jp.abs(action))
        saturation = jp.mean((jp.abs(action) >= 0.98).astype(jp.float32))
        out = {
            "contact_gated_success": round(float(contact_gated_success), 6),
            "contact_conditioned_lift": round(float(contact_conditioned_lift), 6),
            "contact_engagement": round(float(contact_engagement), 6),
            "contacts_mean": round(float(contacts_mean), 6),
            "fling_fraction": round(float(fling_fraction), 6),
            "palm_obj_dist_min": round(float(palm_dist_min), 6),
            "instant_success": round(float((instant_success.max(axis=0) > 0.5).mean()), 6),
            "lift_max": round(float(jp.max(lift_t)), 6),
            "obj_xy_disp_max": round(float(jp.max(xy_t)), 6),
            "action_abs_mean": round(float(action_abs), 6),
            "saturation_frac": round(float(saturation), 6),
            "base_return": round(float(reward.sum(axis=0).mean()), 6),
            "rollout_envs": int(envs),
            "rollout_steps": int(rollout_steps),
            "seed": int(seed),
        }
        out["objective_score"] = round(_candidate_score(out), 6)
        return out

    return score


def _select_by_objective(candidates: list[dict[str, Any]], scores: list[dict[str, Any]]) -> str:
    if not candidates:
        return ""
    if not scores:
        return str(candidates[0].get("name", "candidate_0"))
    best = max(scores, key=_candidate_score)
    return str(best.get("candidate_name") or candidates[0].get("name", "candidate_0"))


def _candidate_score(score: dict[str, Any]) -> float:
    """Objective, contact-gated candidate score.

    Built only from real-world-observable signals so the same ranking can be reproduced
    from a few rollouts on physical hardware. Lifting without contact (flinging) is
    actively penalized rather than rewarded, which was the defect of the old scorer.
    """
    contact_gated = float(score.get("contact_gated_success", 0.0))
    contact_lift = float(score.get("contact_conditioned_lift", 0.0))
    engagement = float(score.get("contact_engagement", 0.0))
    contacts_mean = float(score.get("contacts_mean", 0.0))
    fling = float(score.get("fling_fraction", 0.0))
    palm_dist = float(score.get("palm_obj_dist_min", 0.0))
    saturation = float(score.get("saturation_frac", 0.0))
    action_abs = float(score.get("action_abs_mean", 0.0))
    return (
        120.0 * contact_gated        # full contact-grasp-lift (the real goal)
        + 30.0 * contact_lift        # partial lift, credited only while in contact
        + 6.0 * engagement           # time spent in fingertip contact
        + 1.0 * contacts_mean        # contact richness
        - 25.0 * fling               # suppress non-prehensile flinging
        - 4.0 * palm_dist            # reward getting the palm near the object
        - 6.0 * saturation           # avoid clipped/over-strong priors
        - 0.5 * max(action_abs - 0.55, 0.0)
    )


def _call_llm(backend: str, prompt: str, *, model: str | None, log_dir: Path, tag: str) -> str:
    if str(BOOTSTRAPPING) not in sys.path:
        sys.path.insert(0, str(BOOTSTRAPPING))
    from llm_backend import call_llm

    response = call_llm(backend, prompt, model=model, timeout_s=600.0, log_dir=log_dir, tag=tag)
    if not response.ok:
        return ""
    return response.text


def _parse_json(text: str) -> dict[str, Any] | None:
    candidates = [m.group(1) for m in re.finditer(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S)]
    m = re.search(r"\{.*\}", text or "", flags=re.S)
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


def _weights_by_rule(rules: list[dict[str, Any]], weights: Any) -> dict[str, float]:
    return {
        str(rule.get("name", f"prior_{idx}")): round(float(value), 6)
        for idx, (rule, value) in enumerate(zip(rules, weights))
    }


def _safe_diagnostics(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "iter": row.get("iter"),
        "elapsed_seconds": row.get("elapsed_seconds"),
        "success": row.get("success"),
        "instant_success": row.get("instant_success"),
        "lift_max": row.get("lift_max"),
        "eval_summary": row.get("eval_summary"),
        "action_abs_mean": row.get("action_abs_mean"),
        "saturation_frac": row.get("saturation_frac"),
        "action_prior_weights": row.get("action_prior_weights"),
    }

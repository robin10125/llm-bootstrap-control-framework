from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax.numpy as jp
import numpy as np

from policy_bias_lab.schema import FIELD_INDEX, validate_bias_spec

CORE_REWARD_TEMPLATE_NAMES: tuple[str, ...] = (
    "lift_basin_curriculum",
    "anti_knockaway",
    "contact_persistence",
    "lift_with_contact",
    "stable_lift_hold",
    "grasp_before_transport",
    "finger_approach",
    "closure_contact_consistency",
)
ADAPTIVE_REWARD_SLOT_COUNT = 8
ADAPTIVE_REWARD_TEMPLATE_NAMES: tuple[str, ...] = tuple(
    f"adaptive_{idx:02d}" for idx in range(ADAPTIVE_REWARD_SLOT_COUNT)
)
REWARD_TEMPLATE_NAMES: tuple[str, ...] = CORE_REWARD_TEMPLATE_NAMES + ADAPTIVE_REWARD_TEMPLATE_NAMES
CORE_REWARD_TEMPLATE_COUNT = len(CORE_REWARD_TEMPLATE_NAMES)
REWARD_TEMPLATE_COUNT = len(REWARD_TEMPLATE_NAMES)


@dataclass(frozen=True)
class CompiledBias:
    spec: dict[str, Any]
    reward_terms: tuple[dict[str, Any], ...]
    adaptive_reward_terms: tuple[dict[str, Any], ...]
    action_names: tuple[str, ...]
    base_ids: tuple[int, ...]
    hand_ids: tuple[int, ...]
    ctrl_open: jp.ndarray
    ctrl_close: jp.ndarray
    noise_scale: jp.ndarray

    def shaped_reward(self, prev_eval: jp.ndarray, eval_vec: jp.ndarray, task: str) -> jp.ndarray:
        weights = default_reward_template_weights(task)
        return self.dynamic_shaped_reward(prev_eval, eval_vec, weights, task)[0]

    def dynamic_shaped_reward(
        self,
        prev_eval: jp.ndarray,
        eval_vec: jp.ndarray,
        weights: jp.ndarray,
        task: str,
    ) -> tuple[jp.ndarray, jp.ndarray]:
        core_contrib = jp.zeros((CORE_REWARD_TEMPLATE_COUNT,), dtype=jp.float32)
        if task == "lift":
            core_contrib = _lift_template_contributions(prev_eval, eval_vec)
        adaptive_contrib = _adaptive_reward_contributions(self.adaptive_reward_terms, prev_eval, eval_vec, task)
        contrib = jp.concatenate([core_contrib, adaptive_contrib])
        if task != "lift":
            legacy_reward = jp.float32(0.0)
            for term in self.reward_terms:
                if not _term_applies_to_task(term, task):
                    continue
                gate = _term_gate(term, eval_vec, task)
                prev_phi = _potential(prev_eval, term)
                cur_phi = _potential(eval_vec, term)
                delta = jp.clip(cur_phi - prev_phi, -float(term["max_step"]), float(term["max_step"]))
                legacy_reward = legacy_reward + gate * float(term["weight"]) * delta
            contrib = contrib.at[0].set(legacy_reward)
        reward = jp.sum(jp.asarray(weights[:REWARD_TEMPLATE_COUNT]) * contrib)
        return jp.clip(reward, -0.75, 0.60), contrib

    def action_target_reward(self, obs: jp.ndarray, action: jp.ndarray, task: str) -> jp.ndarray:
        if task != "lift":
            return jp.float32(0.0)
        target = self.supervised_target(obs, task)
        # Small dense behavioral-basin reward. This does not replace the environment reward:
        # it nudges exploration toward the same gross motion basin as supervised_init.
        mse = jp.mean((jp.clip(action, -1.0, 1.0) - target) ** 2)
        return 0.12 * jp.clip(1.0 - mse, 0.0, 1.0)

    def action_prior(self, obs: jp.ndarray, task: str) -> jp.ndarray:
        return self.weighted_action_prior(obs, self.default_action_prior_weights(), task)

    def weighted_action_prior(self, obs: jp.ndarray, weights: jp.ndarray, task: str) -> jp.ndarray:
        prior = jp.zeros((len(self.action_names),), dtype=jp.float32)
        for idx, rule in enumerate(self.spec.get("action_priors", [])):
            prior = prior + _rule_vector(rule, obs, self, task, weight_override=weights[idx])
        return jp.clip(prior, -1.0, 1.0)

    def default_action_prior_weights(self) -> jp.ndarray:
        return jp.asarray([
            float(rule.get("weight", 0.0)) for rule in self.spec.get("action_priors", [])
        ], dtype=jp.float32)

    def supervised_target(self, obs: jp.ndarray, task: str) -> jp.ndarray:
        target = jp.zeros((len(self.action_names),), dtype=jp.float32)
        for rule in self.spec.get("supervised_targets", []):
            target = target + _rule_vector(rule, obs, self, task)
        return jp.clip(target, -1.0, 1.0)


def compile_bias(spec: dict[str, Any], env: Any) -> CompiledBias:
    validation = validate_bias_spec(spec)
    if not validation.ok:
        raise ValueError("; ".join(validation.errors))
    names = tuple(env.model.actuator(i).name for i in range(env.nu))
    base_ids = tuple(int(i) for i in getattr(env, "base_act_ids", ()))
    hand_ids = tuple(int(i) for i in getattr(env, "hand_act_ids", ()))
    noise = np.ones((env.action_size,), dtype=np.float32)
    for group in spec.get("exploration_groups", []):
        ids = _ids_for_group(str(group["group"]), names, base_ids, hand_ids)
        noise[list(ids)] *= float(group["scale"])
    return CompiledBias(
        spec=spec,
        reward_terms=_sanitize_reward_terms(spec.get("reward_terms", [])),
        adaptive_reward_terms=_sanitize_adaptive_reward_terms(spec.get("adaptive_reward_terms", [])),
        action_names=names,
        base_ids=base_ids,
        hand_ids=hand_ids,
        ctrl_open=jp.asarray(env.ctrl_open),
        ctrl_close=jp.asarray(env.ctrl_close),
        noise_scale=jp.asarray(np.clip(noise, 0.05, 4.0)),
    )


def default_reward_template_weights(task: str) -> jp.ndarray:
    weights = np.zeros((REWARD_TEMPLATE_COUNT,), dtype=np.float32)
    if task == "lift":
        weights[REWARD_TEMPLATE_NAMES.index("lift_basin_curriculum")] = 1.0
    return jp.asarray(weights)


def reward_template_metadata() -> list[dict[str, Any]]:
    core = [
        {
            "name": "lift_basin_curriculum",
            "description": "Sequential basin reward: align, contact, lift, then stabilize.",
            "risk": "Contains lift progress and should be treated as task-specific shaping, not held-out reward.",
        },
        {
            "name": "anti_knockaway",
            "description": "Penalize new xy displacement once interacting with or lifting the object.",
            "risk": "Too much weight can prevent legitimate transport tasks; safe for lift only.",
        },
        {
            "name": "contact_persistence",
            "description": "Reward maintaining or increasing fingertip contacts once fingers are near the object.",
            "risk": "Can overemphasize touching without useful lift if used alone.",
        },
        {
            "name": "lift_with_contact",
            "description": "Reward lift progress only when contact exists and xy displacement remains bounded.",
            "risk": "Overlaps lift reward; must stay gated and conservative.",
        },
        {
            "name": "stable_lift_hold",
            "description": "Small absolute reward for lifted, contacted, low-drift object states.",
            "risk": "Should stay small to avoid replacing the held-out env objective.",
        },
        {
            "name": "grasp_before_transport",
            "description": "Reward contact/closure progress before meaningful lift or drift starts.",
            "risk": "Can slow exploration if activated too early or too strongly.",
        },
        {
            "name": "finger_approach",
            "description": "Reward fingertip approach progress without using palm reach.",
            "risk": "May duplicate early approach pressure if basin curriculum is active.",
        },
        {
            "name": "closure_contact_consistency",
            "description": "Penalize high closure without contact, indicating empty-hand squeezing.",
            "risk": "Can discourage useful pre-shaping if too strong.",
        },
    ]
    return core + [
        {
            "name": name,
            "description": "Runtime-compiled adaptive reward-code slot emitted by the troubleshooting coach.",
            "risk": "Generated reward clauses must stay bounded, observable-only, and task-priority aligned.",
        }
        for name in ADAPTIVE_REWARD_TEMPLATE_NAMES
    ]


_DEFAULT_SCALES = {
    "palm_obj_dist": 0.12,
    "min_finger_dist": 0.08,
    "n_contacts": 3.0,
    "closure": 1.0,
    "lift": 0.05,
    "obj_xy_disp": 0.10,
}


def _sanitize_reward_terms(raw_terms: list[dict[str, Any]]) -> tuple[dict[str, Any], ...]:
    terms: list[dict[str, Any]] = []
    seen: set[tuple[str, str, tuple[str, ...]]] = set()
    for raw in raw_terms:
        if not isinstance(raw, dict):
            continue
        observable = str(raw.get("observable", ""))
        direction = str(raw.get("direction", ""))
        if observable not in FIELD_INDEX or direction not in {"minimize", "maximize"}:
            continue
        tasks = tuple(str(item) for item in raw.get("tasks", []) if isinstance(item, str))
        if not tasks:
            tasks = _default_tasks_for_term(observable, direction)
        allowed_tasks = tuple(task for task in tasks if _task_allows_reward_term(task, observable, direction))
        if not allowed_tasks:
            continue
        key = (observable, direction, tuple(sorted(allowed_tasks)))
        if key in seen:
            continue
        seen.add(key)
        scale = abs(float(raw.get("scale", _DEFAULT_SCALES[observable])) or _DEFAULT_SCALES[observable])
        weight = min(abs(float(raw.get("weight", 0.0))), 0.5)
        if weight <= 0.0:
            continue
        terms.append({
            "name": str(raw.get("name") or observable),
            "observable": observable,
            "direction": direction,
            "weight": weight,
            "scale": scale,
            "tasks": allowed_tasks,
            "max_step": min(abs(float(raw.get("max_step", 0.1)) or 0.1), 0.2),
        })
    return tuple(terms)


def _sanitize_adaptive_reward_terms(raw_terms: list[dict[str, Any]]) -> tuple[dict[str, Any], ...]:
    terms: list[dict[str, Any]] = []
    if not isinstance(raw_terms, list):
        return tuple()
    allowed_kinds = {
        "absolute_bound_penalty",
        "progress_penalty",
        "progress_reward",
        "absolute_good_reward",
    }
    for raw in raw_terms:
        if len(terms) >= ADAPTIVE_REWARD_SLOT_COUNT:
            break
        if not isinstance(raw, dict):
            continue
        observable = str(raw.get("observable", ""))
        direction = str(raw.get("direction", ""))
        kind = str(raw.get("kind", ""))
        if observable not in FIELD_INDEX or direction not in {"minimize", "maximize"} or kind not in allowed_kinds:
            continue
        tasks = tuple(str(item) for item in raw.get("tasks", []) if isinstance(item, str))
        if not tasks:
            tasks = _default_tasks_for_term(observable, direction)
        allowed_tasks = tuple(task for task in tasks if _adaptive_task_allows_reward_term(task, observable, direction))
        if not allowed_tasks:
            continue
        gates = _sanitize_gate(raw.get("gate", []))
        weight = min(max(abs(float(raw.get("weight", 0.0))), 0.0), 1.0)
        if weight <= 0.0:
            continue
        scale = abs(float(raw.get("scale", _DEFAULT_SCALES.get(observable, 0.1))) or 0.1)
        threshold = float(raw.get("threshold", _DEFAULT_SCALES.get(observable, 0.1)))
        terms.append({
            "name": str(raw.get("name") or f"adaptive_{len(terms):02d}"),
            "kind": kind,
            "observable": observable,
            "direction": direction,
            "weight": weight,
            "scale": max(scale, 1e-6),
            "threshold": threshold,
            "tasks": allowed_tasks,
            "gate": gates,
            "max_step": min(abs(float(raw.get("max_step", 0.12)) or 0.12), 0.25),
        })
    return tuple(terms)


def _sanitize_gate(raw_gate: Any) -> tuple[dict[str, Any], ...]:
    raw_items = raw_gate if isinstance(raw_gate, list) else []
    gates: list[dict[str, Any]] = []
    for raw in raw_items[:6]:
        if not isinstance(raw, dict):
            continue
        field = str(raw.get("field", ""))
        op = str(raw.get("op", ""))
        if field not in FIELD_INDEX or op not in {"<", "<=", ">", ">="}:
            continue
        try:
            value = float(raw["value"])
        except (KeyError, TypeError, ValueError):
            continue
        gates.append({"field": field, "op": op, "value": value})
    return tuple(gates)


def _default_tasks_for_term(observable: str, direction: str) -> tuple[str, ...]:
    if observable == "obj_xy_disp" and direction == "maximize":
        return ("push",)
    if observable == "obj_xy_disp" and direction == "minimize":
        return ("lift", "stabilize")
    return ("lift", "push", "stabilize")


def _task_allows_reward_term(task: str, observable: str, direction: str) -> bool:
    # The Shadow env reward already contains palm reach, gated closure, lift, and success.
    # LLM shaping is restricted to auxiliary progress signals so it does not redefine the
    # task or duplicate the held-out base reward used for comparison.
    if task == "lift":
        return (observable, direction) in {
            ("min_finger_dist", "minimize"),
            ("n_contacts", "maximize"),
            ("obj_xy_disp", "minimize"),
        }
    if task == "push":
        return (observable, direction) in {
            ("min_finger_dist", "minimize"),
            ("n_contacts", "maximize"),
        }
    if task == "stabilize":
        return (observable, direction) in {
            ("min_finger_dist", "minimize"),
            ("n_contacts", "maximize"),
            ("obj_xy_disp", "minimize"),
        }
    return False


def _adaptive_task_allows_reward_term(task: str, observable: str, direction: str) -> bool:
    # Adaptive rewards are allowed to address failure modes more directly than the static
    # template set, but only through logged eval observables. Task contradictions remain blocked.
    if task == "lift":
        return (observable, direction) in {
            ("min_finger_dist", "minimize"),
            ("n_contacts", "maximize"),
            ("obj_xy_disp", "minimize"),
            ("closure", "minimize"),
            ("lift", "maximize"),
        }
    return _task_allows_reward_term(task, observable, direction)


def _term_applies_to_task(term: dict[str, Any], task: str) -> bool:
    return task in term.get("tasks", ())


def _potential(eval_vec: jp.ndarray, term: dict[str, Any]) -> jp.ndarray:
    value = eval_vec[FIELD_INDEX[str(term["observable"])]]
    scale = float(term["scale"])
    if term["direction"] == "minimize":
        return jp.clip((scale - value) / scale, 0.0, 1.0)
    return jp.clip(value / scale, 0.0, 1.0)


def _term_gate(term: dict[str, Any], eval_vec: jp.ndarray, task: str) -> jp.ndarray:
    if task != "lift":
        return jp.float32(1.0)
    observable = str(term["observable"])
    if observable == "n_contacts":
        near = eval_vec[FIELD_INDEX["min_finger_dist"]] < 0.08
        return near.astype(jp.float32)
    if observable == "obj_xy_disp":
        interacting = (eval_vec[FIELD_INDEX["n_contacts"]] >= 1.0) | (eval_vec[FIELD_INDEX["lift"]] > 0.005)
        return interacting.astype(jp.float32)
    return jp.float32(1.0)


def _lift_basin_curriculum_reward(prev_eval: jp.ndarray, eval_vec: jp.ndarray) -> jp.ndarray:
    """Four-phase sequential lift shaping.

    The phase gates are inferred from current progress rather than stored as a global stage:
    each later phase comes in only when the prerequisite phase is mostly saturated. Previous
    phases decay but remain weakly active so the policy is still paid to preserve prerequisites.
    """
    p1 = _phase1_alignment(eval_vec)
    p2 = _phase2_grasp(eval_vec)
    p3 = _phase3_lift(eval_vec)

    g1 = 1.0
    g2 = p1
    g3 = p1 * p2
    g4 = p1 * p2 * p3
    retain = 0.20

    w1 = jp.maximum(retain, 1.0 - g2)
    w2 = g2 * jp.maximum(retain, 1.0 - g3)
    w3 = g3 * jp.maximum(retain, 1.0 - g4)
    w4 = g4

    r1 = 0.70 * _progress(prev_eval, eval_vec, "palm_obj_dist", "minimize", 0.10)
    r1 = r1 + 0.90 * _progress(prev_eval, eval_vec, "min_finger_dist", "minimize", 0.07)

    r2 = 1.00 * _progress(prev_eval, eval_vec, "n_contacts", "maximize", 3.0)
    r2 = r2 + 0.50 * _progress(prev_eval, eval_vec, "closure", "maximize", 0.8)
    r2 = r2 - 0.20 * _progress(prev_eval, eval_vec, "obj_xy_disp", "maximize", 0.05)

    r3 = 2.00 * _progress(prev_eval, eval_vec, "lift", "maximize", 0.08)
    r3 = r3 + 0.40 * _progress(prev_eval, eval_vec, "n_contacts", "maximize", 3.0)
    r3 = r3 - 0.25 * _progress(prev_eval, eval_vec, "obj_xy_disp", "maximize", 0.08)

    hold = _potential(eval_vec, {"observable": "lift", "direction": "maximize", "scale": 0.10})
    stable = _potential(eval_vec, {"observable": "obj_xy_disp", "direction": "minimize", "scale": 0.08})
    contact = _potential(eval_vec, {"observable": "n_contacts", "direction": "maximize", "scale": 3.0})
    r4 = 0.06 * hold + 0.04 * stable + 0.03 * contact

    return jp.clip(w1 * r1 + w2 * r2 + w3 * r3 + w4 * r4, -0.25, 0.35)


def _lift_template_contributions(prev_eval: jp.ndarray, eval_vec: jp.ndarray) -> jp.ndarray:
    basin = _lift_basin_curriculum_reward(prev_eval, eval_vec)
    contact_gate = _soft_saturate(eval_vec[FIELD_INDEX["n_contacts"]], low=0.5, high=2.0)
    near_gate = 1.0 - _soft_threshold(eval_vec[FIELD_INDEX["min_finger_dist"]], good=0.045, bad=0.10)
    stable_gate = _potential(eval_vec, {"observable": "obj_xy_disp", "direction": "minimize", "scale": 0.08})
    interacting = jp.maximum(contact_gate, _soft_saturate(eval_vec[FIELD_INDEX["lift"]], low=0.005, high=0.02))

    anti_knock = -0.16 * interacting * _progress(prev_eval, eval_vec, "obj_xy_disp", "maximize", 0.08)
    contact_persist = 0.10 * near_gate * _progress(prev_eval, eval_vec, "n_contacts", "maximize", 3.0)
    lift_contact = 0.18 * contact_gate * stable_gate * _progress(prev_eval, eval_vec, "lift", "maximize", 0.08)

    held_lift = _soft_saturate(eval_vec[FIELD_INDEX["lift"]], low=0.04, high=0.10)
    stable_hold = 0.04 * held_lift * contact_gate * stable_gate

    pre_transport = 1.0 - _soft_saturate(eval_vec[FIELD_INDEX["lift"]], low=0.005, high=0.03)
    pre_transport = pre_transport * _potential(eval_vec, {"observable": "obj_xy_disp", "direction": "minimize", "scale": 0.06})
    grasp_before_transport = pre_transport * (
        0.07 * _progress(prev_eval, eval_vec, "n_contacts", "maximize", 3.0)
        + 0.04 * _progress(prev_eval, eval_vec, "closure", "maximize", 0.7)
    )

    finger_approach = 0.08 * _progress(prev_eval, eval_vec, "min_finger_dist", "minimize", 0.08)
    empty_squeeze = jp.maximum(eval_vec[FIELD_INDEX["closure"]] - 0.55, 0.0) * (1.0 - contact_gate)
    closure_contact_consistency = -0.06 * empty_squeeze

    return jp.asarray([
        basin,
        anti_knock,
        contact_persist,
        lift_contact,
        stable_hold,
        grasp_before_transport,
        finger_approach,
        closure_contact_consistency,
    ], dtype=jp.float32)


def _adaptive_reward_contributions(
    terms: tuple[dict[str, Any], ...],
    prev_eval: jp.ndarray,
    eval_vec: jp.ndarray,
    task: str,
) -> jp.ndarray:
    values: list[jp.ndarray] = []
    for term in terms[:ADAPTIVE_REWARD_SLOT_COUNT]:
        if not _term_applies_to_task(term, task):
            values.append(jp.float32(0.0))
        else:
            values.append(_adaptive_reward_contribution(term, prev_eval, eval_vec))
    while len(values) < ADAPTIVE_REWARD_SLOT_COUNT:
        values.append(jp.float32(0.0))
    return jp.asarray(values, dtype=jp.float32)


def _adaptive_reward_contribution(term: dict[str, Any], prev_eval: jp.ndarray, eval_vec: jp.ndarray) -> jp.ndarray:
    gate = _adaptive_gate(term, eval_vec)
    value = eval_vec[FIELD_INDEX[str(term["observable"])]]
    threshold = float(term["threshold"])
    scale = float(term["scale"])
    weight = float(term["weight"])
    direction = str(term["direction"])
    kind = str(term["kind"])

    if kind == "absolute_bound_penalty":
        if direction == "minimize":
            magnitude = jp.clip((value - threshold) / scale, 0.0, 1.0)
        else:
            magnitude = jp.clip((threshold - value) / scale, 0.0, 1.0)
        return -weight * gate * magnitude

    if kind == "absolute_good_reward":
        if direction == "minimize":
            magnitude = jp.clip((threshold - value) / scale, 0.0, 1.0)
        else:
            magnitude = jp.clip((value - threshold) / scale, 0.0, 1.0)
        return weight * gate * magnitude

    if kind == "progress_penalty":
        opposite = "maximize" if direction == "minimize" else "minimize"
        worsening = jp.clip(_progress(prev_eval, eval_vec, str(term["observable"]), opposite, scale), 0.0, float(term["max_step"]))
        return -weight * gate * worsening

    if kind == "progress_reward":
        progress = jp.clip(_progress(prev_eval, eval_vec, str(term["observable"]), direction, scale), 0.0, float(term["max_step"]))
        return weight * gate * progress

    return jp.float32(0.0)


def _adaptive_gate(term: dict[str, Any], eval_vec: jp.ndarray) -> jp.ndarray:
    gate = jp.float32(1.0)
    for cond in term.get("gate", ()):
        value = eval_vec[FIELD_INDEX[str(cond["field"])]]
        threshold = float(cond["value"])
        op = str(cond["op"])
        if op == "<":
            ok = value < threshold
        elif op == "<=":
            ok = value <= threshold
        elif op == ">":
            ok = value > threshold
        else:
            ok = value >= threshold
        gate = gate * ok.astype(jp.float32)
    return gate


def _phase1_alignment(eval_vec: jp.ndarray) -> jp.ndarray:
    palm = 1.0 - _soft_threshold(eval_vec[FIELD_INDEX["palm_obj_dist"]], good=0.07, bad=0.14)
    finger = 1.0 - _soft_threshold(eval_vec[FIELD_INDEX["min_finger_dist"]], good=0.045, bad=0.10)
    return jp.minimum(palm, finger)


def _phase2_grasp(eval_vec: jp.ndarray) -> jp.ndarray:
    contacts = _soft_saturate(eval_vec[FIELD_INDEX["n_contacts"]], low=0.5, high=2.5)
    closure = _soft_saturate(eval_vec[FIELD_INDEX["closure"]], low=0.20, high=0.70)
    return jp.minimum(contacts, closure)


def _phase3_lift(eval_vec: jp.ndarray) -> jp.ndarray:
    return _soft_saturate(eval_vec[FIELD_INDEX["lift"]], low=0.015, high=0.055)


def _soft_threshold(value: jp.ndarray, *, good: float, bad: float) -> jp.ndarray:
    return jp.clip((value - good) / (bad - good), 0.0, 1.0)


def _soft_saturate(value: jp.ndarray, *, low: float, high: float) -> jp.ndarray:
    return jp.clip((value - low) / (high - low), 0.0, 1.0)


def _progress(prev_eval: jp.ndarray, eval_vec: jp.ndarray, observable: str, direction: str, scale: float) -> jp.ndarray:
    term = {"observable": observable, "direction": direction, "scale": scale}
    return jp.clip(_potential(eval_vec, term) - _potential(prev_eval, term), -0.10, 0.10)


def _rule_vector(
    rule: dict[str, Any],
    obs: jp.ndarray,
    bias: CompiledBias,
    task: str,
    weight_override: jp.ndarray | float | None = None,
) -> jp.ndarray:
    names = bias.action_names
    ids = _ids_for_group(str(rule["group"]), names, bias.base_ids, bias.hand_ids)
    direction = str(rule["direction"])
    weight = jp.asarray(float(rule["weight"]) if weight_override is None else weight_override, dtype=jp.float32)
    out = jp.zeros((len(names),), dtype=jp.float32)
    if direction == "toward_object_xy":
        obj_rel = _obj_rel(obs, len(names))
        if "base_x" in names:
            out = out.at[names.index("base_x")].set(jp.clip(obj_rel[0] * 8.0, -1.0, 1.0) * weight)
        if "base_y" in names:
            out = out.at[names.index("base_y")].set(jp.clip(obj_rel[1] * 8.0, -1.0, 1.0) * weight)
        return out
    if direction == "lower_base":
        return _set_ids(out, ids, -abs(weight))
    if direction == "raise_base":
        return _set_ids(out, ids, abs(weight))
    if direction == "close_hand":
        sign = jp.sign(bias.ctrl_close - bias.ctrl_open)
        return _set_ids(out, ids, weight) * sign
    if direction == "open_hand":
        sign = jp.sign(bias.ctrl_open - bias.ctrl_close)
        return _set_ids(out, ids, weight) * sign
    if direction == "stabilize":
        return _set_ids(out, ids, 0.0)
    return out


def _ids_for_group(group: str, names: tuple[str, ...], base_ids: tuple[int, ...], hand_ids: tuple[int, ...]) -> tuple[int, ...]:
    if group == "all":
        return tuple(range(len(names)))
    if group == "base_xy":
        return tuple(i for i, name in enumerate(names) if name in {"base_x", "base_y"})
    if group == "base_z":
        return tuple(i for i, name in enumerate(names) if name == "base_z")
    if group == "hand":
        return hand_ids
    prefixes = {
        "thumb": "rh_A_TH",
        "index": "rh_A_FF",
        "middle": "rh_A_MF",
        "ring": "rh_A_RF",
        "little": "rh_A_LF",
    }
    prefix = prefixes.get(group)
    if prefix:
        return tuple(i for i, name in enumerate(names) if name.startswith(prefix))
    return base_ids


def _set_ids(vector: jp.ndarray, ids: tuple[int, ...], value: float) -> jp.ndarray:
    if not ids:
        return vector
    return vector.at[jp.asarray(ids)].set(jp.asarray(value, dtype=jp.float32))


def _obj_rel(obs: jp.ndarray, action_dim: int) -> jp.ndarray:
    # mjx_env observation layout ends with ctrl[action_dim]; obj_rel is immediately before ctrl.
    rel_start = obs.shape[-1] - action_dim - 3
    return obs[rel_start: rel_start + 3]

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax.numpy as jp
import numpy as np

from policy_bias_lab.schema import FIELD_INDEX, validate_bias_spec


@dataclass(frozen=True)
class CompiledBias:
    spec: dict[str, Any]
    reward_terms: tuple[dict[str, Any], ...]
    action_names: tuple[str, ...]
    base_ids: tuple[int, ...]
    hand_ids: tuple[int, ...]
    ctrl_open: jp.ndarray
    ctrl_close: jp.ndarray
    noise_scale: jp.ndarray

    def shaped_reward(self, prev_eval: jp.ndarray, eval_vec: jp.ndarray, task: str) -> jp.ndarray:
        if task == "lift":
            return _lift_basin_curriculum_reward(prev_eval, eval_vec)
        reward = jp.float32(0.0)
        for term in self.reward_terms:
            if not _term_applies_to_task(term, task):
                continue
            gate = _term_gate(term, eval_vec, task)
            prev_phi = _potential(prev_eval, term)
            cur_phi = _potential(eval_vec, term)
            delta = jp.clip(cur_phi - prev_phi, -float(term["max_step"]), float(term["max_step"]))
            reward = reward + gate * float(term["weight"]) * delta
        return reward

    def action_target_reward(self, obs: jp.ndarray, action: jp.ndarray, task: str) -> jp.ndarray:
        if task != "lift":
            return jp.float32(0.0)
        target = self.supervised_target(obs, task)
        # Small dense behavioral-basin reward. This does not replace the environment reward:
        # it nudges exploration toward the same gross motion basin as supervised_init.
        mse = jp.mean((jp.clip(action, -1.0, 1.0) - target) ** 2)
        return 0.12 * jp.clip(1.0 - mse, 0.0, 1.0)

    def action_prior(self, obs: jp.ndarray, task: str) -> jp.ndarray:
        prior = jp.zeros((len(self.action_names),), dtype=jp.float32)
        for rule in self.spec.get("action_priors", []):
            prior = prior + _rule_vector(rule, obs, self, task)
        return jp.clip(prior, -1.0, 1.0)

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
        action_names=names,
        base_ids=base_ids,
        hand_ids=hand_ids,
        ctrl_open=jp.asarray(env.ctrl_open),
        ctrl_close=jp.asarray(env.ctrl_close),
        noise_scale=jp.asarray(np.clip(noise, 0.05, 4.0)),
    )


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


def _rule_vector(rule: dict[str, Any], obs: jp.ndarray, bias: CompiledBias, task: str) -> jp.ndarray:
    names = bias.action_names
    ids = _ids_for_group(str(rule["group"]), names, bias.base_ids, bias.hand_ids)
    direction = str(rule["direction"])
    weight = float(rule["weight"])
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
    return vector.at[jp.asarray(ids)].set(jp.float32(value))


def _obj_rel(obs: jp.ndarray, action_dim: int) -> jp.ndarray:
    # mjx_env observation layout ends with ctrl[action_dim]; obj_rel is immediately before ctrl.
    rel_start = obs.shape[-1] - action_dim - 3
    return obs[rel_start: rel_start + 3]

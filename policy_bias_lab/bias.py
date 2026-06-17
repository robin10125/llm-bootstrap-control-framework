from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax.numpy as jp
import numpy as np

from policy_bias_lab.schema import FIELD_INDEX, validate_bias_spec


@dataclass(frozen=True)
class CompiledBias:
    spec: dict[str, Any]
    action_names: tuple[str, ...]
    base_ids: tuple[int, ...]
    hand_ids: tuple[int, ...]
    ctrl_open: jp.ndarray
    ctrl_close: jp.ndarray
    noise_scale: jp.ndarray

    def shaped_reward(self, eval_vec: jp.ndarray, task: str) -> jp.ndarray:
        reward = jp.float32(0.0)
        for term in self.spec.get("reward_terms", []):
            idx = FIELD_INDEX[str(term["observable"])]
            value = eval_vec[idx]
            scale = float(term.get("scale", 1.0)) or 1.0
            weight = float(term["weight"])
            normalized = value / scale
            if term["direction"] == "minimize":
                reward = reward - weight * normalized
            else:
                reward = reward + weight * normalized
        if task == "push":
            reward = reward + 0.8 * eval_vec[FIELD_INDEX["obj_xy_disp"]]
        if task == "stabilize":
            reward = reward - 1.0 * eval_vec[FIELD_INDEX["obj_xy_disp"]]
        return reward

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
        action_names=names,
        base_ids=base_ids,
        hand_ids=hand_ids,
        ctrl_open=jp.asarray(env.ctrl_open),
        ctrl_close=jp.asarray(env.ctrl_close),
        noise_scale=jp.asarray(np.clip(noise, 0.05, 4.0)),
    )


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

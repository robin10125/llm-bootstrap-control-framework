from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jp
import numpy as np


Params = dict[str, jp.ndarray]


def init_params(key: jp.ndarray, obs_dim: int, action_dim: int, hidden: int = 128) -> Params:
    k1, k2 = jax.random.split(key)
    return {
        "w1": jax.random.normal(k1, (obs_dim, hidden)) * jp.sqrt(2.0 / max(obs_dim, 1)),
        "b1": jp.zeros((hidden,)),
        "w2": jax.random.normal(k2, (hidden, action_dim)) * 0.01,
        "b2": jp.zeros((action_dim,)),
    }


def apply_policy(params: Params, obs: jp.ndarray, action_prior: jp.ndarray | None = None) -> jp.ndarray:
    x = jp.tanh(obs @ params["w1"] + params["b1"])
    mean = x @ params["w2"] + params["b2"]
    if action_prior is not None:
        mean = mean + action_prior
    return jp.tanh(mean)


def flatten_params(params: Params) -> tuple[jp.ndarray, dict[str, tuple[int, ...]]]:
    shapes = {name: tuple(value.shape) for name, value in params.items()}
    flat = jp.concatenate([jp.ravel(params[name]) for name in ("w1", "b1", "w2", "b2")])
    return flat, shapes


def unflatten_params(flat: jp.ndarray, shapes: dict[str, tuple[int, ...]]) -> Params:
    out: dict[str, jp.ndarray] = {}
    pos = 0
    for name in ("w1", "b1", "w2", "b2"):
        size = int(np.prod(shapes[name]))
        out[name] = jp.reshape(flat[pos: pos + size], shapes[name])
        pos += size
    return out


def numpy_params(params: Params) -> dict[str, Any]:
    return {name: np.asarray(value).tolist() for name, value in params.items()}

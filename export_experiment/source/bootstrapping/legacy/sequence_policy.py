#!/usr/bin/env python3
"""Variable-length primitive-sequence policies.

The policy is a sequence of up to MAX_STEPS tokens. Each token selects a primitive
(discrete head), its parameters and duration (continuous heads), whether to attach a
contact `until` condition, and whether to stop. Unlike the fixed schedule template,
nothing about step order, count, or primitive choice is baked in: search and the neural
policy must discover structure, and LLM-authored schedules keep their structure when
encoded.

Two consumers:

- CEM/random search operates on a flat latent of MAX_STEPS * TOKEN_DIM values
  (`flat_to_schedule` / `schedule_to_flat`).
- `SeqPolicyNet` is an autoregressive NumPy MLP trained by behavior cloning on token
  sequences (`schedule_to_tokens`), generating schedules token by token.

Token layout (TOKEN_DIM = 12):

    [0]     stop logit (sigmoid > 0.5 ends the schedule before this step)
    [1:7]   primitive logits: set_base, move_delta, approach_object, hand_pose, grasp, wait
    [7:10]  p0, p1, p2 raw params (tanh-squashed, meaning depends on primitive)
    [10]    duration raw
    [11]    until logit (sigmoid > 0.5 attaches `until: contacts >= 2`)
"""
from __future__ import annotations

from typing import Any

import numpy as np

from tasks import OBS_DIM


PRIMITIVES = ["set_base", "move_delta", "approach_object", "hand_pose", "grasp", "wait"]
POSE_NAMES = [
    "open hand", "pre grasp", "grasp soft", "grasp hard",
    "grasp sphere", "two finger pinch", "three finger pinch", "close hand",
]
TOKEN_DIM = 12
MAX_STEPS = 10
FLAT_DIM = MAX_STEPS * TOKEN_DIM

UNTIL_CONTACT = [{"metric": "contacts.hand_object_count", "op": ">=", "value": 2}]

_STOP = 0
_PRIM = slice(1, 7)
_P = slice(7, 10)
_DUR = 10
_UNTIL = 11


def _squash(raw: float) -> float:
    return float(np.tanh(raw))


def _unsquash(value: float) -> float:
    return float(np.arctanh(np.clip(value, -0.98, 0.98)))


def token_to_step(token: np.ndarray) -> dict[str, Any] | None:
    """Decode one token into a primitive step; None means stop."""
    if 1.0 / (1.0 + np.exp(-token[_STOP])) > 0.5:
        return None
    prim = PRIMITIVES[int(np.argmax(token[_PRIM]))]
    t0, t1, t2 = (_squash(v) for v in token[_P])
    duration = round(float(np.clip(1.55 + 1.45 * _squash(token[_DUR]), 0.1, 3.0)), 3)

    if prim == "set_base":
        params = {"x": round(0.15 * t0, 4), "y": round(0.15 * t1, 4), "z": round(0.02 + 0.17 * t2, 4)}
    elif prim == "move_delta":
        params = {"dx": round(0.12 * t0, 4), "dy": round(0.12 * t1, 4), "dz": round(0.12 * t2, 4)}
    elif prim == "approach_object":
        params = {"ox": round(-0.035 + 0.05 * t0, 4), "oy": round(0.05 * t1, 4), "z": round(0.11 + 0.08 * t2, 4)}
    elif prim == "hand_pose":
        idx = int(np.clip(round((t0 + 1.0) / 2.0 * (len(POSE_NAMES) - 1)), 0, len(POSE_NAMES) - 1))
        params = {"name": POSE_NAMES[idx]}
    elif prim == "grasp":
        params = {"closure": round(float(np.clip((t0 + 1.0) / 2.0, 0.0, 1.0)), 4)}
    else:
        params = {}

    step: dict[str, Any] = {"primitive": prim, "params": params, "duration_s": duration}
    if 1.0 / (1.0 + np.exp(-token[_UNTIL])) > 0.5:
        step["until"] = list(UNTIL_CONTACT)
    return step


def step_to_token(step: dict[str, Any]) -> np.ndarray:
    token = np.zeros(TOKEN_DIM)
    token[_STOP] = -3.0
    prim = step["primitive"]
    if prim not in PRIMITIVES:
        raise ValueError(f"primitive {prim!r} not in sequence vocabulary")
    logits = np.full(len(PRIMITIVES), -2.0)
    logits[PRIMITIVES.index(prim)] = 2.0
    token[_PRIM] = logits

    params = step.get("params", {})
    if prim == "set_base":
        token[7] = _unsquash(float(params["x"]) / 0.15)
        token[8] = _unsquash(float(params["y"]) / 0.15)
        token[9] = _unsquash((float(params["z"]) - 0.02) / 0.17)
    elif prim == "move_delta":
        token[7] = _unsquash(float(params.get("dx", 0.0)) / 0.12)
        token[8] = _unsquash(float(params.get("dy", 0.0)) / 0.12)
        token[9] = _unsquash(float(params.get("dz", 0.0)) / 0.12)
    elif prim == "approach_object":
        token[7] = _unsquash((float(params.get("ox", 0.0)) + 0.035) / 0.05)
        token[8] = _unsquash(float(params.get("oy", 0.0)) / 0.05)
        token[9] = _unsquash((float(params["z"]) - 0.11) / 0.08)
    elif prim == "hand_pose":
        idx = POSE_NAMES.index(params["name"])
        token[7] = _unsquash(idx / (len(POSE_NAMES) - 1) * 2.0 - 1.0)
    elif prim == "grasp":
        token[7] = _unsquash(float(params["closure"]) * 2.0 - 1.0)

    token[_DUR] = _unsquash((float(step["duration_s"]) - 1.55) / 1.45)
    token[_UNTIL] = 2.0 if step.get("until") else -2.0
    return token


STOP_TOKEN = np.zeros(TOKEN_DIM)
STOP_TOKEN[_STOP] = 3.0


def schedule_to_tokens(policy: dict[str, Any]) -> np.ndarray:
    """Encode a policy's steps as a (T+1, TOKEN_DIM) token array ending in a stop row."""
    rows = [step_to_token(step) for step in policy["steps"][:MAX_STEPS]]
    if len(rows) < MAX_STEPS:
        rows.append(STOP_TOKEN.copy())
    return np.vstack(rows)


def tokens_to_schedule(
    tokens: np.ndarray,
    setup: dict[str, Any],
    *,
    name: str,
    goal: str,
    success: list[dict[str, Any]],
) -> dict[str, Any]:
    steps = []
    for row in tokens[:MAX_STEPS]:
        step = token_to_step(np.asarray(row, dtype=float))
        if step is None:
            break
        steps.append(step)
    if not steps:
        steps = [{"primitive": "wait", "params": {}, "duration_s": 0.5}]
    return {"name": name, "goal": goal, "setup": setup, "steps": steps, "success": success}


def flat_to_schedule(vec: np.ndarray, setup: dict[str, Any], *, name: str, goal: str, success: list[dict[str, Any]]) -> dict[str, Any]:
    return tokens_to_schedule(vec.reshape(MAX_STEPS, TOKEN_DIM), setup, name=name, goal=goal, success=success)


def schedule_to_flat(policy: dict[str, Any]) -> np.ndarray:
    flat = np.zeros((MAX_STEPS, TOKEN_DIM))
    tokens = schedule_to_tokens(policy)
    flat[: len(tokens)] = tokens
    if len(tokens) < MAX_STEPS:
        flat[len(tokens):] = STOP_TOKEN
    return flat.reshape(-1)


# Per-primitive masks for which continuous param dims carry meaning (BC loss masking).
_PARAM_MASK = {
    "set_base": (1.0, 1.0, 1.0),
    "move_delta": (1.0, 1.0, 1.0),
    "approach_object": (1.0, 1.0, 1.0),
    "hand_pose": (1.0, 0.0, 0.0),
    "grasp": (1.0, 0.0, 0.0),
    "wait": (0.0, 0.0, 0.0),
}


class SeqPolicyNet:
    """Autoregressive MLP over primitive tokens, trained by behavior cloning.

    Input per step: [obs, previous token, step_index / MAX_STEPS]; output: next token.
    Teacher forcing during training; greedy (optionally noisy) decoding at rollout.
    """

    def __init__(self, seed: int, obs_dim: int = OBS_DIM, hidden: int = 64):
        rng = np.random.default_rng(seed)
        self.obs_dim = obs_dim
        self.in_dim = obs_dim + TOKEN_DIM + 1
        self.hidden = hidden
        self.w1 = rng.normal(0.0, 0.12, (self.in_dim, hidden))
        self.b1 = np.zeros(hidden)
        self.w2 = rng.normal(0.0, 0.12, (hidden, TOKEN_DIM))
        self.b2 = np.zeros(TOKEN_DIM)

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h = np.tanh(x @ self.w1 + self.b1)
        return h @ self.w2 + self.b2, h

    @staticmethod
    def _rows_from_sequence(obs: np.ndarray, tokens: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        xs, ys = [], []
        prev = np.zeros(TOKEN_DIM)
        for t, token in enumerate(tokens):
            xs.append(np.concatenate([obs, prev, [t / MAX_STEPS]]))
            ys.append(token)
            prev = token
        return np.vstack(xs), np.vstack(ys)

    def train_supervised(
        self,
        sequences: list[tuple[np.ndarray, np.ndarray]],
        *,
        epochs: int,
        lr: float,
    ) -> list[float]:
        """sequences: list of (obs, tokens) pairs; tokens include the trailing stop row."""
        if not sequences:
            return []
        xs_list, ys_list = zip(*[self._rows_from_sequence(obs, tokens) for obs, tokens in sequences])
        xs = np.vstack(xs_list)
        ys = np.vstack(ys_list)
        n = len(xs)

        stop_target = (ys[:, _STOP] > 0).astype(float)
        prim_target = np.argmax(ys[:, _PRIM], axis=1)
        not_stop = 1.0 - stop_target
        cont_mask = np.zeros((n, 3))
        for i in range(n):
            if not_stop[i]:
                cont_mask[i] = _PARAM_MASK[PRIMITIVES[prim_target[i]]]

        losses = []
        for _ in range(epochs):
            y, h = self.forward(xs)

            stop_p = 1.0 / (1.0 + np.exp(-y[:, _STOP]))
            bce = -np.mean(stop_target * np.log(stop_p + 1e-9) + (1 - stop_target) * np.log(1 - stop_p + 1e-9))

            logits = y[:, _PRIM]
            logits = logits - logits.max(axis=1, keepdims=True)
            soft = np.exp(logits)
            soft /= soft.sum(axis=1, keepdims=True)
            ce = -np.mean(not_stop * np.log(soft[np.arange(n), prim_target] + 1e-9))

            cont_diff = (y[:, _P] - ys[:, _P]) * cont_mask
            dur_diff = (y[:, _DUR] - ys[:, _DUR]) * not_stop
            until_diff = (y[:, _UNTIL] - ys[:, _UNTIL]) * not_stop
            mse = float(np.mean(cont_diff**2) + np.mean(dur_diff**2) + 0.25 * np.mean(until_diff**2))

            losses.append(float(bce + ce + 0.5 * mse))

            grad_y = np.zeros_like(y)
            grad_y[:, _STOP] = (stop_p - stop_target) / n
            onehot = np.zeros_like(soft)
            onehot[np.arange(n), prim_target] = 1.0
            grad_y[:, _PRIM] = (soft - onehot) * not_stop[:, None] / n
            grad_y[:, _P] = 0.5 * 2.0 * cont_diff / (n * 3)
            grad_y[:, _DUR] = 0.5 * 2.0 * dur_diff / n
            grad_y[:, _UNTIL] = 0.5 * 0.25 * 2.0 * until_diff / n

            grad_w2 = h.T @ grad_y
            grad_b2 = grad_y.sum(axis=0)
            grad_h = grad_y @ self.w2.T
            grad_z1 = grad_h * (1.0 - h * h)
            grad_w1 = xs.T @ grad_z1
            grad_b1 = grad_z1.sum(axis=0)

            self.w2 -= lr * grad_w2
            self.b2 -= lr * grad_b2
            self.w1 -= lr * grad_w1
            self.b1 -= lr * grad_b1
        return losses

    def generate_tokens(self, obs: np.ndarray, *, noise: float, rng: np.random.Generator) -> np.ndarray:
        rows = []
        prev = np.zeros(TOKEN_DIM)
        for t in range(MAX_STEPS):
            x = np.concatenate([obs, prev, [t / MAX_STEPS]])
            y, _ = self.forward(x[None, :])
            token = y[0].copy()
            if noise > 0:
                token[_PRIM] = token[_PRIM] + rng.normal(0.0, noise * 1.5, len(PRIMITIVES))
                token[_P] = token[_P] + rng.normal(0.0, noise, 3)
                token[_DUR] += rng.normal(0.0, noise)
                token[_UNTIL] += rng.normal(0.0, noise)
                token[_STOP] += rng.normal(0.0, noise * 0.5)
            rows.append(token)
            if 1.0 / (1.0 + np.exp(-token[_STOP])) > 0.5:
                break
            # Feed back the canonicalized token (re-encoded from the decoded step) so
            # autoregressive inputs match the clean target tokens seen in training.
            step = token_to_step(token)
            prev = step_to_token(step) if step is not None else token
        return np.vstack(rows)

    def save(self, path) -> None:
        np.savez(path, w1=self.w1, b1=self.b1, w2=self.w2, b2=self.b2)

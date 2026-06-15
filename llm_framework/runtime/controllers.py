from __future__ import annotations

from typing import Any

import numpy as np


def actuator_index(env: Any) -> dict[str, int]:
    return {env.model.actuator(i).name: i for i in range(env.nu)}


def ctrl_open(env: Any) -> np.ndarray:
    return np.asarray(_device_get(env.ctrl_open), dtype=np.float32)


def ctrl_close(env: Any) -> np.ndarray:
    return np.asarray(_device_get(env.ctrl_close), dtype=np.float32)


def ctrl_limits(env: Any) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray(env.model.actuator_ctrlrange[:, 0], dtype=np.float32),
        np.asarray(env.model.actuator_ctrlrange[:, 1], dtype=np.float32),
    )


def hand_shape_target(env: Any, fraction_closed: float) -> np.ndarray:
    fraction_closed = float(np.clip(fraction_closed, 0.0, 1.0))
    return (1.0 - fraction_closed) * ctrl_open(env) + fraction_closed * ctrl_close(env)


def targets_to_actions(env: Any, targets: np.ndarray, initial_ctrl: np.ndarray | None = None) -> np.ndarray:
    lo, hi = ctrl_limits(env)
    scale = float(env.cfg.action_scale)
    cur = ctrl_open(env).copy() if initial_ctrl is None else np.asarray(initial_ctrl, dtype=np.float32).copy()
    actions = np.zeros_like(targets, dtype=np.float32)
    for t, target in enumerate(np.asarray(targets, dtype=np.float32)):
        target = np.clip(target, lo, hi)
        action = np.clip((target - cur) / scale, -1.0, 1.0)
        actions[t] = action
        cur = np.clip(cur + action * scale, lo, hi)
    return actions


def repeat_target(current: np.ndarray, horizon_steps: int) -> np.ndarray:
    return np.repeat(current[None, :], max(0, int(horizon_steps)), axis=0).astype(np.float32)


def duration_to_steps(env: Any, duration_s: float) -> int:
    return max(1, int(round(float(duration_s) / float(env.cfg.control_dt))))


def latent_decode_target(env: Any, token: str, gain: float, current: np.ndarray, world_object_xy: tuple[float, float]) -> np.ndarray:
    target = current.copy()
    idx = actuator_index(env)
    gain = float(np.clip(gain, 0.0, 1.0))
    if token in {"oppose_and_stabilize", "close_around_object"}:
        target = (1.0 - gain) * target + gain * hand_shape_target(env, 0.75)
    elif token in {"open_and_clear", "release"}:
        target = (1.0 - gain) * target + gain * ctrl_open(env)
    elif token in {"approach_object", "center_over_object"}:
        if {"base_x", "base_y"} <= idx.keys():
            target[idx["base_x"]] = world_object_xy[0]
            target[idx["base_y"]] = world_object_xy[1]
    elif token in {"raise", "stabilize_height"} and "base_z" in idx:
        # In the Shadow/gripper scene, lower base_z is a higher palm.
        target[idx["base_z"]] = min(target[idx["base_z"]], 0.0)
    return np.clip(target, *ctrl_limits(env))


def _device_get(value: Any) -> Any:
    try:
        import jax

        return jax.device_get(value)
    except Exception:
        return value


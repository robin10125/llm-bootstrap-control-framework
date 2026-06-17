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
        # In the Shadow scene, higher base_z raises/retracts the palm.
        target[idx["base_z"]] = max(target[idx["base_z"]], 0.10)
    return np.clip(target, *ctrl_limits(env))


def frame_target_to_base(env: Any, world: Any, current: np.ndarray, block: dict[str, Any]) -> np.ndarray:
    """Convert an actual world-frame palm/grasp target into raw base actuator targets."""
    idx = actuator_index(env)
    target = np.asarray(current, dtype=np.float32).copy()
    if not {"base_x", "base_y", "base_z"} <= idx.keys():
        return target

    desired = _desired_world_position(world, block)
    if desired is None:
        return target

    frame = str(block.get("frame", block.get("target_frame", "grasp_site"))).lower()
    frame_pos = _frame_position(world, frame)
    if frame_pos is None:
        return target

    base_origin = np.asarray([world.base_q[0], world.base_q[1], world.base_q[2]], dtype=float)
    frame_offset = frame_pos - base_origin
    base_target = desired - frame_offset
    for axis, name in enumerate(("base_x", "base_y", "base_z")):
        target[idx[name]] = float(base_target[axis])
    return target


def _desired_world_position(world: Any, block: dict[str, Any]) -> np.ndarray | None:
    raw = block.get("target", {})
    if isinstance(raw, (list, tuple, np.ndarray)):
        pos = np.asarray(raw, dtype=float)
        if len(pos) >= 3:
            return pos[:3]
    if not isinstance(raw, dict):
        raw = {}
    if raw.get("object") == "object" or block.get("object") == "object":
        offset = np.asarray(raw.get("offset", block.get("offset", [0.0, 0.0, 0.0])), dtype=float)
        if offset.shape[0] < 3:
            offset = np.pad(offset, (0, 3 - offset.shape[0]))
        return np.asarray(world.object_pos, dtype=float) + offset[:3]
    if "pos" in raw:
        pos = np.asarray(raw["pos"], dtype=float)
    elif "position" in raw:
        pos = np.asarray(raw["position"], dtype=float)
    else:
        pos = np.asarray([
            raw.get("x", block.get("x")),
            raw.get("y", block.get("y")),
            raw.get("z", block.get("z")),
        ], dtype=object)
    if len(pos) < 3 or any(v is None for v in pos[:3]):
        return None
    return np.asarray(pos[:3], dtype=float)


def _frame_position(world: Any, frame: str) -> np.ndarray | None:
    if frame in {"base", "base_q", "base_origin", "raw_base"}:
        return np.asarray(world.base_q[:3], dtype=float)
    derived = getattr(world, "derived", {}) or {}
    if frame in {"palm", "palm_body", "rh_palm"} and "palm_pos" in derived:
        return np.asarray(derived["palm_pos"], dtype=float)
    if frame in {"grasp", "grasp_site", "grasp_frame"} and "grasp_site_pos" in derived:
        return np.asarray(derived["grasp_site_pos"], dtype=float)
    return None


def _device_get(value: Any) -> Any:
    try:
        import jax

        return jax.device_get(value)
    except Exception:
        return value

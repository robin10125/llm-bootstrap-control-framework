from __future__ import annotations

from typing import Any

import numpy as np

from llm_framework.core.state import WorldState
from llm_framework.core.derived import derived_context
from llm_framework.runtime.appendages import appendage_joint_map
from llm_framework.runtime.unit_schema import joint_schema_for_state


def world_from_env_state(env: Any, state: Any) -> WorldState:
    try:
        import jax

        get = jax.device_get
    except Exception:
        get = lambda x: x

    data = state.data
    model = env.model
    act_names = [model.actuator(i).name for i in range(env.nu)]
    object_pos = np.asarray(get(data.xpos[env.object_bid]), dtype=np.float32)
    try:
        object_vel = np.asarray(get(data.cvel[env.object_bid, 3:6]), dtype=np.float32)
    except Exception:
        object_vel = np.zeros(3, dtype=np.float32)
    base_q = np.asarray([float(get(data.qpos[a])) for a in env.base_qadr], dtype=np.float32)
    hand_q = np.asarray([float(get(data.qpos[a])) for a in env.hand_qadr], dtype=np.float32)
    ctrl = np.asarray(get(data.ctrl), dtype=np.float32)
    ctrl_lo = np.asarray(model.actuator_ctrlrange[:, 0], dtype=np.float32)
    ctrl_hi = np.asarray(model.actuator_ctrlrange[:, 1], dtype=np.float32)

    fingertips: dict[str, np.ndarray] = {}
    for name in ("rh_ffdistal", "rh_mfdistal", "rh_rfdistal", "rh_lfdistal", "rh_thdistal"):
        try:
            bid = model.body(name).id
            fingertips[name] = np.asarray(get(data.xpos[bid]), dtype=np.float32)
        except Exception:
            continue

    world = WorldState(
        time_s=float(get(state.step)) * float(env.cfg.control_dt),
        object_pos=object_pos,
        object_vel=object_vel,
        base_q=base_q,
        hand_q=hand_q,
        ctrl=ctrl,
        ctrl_lo=ctrl_lo,
        ctrl_hi=ctrl_hi,
        actuator_names=act_names,
        appendages=appendage_joint_map(env),
        fingertip_pos=fingertips,
    )
    derived = derived_context(world)
    try:
        palm_pos = np.asarray(get(data.xpos[env.palm_bid]), dtype=np.float32)
        derived["palm_pos"] = palm_pos.round(4).tolist()
        derived["object_to_palm_distance"] = round(float(np.linalg.norm(object_pos - palm_pos)), 4)
        derived["object_to_palm_xy_distance"] = round(float(np.linalg.norm(object_pos[:2] - palm_pos[:2])), 4)
    except Exception:
        pass
    try:
        grasp_pos = np.asarray(get(data.site_xpos[env.grasp_sid]), dtype=np.float32)
        derived["grasp_site_pos"] = grasp_pos.round(4).tolist()
        derived["object_to_grasp_site_distance"] = round(float(np.linalg.norm(object_pos - grasp_pos)), 4)
        derived["object_to_grasp_site_xy_distance"] = round(float(np.linalg.norm(object_pos[:2] - grasp_pos[:2])), 4)
    except Exception:
        pass
    if len(world.base_q) >= 2 and len(object_pos) >= 2:
        derived["object_to_base_xy_distance"] = round(float(np.linalg.norm(object_pos[:2] - world.base_q[:2])), 4)
    try:
        ctrl_open = np.asarray(get(env.ctrl_open), dtype=np.float32)
        ctrl_close = np.asarray(get(env.ctrl_close), dtype=np.float32)
        open_by_name = {name: float(value) for name, value in zip(act_names, ctrl_open, strict=False)}
        close_by_name = {name: float(value) for name, value in zip(act_names, ctrl_close, strict=False)}
    except Exception:
        open_by_name = None
        close_by_name = None
    return WorldState(
        time_s=world.time_s,
        object_pos=world.object_pos,
        object_vel=world.object_vel,
        base_q=world.base_q,
        hand_q=world.hand_q,
        ctrl=world.ctrl,
        ctrl_lo=world.ctrl_lo,
        ctrl_hi=world.ctrl_hi,
        actuator_names=world.actuator_names,
        appendages=world.appendages,
        joint_schema=joint_schema_for_state(world, ctrl_open=open_by_name, ctrl_close=close_by_name),
        derived=derived,
        fingertip_pos=world.fingertip_pos,
        contacts=world.contacts,
        history=world.history,
    )

"""Portable symbolic action primitives shared by action priors and phase controllers.

Everything here is task- and robot-agnostic: behavior is expressed in terms of symbolic
action *groups* (which actuators) and *directions* (how to move them), never in terms of
absolute coordinates, fixed start states, or one-reset constants. A new task or a new
embodied robot is supported by satisfying the small ENV CONTRACT below; no code here
needs to change.

ENV CONTRACT (what an environment must expose to use these primitives):
  - env.action_size: int                       number of normalized actuators in [-1, 1]
  - env.nu, env.model.actuator(i).name         actuator names (for group resolution)
  - env.base_act_ids / env.hand_act_ids        optional id groupings (fallbacks if absent)
  - env.ctrl_open, env.ctrl_close              per-actuator open/closed normalized targets
  - obj-relative vector in the observation     the 3 entries immediately preceding the last
                                               `action_size` entries of `obs` (see
                                               obj_rel_from_obs); the runtime "where is the
                                               object" signal, recoverable from perception.
  - current ctrl target in the observation     the trailing `action_size` entries of `obs` are
                                               the current actuator targets (incremental control
                                               means the next target = these + action*scale).
  - env.cfg.action_scale                       per-step max change in each ctrl target (default
                                               0.05 if absent); makes toward_object_xy settle.
  - env.base_pos_obs_idx                        obs indices of the base carriage (x,y,z) position
                                               for position-seeking approach; defaults to (0,1,2),
                                               which matches an obs that leads with base joint pos.

The group/direction vocabularies live in schema.ACTION_GROUPS / schema.PRIOR_DIRECTIONS so
they are defined in one place and reused everywhere.
"""

from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jp
import numpy as np

from policy_bias_lab.schema import ACTION_GROUPS, PRIOR_DIRECTIONS

# Direction ids (index into schema.PRIOR_DIRECTIONS). Kept as named constants so callers
# that build programs do not hard-code integers.
DIR_TOWARD_OBJECT_XY = PRIOR_DIRECTIONS.index("toward_object_xy")
DIR_LOWER_BASE = PRIOR_DIRECTIONS.index("lower_base")
DIR_RAISE_BASE = PRIOR_DIRECTIONS.index("raise_base")
DIR_CLOSE_HAND = PRIOR_DIRECTIONS.index("close_hand")
DIR_OPEN_HAND = PRIOR_DIRECTIONS.index("open_hand")
DIR_STABILIZE = PRIOR_DIRECTIONS.index("stabilize")


def group_masks(env: Any) -> np.ndarray:
    """[n_groups, action_size] 0/1 masks selecting which actuators each symbolic group drives.

    Group resolution is name-based with graceful fallbacks, so a robot that does not have,
    say, named finger groups simply yields empty per-finger masks rather than erroring.
    """
    names = tuple(env.model.actuator(i).name for i in range(env.nu))
    base_ids = tuple(int(i) for i in getattr(env, "base_act_ids", ()))
    hand_ids = tuple(int(i) for i in getattr(env, "hand_act_ids", ()))
    masks = np.zeros((len(ACTION_GROUPS), env.action_size), dtype=np.float32)
    for group_idx, group in enumerate(ACTION_GROUPS):
        if group == "all":
            ids: tuple[int, ...] = tuple(range(len(names)))
        elif group == "base_xy":
            ids = tuple(i for i, name in enumerate(names) if name in {"base_x", "base_y"})
        elif group == "base_z":
            ids = tuple(i for i, name in enumerate(names) if name == "base_z")
        elif group == "hand":
            ids = hand_ids
        else:
            prefixes = {
                "thumb": "rh_A_TH",
                "index": "rh_A_FF",
                "middle": "rh_A_MF",
                "ring": "rh_A_RF",
                "little": "rh_A_LF",
            }
            prefix = prefixes.get(group)
            ids = tuple(i for i, name in enumerate(names) if prefix and name.startswith(prefix))
            if not ids:
                ids = base_ids
        if ids:
            masks[group_idx, list(ids)] = 1.0
    return masks


def obj_rel_from_obs(obs: jp.ndarray, action_dim: int) -> jp.ndarray:
    """Object-relative xyz vector convention: the 3 obs entries just before the trailing
    `action_dim` proprioceptive entries. A real robot supplies this from its perception
    stack; nothing here depends on simulator-only state."""
    rel_start = obs.shape[-1] - action_dim - 3
    return obs[rel_start: rel_start + 3]


def encode_rules(rules: list[dict[str, Any]], *, max_rules: int = 12) -> dict[str, np.ndarray]:
    """Pad a list of {group, direction, weight} rules into fixed-shape id/weight arrays."""
    group_ids = np.zeros((max_rules,), dtype=np.int32)
    direction_ids = np.full((max_rules,), DIR_STABILIZE, dtype=np.int32)
    weights = np.zeros((max_rules,), dtype=np.float32)
    for idx, rule in enumerate(rules[:max_rules]):
        group = str(rule.get("group", "all"))
        direction = str(rule.get("direction", "stabilize"))
        if group not in ACTION_GROUPS or direction not in PRIOR_DIRECTIONS:
            continue
        group_ids[idx] = ACTION_GROUPS.index(group)
        direction_ids[idx] = PRIOR_DIRECTIONS.index(direction)
        weights[idx] = max(0.0, float(rule.get("weight", 0.0)))
    return {"group_ids": group_ids, "direction_ids": direction_ids, "weights": weights}


def sustained_bool(mask_te: jp.ndarray, hold_steps: int) -> jp.ndarray:
    """Fraction of envs (axis 1) with a run of >= hold_steps consecutive True steps (axis 0).

    Used for contact-gated "held a grasp-lift for long enough" style metrics. mask_te has
    shape [T, E].
    """
    def episode(ep: jp.ndarray) -> jp.ndarray:
        def body(run_len, flag):
            next_len = jp.where(flag, run_len + 1, 0)
            return next_len, next_len

        _last, run_lengths = jax.lax.scan(body, jp.int32(0), ep)
        return (run_lengths.max() >= int(hold_steps)).astype(jp.float32)

    return jax.vmap(episode, in_axes=1)(mask_te).mean()


def base_world_jacobian(env: Any) -> np.ndarray | None:
    """3x3 map M from base-actuator ctrl deltas to grasp-site WORLD displacement.

    Column j is d(grasp_world_xyz)/d(base_actuator_j ctrl), in actuator order base_x, base_y,
    base_z. Computed once from the model via mj_jacSite at the home pose (the slide Jacobian is
    configuration-independent for a translating base). This captures any base-body rotation /
    axis inversion -- e.g. a base mounted upside down makes M = diag(1,-1,-1) -- which a naive
    jnt_axis read misses. Returns None if the model lacks the expected grasp site / base
    actuators, in which case callers fall back to identity (legacy behavior).
    """
    try:
        import mujoco
        m = env.model
        sid = int(getattr(env, "grasp_sid", m.site("grasp_site").id))
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        jacp = np.zeros((3, m.nv))
        jacr = np.zeros((3, m.nv))
        mujoco.mj_jacSite(m, d, jacp, jacr, sid)
        names = tuple(m.actuator(i).name for i in range(m.nu))
        cols = []
        for act_name in ("base_x", "base_y", "base_z"):
            if act_name not in names:
                return None
            aid = names.index(act_name)
            jid = int(m.actuator_trnid[aid, 0])
            dof = int(m.jnt_dofadr[jid])
            cols.append(jacp[:, dof])
        return np.stack(cols, axis=1).astype(np.float32)  # [world_xyz, base_xyz]
    except Exception:
        return None


def make_rule_action_fn(env: Any) -> tuple[Callable[..., jp.ndarray], dict[str, Any]]:
    """Build the single source of truth for "symbolic rules -> normalized action vector".

    Returns (action_from_obs, info). `action_from_obs(obs, group_ids, direction_ids, weights)`
    maps one observation plus padded rule arrays to a clipped action in [-1, 1]. The same
    function is used by the open-loop prior scorer and by the phase controller, so their
    action semantics can never drift apart.
    """
    masks = jp.asarray(group_masks(env), dtype=jp.float32)
    close_sign = jp.sign(jp.asarray(env.ctrl_close) - jp.asarray(env.ctrl_open))
    open_sign = jp.sign(jp.asarray(env.ctrl_open) - jp.asarray(env.ctrl_close))
    action_dim = int(env.action_size)
    action_names = tuple(env.model.actuator(i).name for i in range(env.nu))
    base_x_idx = action_names.index("base_x") if "base_x" in action_names else -1
    base_y_idx = action_names.index("base_y") if "base_y" in action_names else -1
    base_z_idx = action_names.index("base_z") if "base_z" in action_names else -1
    # Actions are INCREMENTAL deltas to the position target (env: target = ctrl + action*scale).
    # `toward_object_xy` is therefore position-SEEKING, not a constant push: we drive the target
    # toward the object's absolute carriage position so it settles (delta -> 0 as the hand aligns)
    # instead of integrating to the travel limit and oscillating. Base carriage position is at
    # obs[base_pos_obs_idx], current ctrl is the trailing action_dim block of obs -- both
    # observable on hardware. The base->world Jacobian (base_world_jacobian) calibrates per-axis
    # sign/scale so the seek and the vertical directions move the grasp the correct way even when
    # the base body is rotated (here M = diag(1,-1,-1): slide_y/z are world-inverted, so a naive
    # seek drives away on y and `lower_base` would raise the hand). Falls back to identity.
    action_scale = float(getattr(getattr(env, "cfg", None), "action_scale", 0.05)) or 0.05
    base_pos_obs_idx = tuple(getattr(env, "base_pos_obs_idx", (0, 1, 2)))
    M = base_world_jacobian(env)
    if M is None:
        M = np.eye(3, dtype=np.float32)
    diagM = np.diag(M)
    sgn = np.where(np.abs(diagM) > 1e-6, np.sign(diagM), 1.0).astype(np.float32)  # grasp move / ctrl per axis
    sgn_x, sgn_y = float(sgn[0]), float(sgn[1])
    z_down_ctrl_sign = float(-sgn[2])  # ctrl-delta sign that moves the grasp DOWN in world z

    def action_from_obs(obs, group_ids, direction_ids, weights):
        obj_rel = obj_rel_from_obs(obs, action_dim)
        cur_ctrl = obs[obs.shape[-1] - action_dim:]  # trailing dx.ctrl block

        def seek_delta(act_idx, base_pos_idx, rel, axis_sign):
            # Convert the world offset to a slide-coordinate target (divide by the world sign),
            # then drive the ctrl toward it so the delta vanishes as the hand aligns.
            desired = obs[base_pos_idx] + rel * axis_sign
            return jp.clip((desired - cur_ctrl[act_idx]) / action_scale, -1.0, 1.0)

        def add_rule(carry, item):
            out = carry
            group_id, direction_id, weight = item
            mask = masks[group_id]
            base_xy = jp.zeros((action_dim,), dtype=jp.float32)
            if base_x_idx >= 0:
                base_xy = base_xy.at[base_x_idx].set(seek_delta(base_x_idx, base_pos_obs_idx[0], obj_rel[0], sgn_x) * weight)
            if base_y_idx >= 0:
                base_xy = base_xy.at[base_y_idx].set(seek_delta(base_y_idx, base_pos_obs_idx[1], obj_rel[1], sgn_y) * weight)
            # Vertical directions are calibrated to world down/up via the base Jacobian.
            lower = jp.zeros((action_dim,), dtype=jp.float32)
            raise_base = jp.zeros((action_dim,), dtype=jp.float32)
            if base_z_idx >= 0:
                lower = lower.at[base_z_idx].set(z_down_ctrl_sign * jp.abs(weight))
                raise_base = raise_base.at[base_z_idx].set(-z_down_ctrl_sign * jp.abs(weight))
            close = weight * mask * close_sign
            open_hand = weight * mask * open_sign
            vector = jp.where(direction_id == DIR_TOWARD_OBJECT_XY, base_xy, jp.zeros_like(base_xy))
            vector = jp.where(direction_id == DIR_LOWER_BASE, lower, vector)
            vector = jp.where(direction_id == DIR_RAISE_BASE, raise_base, vector)
            vector = jp.where(direction_id == DIR_CLOSE_HAND, close, vector)
            vector = jp.where(direction_id == DIR_OPEN_HAND, open_hand, vector)
            return out + vector, None

        out, _ = jax.lax.scan(add_rule, jp.zeros((action_dim,), dtype=jp.float32), (group_ids, direction_ids, weights))
        return jp.clip(out, -1.0, 1.0)

    info = {"action_dim": action_dim, "masks": masks}
    return action_from_obs, info

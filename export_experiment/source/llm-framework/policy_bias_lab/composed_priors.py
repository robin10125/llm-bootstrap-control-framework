"""Situation-dependent action priors compiled into a stateless per-step mean-shift.

A *prior program* (a dict spec) is compiled against an environment into a pure JAX function
``prior_fn(obs, weights) -> mean_shift`` that is added to the policy's pre-tanh action each
step. See policy_bias_lab/EXPERIMENT_situation_dependent_priors.md for the design.

Modes
-----
- ``monolithic``   : one always-on legacy rule-set (today's behaviour; for control arms).
- ``gated``        : a small library of legacy sub-priors {approach, grasp, lift} composed by
                     a gating discipline (``soft`` / ``subgoal`` / ``options`` / ``stacked``).
- ``reactive_law`` : one continuous controller built from observable signals (B.5).
- ``dmp``          : a goal-bound, progress-phased second-order servo (B.6).

STATELESSNESS (hard requirement): PPO recomputes the prior on shuffled minibatches at update
time, so the mean-shift MUST be a deterministic function of (obs, weights) alone. No phase
pointer, DMP integrator, or hysteresis latch may carry state across steps -- every "situation"
is read from the *current* observable signals. (True temporal commitment would need a phase
feature appended to obs; that is a documented future extension.)

PORTABILITY: gates use only signals derivable from obs and measurable on hardware
(object-relative position, finger closure from proprioception, object height). The base-motion
operators use the same calibrated base->world Jacobian as symbolic_control / bias, so a prior
behaves identically whether scored, trained, or run on the robot.
"""

from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jp
import numpy as np

from policy_bias_lab.symbolic_control import (
    base_world_jacobian,
    encode_rules,
    make_rule_action_fn,
    obj_rel_from_obs,
)

# ----------------------------------------------------------------------------------------------
# Observable signals
# ----------------------------------------------------------------------------------------------

def _object_rest_z(env: Any) -> float:
    """Object height at the home pose (CPU forward kinematics, no MJX)."""
    try:
        import mujoco
        m = env.model
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        return float(d.xpos[int(env.object_bid)][2])
    except Exception:
        return 0.0


def make_signal_fn(env: Any) -> tuple[Callable[[jp.ndarray], dict[str, jp.ndarray]], dict[str, Any]]:
    """Build ``signals(obs) -> {palm_obj_dist, obj_rel, closure, lift}`` from obs alone.

    obs layout: [<robot prefix: joint blocks + per-joint constraint forces>, obj_pos(3),
    obj_vel(3), palm_pos(3), obj_rel(3), ctrl(action_dim)]. palm_obj_dist == ||obj_rel|| exactly
    (obj_rel = obj - grasp site). closure == normalized hand ctrl. lift == obj_pos_z - rest_z
    (rest_z ~ const).
    """
    action_dim = int(env.action_size)
    # Prefix length derived from the ACTUAL obs size, not a re-listing of the prefix blocks --
    # the tail [obj_pos, obj_vel, palm_pos, obj_rel, ctrl] is the layout contract (12 + nu), so
    # this stays correct when the robot prefix grows (e.g. the constraint-force block).
    n_pre = int(env.obs_size) - 12 - action_dim
    obj_z_idx = n_pre + 2
    hand_ids = np.asarray([int(i) for i in getattr(env, "hand_act_ids", ())], dtype=np.int32)
    ctrl_open = np.asarray(env.ctrl_open, dtype=np.float32)
    ctrl_close = np.asarray(env.ctrl_close, dtype=np.float32)
    close_dir = ctrl_close - ctrl_open
    z0 = _object_rest_z(env)
    has_hand = hand_ids.size > 0
    hand_ids_j = jp.asarray(hand_ids)
    open_j = jp.asarray(ctrl_open[hand_ids]) if has_hand else jp.zeros(0)
    dir_j = jp.asarray(close_dir[hand_ids]) if has_hand else jp.ones(0)

    def signals(obs: jp.ndarray) -> dict[str, jp.ndarray]:
        obj_rel = obj_rel_from_obs(obs, action_dim)
        palm_obj_dist = jp.linalg.norm(obj_rel)
        cur_ctrl = obs[obs.shape[-1] - action_dim:]
        if has_hand:
            closure = jp.mean(jp.clip((cur_ctrl[hand_ids_j] - open_j) / (dir_j + 1e-6), 0.0, 1.0))
        else:
            closure = jp.float32(0.0)
        lift = obs[obj_z_idx] - z0
        return {"palm_obj_dist": palm_obj_dist, "obj_rel": obj_rel, "closure": closure, "lift": lift}

    info = {"z0": z0, "obj_z_idx": obj_z_idx, "action_dim": action_dim, "n_pre": n_pre}
    return signals, info


# ----------------------------------------------------------------------------------------------
# Gating disciplines (stateless functions of current signals -> [g_approach, g_grasp, g_lift])
# ----------------------------------------------------------------------------------------------

def _sigmoid(x: jp.ndarray) -> jp.ndarray:
    return 1.0 / (1.0 + jp.exp(-x))


def _gates(discipline: str, sig: dict[str, jp.ndarray], g: dict[str, float]) -> jp.ndarray:
    """Return a length-3 nonnegative gate vector for [approach, grasp, lift].

    Signals: d = palm_obj_dist (target eps), c = closure (target kappa), z = lift (target h).
    All four disciplines are stateless partitions of the current signal space.
    """
    d = sig["palm_obj_dist"]
    c = sig["closure"]
    eps = float(g.get("approach_eps", 0.04))
    kappa = float(g.get("grasp_kappa", 0.6))
    tau_d = float(g.get("tau_d", 0.02))
    tau_c = float(g.get("tau_c", 0.15))
    margin = float(g.get("commit_margin", 0.02))

    if discipline == "soft":
        # Smooth partition that sums to exactly 1: near gates grasp/lift, gripped splits them.
        near = _sigmoid((eps - d) / tau_d)
        gripped = _sigmoid((c - kappa) / tau_c)
        return jp.asarray([1.0 - near, near * (1.0 - gripped), near * gripped])

    if discipline == "subgoal":
        # Hard one-hot on the current situation: far -> approach; near & open -> grasp;
        # near & gripped -> lift.
        near = (d < eps).astype(jp.float32)
        gripped = (c > kappa).astype(jp.float32)
        return jp.asarray([1.0 - near, near * (1.0 - gripped), near * gripped])

    if discipline == "options":
        # Hard, with a stateless commitment flavour: once gripped (within a margin) commit to
        # LIFT regardless of small proximity changes; widen the near region by `margin` so the
        # grasp option is sticky. Priority lift > grasp > approach.
        gripped = (c > (kappa - margin)).astype(jp.float32)
        near = (d < (eps + margin)).astype(jp.float32)
        g_lift = gripped
        g_grasp = (1.0 - gripped) * near
        g_approach = (1.0 - gripped) * (1.0 - near)
        return jp.asarray([g_approach, g_grasp, g_lift])

    if discipline == "stacked":
        # A.2 structure + A.3 commitment + A.1 softness: soft transitions, but `gripped` uses a
        # sharper sigmoid and commits to lift (object-grip dominates), grasp/approach soft below.
        near = _sigmoid((eps - d) / tau_d)
        gripped = _sigmoid((c - (kappa - margin)) / (0.5 * tau_c))  # sharper -> commitment
        g_lift = gripped
        g_grasp = (1.0 - gripped) * near
        g_approach = (1.0 - gripped) * (1.0 - near)
        return jp.asarray([g_approach, g_grasp, g_lift])

    raise ValueError(f"unknown gating discipline {discipline!r}")


# ----------------------------------------------------------------------------------------------
# Default hand-authored library (shared & constant across gating-bake-off arms)
# ----------------------------------------------------------------------------------------------

def default_library() -> list[dict[str, Any]]:
    """Three legacy sub-priors: approach, grasp, lift. Order MUST match _gates' [a,g,l]."""
    return [
        {"name": "approach", "rules": [
            {"group": "base_xy", "direction": "toward_object_xy", "weight": 0.5},
            {"group": "base_z", "direction": "lower_base", "weight": 0.3},
        ]},
        {"name": "grasp", "rules": [
            {"group": "thumb", "direction": "close_hand", "weight": 0.5},
            {"group": "index", "direction": "close_hand", "weight": 0.45},
            {"group": "middle", "direction": "close_hand", "weight": 0.4},
        ]},
        {"name": "lift", "rules": [
            {"group": "base_z", "direction": "raise_base", "weight": 0.4},
            {"group": "thumb", "direction": "close_hand", "weight": 0.2},
            {"group": "index", "direction": "close_hand", "weight": 0.2},
            {"group": "middle", "direction": "close_hand", "weight": 0.2},
        ]},
    ]


PRIOR_PROGRAM_ARMS: tuple[str, ...] = (
    "prior_monolithic",
    "prior_gate_soft",
    "prior_gate_subgoal",
    "prior_gate_options",
    "prior_gate_stacked",
    "prior_reactive_law",
    "prior_dmp",
)


def prior_program_for_arm(
    arm: str,
    library: list[dict[str, Any]] | None = None,
    gates: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Map an experiment arm name to its prior program (None if the arm isn't a prior arm).

    The bake-off shares one `library` across all gated arms so only the gating discipline
    varies; `prior_monolithic` is the control (today's single always-on prior = the union of
    the library's rules, no gating).
    """
    lib = library or default_library()
    gates = dict(gates or {})
    if arm == "prior_monolithic":
        rules: list[dict[str, Any]] = []
        for sub in lib:
            rules.extend(sub.get("rules", []))
        return {"mode": "monolithic", "rules": rules}
    if arm.startswith("prior_gate_"):
        return {"mode": "gated", "discipline": arm[len("prior_gate_"):], "library": lib, "gates": gates}
    if arm == "prior_reactive_law":
        return {"mode": "reactive_law", "gates": gates}
    if arm == "prior_dmp":
        return {"mode": "dmp", "gates": gates}
    return None


# ----------------------------------------------------------------------------------------------
# Base->world calibration helpers (shared with bias / symbolic_control)
# ----------------------------------------------------------------------------------------------

def _calibration(env: Any) -> dict[str, Any]:
    action_names = tuple(env.model.actuator(i).name for i in range(env.nu))
    idx = {n: (action_names.index(n) if n in action_names else -1) for n in ("base_x", "base_y", "base_z")}
    M = base_world_jacobian(env)
    if M is None:
        sgn = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    else:
        diag = np.diag(M)
        sgn = np.where(np.abs(diag) > 1e-6, np.sign(diag), 1.0).astype(np.float32)
    return {
        "bx": idx["base_x"], "by": idx["base_y"], "bz": idx["base_z"],
        "sgn": (float(sgn[0]), float(sgn[1]), float(sgn[2])),
        "base_pos_obs_idx": tuple(int(i) for i in getattr(env, "base_pos_obs_idx", (0, 1, 2))),
        "action_scale": float(getattr(getattr(env, "cfg", None), "action_scale", 0.05)) or 0.05,
    }


# ----------------------------------------------------------------------------------------------
# Prior-program compilation
# ----------------------------------------------------------------------------------------------

def make_composed_prior_fn(env: Any, program: dict[str, Any]) -> tuple[Callable[..., jp.ndarray], jp.ndarray, dict[str, Any]]:
    """Compile a prior program into (prior_fn(obs, weights)->mean_shift, default_weights, info).

    `weights` is the runtime-tunable knob threaded by PPO (the coach may scale it):
      - gated     : per-sub-prior gain (length = #sub-priors), default ones.
      - reactive_law / dmp : per-channel gains (length = #gains), default from the program.
      - monolithic: legacy per-rule weights.
    """
    mode = str(program.get("mode", "gated"))
    action_dim = int(env.action_size)
    rule_action_fn, _ = make_rule_action_fn(env)  # shared calibrated operator evaluator

    if mode == "monolithic":
        rules = list(program.get("rules", []))
        enc = encode_rules(rules)
        gids = jp.asarray(enc["group_ids"], jp.int32)
        dids = jp.asarray(enc["direction_ids"], jp.int32)
        base_w = jp.asarray(enc["weights"], jp.float32)

        def prior_fn(obs, weights):  # weights overrides per-rule weight (legacy semantics)
            w = base_w if weights is None else jp.asarray(weights, jp.float32)[:base_w.shape[0]]
            return rule_action_fn(obs, gids, dids, w)

        return prior_fn, base_w, {"mode": mode, "n_weights": int(base_w.shape[0])}

    if mode == "gated":
        discipline = str(program.get("discipline", "soft"))
        library = list(program.get("library") or default_library())
        gate_params = dict(program.get("gates", {}))
        signals, _sinfo = make_signal_fn(env)
        # Pre-encode each sub-prior's rules.
        enc_list = [encode_rules(list(sub.get("rules", []))) for sub in library]
        gids = [jp.asarray(e["group_ids"], jp.int32) for e in enc_list]
        dids = [jp.asarray(e["direction_ids"], jp.int32) for e in enc_list]
        wts = [jp.asarray(e["weights"], jp.float32) for e in enc_list]
        n_sub = len(library)

        def prior_fn(obs, weights):
            gains = jp.ones((n_sub,), jp.float32) if weights is None else jp.asarray(weights, jp.float32)[:n_sub]
            sig = signals(obs)
            g = _gates(discipline, sig, gate_params)  # [n_sub]
            out = jp.zeros((action_dim,), jp.float32)
            for i in range(n_sub):
                sub_action = rule_action_fn(obs, gids[i], dids[i], wts[i])
                out = out + (g[i] * gains[i]) * sub_action
            return jp.clip(out, -1.0, 1.0)

        return prior_fn, jp.ones((n_sub,), jp.float32), {
            "mode": mode, "discipline": discipline, "n_weights": n_sub,
            "sub_names": [str(s.get("name", f"sub_{i}")) for i, s in enumerate(library)],
        }

    if mode in ("stacked", "freeform", "freeform_staged"):
        # Lazy import avoids the composed_priors <-> freeform_priors cycle. `stacked` composes
        # DSL/free-form phase sub-priors via the FIXED stacked gate; `freeform` is a single free-form
        # law; `freeform_staged` is a free-form law composed of N author-defined stages, each with a
        # free-form GATE expression (soft/hard blend) -- generalizes the fixed 3-phase gate.
        from policy_bias_lab.freeform_priors import (
            make_stacked_prior_fn, make_freeform_prior_fn, make_freeform_staged_prior_fn)
        if mode == "stacked":
            return make_stacked_prior_fn(env, program)
        if mode == "freeform_staged":
            return make_freeform_staged_prior_fn(env, program)
        return make_freeform_prior_fn(env, program)

    if mode == "reactive_law":
        return _make_reactive_law(env, program, action_dim)

    if mode == "dmp":
        return _make_dmp(env, program, action_dim)

    raise ValueError(f"unknown prior mode {mode!r}")


def _make_reactive_law(env, program, action_dim):
    """B.5: one continuous controller. mean-shift built from observable signals with continuous
    gains (no discrete gating). Gains are the tunable `weights` vector, order:
    [seek_xy, descend, close, raise]."""
    signals, _ = make_signal_fn(env)
    cal = _calibration(env)
    bx, by, bz = cal["bx"], cal["by"], cal["bz"]
    sgn_x, sgn_y, sgn_z = cal["sgn"]
    z_down = -sgn_z
    px, py, _pz = cal["base_pos_obs_idx"]
    scale = cal["action_scale"]
    # Hand masks (which actuators are fingers) and close direction.
    names = tuple(env.model.actuator(i).name for i in range(env.nu))
    hand_ids = [i for i in range(env.nu) if names[i].startswith("rh_A_")]
    hand_mask = np.zeros((action_dim,), np.float32)
    if hand_ids:
        hand_mask[hand_ids] = 1.0
    hand_mask_j = jp.asarray(hand_mask)
    close_sign = jp.sign(jp.asarray(env.ctrl_close) - jp.asarray(env.ctrl_open))
    gp = dict(program.get("gains", {}))
    default_gains = jp.asarray([
        float(gp.get("seek_xy", 1.0)), float(gp.get("descend", 0.6)),
        float(gp.get("close", 0.8)), float(gp.get("raise", 0.6)),
    ], jp.float32)
    eps = float(program.get("gates", {}).get("approach_eps", 0.04))
    kappa = float(program.get("gates", {}).get("grasp_kappa", 0.6))
    tau_d = float(program.get("gates", {}).get("tau_d", 0.02))
    tau_c = float(program.get("gates", {}).get("tau_c", 0.15))

    def prior_fn(obs, weights):
        gains = default_gains if weights is None else jp.asarray(weights, jp.float32)[:4]
        sig = signals(obs)
        d = sig["palm_obj_dist"]; c = sig["closure"]; obj_rel = sig["obj_rel"]
        cur_ctrl = obs[obs.shape[-1] - action_dim:]
        near = _sigmoid((eps - d) / tau_d)
        gripped = _sigmoid((c - kappa) / tau_c)
        out = jp.zeros((action_dim,), jp.float32)
        # seek object xy (position-seeking, calibrated) -- active until gripped.
        if bx >= 0:
            out = out.at[bx].set(jp.clip((obs[px] + obj_rel[0] * sgn_x - cur_ctrl[bx]) / scale, -1.0, 1.0)
                                 * gains[0] * (1.0 - gripped))
        if by >= 0:
            out = out.at[by].set(jp.clip((obs[py] + obj_rel[1] * sgn_y - cur_ctrl[by]) / scale, -1.0, 1.0)
                                 * gains[0] * (1.0 - gripped))
        # descend while above the object and not yet gripped; raise once gripped.
        if bz >= 0:
            descend = z_down * gains[1] * (1.0 - near) * (1.0 - gripped)
            lift_up = (-z_down) * gains[3] * gripped
            out = out.at[bz].set(descend + lift_up)
        # close fingers as fingertips arrive (near) and not yet gripped.
        out = out + (gains[2] * near * (1.0 - gripped)) * hand_mask_j * close_sign
        return jp.clip(out, -1.0, 1.0)

    return prior_fn, default_gains, {"mode": "reactive_law", "n_weights": 4,
                                     "gain_names": ["seek_xy", "descend", "close", "raise"]}


def _make_dmp(env, program, action_dim):
    """B.6: goal-bound, progress-phased second-order servo (stateless: x and xdot are read from
    obs each step, goal from the observed object, phase from progress). mean-shift =
    clip(gain * (alpha*(beta*(g - x) - xdot_term))). Gains are [base_kp, base_kd, hand_kp]."""
    signals, sinfo = make_signal_fn(env)
    cal = _calibration(env)
    bx, by, bz = cal["bx"], cal["by"], cal["bz"]
    sgn_x, sgn_y, sgn_z = cal["sgn"]
    z_down = -sgn_z
    px, py, _pz = cal["base_pos_obs_idx"]
    scale = cal["action_scale"]
    n_pre = sinfo["n_pre"]
    # base velocity indices in obs (base_v follows base_q at [n_base : 2*n_base]); we only need
    # the base dofs that map to base_x/y/z. base_q occupies [0:n_base], base_v [n_base:2*n_base].
    n_base = len(env.base_qadr)
    names = tuple(env.model.actuator(i).name for i in range(env.nu))
    base_order = [n for n in ("base_x", "base_y", "base_z")]
    base_vel_idx = {}
    for k, nm in enumerate(base_order):
        # base actuators are the first base_qadr joints in obs order; map by name position.
        base_vel_idx[nm] = n_base + k if k < n_base else -1
    hand_ids = [i for i in range(env.nu) if names[i].startswith("rh_A_")]
    hand_mask = np.zeros((action_dim,), np.float32)
    if hand_ids:
        hand_mask[hand_ids] = 1.0
    hand_mask_j = jp.asarray(hand_mask)
    close_sign = jp.sign(jp.asarray(env.ctrl_close) - jp.asarray(env.ctrl_open))
    dp = dict(program.get("gains", {}))
    default_gains = jp.asarray([
        float(dp.get("base_kp", 1.0)), float(dp.get("base_kd", 0.3)), float(dp.get("hand_kp", 0.8)),
    ], jp.float32)
    alpha = float(program.get("dmp", {}).get("alpha", 4.0))
    beta = float(program.get("dmp", {}).get("beta", 1.0))
    eps = float(program.get("gates", {}).get("approach_eps", 0.04))
    kappa = float(program.get("gates", {}).get("grasp_kappa", 0.6))
    tau_d = float(program.get("gates", {}).get("tau_d", 0.02))
    tau_c = float(program.get("gates", {}).get("tau_c", 0.15))

    def prior_fn(obs, weights):
        gains = default_gains if weights is None else jp.asarray(weights, jp.float32)[:3]
        base_kp, base_kd, hand_kp = gains[0], gains[1], gains[2]
        sig = signals(obs)
        d = sig["palm_obj_dist"]; c = sig["closure"]; obj_rel = sig["obj_rel"]
        cur_ctrl = obs[obs.shape[-1] - action_dim:]
        near = _sigmoid((eps - d) / tau_d)
        gripped = _sigmoid((c - kappa) / tau_c)
        out = jp.zeros((action_dim,), jp.float32)
        # Goal-seeking attractor on the base: goal = object carriage target (xy), descend target
        # (z) until gripped then lift target. xdot from base_v (damping).
        bvx = base_vel_idx.get("base_x", -1); bvy = base_vel_idx.get("base_y", -1); bvz = base_vel_idx.get("base_z", -1)
        if bx >= 0:
            err = (obs[px] + obj_rel[0] * sgn_x) - cur_ctrl[bx]
            xdot = obs[bvx] if bvx >= 0 else 0.0
            out = out.at[bx].set(jp.clip(alpha * (beta * err - base_kd * xdot) / (scale * 10.0), -1.0, 1.0)
                                 * base_kp * (1.0 - gripped))
        if by >= 0:
            err = (obs[py] + obj_rel[1] * sgn_y) - cur_ctrl[by]
            xdot = obs[bvy] if bvy >= 0 else 0.0
            out = out.at[by].set(jp.clip(alpha * (beta * err - base_kd * xdot) / (scale * 10.0), -1.0, 1.0)
                                 * base_kp * (1.0 - gripped))
        if bz >= 0:
            # progress-phased: descend while not gripped, lift once gripped (goal flips).
            descend = z_down * base_kp * (1.0 - near) * (1.0 - gripped)
            lift_up = (-z_down) * base_kp * gripped
            out = out.at[bz].set(descend + lift_up)
        # finger attractor toward closed pose, phased by proximity, until gripped.
        out = out + (hand_kp * near * (1.0 - gripped)) * hand_mask_j * close_sign
        return jp.clip(out, -1.0, 1.0)

    return prior_fn, default_gains, {"mode": "dmp", "n_weights": 3,
                                     "gain_names": ["base_kp", "base_kd", "hand_kp"]}

"""Motor-tape priors: reset-anchored feedforward command tapes (see DESIGN.md).

Brain isomorphism: the LLM is the motor cortex -- it authors a FULL SEQUENCE OF COMMANDS for the
whole episode (a plan, generated once, conditioned on the state at planning time). Downstream
learned networks are the cerebellum -- they see the plan (efference copy) plus sensory state and
correct it online (residual), retime it (playback rate), and optionally bend it (modulation).

The authored artifact is a list of KEYFRAMES. Each keyframe's per-actuator targets are EXPRESSIONS
over the RESET observation (raw observables + the candidate's own `signals`), evaluated ONCE per
episode at reset -- so the plan adapts to where the object actually spawned -- then interpolated
into a concrete numeric command tape q_des[T+1, nu] of ABSOLUTE actuator ctrl targets. Playback is
feedforward: no expression reads sensors after reset. All feedback lives in the learned networks.

IR (mode "motor_tape"):
  {
    "mode": "motor_tape",
    "name": "...",
    "signals":    {"<name>": "<expr over raw RESET observables>"},   # evaluated once, in order
    "parameters": {"<name>": 0.5 | {"init": .., "range": [lo, hi]}}, # scalar constants
    "defaults":   {"interp": "minjerk"},                             # optional
    "keyframes": [
      {"t": 2.5, "label": "free text",
       "targets": {"<actuator or semantic group>": "<expr -> ABSOLUTE ctrl target>"},
       "interp": "linear" | "minjerk"}                               # segment ENDING here
    ]
  }

Semantics:
  - `t` (seconds) means ARRIVE AT this value BY time t. Every actuator that appears in any
    keyframe gets an implicit knot at t=0 holding its reset ctrl, so the first authored segment
    interpolates from the spawn pose (no step discontinuity). Actuators in NO keyframe hold their
    reset ctrl for the whole episode. After an actuator's last knot its target holds (hold-last).
  - `targets` keys are actuator names or semantic-group tags (resolved by _resolve_actuators);
    a group key binds all members to the SAME expr; an exact-name key in the same keyframe wins
    over a group key. Per-member individuality inside a group comes from the RESET-SELF signals:
    `ctrl_init_self` (that actuator's reset ctrl), `ctrl_lo_self` / `ctrl_hi_self` (its range).
  - Interpolation per actuator between consecutive knots: "linear" or "minjerk" (quintic
    10u^3-15u^4+6u^5 -- zero velocity/acceleration at both knots, never leaves [v0, v1]).
  - Knot values are clipped to the actuator's ctrlrange at tape build (report warns on excess).

Compilation: `compile_tape` returns a pure JAX function tape_from_obs(obs0) -> q_des[T+1, nu],
vmap/jit-safe, so per-env tapes are built from the batched reset obs inside the trainer.

Task knowledge enters ONLY through the LLM-authored program; this module stays task-agnostic.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import jax
import jax.numpy as jp
import numpy as np

from policy_bias_lab.freeform_priors import (
    _parameter_values,
    _resolve_actuators,
    all_actuator_names,
    _semantic_group,
    compile_expr,
    raw_obs_entries,
    raw_signal_fn,
)
from policy_bias_lab.llm_util import candidate_score
from policy_bias_lab.symbolic_control import sustained_bool
from policy_bias_lab.tasks import task_graded_objective, task_success, task_fitness
from policy_bias_lab.training.fragmented_ppo import _eval_summary

RESET_SELF_SIGNALS = {"ctrl_init_self", "ctrl_lo_self", "ctrl_hi_self"}
INTERP_KINDS = ("linear", "minjerk")


# --------------------------------------------------------------------------------------------
# Signals (reset-time vocabulary)
# --------------------------------------------------------------------------------------------

def tape_signal_fn(env: Any, program: dict[str, Any]) -> tuple[Callable, set[str]]:
    """signals(obs0) -> dict for RESET-time evaluation: raw observables + `parameters` +
    the candidate's own `signals` (compiled in insertion order; later may reference earlier).
    STRICT vocabulary -- no legacy derived signals; every derived quantity must be authored."""
    raw_fn, raw_names, _doc = raw_signal_fn(env)
    params = _parameter_values(program)
    avail = set(raw_names) | set(params)
    compiled: list[tuple[str, Callable]] = []
    authored = program.get("signals")
    if isinstance(authored, dict):
        for name, expr in authored.items():
            ev = compile_expr(str(expr), avail)  # raises -> validation rejects the candidate
            compiled.append((str(name), ev))
            avail.add(str(name))

    def signals(obs0: jp.ndarray) -> dict[str, jp.ndarray]:
        sig = raw_fn(obs0)
        for name, value in params.items():
            sig[name] = jp.asarray(value, jp.float32)
        for name, ev in compiled:
            sig[name] = jp.asarray(ev(sig), jp.float32)
        return sig

    return signals, avail


# --------------------------------------------------------------------------------------------
# Keyframe expansion and interpolation
# --------------------------------------------------------------------------------------------

def expand_keyframes(env: Any, program: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
    """Per-actuator knot timelines from the keyframe list.

    Returns {act_idx: [ {t, expr(str|None), interp, kf_index} ... ]} sorted by t, where expr=None
    is the implicit t=0 anchor (value = that actuator's reset ctrl). Within one keyframe an
    exact-name target overrides a group target for the same actuator. `interp` on a knot is the
    kind of the segment ENDING at that knot.
    """
    names = all_actuator_names(env)
    default_interp = str((program.get("defaults") or {}).get("interp", "minjerk"))
    per_act: dict[int, list[dict[str, Any]]] = {}
    for ki, kf in enumerate(program.get("keyframes", [])):
        t = float(kf["t"])
        interp = str(kf.get("interp", default_interp))
        targets = kf.get("targets") or {}
        # group entries first, exact names second (member-over-group precedence)
        entry: dict[int, str] = {}
        for key, expr in targets.items():
            if str(key) in names:
                continue
            for a in _resolve_actuators(env, [str(key)]):
                entry[a] = str(expr)
        for key, expr in targets.items():
            if str(key) in names:
                entry[names.index(str(key))] = str(expr)
        for a, expr in entry.items():
            per_act.setdefault(a, []).append(
                {"t": t, "expr": expr, "interp": interp, "kf_index": ki})
    # implicit t=0 anchor from the reset pose
    for a, knots in per_act.items():
        knots.sort(key=lambda k: k["t"])
        if knots[0]["t"] > 0.0:
            knots.insert(0, {"t": 0.0, "expr": None, "interp": default_interp, "kf_index": -1})
    return per_act


def interp_weights(knot_times: np.ndarray, interps: list[str], n_steps: int,
                   control_dt: float) -> np.ndarray:
    """W[n_steps+1, K] with rows summing to 1: q(t_i) = W @ knot_values.

    Piecewise pairs (1-w, w) on the bracketing knots; w = u (linear) or the quintic minimum-jerk
    ease 10u^3 - 15u^4 + 6u^5 (interps[k+1] governs the segment k -> k+1; interps[0] is unused).
    Weight 1 on the first knot before it (hold from spawn -- only reachable when the first knot
    is the implicit t=0 anchor) and on the last knot after it (hold-last). Pure NumPy.
    """
    kt = np.asarray(knot_times, dtype=np.float64)
    K = len(kt)
    W = np.zeros((n_steps + 1, K), dtype=np.float32)
    ts = np.arange(n_steps + 1, dtype=np.float64) * float(control_dt)
    for i, t in enumerate(ts):
        if t <= kt[0]:
            W[i, 0] = 1.0
        elif t >= kt[-1]:
            W[i, -1] = 1.0
        else:
            k = int(np.searchsorted(kt, t, side="right") - 1)
            u = (t - kt[k]) / (kt[k + 1] - kt[k])
            if interps[k + 1] == "linear":
                w = u
            else:  # minjerk
                w = u * u * u * (10.0 - 15.0 * u + 6.0 * u * u)
            W[i, k] = 1.0 - w
            W[i, k + 1] = w
    return W


# --------------------------------------------------------------------------------------------
# Compile
# --------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class CompiledTape:
    tape_from_obs: Callable[[jp.ndarray], jp.ndarray]  # obs0[obs] -> q_des[T+1, nu]; vmap-safe
    knots_from_obs: Callable[[jp.ndarray], jp.ndarray]  # obs0 -> UNCLIPPED knot values (flat)
    knot_index: tuple[tuple[str, float], ...]           # (actuator name, t) per flat knot
    n_steps: int                                        # T = env.horizon
    nu: int
    covered: tuple[str, ...]                            # actuators with >= 1 authored knot
    info: dict[str, Any]


def compile_tape(env: Any, program: dict[str, Any]) -> CompiledTape:
    names = all_actuator_names(env)
    nu = int(env.action_size)
    T = int(env.horizon)
    dt = float(env.cfg.control_dt)
    lo = np.asarray(env.ctrl_lo, dtype=np.float32)
    hi = np.asarray(env.ctrl_hi, dtype=np.float32)
    signals, avail = tape_signal_fn(env, program)
    obs_idx = dict(raw_obs_entries(env)[0])
    ctrl_idx = [obs_idx[f"ctrl_{n}"] for n in names]

    per_act = expand_keyframes(env, program)
    # compile each distinct (kf_index, expr) once; bind self per actuator at eval time
    expr_cache: dict[tuple[int, str], Callable] = {}
    for knots in per_act.values():
        for k in knots:
            if k["expr"] is not None and (k["kf_index"], k["expr"]) not in expr_cache:
                expr_cache[(k["kf_index"], k["expr"])] = compile_expr(
                    k["expr"], avail | RESET_SELF_SIGNALS)

    # static per-actuator structure
    acts = sorted(per_act)
    plans = []  # (act_idx, W[jp], [(ev|None)], knot ts)
    knot_index: list[tuple[str, float]] = []
    for a in acts:
        knots = per_act[a]
        kt = np.asarray([k["t"] for k in knots])
        interps = [k["interp"] for k in knots]
        W = jp.asarray(interp_weights(kt, interps, T, dt))
        evs = [None if k["expr"] is None else expr_cache[(k["kf_index"], k["expr"])]
               for k in knots]
        plans.append((a, W, evs))
        knot_index += [(names[a], float(k["t"])) for k in knots]

    def _knot_values(obs0: jp.ndarray, sig: dict[str, jp.ndarray]) -> list[jp.ndarray]:
        """Unclipped knot values per plan, self signals bound per actuator."""
        out = []
        for a, _W, evs in plans:
            init = obs0[ctrl_idx[a]]
            s = dict(sig)
            s["ctrl_init_self"] = init
            s["ctrl_lo_self"] = jp.float32(lo[a])
            s["ctrl_hi_self"] = jp.float32(hi[a])
            out.append(jp.stack([init if ev is None else jp.asarray(ev(s), jp.float32)
                                 for ev in evs]))
        return out

    def tape_from_obs(obs0: jp.ndarray) -> jp.ndarray:
        sig = signals(obs0)
        vals = _knot_values(obs0, sig)
        cols = {}
        for (a, W, _evs), v in zip(plans, vals):
            cols[a] = W @ jp.clip(v, lo[a], hi[a])
        untouched = obs0[jp.asarray(ctrl_idx, jp.int32)]  # [nu] reset ctrl
        return jp.stack([cols.get(a, jp.broadcast_to(untouched[a], (T + 1,)))
                         for a in range(nu)], axis=1)

    def knots_from_obs(obs0: jp.ndarray) -> jp.ndarray:
        sig = signals(obs0)
        return jp.concatenate(_knot_values(obs0, sig)) if plans else jp.zeros((0,))

    covered = tuple(names[a] for a in acts)
    info = {
        "n_keyframes": len(program.get("keyframes", [])),
        "n_knots": len(knot_index),
        "covered": covered,
        "uncovered": tuple(n for n in names if n not in covered),
        "knot_times": {names[a]: [k["t"] for k in per_act[a]] for a in acts},
        "labels": [str(kf.get("label", "")) for kf in program.get("keyframes", [])],
    }
    return CompiledTape(tape_from_obs=tape_from_obs, knots_from_obs=knots_from_obs,
                        knot_index=tuple(knot_index), n_steps=T, nu=nu,
                        covered=covered, info=info)


# --------------------------------------------------------------------------------------------
# Validation (hard gate) and reporting (warnings)
# --------------------------------------------------------------------------------------------

def validate_motor_tape(env: Any, cand: dict[str, Any],
                        errors: list[str] | None = None) -> dict[str, Any] | None:
    """Structural + compile gate. Returns the normalized program dict or None (messages appended
    to `errors`). Never task judgment -- numeric quality goes to tape_report."""
    errs: list[str] = []

    def fail() -> None:
        if errors is not None:
            errors.extend(errs)

    if cand.get("mode") not in (None, "motor_tape"):
        errs.append(f"mode must be 'motor_tape', got {cand.get('mode')!r}")
        fail()
        return None
    keyframes = cand.get("keyframes")
    if not isinstance(keyframes, list) or not keyframes:
        errs.append("candidate has no 'keyframes' list")
        fail()
        return None

    episode_seconds = float(getattr(env.cfg, "episode_seconds", 0.0) or 0.0)
    names = set(all_actuator_names(env))
    prog: dict[str, Any] = {"mode": "motor_tape", "name": str(cand.get("name", "motor_tape"))}
    if isinstance(cand.get("signals"), dict) and cand["signals"]:
        prog["signals"] = {str(k): str(v) for k, v in cand["signals"].items()}
    if isinstance(cand.get("parameters"), dict) and cand["parameters"]:
        prog["parameters"] = cand["parameters"]
        for k, v in cand["parameters"].items():
            vv = v.get("value", v.get("init")) if isinstance(v, dict) else v
            if not isinstance(vv, (int, float)) or isinstance(vv, bool):
                errs.append(f"parameter {k!r}: value must be numeric, got {vv!r}")
    defaults = cand.get("defaults") or {}
    di = str(defaults.get("interp", "minjerk"))
    if di not in INTERP_KINDS:
        errs.append(f"defaults.interp must be one of {INTERP_KINDS}, got {di!r}")
    prog["defaults"] = {"interp": di}

    prev_t = -1.0
    kfs = []
    for i, kf in enumerate(keyframes):
        where = f"keyframe {i}"
        if not isinstance(kf, dict):
            errs.append(f"{where}: must be an object")
            continue
        t = kf.get("t")
        if not isinstance(t, (int, float)) or isinstance(t, bool):
            errs.append(f"{where}: 't' must be numeric, got {t!r}")
            continue
        t = float(t)
        if t < 0.0 or (episode_seconds > 0 and t > episode_seconds):
            errs.append(f"{where}: t={t} outside [0, {episode_seconds}] (the episode budget)")
        if t < prev_t:
            errs.append(f"{where}: t={t} decreases (keyframes must be ordered by t)")
        prev_t = max(prev_t, t)
        targets = kf.get("targets")
        if not isinstance(targets, dict) or not targets:
            errs.append(f"{where}: 'targets' must be a non-empty object")
            continue
        interp = str(kf.get("interp", di))
        if interp not in INTERP_KINDS:
            errs.append(f"{where}: interp must be one of {INTERP_KINDS}, got {interp!r}")
        for key in targets:
            if str(key) not in names and not _resolve_actuators(env, [str(key)]):
                errs.append(f"{where}: target key {key!r} matches no actuator or group")
        out = {"t": t, "targets": {str(k): str(v) for k, v in targets.items()},
               "interp": interp}
        if kf.get("label"):
            out["label"] = str(kf["label"])
        kfs.append(out)
    prog["keyframes"] = kfs
    if errs:
        fail()
        return None

    # per-actuator strictly-increasing knot times
    try:
        per_act = expand_keyframes(env, prog)
    except Exception as e:  # noqa: BLE001
        errs.append(f"keyframe expansion failed: {e}")
        fail()
        return None
    act_names = all_actuator_names(env)
    for a, knots in per_act.items():
        ts = [k["t"] for k in knots]
        if any(t1 <= t0 for t0, t1 in zip(ts, ts[1:])):
            errs.append(f"actuator {act_names[a]}: duplicate/decreasing knot times {ts} "
                        "(each actuator may appear at most once per instant)")
    if errs:
        fail()
        return None

    # hard compile gate: full compile + one tape evaluation on a real reset obs
    try:
        compiled = compile_tape(env, prog)
        obs0 = env.reset(jax.random.PRNGKey(0)).obs
        tape = np.asarray(compiled.tape_from_obs(obs0))
        if not np.all(np.isfinite(tape)):
            errs.append("compiled tape contains non-finite values on a real reset")
    except Exception as e:  # noqa: BLE001
        errs.append(f"compile failed: {e}")
    if errs:
        fail()
        return None
    return prog


def tape_report(env: Any, program: dict[str, Any], envs: int = 32, seed: int = 0) -> dict:
    """Numeric accounting over real randomized resets (warnings, never failures):
    coverage, out-of-range knot values, implied segment speeds vs the slew ceiling, end time."""
    compiled = compile_tape(env, program)
    reset = jax.jit(lambda k: jax.vmap(env.reset)(k))
    obs0 = reset(jax.random.split(jax.random.PRNGKey(seed), envs)).obs
    knots = np.asarray(jax.vmap(compiled.knots_from_obs)(obs0))  # [envs, n_knots]
    names = all_actuator_names(env)
    lo = np.asarray(env.ctrl_lo)
    hi = np.asarray(env.ctrl_hi)
    span = hi - lo
    idx_of = {n: i for i, n in enumerate(names)}
    dt = float(env.cfg.control_dt)
    slew = float(env.cfg.action_scale) / dt  # max ctrl-units per second per actuator

    out_of_range = []
    for j, (an, t) in enumerate(compiled.knot_index):
        a = idx_of[an]
        excess = np.maximum(knots[:, j] - hi[a], 0.0) + np.maximum(lo[a] - knots[:, j], 0.0)
        frac = float(np.mean(excess > 0.05 * span[a]))
        if frac > 0.0:
            out_of_range.append({"actuator": an, "t": t, "frac_beyond_5pct": round(frac, 3),
                                 "mean_excess": round(float(excess.mean()), 4)})
    # implied mean segment speed (on the mean tape values, clipped as compiled)
    fast = []
    j = 0
    per: dict[str, list[tuple[float, float]]] = {}
    for an, t in compiled.knot_index:
        a = idx_of[an]
        v = float(np.mean(np.clip(knots[:, j], lo[a], hi[a])))
        per.setdefault(an, []).append((t, v))
        j += 1
    for an, kv in per.items():
        for (t0, v0), (t1, v1) in zip(kv, kv[1:]):
            if t1 > t0:
                speed = abs(v1 - v0) / (t1 - t0)
                if speed > slew:
                    fast.append({"actuator": an, "segment": [t0, t1],
                                 "implied_speed": round(speed, 3), "slew_ceiling": round(slew, 3)})
    end_t = max((k["t"] for kf in program.get("keyframes", []) for k in [kf]), default=0.0)
    by_group: dict[str, list[str]] = {}
    for n in compiled.info["uncovered"]:
        by_group.setdefault(_semantic_group(n), []).append(n)
    return {
        "n_keyframes": compiled.info["n_keyframes"],
        "n_knots": compiled.info["n_knots"],
        "covered": len(compiled.covered),
        "uncovered_by_group": by_group,
        "out_of_range_knots": out_of_range,
        "over_slew_segments": fast,
        "last_keyframe_t": end_t,
        "episode_seconds": float(getattr(env.cfg, "episode_seconds", 0.0) or 0.0),
        "slew_ceiling_units_per_s": round(slew, 3),
    }


def render_tape_report(rep: dict) -> str:
    lines = [f"tape: {rep['n_keyframes']} keyframes, {rep['n_knots']} knots, "
             f"{rep['covered']} actuators covered; last keyframe at {rep['last_keyframe_t']}s "
             f"of {rep['episode_seconds']}s"]
    if rep["uncovered_by_group"]:
        lines.append("  uncovered (hold reset pose): "
                     + "; ".join(f"{g}: {', '.join(a)}" for g, a in
                                 sorted(rep["uncovered_by_group"].items())))
    for w in rep["out_of_range_knots"]:
        lines.append(f"  WARN out-of-range knot {w['actuator']}@{w['t']}s: beyond 5% of range in "
                     f"{w['frac_beyond_5pct']:.0%} of resets (mean excess {w['mean_excess']})")
    for w in rep["over_slew_segments"]:
        lines.append(f"  WARN over-slew {w['actuator']} {w['segment']}: implied "
                     f"{w['implied_speed']} > ceiling {w['slew_ceiling']} units/s "
                     "(will saturate into a max-rate ramp; arrival will be late)")
    return "\n".join(lines)


# --------------------------------------------------------------------------------------------
# Autopilot scoring (tape only, no networks)
# --------------------------------------------------------------------------------------------

def make_autopilot_rollout(env: Any, program: dict[str, Any], envs: int, rate: float = 1.0,
                           return_obs: bool = False) -> Callable:
    """rollout(rng) -> per-step eval trajectory [T, E, 6] under pure tape playback:
    a = clip((q_des(s) - ctrl) / action_scale, -1, 1), s advancing `rate` tape-steps per step.
    With return_obs=True: rollout(rng) -> (ev, obs0[E, obs], obs_end[E, obs])."""
    compiled = compile_tape(env, program)
    T = int(env.horizon)
    scale = float(env.cfg.action_scale)
    reset = jax.jit(lambda k: jax.vmap(env.reset)(k))
    step = jax.vmap(env.step)
    build = jax.vmap(compiled.tape_from_obs)

    def lookup(tape, s):  # tape [E, T+1, nu], s [E]
        i0 = jp.clip(jp.floor(s).astype(jp.int32), 0, T)
        i1 = jp.clip(i0 + 1, 0, T)
        f = (s - i0.astype(jp.float32))[:, None]
        g0 = jp.take_along_axis(tape, i0[:, None, None], axis=1)[:, 0]
        g1 = jp.take_along_axis(tape, i1[:, None, None], axis=1)[:, 0]
        return (1.0 - f) * g0 + f * g1

    def rollout(rng):
        st = reset(jax.random.split(rng, envs))
        obs0 = st.obs
        tape = build(st.obs)

        def body(carry, _):
            s, phase = carry
            q = lookup(tape, phase)
            a = jp.clip((q - s.data.ctrl) / scale, -1.0, 1.0)
            ns = step(s, a)
            return (ns, jp.minimum(phase + float(rate), float(T))), ns.metrics["eval"]

        (st_end, _p), ev = jax.lax.scan(body, (st, jp.zeros((envs,), jp.float32)), None, length=T)
        if return_obs:
            return ev, obs0, st_end.obs
        return ev

    return jax.jit(rollout)


def tape_grounding_report(env: Any, program: dict[str, Any], envs: int = 4,
                          seed: int = 0) -> dict:
    """Per-spawn numeric grounding evidence for the feedback loop: where the plan actually left
    the effector relative to the object, on a few real randomized resets. Mechanical measurement
    only -- interpretation stays with the LLM."""
    ev, obs0, obs_end = make_autopilot_rollout(env, program, envs,
                                               return_obs=True)(jax.random.PRNGKey(seed))
    evn = np.asarray(ev)
    o0 = np.asarray(obs0)
    o1 = np.asarray(obs_end)
    obs_idx = dict(raw_obs_entries(env)[0])
    rel = [f"obj_rel_{a}" for a in "xyz"]
    spawns = []
    for e in range(envs):
        spawns.append({
            "spawn_obj_rel": {k: round(float(o0[e, obs_idx[k]]), 4) for k in rel},
            "end_obj_rel": {k: round(float(o1[e, obs_idx[k]]), 4) for k in rel},
            "min_palm_obj_dist": round(float(evn[:, e, 0].min()), 4),
            "max_contacts": round(float(evn[:, e, 2].max()), 3),
            "lift_max": round(float(evn[:, e, 4].max()), 4),
        })
    return {"spawns": spawns,
            "contact_engagement": round(float(np.mean(evn[:, :, 2] >= 1.0)), 4)}


def render_grounding_report(rep: dict) -> str:
    lines = [f"GROUNDING (measured by playing your plan on {len(rep['spawns'])} real randomized "
             f"episodes; obj_rel = object position MINUS grasp-point position, so a plan that "
             f"reaches the object drives every obj_rel component toward 0):",
             f"  fraction of steps with real object contact: {rep['contact_engagement']:.1%}"]
    for i, s in enumerate(rep["spawns"]):
        sp = s["spawn_obj_rel"]
        en = s["end_obj_rel"]
        lines.append(
            f"  episode {i}: spawn obj_rel=({sp['obj_rel_x']:+.3f}, {sp['obj_rel_y']:+.3f}, "
            f"{sp['obj_rel_z']:+.3f}) -> end obj_rel=({en['obj_rel_x']:+.3f}, "
            f"{en['obj_rel_y']:+.3f}, {en['obj_rel_z']:+.3f}); min palm-object distance "
            f"{s['min_palm_obj_dist']:.3f} m; max simultaneous contacts {s['max_contacts']:.0f}; "
            f"max lift {s['lift_max']:.3f} m")
    return "\n".join(lines)


# --------------------------------------------------------------------------------------------
# Mechanical sign calibration (probed from the sim, never authored)
# --------------------------------------------------------------------------------------------

def probe_base_calibration(env: Any, n_steps: int = 8,
                           settle_steps: int = 40) -> dict[str, dict[str, float]]:
    """How each base actuator's ctrl actually moves the world-position observables.

    From the settled spawn state, drive ONE base actuator at full action for n_steps, hold for
    settle_steps (so the servo finishes tracking and the slope is steady-state, not mid-slew),
    and measure the change of every palm_pos_* / obj_rel_* / base_pos_* observable per ctrl unit
    actually moved. Purely mechanical (simulator probe) -- the injectable antidote to sign errors
    in authored position arithmetic. Returns {actuator: {observable: delta_per_ctrl_unit}}."""
    obs_idx = dict(raw_obs_entries(env)[0])
    watch = [n for n in obs_idx if n.startswith(("palm_pos_", "obj_rel_", "base_pos_"))]
    names = all_actuator_names(env)
    state0 = env.reset(jax.random.PRNGKey(0))
    o0 = np.asarray(state0.obs)
    step = jax.jit(env.step)
    hold = jp.zeros((int(env.action_size),), jp.float32)

    def _slopes(a: int, sign: float) -> dict[str, float] | None:
        action = hold.at[a].set(sign)
        st = state0
        for _ in range(int(n_steps)):
            st = step(st, action)
        for _ in range(int(settle_steps)):
            st = step(st, hold)
        o1 = np.asarray(st.obs)
        dctrl = float(o1[obs_idx[f"ctrl_{names[a]}"]] - o0[obs_idx[f"ctrl_{names[a]}"]])
        if abs(dctrl) < 1e-6:
            return None
        return {w: float(o1[obs_idx[w]] - o0[obs_idx[w]]) / dctrl for w in sorted(watch)}

    out: dict[str, dict[str, float]] = {}
    for a in [int(i) for i in getattr(env, "base_act_ids", [])]:
        # Probe BOTH directions and keep, per observable, the larger-magnitude slope: motion
        # toward an obstruction (e.g. descending into the table from spawn) truncates the
        # response, so the freer direction carries the true kinematic slope.
        pos = _slopes(a, 1.0)
        neg = _slopes(a, -1.0)
        cands = [s for s in (pos, neg) if s is not None]
        if not cands:
            continue
        deltas = {}
        for w in sorted(watch):
            d = max((s.get(w, 0.0) for s in cands), key=abs)
            if abs(d) > 5e-2:  # keep only observables this actuator meaningfully drives
                deltas[w] = round(d, 3)
        out[names[a]] = deltas
    return out


def render_calibration_block(cal: dict[str, dict[str, float]]) -> str:
    lines = ["SIGN CALIBRATION (probed mechanically from the robot at its spawn state: the "
             "measured change of each world observable PER +1.0 ctrl unit of each base "
             "actuator. Use these signs -- do not guess them):"]
    for act, deltas in cal.items():
        eff = ", ".join(f"{w} {d:+.3f}" for w, d in deltas.items())
        lines.append(f"  {act} +1.0 ctrl => {eff}")
    lines.append(
        "  HOW TO USE: to change an observable <w> by D using actuator <a>, change <a>'s ctrl "
        "by D / slope(a, w) from its start value. Example: if slope(a, obj_rel_k) is NEGATIVE, "
        "driving obj_rel_k toward 0 from a POSITIVE start means INCREASING <a>'s ctrl by "
        "obj_rel_k_start / |slope|; check every axis' sign against the table independently.")
    return "\n".join(lines)


def nearmiss_score(score: dict, lo: float, hi: float, engagement_scale: float = 0.01) -> float:
    """Proximity-WITHOUT-contact selection criterion (from a score_tape result).

    The gen-condition study showed open-loop graded score anti-predicts trainability: revisions
    selected by it make marginal open-loop contact, and policies trained on such tapes never form
    a grasp. This criterion instead rewards a plan whose closest approach (mean per-episode min
    grasp-point-to-object distance) lands INSIDE the near-miss band [lo, hi] metres, and
    penalizes any measured object contact toward zero (full penalty at
    engagement >= engagement_scale). Range [0, 1]; 1 = centered in the band with zero contact."""
    d = float(score["palm_obj_dist_min"])
    eng = float(score["contact_engagement"])
    mid, half = 0.5 * (lo + hi), max(0.5 * (hi - lo), 1e-6)
    prox = float(np.exp(-(((d - mid) / half) ** 2)))
    return prox * max(0.0, 1.0 - eng / max(engagement_scale, 1e-9))


def score_tape(env: Any, program: dict[str, Any], envs: int = 128, seed: int = 0,
               rate: float = 1.0, task: str = "lift") -> dict:
    """Autopilot rollout scored two ways: the contact-gated open-loop columns (comparable to
    prior_eval.score_program) AND the task eval fields + task_graded_objective (comparable to
    every trained arm's eval)."""
    ev = make_autopilot_rollout(env, program, envs, rate)(jax.random.PRNGKey(seed))
    evn = np.asarray(ev)  # [T, E, 6]
    contacts, lift, xy = evn[:, :, 2], evn[:, :, 4], evn[:, :, 5]
    in_c = contacts >= 1.0
    lifted = lift > 0.05
    notflung = xy <= 0.08
    grasp_lift = jp.asarray(in_c & lifted & notflung)
    fling = jp.asarray(lifted & ~in_c)
    out = {
        "contact_gated_success": float(sustained_bool(grasp_lift, 20)),
        "contact_conditioned_lift": float(np.mean(np.where(in_c & notflung, lift, 0.0))),
        "contact_engagement": float(np.mean(in_c)),
        "contacts_mean": float(np.mean(contacts)),
        "fling_fraction": float(sustained_bool(fling, 20)),
        "palm_obj_dist_min": float(np.mean(np.min(evn[:, :, 0], axis=0))),
        "saturation_frac": 0.0, "action_abs_mean": 0.0,
    }
    out["objective_score"] = float(candidate_score(out))
    es = _eval_summary(jp.asarray(ev))  # [E, 6]
    lift_thresh = 0.05
    esn = np.asarray(es)
    tape_eval = {
        "eval_success_rate": round(float(np.asarray(task_success(task, es)).mean()), 6),
        "eval_task_fitness": round(float(np.asarray(task_fitness(task, es)).mean()), 6),
        "eval_reach_rate": round(float((esn[:, 2] >= 1.0).mean()), 6),
        "eval_grasp_rate": round(float(((esn[:, 2] >= 1.0) & (esn[:, 3] >= 0.5)).mean()), 6),
        "eval_lift_reached_rate": round(float((esn[:, 4] >= lift_thresh).mean()), 6),
        "eval_lift_max": round(float(esn[:, 4].mean()), 6),
        "eval_summary": [round(float(x), 6) for x in esn.mean(axis=0)],
    }
    tape_eval["eval_graded_objective"] = round(task_graded_objective(task, tape_eval), 6)
    out["tape_eval"] = tape_eval
    return out

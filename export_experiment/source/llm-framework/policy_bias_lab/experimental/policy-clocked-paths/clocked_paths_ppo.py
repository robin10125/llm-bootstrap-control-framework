"""Policy-clocked action paths: learned switching over authored behavior segments (DESIGN.md).

The LLM authors WHAT (an ordered list of closed-loop behavior segments -- action paths -- plus
optional recovery edges and measurement hints); the learned policy owns WHEN (progression). The
policy's action space is the actuator residual PLUS a progression head:

  rate     -- scalar in [0, rate_max]; within-segment phase integrates
              u += rate * dt / est_seconds[seg]; at u >= 1 the segment advances. Monotone by
              construction, no argmax, no thresholds in the control loop; "hold" is rate ~ 0.
  recover  -- optional scalar; crossing its threshold jumps to the segment's declared fallback
              (only where an edge is authored; bounded per episode; cooldown).

Authored conditions are HINTS, not switches: `done_hint`/`abort_hint` values feed the actor and
critic as continuous features, an annealed imitation loss initializes the progression head toward
the authored decision, and a potential-based term on ladder progress phi = (seg + u) / n shapes
reward (provably cannot change the optimal policy).

Structural guarantees (task-agnostic): min dwell (no chatter), max dwell = slack * est_seconds
(forced advance -- a stall becomes the measured `forced_advance_frac`, never a dead run), bounded
recoveries. `progression = "autopilot"` drives advancement from the hints alone (dwell-filtered)
with the same guarantees -- the hardened-gates ablation arm and the imitation target's semantics.

Everything is fully on-policy: the progression head is an ordinary action with an exact stored
log-probability. Deterministic evaluation runs the complete hybrid (segments + policy), scored
with the SAME eval_summary columns and task_graded_objective as every other arm.

Task knowledge enters ONLY through the LLM-authored program; this module stays task-agnostic.
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
BOOTSTRAPPING = ROOT.parent / "bootstrapping"
if str(BOOTSTRAPPING) not in sys.path:
    sys.path.insert(0, str(BOOTSTRAPPING))

import flax.linen as nn
import jax
import jax.numpy as jp
import numpy as np
import optax

import ppo

from policy_bias_lab.freeform_priors import (
    _compile_channel,
    compile_expr,
    program_signal_fn,
    raw_obs_entries,
)
from policy_bias_lab.ppo_bias import (
    _eval_summary,
    _merge_eval_summaries,
    squashed_gaussian_logp,
)
from policy_bias_lab.tasks import task_fitness, task_graded_objective, task_success

PROGRESSIONS = ("learned", "autopilot")


# --------------------------------------------------------------------------------------------
# IR: clocked_paths programs
# --------------------------------------------------------------------------------------------

def convert_staged_program(program: dict[str, Any],
                           default_est_seconds: float = 2.0) -> dict[str, Any]:
    """Mechanical, task-agnostic conversion of a freeform_staged program to clocked_paths.

    stage k's channels become segment k's path; the NEXT stage's gate (the entry condition of the
    hand-off) becomes segment k's done_hint; authored `success` carries over. No recovery edges
    are synthesized -- only an author can declare those.
    """
    if program.get("mode") == "clocked_paths":
        return program
    if program.get("mode") != "freeform_staged":
        raise ValueError(f"cannot convert mode={program.get('mode')!r} to clocked_paths")
    stages = list(program.get("stages", []))
    segments = []
    for k, st in enumerate(stages):
        done = None
        if k + 1 < len(stages):
            done = stages[k + 1].get("gate")
        elif st.get("success"):
            done = st.get("success")
        segments.append({
            "name": str(st.get("name", f"segment_{k}")),
            "channels": list(st.get("channels", [])),
            "est_seconds": float(st.get("est_seconds", default_est_seconds)),
            "done_hint": done,
            "abort_hint": None,
            "recovery": None,
            "success": st.get("success"),
        })
    out = {k: v for k, v in program.items() if k not in ("mode", "stages")}
    out["mode"] = "clocked_paths"
    out["segments"] = segments
    return out


def validate_clocked_program(env: Any, program: dict[str, Any]) -> list[str]:
    """Structural validation only (compilation, names, edges, budgets) -- never task judgment."""
    errors: list[str] = []
    if program.get("mode") != "clocked_paths":
        return [f"mode must be 'clocked_paths', got {program.get('mode')!r}"]
    segments = list(program.get("segments", []))
    if not segments:
        return ["program has no segments"]
    names = [str(s.get("name", f"segment_{i}")) for i, s in enumerate(segments)]
    if len(set(names)) != len(names):
        errors.append(f"segment names not unique: {names}")
    try:
        signals, signal_names = program_signal_fn(env, program)
        expr_names = set(signal_names) | {"u"}
        obs_idx = dict(raw_obs_entries(env)[0])
    except Exception as e:  # authored signals failed to compile
        return [f"signals: {e}"]
    idx_of = {n: i for i, n in enumerate(names)}
    for i, seg in enumerate(segments):
        where = f"segment {i} ({names[i]})"
        if float(seg.get("est_seconds", 0.0) or 0.0) <= 0.0:
            errors.append(f"{where}: est_seconds must be > 0")
        if not seg.get("channels"):
            errors.append(f"{where}: no channels")
        for ch in seg.get("channels", []):
            try:
                _compile_channel(env, ch, expr_names, obs_idx)
            except Exception as e:
                errors.append(f"{where} channel {ch.get('actuators')}: {e}")
        for key in ("done_hint", "abort_hint", "success"):
            expr = seg.get(key)
            if expr is not None and str(expr).strip() != "":
                try:
                    compile_expr(str(expr), expr_names)
                except Exception as e:
                    errors.append(f"{where} {key}: {e}")
        rec = seg.get("recovery")
        if rec is not None:
            if str(rec) not in idx_of:
                errors.append(f"{where}: recovery target {rec!r} is not a segment name")
            elif idx_of[str(rec)] >= i:
                errors.append(f"{where}: recovery must target an EARLIER segment, got {rec!r}")
            if seg.get("abort_hint") is None:
                errors.append(f"{where}: recovery edge declared without an abort_hint")
    return errors


# --------------------------------------------------------------------------------------------
# Compiled runtime
# --------------------------------------------------------------------------------------------

def make_runtime(env: Any, program: dict[str, Any], cfg: "ClockedPPOConfig"):
    """Compile segments -> (step_eval, static arrays).

    step_eval(obs, seg, u) -> (segment_action[action_dim], done_val, abort_val, success_val)
    for a SINGLE env; callers vmap. Segment selection is a one-hot sum over the (small) segment
    list, indexed by the carried cursor -- there is no gate arithmetic anywhere in the loop.
    """
    action_dim = int(env.action_size)
    segments = list(program.get("segments", []))
    n = len(segments)
    signals, signal_names = program_signal_fn(env, program)
    expr_names = set(signal_names) | {"u"}
    obs_idx = dict(raw_obs_entries(env)[0])
    names = [str(s.get("name", f"segment_{i}")) for i, s in enumerate(segments)]
    idx_of = {nm: i for i, nm in enumerate(names)}

    compiled = []
    for seg in segments:
        chans = [_compile_channel(env, ch, expr_names, obs_idx)
                 for ch in seg.get("channels", [])]

        def _ev(expr, default):
            if expr is None or str(expr).strip() == "":
                return lambda sig, _d=default: jp.float32(_d)
            return compile_expr(str(expr), expr_names)

        compiled.append((chans, _ev(seg.get("done_hint"), -1.0),
                         _ev(seg.get("abort_hint"), -1.0), _ev(seg.get("success"), 0.0)))

    dt = float(cfg.control_dt)
    est_steps = jp.asarray(
        [max(float(s.get("est_seconds", cfg.default_est_seconds)) / dt, 1.0)
         for s in segments], dtype=jp.float32)
    max_dwell = jp.asarray(
        [int(round(float(cfg.dwell_slack) * float(e))) for e in est_steps], dtype=jp.int32)
    rec_target = jp.asarray(
        [idx_of[str(s["recovery"])] if s.get("recovery") is not None else i
         for i, s in enumerate(segments)], dtype=jp.int32)
    has_rec = jp.asarray(
        [1.0 if s.get("recovery") is not None else 0.0 for s in segments], dtype=jp.float32)
    any_recovery = bool(np.asarray(has_rec).sum() > 0)

    def step_eval(obs, seg, u):
        sig = signals(obs)
        sig = dict(sig)
        sig["u"] = u
        outs, dones, aborts, succs = [], [], [], []
        for chans, done_ev, abort_ev, succ_ev in compiled:
            out = jp.zeros((action_dim,), jp.float32)
            for idx, ev in chans:
                out = out.at[idx].add(jp.clip(ev(obs, sig), -1.0, 1.0))
            outs.append(jp.clip(out, -1.0, 1.0))
            dones.append(jp.asarray(done_ev(sig), jp.float32))
            aborts.append(jp.asarray(abort_ev(sig), jp.float32))
            succs.append(jp.asarray(succ_ev(sig), jp.float32))
        w = jax.nn.one_hot(seg, n, dtype=jp.float32)
        action = jp.tensordot(w, jp.stack(outs), axes=1)
        done = jp.dot(w, jp.stack(dones))
        abort = jp.dot(w, jp.stack(aborts))
        succ = jp.dot(w, jp.stack(succs))
        return action, done, abort, succ

    info = {"n_segments": n, "segment_names": names, "any_recovery": any_recovery}
    return step_eval, est_steps, max_dwell, rec_target, has_rec, info


# --------------------------------------------------------------------------------------------
# Config / networks
# --------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class ClockedPPOConfig:
    iters: int = 2000
    envs: int = 256
    eval_envs: int = 256
    fragment_steps: int = 100
    lr: float = 3e-4
    gamma: float = 0.99
    lam: float = 0.95
    hidden: tuple[int, ...] = (256, 256)
    ent_coef: float = 0.0
    target_train_seconds: float | None = None
    max_env_steps: int | None = None
    checkpoint_every: int = 0
    eval_every: int = 25
    base_reward_weight: float = 1.0
    warmup_compile: bool = True
    control_dt: float = 0.025
    default_est_seconds: float = 2.0

    progression: str = "learned"         # learned | autopilot (hardened-hints ablation)
    rate_max: float = 3.0                # max phase speed as a multiple of the authored pace
    min_dwell_steps: int = 5             # segment holds at least this many steps (no chatter)
    dwell_slack: float = 3.0             # max dwell = slack * est_steps -> forced advance
    hint_dwell_steps: int = 3            # autopilot: done_hint>0 must hold this long to advance
    recover_threshold: float = 0.5       # learned recover fires when squashed dim > threshold
    max_recoveries: int = 2              # per episode
    recover_cooldown_steps: int = 20
    residual_scale: float = 1.0          # env_action = clip(segment_action + scale * residual)
    potential_weight: float = 0.5        # potential-based shaping on phi = (seg + u) / n
    imitation_coef: float = 0.5          # progression head pulled to the authored hint decision
    imitation_anneal_iters: int | None = None  # default: iters


class Actor(nn.Module):
    out_dim: int
    hidden: tuple[int, ...] = (256, 256)

    @nn.compact
    def __call__(self, obs, feats):
        a = jp.concatenate([obs, feats], axis=-1)
        for h in self.hidden:
            a = nn.tanh(nn.Dense(h, kernel_init=nn.initializers.orthogonal(jp.sqrt(2.0)))(a))
        mean = nn.Dense(self.out_dim, kernel_init=nn.initializers.orthogonal(0.01))(a)
        log_std = self.param("log_std", nn.initializers.constant(-0.5), (self.out_dim,))
        return mean, log_std


class Critic(nn.Module):
    hidden: tuple[int, ...] = (256, 256)

    @nn.compact
    def __call__(self, obs, feats):
        v = jp.concatenate([obs, feats], axis=-1)
        for h in self.hidden:
            v = nn.tanh(nn.Dense(h, kernel_init=nn.initializers.orthogonal(jp.sqrt(2.0)))(v))
        return jp.squeeze(nn.Dense(1, kernel_init=nn.initializers.orthogonal(1.0))(v), -1)


def _feat_dim(n_segments: int) -> int:
    # seg one-hot + u + dwell_norm + tanh(done_hint) + tanh(abort_hint)
    return n_segments + 4


def _anneal(init: float, final: float, it: int, anneal_iters: int) -> float:
    if anneal_iters <= 0:
        return final
    frac = min(max(it / float(anneal_iters), 0.0), 1.0)
    return float(init + (final - init) * frac)


# --------------------------------------------------------------------------------------------
# Collection
# --------------------------------------------------------------------------------------------

def make_clocked_collect(*, env: Any, actor: Actor, critic: Critic, runtime, cfg: ClockedPPOConfig,
                         deterministic: bool):
    """collect(params, state, prog, knobs, key) -> (state, prog, traj, last_value, eval_summary)

    prog = (seg[E]i32, u[E]f32, dwell[E]i32, recovs[E]i32, cooldown[E]i32, hint_dwell[E]i32) --
    the progression state, carried across fragments exactly like the env state, reset at episode
    boundaries by the caller. knobs = [imitation_coef] (traced; annealing never recompiles).

    Traj layout (each [T, E, ...]):
      0 obs, 1 feats, 2 action, 3 logp, 4 value, 5 train_reward, 6 base_reward, 7 shaping,
      8 phi, 9 seg, 10 rate, 11 done_pos, 12 forced, 13 recovered, 14 imit_target,
      15 eval_metrics
    """
    step_fn = jax.vmap(env.step)
    action_dim = int(env.action_size)
    step_eval, est_steps, max_dwell, rec_target, has_rec, info = runtime
    n = int(info["n_segments"])
    any_rec = bool(info["any_recovery"])
    prog_dims = 1 + (1 if any_rec else 0)       # rate (+ recover)
    out_dim = action_dim + prog_dims
    learned = cfg.progression == "learned"

    def feats_fn(seg, u, dwell, done, abort):
        return jp.concatenate([
            jax.nn.one_hot(seg, n, dtype=jp.float32),
            u[..., None],
            (dwell.astype(jp.float32) / jp.take(max_dwell, seg).astype(jp.float32))[..., None],
            jp.tanh(done)[..., None],
            jp.tanh(abort)[..., None],
        ], axis=-1)

    def policy_step(params, obs, prog, key):
        seg, u, dwell, recovs, cooldown, hint_dwell = prog
        seg_action, done, abort, succ = jax.vmap(step_eval)(obs, seg, u)
        feats = feats_fn(seg, u, dwell, done, abort)
        mean, log_std = actor.apply(params["actor"], obs, feats)
        log_std = jp.clip(log_std, -5.0, 2.0)
        if deterministic:
            pre = mean
        else:
            pre = mean + jp.exp(log_std) * jax.random.normal(key, mean.shape)
        act = jp.tanh(pre)
        # The full action vector (residual + progression head) is on-policy: exact stored logp.
        logp = squashed_gaussian_logp(act, mean, log_std)
        value = critic.apply(params["critic"], obs, feats)

        residual = act[:, :action_dim]
        rate = (act[:, action_dim] + 1.0) * 0.5 * float(cfg.rate_max)
        env_action = jp.clip(seg_action + float(cfg.residual_scale) * residual, -1.0, 1.0)

        # ---- progression dynamics ----
        e = jp.take(est_steps, seg)
        du = (rate if learned else jp.ones_like(rate)) / e
        u_next = u + du
        dwell_next = dwell + 1
        hint_pos = (done > 0.0)
        hint_dwell_next = jp.where(hint_pos, hint_dwell + 1, 0)
        forced = dwell_next >= jp.take(max_dwell, seg)
        if learned:
            adv = ((u_next >= 1.0) & (dwell_next >= int(cfg.min_dwell_steps))) | forced
        else:
            adv = ((hint_dwell_next >= int(cfg.hint_dwell_steps))
                   & (dwell_next >= int(cfg.min_dwell_steps))) | forced
        can_rec = ((jp.take(has_rec, seg) > 0.0)
                   & (recovs < int(cfg.max_recoveries)) & (cooldown <= 0))
        if any_rec and learned:
            rec_sig = (act[:, action_dim + 1] + 1.0) * 0.5
            recover = can_rec & (rec_sig > float(cfg.recover_threshold))
        elif any_rec:
            recover = can_rec & (abort > 0.0)
        else:
            recover = jp.zeros_like(adv)
        adv = adv & ~recover
        seg_next = jp.where(recover, jp.take(rec_target, seg),
                            jp.where(adv, jp.minimum(seg + 1, n - 1), seg))
        switched = adv | recover
        u_next = jp.where(switched, 0.0, jp.minimum(u_next, 1.0))
        dwell_next = jp.where(switched, 0, dwell_next)
        hint_dwell_next = jp.where(switched, 0, hint_dwell_next)
        recovs_next = recovs + recover.astype(jp.int32)
        cooldown_next = jp.where(recover, int(cfg.recover_cooldown_steps),
                                 jp.maximum(cooldown - 1, 0))

        # imitation target for the progression head: the authored hint decision (pre-squash +-0.9)
        imit_parts = [jp.where(hint_pos, 0.9, -0.9)[:, None]]
        if any_rec:
            imit_parts.append(jp.where(abort > 0.0, 0.9, -0.9)[:, None])
        imit_target = jp.concatenate(imit_parts, axis=-1)

        phi_prev = (seg.astype(jp.float32) + u) / float(n)
        phi_next = (seg_next.astype(jp.float32) + u_next) / float(n)
        prog_next = (seg_next, u_next, dwell_next, recovs_next, cooldown_next, hint_dwell_next)
        rec = (env_action, act, logp, value, feats, rate, hint_pos, forced, recover, imit_target,
               phi_prev, phi_next, seg)
        return prog_next, rec

    def collect(params, state, prog, knobs, key):
        del knobs  # traced pass-through; imitation coef is applied in the update

        def body(carry, _t):
            state, prog, key = carry
            key, ak = jax.random.split(key)
            prog_next, rec = policy_step(params, state.obs, prog, ak)
            (env_action, act, logp, value, feats, rate, hint_pos, forced, recover, imit_target,
             phi_prev, phi_next, seg) = rec
            nstate = step_fn(state, env_action)
            shaping = float(cfg.potential_weight) * (float(cfg.gamma) * phi_next - phi_prev)
            train_reward = float(cfg.base_reward_weight) * nstate.reward + shaping
            return (nstate, prog_next, key), (
                state.obs, feats, act, logp, value, train_reward, nstate.reward, shaping,
                phi_prev, seg.astype(jp.float32), rate, hint_pos.astype(jp.float32),
                forced.astype(jp.float32), recover.astype(jp.float32), imit_target,
                nstate.metrics["eval"])

        (state, prog, key), traj = jax.lax.scan(
            body, (state, prog, key), jp.arange(int(cfg.fragment_steps)))
        seg, u, dwell, recovs, cooldown, hint_dwell = prog
        seg_action, done, abort, _succ = jax.vmap(step_eval)(state.obs, seg, u)
        feats = feats_fn(seg, u, dwell, done, abort)
        last_value = critic.apply(params["critic"], state.obs, feats)
        eval_summary = _eval_summary(traj[15])
        return state, prog, traj, last_value, eval_summary

    return jax.jit(collect), out_dim, prog_dims


# --------------------------------------------------------------------------------------------
# Update
# --------------------------------------------------------------------------------------------

def make_clocked_update(*, actor: Actor, critic: Critic, optimizer: Any, cfg: ClockedPPOConfig,
                        action_dim: int):

    def loss_fn(params, batch, imit_coef):
        obs, feats, action, old_logp, adv, ret, imit_target = batch
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        mean, log_std = actor.apply(params["actor"], obs, feats)
        log_std = jp.clip(log_std, -5.0, 2.0)
        logp = squashed_gaussian_logp(action, mean, log_std)
        ratio = jp.exp(logp - old_logp)
        pg_loss = -jp.minimum(ratio * adv, jp.clip(ratio, 0.8, 1.2) * adv).mean()
        value = critic.apply(params["critic"], obs, feats)
        v_loss = 0.5 * ((value - ret) ** 2).mean()
        entropy = ppo.gaussian_entropy(log_std).mean()
        # Imitation of the authored hint decision, on the progression head's pre-threshold mean:
        # initialization pressure only -- the coefficient anneals to zero.
        prog_mean = jp.tanh(mean[:, action_dim:])
        imit_loss = ((prog_mean - imit_target) ** 2).mean()
        loss = pg_loss + 0.5 * v_loss - float(cfg.ent_coef) * entropy + imit_coef * imit_loss
        return loss, {
            "pg_loss": pg_loss,
            "v_loss": v_loss,
            "imit_loss": imit_loss,
            "entropy": entropy,
            "approx_kl": (old_logp - logp).mean(),
        }

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    def update(params, opt_state, data, key, imit_coef):
        b = data[0].shape[0]
        num_minibatches = min(8, b)
        mb = max(1, b // num_minibatches)

        def epoch(carry, _):
            params, opt_state, key = carry
            key, pk = jax.random.split(key)
            perm = jax.random.permutation(pk, b)

            def minibatch(carry, idx):
                params, opt_state = carry
                sl = jax.lax.dynamic_slice_in_dim(perm, idx * mb, mb)
                batch = tuple(jp.take(x, sl, axis=0) for x in data)
                (_loss, metrics), grads = grad_fn(params, batch, imit_coef)
                updates, opt_state = optimizer.update(grads, opt_state, params)
                params = optax.apply_updates(params, updates)
                return (params, opt_state), metrics

            (params, opt_state), metrics = jax.lax.scan(
                minibatch, (params, opt_state), jp.arange(num_minibatches))
            return (params, opt_state, key), metrics

        (params, opt_state, _key), metrics = jax.lax.scan(
            epoch, (params, opt_state, key), None, length=4)
        return params, opt_state, jax.tree_util.tree_map(lambda x: x.mean(), metrics)

    return jax.jit(update)


def _data_from_traj(traj, last_value, gamma: float, lam: float):
    obs, feats, action, logp, value, train_reward = traj[:6]
    imit_target = traj[14]
    adv, ret = ppo.compute_gae(train_reward, value, last_value, gamma, lam)
    flat = lambda x: x.reshape((-1,) + x.shape[2:])
    return (flat(obs), flat(feats), flat(action), flat(logp), flat(adv), flat(ret),
            flat(imit_target))


# --------------------------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------------------------

def _reset_prog(envs: int):
    z_i = jp.zeros((envs,), dtype=jp.int32)
    return (z_i, jp.zeros((envs,), dtype=jp.float32), z_i, z_i, z_i, z_i)


def train_clocked_ppo(*, env: Any, program: dict[str, Any], task: str, seed: int,
                      cfg: ClockedPPOConfig, out_dir: Path | None = None):
    if cfg.progression not in PROGRESSIONS:
        raise ValueError(f"unknown progression={cfg.progression!r} (expected {PROGRESSIONS})")
    if int(env.horizon) % int(cfg.fragment_steps) != 0:
        raise ValueError(
            f"fragment_steps={cfg.fragment_steps} must divide env.horizon={env.horizon}")
    errors = validate_clocked_program(env, program)
    if errors:
        raise ValueError("invalid clocked_paths program:\n  " + "\n  ".join(errors))
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    action_dim = int(env.action_size)
    runtime = make_runtime(env, program, cfg)
    n_seg = int(runtime[5]["n_segments"])
    out_dim = action_dim + 1 + (1 if runtime[5]["any_recovery"] else 0)

    key = jax.random.PRNGKey(seed)
    key, ak, ck = jax.random.split(key, 3)
    actor = Actor(out_dim=out_dim, hidden=cfg.hidden)
    critic = Critic(hidden=cfg.hidden)
    params = {
        "actor": actor.init(ak, jp.zeros((1, env.obs_size)), jp.zeros((1, _feat_dim(n_seg)))),
        "critic": critic.init(ck, jp.zeros((1, env.obs_size)), jp.zeros((1, _feat_dim(n_seg)))),
    }
    optimizer = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(cfg.lr))
    opt_state = optimizer.init(params)
    reset = jax.jit(lambda keys: jax.vmap(env.reset)(keys))
    collect, _od, _pd = make_clocked_collect(env=env, actor=actor, critic=critic, runtime=runtime,
                                             cfg=cfg, deterministic=False)
    update = make_clocked_update(actor=actor, critic=critic, optimizer=optimizer, cfg=cfg,
                                 action_dim=action_dim)

    fragments_per_episode = int(env.horizon) // int(cfg.fragment_steps)
    imit_anneal = (cfg.imitation_anneal_iters if cfg.imitation_anneal_iters is not None
                   else cfg.iters)

    if cfg.warmup_compile:
        state0 = reset(jax.random.split(jax.random.PRNGKey(seed + 99), cfg.envs))
        _s, _p, traj, last_value, _es = collect(
            params, state0, _reset_prog(cfg.envs), jp.zeros((1,), jp.float32),
            jax.random.PRNGKey(seed + 100))
        last_value.block_until_ready()
        data = _data_from_traj(traj, last_value, cfg.gamma, cfg.lam)
        p2, _o2, _m2 = update(params, opt_state, data, jax.random.PRNGKey(seed + 101),
                              jp.float32(cfg.imitation_coef))
        jax.block_until_ready(p2)

    key, rk = jax.random.split(key)
    state = reset(jax.random.split(rk, cfg.envs))
    prog = _reset_prog(cfg.envs)

    rows: list[dict[str, Any]] = []
    best_score = -1.0e30
    best_iter = -1
    best_params = params
    start = time.monotonic()

    for it in range(int(cfg.iters)):
        elapsed = time.monotonic() - start
        if cfg.target_train_seconds is not None and it > 0 and elapsed >= cfg.target_train_seconds:
            break
        env_steps = (it + 1) * int(cfg.envs) * int(cfg.fragment_steps)
        if cfg.max_env_steps is not None and it > 0 and env_steps >= int(cfg.max_env_steps):
            break

        imit_coef = _anneal(cfg.imitation_coef, 0.0, it, imit_anneal)
        fragment_index = it % fragments_per_episode

        key, ck2, uk = jax.random.split(key, 3)
        state, prog, traj, last_value, eval_summary = collect(
            params, state, prog, jp.zeros((1,), jp.float32), ck2)
        data = _data_from_traj(traj, last_value, cfg.gamma, cfg.lam)
        params, opt_state, metrics = update(params, opt_state, data, uk, jp.float32(imit_coef))
        jax.block_until_ready(params)

        if fragment_index == fragments_per_episode - 1:
            key, rk = jax.random.split(key)
            state = reset(jax.random.split(rk, cfg.envs))
            prog = _reset_prog(cfg.envs)

        row = _training_row(it=it, env_steps=env_steps, fragment_index=fragment_index, task=task,
                            traj=traj, eval_summary=eval_summary, metrics=metrics,
                            elapsed=time.monotonic() - start, imit_coef=imit_coef, n_seg=n_seg,
                            cfg=cfg)
        rows.append(row)

        if cfg.eval_every > 0 and (it == 0 or (it + 1) % int(cfg.eval_every) == 0):
            ev = evaluate_clocked_policy(env=env, params=params, program=program, task=task,
                                         seed=seed + 10_000 + it, cfg=cfg)
            score = float(ev["eval_task_fitness"])
            if score > best_score:
                best_score = float(score)
                best_iter = int(it)
                best_params = params
                if out_dir is not None:
                    with (out_dir / "best_params.pkl").open("wb") as f:
                        pickle.dump(jax.device_get(best_params), f)
            rows[-1]["eval_objective"] = round(float(score), 6)
            rows[-1]["best_objective"] = round(float(best_score), 6)
            rows[-1]["eval_success"] = round(float(ev["eval_success_rate"]), 6)
            rows[-1]["eval_graded"] = round(float(ev["eval_graded_objective"]), 6)

        # Written AFTER the eval block so eval_objective/best_objective land in the file too.
        if out_dir is not None:
            with (out_dir / "metrics.jsonl").open("a") as f:
                f.write(json.dumps(rows[-1]) + "\n")

        if out_dir is not None and cfg.checkpoint_every > 0 and (it + 1) % cfg.checkpoint_every == 0:
            with (out_dir / f"params_iter{it + 1:05d}.pkl").open("wb") as f:
                pickle.dump(jax.device_get(params), f)

    return params, rows, best_params, best_score, best_iter


# --------------------------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------------------------

def evaluate_clocked_policy(*, env: Any, params: Any, program: dict[str, Any], task: str,
                            seed: int, cfg: ClockedPPOConfig) -> dict[str, Any]:
    """Deterministic full-episode evaluation of the COMPLETE hybrid (segments + policy).

    The segment machinery ships in the deployed controller by design (minimality is explicitly
    not a goal here); scored with the SAME eval_summary columns and task_graded_objective as the
    ppo_bias baseline and the alternative-methods arms.
    """
    eval_cfg = ClockedPPOConfig(**{**cfg.__dict__, "envs": int(cfg.eval_envs)})
    runtime = make_runtime(env, program, eval_cfg)
    n_seg = int(runtime[5]["n_segments"])
    out_dim = int(env.action_size) + 1 + (1 if runtime[5]["any_recovery"] else 0)
    actor = Actor(out_dim=out_dim, hidden=cfg.hidden)
    critic = Critic(hidden=cfg.hidden)
    collect, _od, _pd = make_clocked_collect(env=env, actor=actor, critic=critic, runtime=runtime,
                                             cfg=eval_cfg, deterministic=True)
    reset = jax.jit(lambda keys: jax.vmap(env.reset)(keys))
    state = reset(jax.random.split(jax.random.PRNGKey(seed), eval_cfg.envs))
    prog = _reset_prog(eval_cfg.envs)
    traj_parts, summary_parts = [], []
    for i in range(int(env.horizon) // int(eval_cfg.fragment_steps)):
        state, prog, traj, _lv, eval_summary = collect(
            params, state, prog, jp.zeros((1,), jp.float32), jax.random.PRNGKey(seed + 1 + i))
        traj_parts.append(jax.device_get(traj))
        summary_parts.append(jax.device_get(eval_summary))
    merged = _merge_eval_summaries(summary_parts)
    es = jp.asarray(merged)
    success_rate = float(task_success(task, es).mean())
    fitness = float(task_fitness(task, es).mean())
    base_ret = sum(float(t[6].sum(axis=0).mean()) for t in traj_parts)
    shaping_ret = sum(float(t[7].sum(axis=0).mean()) for t in traj_parts)
    train_ret = sum(float(t[5].sum(axis=0).mean()) for t in traj_parts)
    seg_all = np.concatenate([np.asarray(t[9]) for t in traj_parts], axis=0)
    occupancy = [round(float((seg_all == k).mean()), 6) for k in range(n_seg)]
    lift_thresh = 0.05
    out = {
        "eval_success_rate": round(success_rate, 6),
        "eval_task_fitness": round(fitness, 6),
        "eval_reach_rate": round(float((es[:, 2] >= 1.0).mean()), 6),
        "eval_grasp_rate": round(float(((es[:, 2] >= 1.0) & (es[:, 3] >= 0.5)).mean()), 6),
        "eval_grasp_lift_rate": round(float(((es[:, 2] >= 1.0) & (es[:, 3] >= 0.5)
                                             & (es[:, 4] >= lift_thresh)).mean()), 6),
        "eval_lift_reached_rate": round(float((es[:, 4] >= lift_thresh).mean()), 6),
        "eval_lift_max": round(float(es[:, 4].mean()), 6),
        "eval_train_return": round(train_ret, 6),
        "eval_base_return": round(base_ret, 6),
        "eval_shaping_return": round(shaping_ret, 6),
        "eval_seg_occupancy": occupancy,
        "eval_forced_advance_frac": round(
            float(np.mean([np.asarray(t[12]).mean() for t in traj_parts])), 6),
        "eval_recover_frac": round(
            float(np.mean([np.asarray(t[13]).mean() for t in traj_parts])), 6),
        "eval_summary": [round(float(x), 6) for x in es.mean(axis=0)],
    }
    out["eval_graded_objective"] = round(task_graded_objective(task, out), 6)
    return out


def _training_row(*, it, env_steps, fragment_index, task, traj, eval_summary, metrics, elapsed,
                  imit_coef, n_seg, cfg: ClockedPPOConfig) -> dict[str, Any]:
    train_reward, base_reward, shaping = traj[5], traj[6], traj[7]
    seg, rate, hint_pos, forced, recovered = traj[9], traj[10], traj[11], traj[12], traj[13]
    success = task_success(task, eval_summary).mean()
    seg_np = np.asarray(seg)
    rate_np = np.asarray(rate)
    hint_np = np.asarray(hint_pos)
    # agreement: progression head pushes (rate above half speed) exactly when the hint says done
    agree = float(((rate_np > 0.5 * float(cfg.rate_max)) == (hint_np > 0.5)).mean())
    row = {
        "iter": int(it),
        "env_steps": int(env_steps),
        "fragment_index": int(fragment_index),
        "task": task,
        "progression": cfg.progression,
        "fragment_return": round(float(train_reward.sum(axis=0).mean()), 6),
        "base_return": round(float(base_reward.sum(axis=0).mean()), 6),
        "shaping_return": round(float(shaping.sum(axis=0).mean()), 6),
        "success": round(float(success), 6),
        "seg_mean": round(float(seg_np.mean()), 6),
        "seg_occupancy": [round(float((seg_np == k).mean()), 6) for k in range(n_seg)],
        "rate_mean": round(float(rate_np.mean()), 6),
        "hint_agreement": round(agree, 6),
        "forced_advance_frac": round(float(np.asarray(forced).mean()), 6),
        "recover_frac": round(float(np.asarray(recovered).mean()), 6),
        "imit_coef": round(float(imit_coef), 6),
        "eval_summary": [round(float(x), 6) for x in eval_summary.mean(axis=0)],
        "pg_loss": round(float(metrics["pg_loss"]), 6),
        "v_loss": round(float(metrics["v_loss"]), 6),
        "imit_loss": round(float(metrics["imit_loss"]), 6),
        "entropy": round(float(metrics["entropy"]), 6),
        "approx_kl": round(float(metrics["approx_kl"]), 6),
        "elapsed_seconds": round(float(elapsed), 3),
    }
    return row

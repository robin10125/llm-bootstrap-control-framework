"""PPO over an LLM-authored motor tape: cerebellar correction heads on a feedforward plan.

The plan (motor_tape.compile_tape) is a per-episode command tape q_des[T+1, nu] built ONCE from
the reset observation. The policy is the corrective system; its action vector factors into up to
three heads, each individually removable for ablations:

  residual    [nu]  -- efference-copy residual: added to the tape's tracking action each step.
  rate        [1]   -- time-warp: playback advances `rate` tape-steps per control step
                       (rate in [rate_lo, rate_hi]); imitation-annealed toward 1.0 (authored pace).
  modulation  [nu]  -- OPTIONAL plan bender: an offset (bounded fraction of each actuator's
                       half-range) added to the tape COMMAND before the tracking law, i.e. it
                       reshapes the plan rather than perturbing execution. Smoothness-regularized.
                       With use_modulation=False the head does not exist -- the network, logp,
                       and loss are bit-identical to the no-modulation ablation.

Efference copy: actor AND critic receive, besides obs, the plan features -- normalized tracking
error (q_des(s) - ctrl), the same at lookahead offsets (upcoming plan commands), phase s/T and
time-to-end. The corrective heads therefore see what the plan is ABOUT to command, not just the
current sensory state -- the piece a reactive residual cannot have.

Playback composition per step (policy_step):
  feats   = plan features at current phase s
  act     = tanh(mean + sigma * eps)            (single squashed Gaussian; exact logp)
  s'      = min(s + rate, T)
  q       = lerp(tape, s');  q_eff = clip(q + mod_scale * halfrange * mod, lo, hi)
  a_ff    = clip((q_eff - ctrl) / action_scale, -1, 1)     (one-step-exact tracking law)
  action  = clip(a_ff + residual_scale * residual, -1, 1)

Shaping: potential-based term on phi = s/T (cannot change the optimal policy). Evaluation is
deterministic full-episode of the complete hybrid (tape + heads), scored with the SAME
eval_summary columns and task_graded_objective as every other arm.

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

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for p in (str(HERE), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import flax.linen as nn
import jax
import jax.numpy as jp
import numpy as np
import optax

from experiment_runtime import ppo

from motor_tape import CompiledTape, compile_tape
from policy_bias_lab.freeform_priors import all_actuator_names, raw_obs_entries
from policy_bias_lab.tasks import task_fitness, task_graded_objective, task_success
from policy_bias_lab.training.fragmented_ppo import (
    _eval_summary,
    _merge_eval_summaries,
    squashed_gaussian_logp,
)


@dataclass(frozen=True)
class MotorTapePPOConfig:
    iters: int = 2000
    envs: int = 256
    eval_envs: int = 128            # clocked eval at 256 envs OOMed on the 15GB machine
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
    # corrective heads (each removable)
    use_residual: bool = True
    use_rate: bool = True
    use_modulation: bool = False
    residual_scale: float = 1.0
    rate_lo: float = 0.0
    rate_hi: float = 2.0
    mod_scale: float = 0.15          # max plan bend as a fraction of each actuator's half-range
    mod_smooth_coef: float = 1e-2    # ||mod_mean(t) - mod_mean(t+1)||^2
    mod_l2_coef: float = 1e-3
    # efference-copy features
    lookahead_deltas: tuple[float, ...] = (0.1, 0.25, 0.5, 1.0)  # seconds of PLAN time
    # shaping / init
    potential_weight: float = 0.5    # potential on phi = s/T
    imitation_coef: float = 0.5      # rate head pulled toward 1.0 (authored pace), annealed
    imitation_anneal_iters: int | None = None  # default: iters
    # Grasp-gated lift income (contact-harm mitigation, reward side). When set, the caller must
    # build the env with its lift-income terms ZEROED (w_lift=w_lift_pot=w_lift_hold=w_success=0)
    # and this dict carries the original weights; the collect loop re-adds those terms computed
    # from the per-step eval vectors, multiplied by clip(closure/closure_target, 0, 1) -- so lift
    # income only pays in proportion to a formed grasp, deleting the marginal-contact
    # (graze-and-scoop) optimum. Keys: w_lift, w_lift_pot, w_lift_hold, w_success, lift_target,
    # success_height, contact_target, fling_xy_thresh, pbrs_gamma, closure_target.
    grasp_gated_lift: dict | None = None
    # Feedforward handoff (contact-harm mitigation, authority side). When handoff_hi > 0, the
    # tape's tracking action a_ff is faded linearly to 0 as the grasp-point-to-object distance
    # ||obj_rel|| falls from handoff_hi to handoff_lo: transport stays tape-driven, but first
    # contact belongs entirely to the learned residual. Plan features (efference copy) and the
    # phase potential are NOT masked -- the plan stays visible to actor and critic throughout.
    handoff_lo: float = 0.0
    handoff_hi: float = 0.0


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


def action_layout(cfg: MotorTapePPOConfig, nu: int) -> dict[str, Any]:
    """Fixed head order [residual | rate | modulation]; slices are None for absent heads."""
    off = 0
    res = rate = mod = None
    if cfg.use_residual:
        res = slice(off, off + nu)
        off += nu
    if cfg.use_rate:
        rate = off
        off += 1
    if cfg.use_modulation:
        mod = slice(off, off + nu)
        off += nu
    if off == 0:
        raise ValueError("at least one corrective head must be enabled")
    return {"residual": res, "rate": rate, "modulation": mod, "out_dim": off}


def feat_dim(cfg: MotorTapePPOConfig, nu: int) -> int:
    return nu * (1 + len(cfg.lookahead_deltas)) + 2


def rate1_tanh(cfg: MotorTapePPOConfig) -> float:
    """tanh-space value of the rate head at which the decoded rate is exactly 1.0."""
    if cfg.rate_hi <= cfg.rate_lo:
        raise ValueError("rate_hi must exceed rate_lo")
    return float(np.clip(2.0 * (1.0 - cfg.rate_lo) / (cfg.rate_hi - cfg.rate_lo) - 1.0,
                         -0.999, 0.999))


def _anneal(init: float, final: float, it: int, anneal_iters: int) -> float:
    if anneal_iters <= 0:
        return final
    frac = min(max(it / float(anneal_iters), 0.0), 1.0)
    return float(init + (final - init) * frac)


# --------------------------------------------------------------------------------------------
# Collection
# --------------------------------------------------------------------------------------------

def make_gated_lift_fn(gg: dict):
    """(prev_eval [E,6], eval [E,6]) -> grasp-gated lift reward [E].

    Re-implements the env's four lift-income terms (r_lift_pot, r_lift, r_lift_hold, r_success --
    see MjxEnv._builtin_reward) from the eval vectors alone, multiplied by a closure gate
    clip(closure / closure_target, 0, 1). Exactness against the env's own arithmetic is verified
    by test_grasp_gated_matches_env_lift_terms. Eval layout: [0] palm_obj_dist
    [1] min_finger_dist [2] n_contacts [3] closure [4] lift [5] obj_xy_disp."""
    g = float(gg["pbrs_gamma"])

    def fn(prev_e: jp.ndarray, e: jp.ndarray) -> jp.ndarray:
        lift, prev_lift = e[..., 4], prev_e[..., 4]
        contact_gate = jp.clip(e[..., 2] / float(gg["contact_target"]), 0.0, 1.0)
        in_contact = (e[..., 2] >= 1.0).astype(jp.float32)
        not_flung = jp.clip((float(gg["fling_xy_thresh"]) - e[..., 5])
                            / (0.5 * float(gg["fling_xy_thresh"]) + 1e-6), 0.0, 1.0)
        lift_frac = jp.clip(lift / float(gg["success_height"]), 0.0, 1.0)
        prev_lift_frac = jp.clip(prev_lift / float(gg["success_height"]), 0.0, 1.0)
        r_lift_pot = float(gg["w_lift_pot"]) * contact_gate * not_flung * (g * lift - prev_lift)
        r_lift = (float(gg["w_lift"]) * jp.clip(lift, 0.0, float(gg["lift_target"]))
                  * contact_gate * not_flung)
        r_lift_hold = (float(gg["w_lift_hold"]) * contact_gate * not_flung
                       * lift_frac * prev_lift_frac)
        r_success = (float(gg["w_success"])
                     * jp.where(lift > float(gg["success_height"]), 1.0, 0.0)
                     * in_contact * not_flung)
        ct = float(gg["closure_target"])
        # closure_target <= 0 means "no gate" (recovers the env's own lift terms exactly --
        # used by the exactness unit test)
        grasp_gate = jp.clip(e[..., 3] / ct, 0.0, 1.0) if ct > 0 else jp.ones_like(lift)
        return grasp_gate * (r_lift_pot + r_lift + r_lift_hold + r_success)

    return fn


def make_tape_collect(*, env: Any, actor: Actor, critic: Critic, compiled: CompiledTape,
                      cfg: MotorTapePPOConfig, deterministic: bool):
    """collect(params, state, s, tape, key) -> (state, s, traj, last_value, eval_summary).

    Carried per env across fragments: phase s [E] f32 and the tape [E, T+1, nu] (static between
    Python-level episode resets -- the caller rebuilds it from the fresh reset obs; env.step never
    auto-resets mid-fragment, done fires only at horizon).

    Traj layout ([T, E, ...]): 0 obs, 1 feats, 2 action, 3 logp, 4 value, 5 train_reward,
    6 base_reward, 7 shaping, 8 phase, 9 rate, 10 eval_metrics, 11 gated_lift_reward
    """
    step_fn = jax.vmap(env.step)
    nu = int(env.action_size)
    T = int(env.horizon)
    dt = float(env.cfg.control_dt)
    scale = float(env.cfg.action_scale)
    lay = action_layout(cfg, nu)
    obs_idx = dict(raw_obs_entries(env)[0])
    ctrl_idx = jp.asarray([obs_idx[f"ctrl_{n}"] for n in all_actuator_names(env)], jp.int32)
    rel_idx = jp.asarray([obs_idx[f"obj_rel_{a}"] for a in "xyz"], jp.int32)
    halfrange = jp.maximum(0.5 * (env.ctrl_hi - env.ctrl_lo), 1e-6)
    delta_steps = [int(round(d / dt)) for d in cfg.lookahead_deltas]
    gated_lift = make_gated_lift_fn(cfg.grasp_gated_lift) if cfg.grasp_gated_lift else None
    if cfg.handoff_hi > 0 and not (cfg.handoff_hi > cfg.handoff_lo >= 0):
        raise ValueError("handoff requires handoff_hi > handoff_lo >= 0")

    def lookup(tape, s):
        """tape [E, T+1, nu], s [E] fractional tape-step -> command [E, nu] (linear lookup)."""
        i0 = jp.clip(jp.floor(s).astype(jp.int32), 0, T)
        i1 = jp.clip(i0 + 1, 0, T)
        f = (s - i0.astype(jp.float32))[:, None]
        g0 = jp.take_along_axis(tape, i0[:, None, None], axis=1)[:, 0]
        g1 = jp.take_along_axis(tape, i1[:, None, None], axis=1)[:, 0]
        return (1.0 - f) * g0 + f * g1

    def feats_fn(obs, s, tape):
        """Efference copy: normalized tracking error now and at plan lookaheads, phase, time-left.
        Lookahead offsets are in TAPE steps (what the plan will command), clamped to hold-last."""
        ctrl = obs[:, ctrl_idx]
        blocks = []
        for ds in (0, *delta_steps):
            q = lookup(tape, jp.minimum(s + float(ds), float(T)))
            blocks.append(jp.tanh((q - ctrl) / halfrange))
        blocks.append((s / float(T))[:, None])
        blocks.append(((float(T) - s) / float(T))[:, None])
        return jp.concatenate(blocks, axis=-1)

    def policy_step(params, obs, s, tape, key):
        feats = feats_fn(obs, s, tape)
        mean, log_std = actor.apply(params["actor"], obs, feats)
        log_std = jp.clip(log_std, -5.0, 2.0)
        pre = mean if deterministic else mean + jp.exp(log_std) * jax.random.normal(key, mean.shape)
        act = jp.tanh(pre)
        logp = squashed_gaussian_logp(act, mean, log_std)
        value = critic.apply(params["critic"], obs, feats)

        if lay["rate"] is not None:
            rate = cfg.rate_lo + (act[:, lay["rate"]] + 1.0) * 0.5 * (cfg.rate_hi - cfg.rate_lo)
        else:
            rate = jp.ones_like(s)
        s_next = jp.minimum(s + rate, float(T))
        q = lookup(tape, s_next)
        if lay["modulation"] is not None:
            q = jp.clip(q + float(cfg.mod_scale) * halfrange * act[:, lay["modulation"]],
                        env.ctrl_lo, env.ctrl_hi)
        a_ff = jp.clip((q - obs[:, ctrl_idx]) / scale, -1.0, 1.0)
        if cfg.handoff_hi > 0:
            # feedforward handoff: the tape's tracking action fades to 0 as the grasp point
            # nears the object; features/phase are untouched (the plan stays visible).
            dist = jp.linalg.norm(obs[:, rel_idx], axis=-1)
            w_ff = jp.clip((dist - float(cfg.handoff_lo))
                           / (float(cfg.handoff_hi) - float(cfg.handoff_lo)), 0.0, 1.0)
            a_ff = a_ff * w_ff[:, None]
        if lay["residual"] is not None:
            env_action = jp.clip(a_ff + float(cfg.residual_scale) * act[:, lay["residual"]],
                                 -1.0, 1.0)
        else:
            env_action = a_ff
        phi_prev = s / float(T)
        phi_next = s_next / float(T)
        return s_next, (env_action, act, logp, value, feats, rate, phi_prev, phi_next)

    def collect(params, state, s, tape, key):
        def body(carry, _t):
            state, s, key = carry
            key, ak = jax.random.split(key)
            s_next, (env_action, act, logp, value, feats, rate, phi_prev, phi_next) = \
                policy_step(params, state.obs, s, tape, ak)
            nstate = step_fn(state, env_action)
            shaping = float(cfg.potential_weight) * (float(cfg.gamma) * phi_next - phi_prev)
            if gated_lift is not None:
                gl = gated_lift(state.metrics["eval"], nstate.metrics["eval"])
            else:
                gl = jp.zeros_like(nstate.reward)
            train_reward = float(cfg.base_reward_weight) * nstate.reward + shaping + gl
            return (nstate, s_next, key), (
                state.obs, feats, act, logp, value, train_reward, nstate.reward, shaping,
                s, rate, nstate.metrics["eval"], gl)

        (state, s, key), traj = jax.lax.scan(
            body, (state, s, key), jp.arange(int(cfg.fragment_steps)))
        feats = feats_fn(state.obs, s, tape)
        last_value = critic.apply(params["critic"], state.obs, feats)
        eval_summary = _eval_summary(traj[10])
        return state, s, traj, last_value, eval_summary

    return jax.jit(collect)


# --------------------------------------------------------------------------------------------
# Update
# --------------------------------------------------------------------------------------------

def _data_from_traj(traj, last_value, cfg: MotorTapePPOConfig):
    obs, feats, action, logp, value, train_reward = traj[:6]
    adv, ret = ppo.compute_gae(train_reward, value, last_value, cfg.gamma, cfg.lam)
    flat = lambda x: x.reshape((-1,) + x.shape[2:])
    data = [flat(obs), flat(feats), flat(action), flat(logp), flat(adv), flat(ret)]
    if cfg.use_modulation:
        # adjacent (t, t+1) pairs for the modulation smoothness term; the final step of the
        # fragment has no in-fragment successor, so its pair weight is 0. Fragments never
        # straddle an episode boundary (resets happen only between fragments), so within-fragment
        # adjacency is always same-episode.
        obs2 = jp.concatenate([obs[1:], obs[-1:]], axis=0)
        feats2 = jp.concatenate([feats[1:], feats[-1:]], axis=0)
        pairw = jp.ones(obs.shape[:2], jp.float32).at[-1].set(0.0)
        data += [flat(obs2), flat(feats2), flat(pairw)]
    return tuple(data)


def make_tape_update(*, actor: Actor, critic: Critic, optimizer: Any,
                     cfg: MotorTapePPOConfig, nu: int):
    lay = action_layout(cfg, nu)
    r1 = rate1_tanh(cfg) if cfg.use_rate else 0.0

    def loss_fn(params, batch, imit_coef):
        obs, feats, action, old_logp, adv, ret = batch[:6]
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        mean, log_std = actor.apply(params["actor"], obs, feats)
        log_std = jp.clip(log_std, -5.0, 2.0)
        logp = squashed_gaussian_logp(action, mean, log_std)
        ratio = jp.exp(logp - old_logp)
        pg_loss = -jp.minimum(ratio * adv, jp.clip(ratio, 0.8, 1.2) * adv).mean()
        value = critic.apply(params["critic"], obs, feats)
        v_loss = 0.5 * ((value - ret) ** 2).mean()
        entropy = ppo.gaussian_entropy(log_std).mean()
        loss = pg_loss + 0.5 * v_loss - float(cfg.ent_coef) * entropy
        imit_loss = jp.float32(0.0)
        if cfg.use_rate:
            # initialization pressure toward the authored pace (rate = 1.0); anneals to zero
            imit_loss = ((jp.tanh(mean[:, lay["rate"]]) - r1) ** 2).mean()
            loss = loss + imit_coef * imit_loss
        mod_smooth = jp.float32(0.0)
        if cfg.use_modulation:
            obs2, feats2, pairw = batch[6:9]
            m1 = jp.tanh(mean[:, lay["modulation"]])
            mean2, _ = actor.apply(params["actor"], obs2, feats2)
            m2 = jp.tanh(mean2[:, lay["modulation"]])
            mod_smooth = (pairw * ((m1 - m2) ** 2).sum(axis=-1)).sum() / (pairw.sum() + 1e-8)
            loss = (loss + float(cfg.mod_smooth_coef) * mod_smooth
                    + float(cfg.mod_l2_coef) * (m1 ** 2).mean())
        return loss, {
            "pg_loss": pg_loss,
            "v_loss": v_loss,
            "imit_loss": imit_loss,
            "mod_smooth": mod_smooth,
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


# --------------------------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------------------------

def train_motor_tape_ppo(*, env: Any, program: dict[str, Any], task: str, seed: int,
                         cfg: MotorTapePPOConfig, out_dir: Path | None = None):
    if int(env.horizon) % int(cfg.fragment_steps) != 0:
        raise ValueError(
            f"fragment_steps={cfg.fragment_steps} must divide env.horizon={env.horizon}")
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    nu = int(env.action_size)
    T = int(env.horizon)
    compiled = compile_tape(env, program)
    lay = action_layout(cfg, nu)
    fd = feat_dim(cfg, nu)

    key = jax.random.PRNGKey(seed)
    key, ak, ck = jax.random.split(key, 3)
    actor = Actor(out_dim=lay["out_dim"], hidden=cfg.hidden)
    critic = Critic(hidden=cfg.hidden)
    params = {
        "actor": actor.init(ak, jp.zeros((1, env.obs_size)), jp.zeros((1, fd))),
        "critic": critic.init(ck, jp.zeros((1, env.obs_size)), jp.zeros((1, fd))),
    }
    optimizer = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(cfg.lr))
    opt_state = optimizer.init(params)
    reset = jax.jit(lambda keys: jax.vmap(env.reset)(keys))
    build_tape = jax.jit(jax.vmap(compiled.tape_from_obs))
    collect = make_tape_collect(env=env, actor=actor, critic=critic, compiled=compiled,
                                cfg=cfg, deterministic=False)
    update = make_tape_update(actor=actor, critic=critic, optimizer=optimizer, cfg=cfg, nu=nu)

    fragments_per_episode = T // int(cfg.fragment_steps)
    imit_anneal = (cfg.imitation_anneal_iters if cfg.imitation_anneal_iters is not None
                   else cfg.iters)

    def fresh(rk):
        state = reset(jax.random.split(rk, cfg.envs))
        return state, build_tape(state.obs), jp.zeros((cfg.envs,), jp.float32)

    if cfg.warmup_compile:
        state0, tape0, s0 = fresh(jax.random.PRNGKey(seed + 99))
        _s, _p, traj, last_value, _es = collect(params, state0, s0, tape0,
                                                jax.random.PRNGKey(seed + 100))
        last_value.block_until_ready()
        data = _data_from_traj(traj, last_value, cfg)
        p2, _o2, _m2 = update(params, opt_state, data, jax.random.PRNGKey(seed + 101),
                              jp.float32(cfg.imitation_coef))
        jax.block_until_ready(p2)

    key, rk = jax.random.split(key)
    state, tape, s = fresh(rk)

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
        state, s, traj, last_value, eval_summary = collect(params, state, s, tape, ck2)
        data = _data_from_traj(traj, last_value, cfg)
        params, opt_state, metrics = update(params, opt_state, data, uk, jp.float32(imit_coef))
        jax.block_until_ready(params)

        if fragment_index == fragments_per_episode - 1:
            # Python-level episode reset: rebuild each env's tape from ITS fresh reset obs
            # (the plan is re-generated for the new spawn state -- reset-anchored by design).
            key, rk = jax.random.split(key)
            state, tape, s = fresh(rk)

        row = _training_row(it=it, env_steps=env_steps, fragment_index=fragment_index, task=task,
                            traj=traj, eval_summary=eval_summary, metrics=metrics,
                            elapsed=time.monotonic() - start, imit_coef=imit_coef, T=T, cfg=cfg)
        rows.append(row)

        if cfg.eval_every > 0 and (it == 0 or (it + 1) % int(cfg.eval_every) == 0):
            ev = evaluate_motor_tape_policy(env=env, params=params, program=program, task=task,
                                            seed=seed + 10_000 + it, cfg=cfg, compiled=compiled)
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

def evaluate_motor_tape_policy(*, env: Any, params: Any, program: dict[str, Any], task: str,
                               seed: int, cfg: MotorTapePPOConfig,
                               compiled: CompiledTape | None = None) -> dict[str, Any]:
    """Deterministic full-episode evaluation of the COMPLETE hybrid (tape + corrective heads),
    scored with the same eval_summary columns and task_graded_objective as every other arm."""
    eval_cfg = MotorTapePPOConfig(**{**cfg.__dict__, "envs": int(cfg.eval_envs)})
    if compiled is None:
        compiled = compile_tape(env, program)
    nu = int(env.action_size)
    lay = action_layout(cfg, nu)
    actor = Actor(out_dim=lay["out_dim"], hidden=cfg.hidden)
    critic = Critic(hidden=cfg.hidden)
    collect = make_tape_collect(env=env, actor=actor, critic=critic, compiled=compiled,
                                cfg=eval_cfg, deterministic=True)
    reset = jax.jit(lambda keys: jax.vmap(env.reset)(keys))
    build_tape = jax.jit(jax.vmap(compiled.tape_from_obs))
    state = reset(jax.random.split(jax.random.PRNGKey(seed), eval_cfg.envs))
    tape = build_tape(state.obs)
    s = jp.zeros((eval_cfg.envs,), jp.float32)
    traj_parts, summary_parts = [], []
    for i in range(int(env.horizon) // int(eval_cfg.fragment_steps)):
        state, s, traj, _lv, eval_summary = collect(params, state, s, tape,
                                                    jax.random.PRNGKey(seed + 1 + i))
        traj_parts.append(jax.device_get(traj))
        summary_parts.append(jax.device_get(eval_summary))
    merged = _merge_eval_summaries(summary_parts)
    es = jp.asarray(merged)
    success_rate = float(task_success(task, es).mean())
    fitness = float(task_fitness(task, es).mean())
    base_ret = sum(float(t[6].sum(axis=0).mean()) for t in traj_parts)
    shaping_ret = sum(float(t[7].sum(axis=0).mean()) for t in traj_parts)
    train_ret = sum(float(t[5].sum(axis=0).mean()) for t in traj_parts)
    rate_mean = float(np.mean([np.asarray(t[9]).mean() for t in traj_parts]))
    final_phase = float(np.asarray(traj_parts[-1][8])[-1].mean()) / float(env.horizon)
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
        "eval_gated_lift_return": round(
            sum(float(t[11].sum(axis=0).mean()) for t in traj_parts), 6),
        "eval_rate_mean": round(rate_mean, 6),
        "eval_final_phase": round(final_phase, 6),
        "eval_summary": [round(float(x), 6) for x in es.mean(axis=0)],
    }
    out["eval_graded_objective"] = round(task_graded_objective(task, out), 6)
    return out


def _training_row(*, it, env_steps, fragment_index, task, traj, eval_summary, metrics, elapsed,
                  imit_coef, T, cfg: MotorTapePPOConfig) -> dict[str, Any]:
    train_reward, base_reward, shaping = traj[5], traj[6], traj[7]
    phase, rate = traj[8], traj[9]
    success = task_success(task, eval_summary).mean()
    return {
        "iter": int(it),
        "env_steps": int(env_steps),
        "fragment_index": int(fragment_index),
        "task": task,
        "fragment_return": round(float(train_reward.sum(axis=0).mean()), 6),
        "base_return": round(float(base_reward.sum(axis=0).mean()), 6),
        "shaping_return": round(float(shaping.sum(axis=0).mean()), 6),
        "gated_lift_return": round(float(traj[11].sum(axis=0).mean()), 6),
        "success": round(float(success), 6),
        "phase_mean": round(float(np.asarray(phase).mean()) / float(T), 6),
        "rate_mean": round(float(np.asarray(rate).mean()), 6),
        "imit_coef": round(float(imit_coef), 6),
        "eval_summary": [round(float(x), 6) for x in eval_summary.mean(axis=0)],
        "pg_loss": round(float(metrics["pg_loss"]), 6),
        "v_loss": round(float(metrics["v_loss"]), 6),
        "imit_loss": round(float(metrics["imit_loss"]), 6),
        "mod_smooth": round(float(metrics["mod_smooth"]), 6),
        "entropy": round(float(metrics["entropy"]), 6),
        "approx_kl": round(float(metrics["approx_kl"]), 6),
        "elapsed_seconds": round(float(elapsed), 3),
    }

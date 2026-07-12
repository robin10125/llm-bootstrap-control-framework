"""Alternative prior-influence methods for PPO (experimental).

Four ways for an LLM-authored action prior to influence training WITHOUT sitting in the final
policy's action path (contrast with policy_bias_lab.ppo_bias, where env_action =
residual + prior_scale * prior):

  proposal        -- Prior as EXPLORATION PROPOSAL. The neural policy defines the action
                     distribution; during collection some steps execute a prior-centered sample
                     instead. The stored action/log-prob are the EXECUTED action scored under the
                     neural distribution, so PPO's importance ratio handles the off-policy data
                     (or those steps are masked out entirely with proposal_offpolicy="mask").
                     Proposal probability anneals to a final value; optional low-confidence gate.
  curriculum      -- Prior as STATE-VISITATION SHAPER. Each episode starts with a per-env
                     random-length warmup prefix driven purely by the prior; those steps are
                     masked out of the PPO loss, so the policy trains only on its own actions
                     from the states the prior reached. Warmup length anneals over training.
  value_shaping   -- Prior as VALUE/REWARD information. No action prior at all. Rewards come from
                     the authored per-stage `success` expressions (progress + completion bonus,
                     same machinery as ppo_bias.make_stage_reward_fn) plus a potential-based term
                     on ladder progress phi(s) = mean_k sigmoid(success_k / temp), plus an
                     optional auxiliary critic head regressing phi.
  critic_features -- Prior diagnostics as CRITIC INPUT ONLY. The actor sees the raw observation;
                     the critic additionally sees stage-cursor one-hot, the prior's suggested
                     action, its norm, prior-policy disagreement, and stage success margins.
                     The policy interface (actor input/output) is unchanged.
  kl_prior        -- Prior as a KL-REGULARIZED EXPLORATION DIRECTION. The prior defines a
                     reference distribution (pre-tanh Gaussian centered at atanh(prior) with width
                     kl_sigma_ref); the PPO loss adds beta * KL(pi || pi_ref) per state. The policy
                     is pulled toward the instruction everywhere -- so its own sampling explores
                     along it -- but diverges at exactly the states where the reward advantage of
                     diverging outbids the price beta. beta anneals to zero (or is servo-controlled
                     toward a growing KL budget via kl_target), so the optimal policy is preserved.

All four share the fragmented-PPO skeleton and the arbiter-parity evaluation of
policy_bias_lab.ppo_bias, so results are directly comparable to the baseline arm.
Task knowledge enters ONLY through the LLM-authored prior program; this module stays
task-agnostic.
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

from policy_bias_lab.bias import CompiledBias
from policy_bias_lab.freeform_priors import compile_expr, program_signal_fn
from policy_bias_lab.ppo_bias import (
    _eval_summary,
    _merge_eval_summaries,
    make_stage_reward_fn,
    squashed_gaussian_logp,
)
from policy_bias_lab.tasks import task_fitness, task_graded_objective, task_success

METHODS = ("proposal", "curriculum", "value_shaping", "critic_features", "kl_prior")


@dataclass(frozen=True)
class AltPPOConfig:
    method: str = "proposal"
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

    # -- proposal --
    proposal_prob: float = 0.3           # initial per-step probability of executing a proposal
    proposal_prob_final: float = 0.0     # annealed-to value
    proposal_anneal_iters: int | None = None  # default: cfg.iters
    proposal_sigma: float = 0.10         # exploration noise around the prior suggestion
    proposal_gate: str = "none"          # none | low_value (propose only below batch-median value)
    proposal_offpolicy: str = "ratio"    # ratio: keep with executed-action logp; mask: drop from loss

    # -- curriculum --
    warmup_frac: float = 0.5             # initial max warmup length as a fraction of the horizon
    warmup_frac_final: float = 0.0
    warmup_anneal_iters: int | None = None
    warmup_mode: str = "uniform"         # uniform: per-env U{0..max}; fixed: exactly max

    # -- value_shaping --
    stage_reward_weight: float = 1.0
    stage_progress_weight: float = 1.0
    stage_completion_bonus: float = 0.05
    stage_success_temperature: float = 1.0
    stage_reward_clip: float = 0.5
    potential_weight: float = 0.5        # potential-based shaping on ladder progress phi
    potential_temp: float = 1.0
    aux_coef: float = 0.0                # >0 adds an auxiliary critic head regressing phi

    # -- kl_prior --
    kl_coef: float = 1.0                 # initial beta: price of diverging from the instruction
    kl_coef_final: float = 0.0           # annealed-to beta (ignored when kl_target is set)
    kl_anneal_iters: int | None = None   # default: cfg.iters
    kl_sigma_ref: float = 0.3            # pre-tanh width of the instruction cone
    kl_ref_clip: float = 3.0             # clip |atanh(prior)| so saturated priors keep gradients
    kl_target: float | None = None       # if set, beta is servo-controlled to this KL budget ...
    kl_target_final: float | None = None # ... annealed toward this (default: kl_target)

    # -- critic_features --
    critic_stage_onehot: bool = True
    critic_prior_action: bool = True
    critic_prior_norms: bool = True      # prior action norm + prior-policy disagreement
    critic_success_margins: bool = True


class Actor(nn.Module):
    action_dim: int
    hidden: tuple[int, ...] = (256, 256)

    @nn.compact
    def __call__(self, obs):
        a = obs
        for h in self.hidden:
            a = nn.tanh(nn.Dense(h, kernel_init=nn.initializers.orthogonal(jp.sqrt(2.0)))(a))
        mean = nn.Dense(self.action_dim, kernel_init=nn.initializers.orthogonal(0.01))(a)
        log_std = self.param("log_std", nn.initializers.constant(-0.5), (self.action_dim,))
        return mean, log_std


class Critic(nn.Module):
    hidden: tuple[int, ...] = (256, 256)
    aux_head: bool = False

    @nn.compact
    def __call__(self, obs, feats):
        v = jp.concatenate([obs, feats], axis=-1)
        for h in self.hidden:
            v = nn.tanh(nn.Dense(h, kernel_init=nn.initializers.orthogonal(jp.sqrt(2.0)))(v))
        value = jp.squeeze(nn.Dense(1, kernel_init=nn.initializers.orthogonal(1.0))(v), -1)
        if self.aux_head:
            aux = jp.squeeze(nn.Dense(1, kernel_init=nn.initializers.orthogonal(1.0))(v), -1)
        else:
            aux = jp.zeros_like(value)
        return value, aux


def make_success_values_fn(env: Any, program: dict[str, Any]):
    """Per-stage authored `success` expression values (only stages that define one)."""
    stages = list(program.get("stages", [])) if program.get("mode") == "freeform_staged" else []
    exprs = [str(st["success"]) for st in stages
             if st.get("success") is not None and str(st.get("success")).strip() != ""]
    if not exprs:
        def zero(obs):
            return jp.zeros((0,), dtype=jp.float32)
        return zero, 0
    signals, signal_names = program_signal_fn(env, program)
    compiled = [compile_expr(e, signal_names) for e in exprs]

    def values(obs):
        sig = signals(obs)
        return jp.stack([jp.asarray(ev(sig), dtype=jp.float32) for ev in compiled])

    return values, len(compiled)


def _anneal(init: float, final: float, it: int, anneal_iters: int) -> float:
    if anneal_iters <= 0:
        return final
    frac = min(max(it / float(anneal_iters), 0.0), 1.0)
    return float(init + (final - init) * frac)


def _critic_feature_dim(cfg: AltPPOConfig, n_stages: int, action_dim: int, n_succ: int) -> int:
    # Minimum width 1 (a constant zero column) so [T*E, F] reshapes stay well-defined.
    dim = 1
    if cfg.method != "critic_features":
        return dim
    if cfg.critic_stage_onehot:
        dim += n_stages
    if cfg.critic_prior_action:
        dim += action_dim
    if cfg.critic_prior_norms:
        dim += 2
    if cfg.critic_success_margins:
        dim += n_succ
    return dim


def make_alt_collect(
    *,
    env: Any,
    actor: Actor,
    critic: Critic,
    bias: CompiledBias,
    program: dict[str, Any],
    cfg: AltPPOConfig,
    deterministic: bool,
):
    """Fragment collector shared by all four methods.

    collect(params, state, cursor, warmup_target, step_offset, knobs, key, action_prior_weights)
      -> (state, cursor, traj, last_value, eval_summary)

    knobs = jp.array([proposal_prob]) traced scalars so annealing does not retrigger compilation.
    Traj layout (each [T, E, ...]):
      0 obs, 1 feats, 2 action, 3 logp, 4 value, 5 train_reward, 6 base_reward, 7 shaping_reward,
      8 mask, 9 phi, 10 prior_used, 11 disagreement, 12 eval_metrics, 13 prior_action
    """
    step_fn = jax.vmap(env.step)
    action_dim = int(env.action_size)
    n_stages = len(program.get("stages", [])) if program.get("mode") == "freeform_staged" else 0
    succ_values, n_succ = make_success_values_fn(env, program)
    temp = max(float(cfg.stage_success_temperature), 1e-6)
    pot_temp = max(float(cfg.potential_temp), 1e-6)

    if cfg.method == "value_shaping":
        stage_reward_fn, _n = make_stage_reward_fn(env, program, cfg)
    else:
        stage_reward_fn = None

    def phi_fn(obs):  # ladder progress in [0, 1]; 0 when no success exprs authored
        if n_succ == 0:
            return jp.float32(0.0)
        return jp.mean(jax.nn.sigmoid(succ_values(obs) / pot_temp))

    def prior_step(obs, cursor, action_prior_weights):
        if bias.prior_step_fn is not None:
            return jax.vmap(lambda o, c: bias.prior_step_fn(o, action_prior_weights, c))(obs, cursor)
        if bias.prior_fn is not None:
            prior = jax.vmap(lambda o: bias.prior_fn(o, action_prior_weights))(obs)
            return prior, cursor
        return jp.zeros((obs.shape[0], action_dim), dtype=jp.float32), cursor

    def critic_feats(obs, cursor, prior, mean_action):
        parts = [jp.zeros((obs.shape[0], 1), dtype=jp.float32)]  # keep feature width >= 1
        if cfg.method == "critic_features":
            if cfg.critic_stage_onehot and n_stages:
                parts.append(jax.nn.one_hot(cursor, n_stages, dtype=jp.float32))
            if cfg.critic_prior_action:
                parts.append(prior)
            if cfg.critic_prior_norms:
                scale = 1.0 / jp.sqrt(float(action_dim))
                parts.append(jp.linalg.norm(prior, axis=-1, keepdims=True) * scale)
                parts.append(jp.linalg.norm(prior - mean_action, axis=-1, keepdims=True) * scale)
            if cfg.critic_success_margins and n_succ:
                margins = jax.vmap(lambda o: jax.nn.sigmoid(succ_values(o) / temp))(obs)
                parts.append(margins)
        return jp.concatenate(parts, axis=-1)

    def policy_step(params, obs, cursor, warmup_target, t_global, knobs, key,
                    action_prior_weights):
        mean, log_std = actor.apply(params["actor"], obs)
        log_std = jp.clip(log_std, -5.0, 2.0)
        pk, nk = jax.random.split(key)
        if deterministic:
            pre_action = mean
        else:
            pre_action = mean + jp.exp(log_std) * jax.random.normal(pk, mean.shape)
        policy_action = jp.tanh(pre_action)
        prior, next_cursor = prior_step(obs, cursor, action_prior_weights)
        prior = jp.clip(prior, -1.0, 1.0)
        feats = critic_feats(obs, cursor, prior, jp.tanh(mean))
        value, aux = critic.apply(params["critic"], obs, feats)

        mask = jp.ones((obs.shape[0],), dtype=jp.float32)
        prior_used = jp.zeros((obs.shape[0],), dtype=jp.float32)
        env_action = policy_action

        if cfg.method == "proposal" and not deterministic:
            prop_key, use_key = jax.random.split(nk)
            proposal = jp.clip(
                prior + float(cfg.proposal_sigma) * jax.random.normal(prop_key, prior.shape),
                -1.0, 1.0)
            p = knobs[0]
            if cfg.proposal_gate == "low_value":
                gate = (value < jp.median(value)).astype(jp.float32)
            else:
                gate = jp.ones_like(value)
            use = (jax.random.uniform(use_key, (obs.shape[0],)) < p * gate).astype(jp.float32)
            env_action = jp.where(use[:, None] > 0.5, proposal, policy_action)
            prior_used = use
            if cfg.proposal_offpolicy == "mask":
                mask = 1.0 - use
        elif cfg.method == "curriculum" and not deterministic:
            in_warmup = (t_global < warmup_target).astype(jp.float32)
            env_action = jp.where(in_warmup[:, None] > 0.5, prior, policy_action)
            prior_used = in_warmup
            mask = 1.0 - in_warmup

        # The stored action is the EXECUTED one, scored under the current neural distribution:
        # PPO's ratio then treats prior-driven steps as off-policy data (masked steps drop out).
        logp = squashed_gaussian_logp(env_action, mean, log_std)
        disagreement = jp.mean(jp.abs(prior - policy_action), axis=-1)
        return (env_action, logp, value, aux, feats, mask, prior_used, disagreement, prior,
                next_cursor)

    def collect(params, state, cursor, warmup_target, step_offset, knobs, key,
                action_prior_weights):
        def body(carry, t):
            state, cursor, key = carry
            key, ak = jax.random.split(key)
            t_global = step_offset + t
            (env_action, logp, value, aux, feats, mask, prior_used, disagreement, prior,
             next_cursor) = (
                policy_step(params, state.obs, cursor, warmup_target, t_global, knobs, ak,
                            action_prior_weights))
            nstate = step_fn(state, env_action)
            phi_prev = jax.vmap(phi_fn)(state.obs)
            shaping = jp.zeros((state.obs.shape[0],), dtype=jp.float32)
            if cfg.method == "value_shaping":
                if stage_reward_fn is not None:
                    stage_r, _contrib = jax.vmap(stage_reward_fn)(state.obs, nstate.obs)
                    shaping = shaping + stage_r
                if float(cfg.potential_weight) != 0.0 and n_succ:
                    phi_next = jax.vmap(phi_fn)(nstate.obs)
                    shaping = shaping + float(cfg.potential_weight) * (
                        float(cfg.gamma) * phi_next - phi_prev)
            train_reward = (jp.asarray(float(cfg.base_reward_weight), dtype=jp.float32)
                            * nstate.reward + shaping)
            return (nstate, next_cursor, key), (
                state.obs, feats, env_action, logp, value, train_reward, nstate.reward,
                shaping, mask, phi_prev, prior_used, disagreement, nstate.metrics["eval"], prior)

        (state, cursor, key), traj = jax.lax.scan(
            body, (state, cursor, key), jp.arange(int(cfg.fragment_steps)))
        mean, log_std = actor.apply(params["actor"], state.obs)
        prior, _c = prior_step(state.obs, cursor, action_prior_weights)
        feats = critic_feats(state.obs, cursor, jp.clip(prior, -1.0, 1.0), jp.tanh(mean))
        last_value, _aux = critic.apply(params["critic"], state.obs, feats)
        eval_summary = _eval_summary(traj[12])
        return state, cursor, traj, last_value, eval_summary

    return jax.jit(collect)


def make_alt_update(*, actor: Actor, critic: Critic, optimizer: Any, cfg: AltPPOConfig):
    aux_coef = float(cfg.aux_coef) if cfg.method == "value_shaping" else 0.0
    sigma_ref = max(float(cfg.kl_sigma_ref), 1e-3)
    ref_clip = float(cfg.kl_ref_clip)

    def _atanh(x):
        return 0.5 * (jp.log1p(x) - jp.log1p(-x))

    def loss_fn(params, batch, kl_coef):
        obs, feats, action, old_logp, adv, ret, mask, phi, prior = batch
        denom = jp.maximum(mask.sum(), 1.0)
        m_mean = (adv * mask).sum() / denom
        m_std = jp.sqrt(((adv - m_mean) ** 2 * mask).sum() / denom)
        adv = (adv - m_mean) / (m_std + 1e-8)
        mean, log_std = actor.apply(params["actor"], obs)
        log_std = jp.clip(log_std, -5.0, 2.0)
        logp = squashed_gaussian_logp(action, mean, log_std)
        ent = ppo.gaussian_entropy(log_std)
        ratio = jp.exp(logp - old_logp)
        pg = -jp.minimum(ratio * adv, jp.clip(ratio, 0.8, 1.2) * adv)
        pg_loss = (pg * mask).sum() / denom
        value, aux = critic.apply(params["critic"], obs, feats)
        v_loss = 0.5 * (((value - ret) ** 2) * mask).sum() / denom
        aux_loss = (((aux - phi) ** 2) * mask).sum() / denom
        entropy = ent.mean()
        # KL(pi || pi_ref) in pre-tanh space (exact for the squashed pair, tanh is a bijection):
        # the reference is a Gaussian centered on the instruction with fixed width sigma_ref.
        mu_ref = jp.clip(_atanh(jp.clip(prior, -0.999999, 0.999999)), -ref_clip, ref_clip)
        var = jp.exp(2.0 * log_std)
        kl_dim = (jp.log(sigma_ref) - log_std
                  + (var + (mean - mu_ref) ** 2) / (2.0 * sigma_ref ** 2) - 0.5)
        prior_kl = (jp.sum(kl_dim, axis=-1) * mask).sum() / denom
        loss = (pg_loss + 0.5 * v_loss - float(cfg.ent_coef) * entropy + aux_coef * aux_loss
                + kl_coef * prior_kl)
        return loss, {
            "pg_loss": pg_loss,
            "v_loss": v_loss,
            "aux_loss": aux_loss,
            "entropy": entropy,
            "approx_kl": ((old_logp - logp) * mask).sum() / denom,
            "prior_kl": prior_kl,
        }

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    def update(params, opt_state, data, key, kl_coef):
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
                (_loss, metrics), grads = grad_fn(params, batch, kl_coef)
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


def _alt_data_from_traj(traj, last_value, gamma: float, lam: float):
    obs, feats, action, logp, value, train_reward = traj[:6]
    mask, phi, prior = traj[8], traj[9], traj[13]
    adv, ret = ppo.compute_gae(train_reward, value, last_value, gamma, lam)
    flat = lambda x: x.reshape((-1,) + x.shape[2:])
    return (flat(obs), flat(feats), flat(action), flat(logp), flat(adv), flat(ret),
            flat(mask), flat(phi), flat(prior))


def _kl_coef_step(cfg: AltPPOConfig, it: int, beta: float, measured_kl: float | None) -> float:
    """Next beta for kl_prior: linear anneal, or a servo toward an annealed KL budget."""
    if cfg.method != "kl_prior":
        return 0.0
    anneal_iters = cfg.kl_anneal_iters if cfg.kl_anneal_iters is not None else cfg.iters
    if cfg.kl_target is None:
        return _anneal(cfg.kl_coef, cfg.kl_coef_final, it, anneal_iters)
    target_final = cfg.kl_target_final if cfg.kl_target_final is not None else cfg.kl_target
    target = max(_anneal(cfg.kl_target, target_final, it, anneal_iters), 1e-6)
    if measured_kl is not None:
        if measured_kl > 1.5 * target:
            beta *= 1.2
        elif measured_kl < target / 1.5:
            beta /= 1.2
    return float(min(max(beta, 1e-4), 1e3))


def _knob_values(cfg: AltPPOConfig, it: int) -> tuple[float, float]:
    """(proposal_prob, warmup_frac) at iteration `it` under the configured anneals."""
    p = _anneal(cfg.proposal_prob, cfg.proposal_prob_final, it,
                cfg.proposal_anneal_iters if cfg.proposal_anneal_iters is not None else cfg.iters)
    w = _anneal(cfg.warmup_frac, cfg.warmup_frac_final, it,
                cfg.warmup_anneal_iters if cfg.warmup_anneal_iters is not None else cfg.iters)
    return p, w


def _draw_warmup_target(key, cfg: AltPPOConfig, horizon: int, warmup_frac: float,
                        envs: int) -> jp.ndarray:
    if cfg.method != "curriculum" or warmup_frac <= 0.0:
        return jp.zeros((envs,), dtype=jp.int32)
    max_steps = int(round(warmup_frac * horizon))
    if max_steps <= 0:
        return jp.zeros((envs,), dtype=jp.int32)
    if cfg.warmup_mode == "fixed":
        return jp.full((envs,), max_steps, dtype=jp.int32)
    return jax.random.randint(key, (envs,), 0, max_steps + 1)


def train_alt_ppo(
    *,
    env: Any,
    bias: CompiledBias,
    program: dict[str, Any],
    task: str,
    seed: int,
    cfg: AltPPOConfig,
    out_dir: Path | None = None,
) -> tuple[Any, list[dict[str, Any]], Any, float, int]:
    """Train one alternative-method PPO arm on fixed-length fragments."""
    if cfg.method not in METHODS:
        raise ValueError(f"unknown method={cfg.method!r} (expected one of {METHODS})")
    if int(env.horizon) % int(cfg.fragment_steps) != 0:
        raise ValueError(
            f"fragment_steps={cfg.fragment_steps} must divide env.horizon={env.horizon}")
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    action_dim = int(env.action_size)
    n_stages = len(program.get("stages", [])) if program.get("mode") == "freeform_staged" else 0
    _sv, n_succ = make_success_values_fn(env, program)
    feat_dim = _critic_feature_dim(cfg, n_stages, action_dim, n_succ)

    key = jax.random.PRNGKey(seed)
    key, ak, ck = jax.random.split(key, 3)
    actor = Actor(action_dim=action_dim, hidden=cfg.hidden)
    critic = Critic(hidden=cfg.hidden,
                    aux_head=(cfg.method == "value_shaping" and float(cfg.aux_coef) > 0.0))
    params = {
        "actor": actor.init(ak, jp.zeros((1, env.obs_size))),
        "critic": critic.init(ck, jp.zeros((1, env.obs_size)), jp.zeros((1, feat_dim))),
    }
    optimizer = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(cfg.lr))
    opt_state = optimizer.init(params)
    reset = jax.jit(lambda keys: jax.vmap(env.reset)(keys))
    collect = make_alt_collect(env=env, actor=actor, critic=critic, bias=bias, program=program,
                               cfg=cfg, deterministic=False)
    update = make_alt_update(actor=actor, critic=critic, optimizer=optimizer, cfg=cfg)
    action_prior_weights = bias.default_action_prior_weights()

    fragments_per_episode = int(env.horizon) // int(cfg.fragment_steps)

    if cfg.warmup_compile:
        state0 = reset(jax.random.split(jax.random.PRNGKey(seed + 99), cfg.envs))
        cursor0 = jp.zeros((cfg.envs,), dtype=jp.int32)
        wt0 = _draw_warmup_target(jax.random.PRNGKey(seed + 98), cfg, int(env.horizon),
                                  cfg.warmup_frac, cfg.envs)
        knobs0 = jp.asarray([cfg.proposal_prob], dtype=jp.float32)
        _s, _c, traj, last_value, _es = collect(
            params, state0, cursor0, wt0, jp.int32(0), knobs0, jax.random.PRNGKey(seed + 100),
            action_prior_weights)
        last_value.block_until_ready()
        data = _alt_data_from_traj(traj, last_value, cfg.gamma, cfg.lam)
        p2, _o2, _m2 = update(params, opt_state, data, jax.random.PRNGKey(seed + 101),
                              jp.float32(cfg.kl_coef if cfg.method == "kl_prior" else 0.0))
        jax.block_until_ready(p2)

    key, rk, wk = jax.random.split(key, 3)
    state = reset(jax.random.split(rk, cfg.envs))
    cursor = jp.zeros((cfg.envs,), dtype=jp.int32)
    p0, w0 = _knob_values(cfg, 0)
    warmup_target = _draw_warmup_target(wk, cfg, int(env.horizon), w0, cfg.envs)

    rows: list[dict[str, Any]] = []
    best_score = -1.0e30
    best_iter = -1
    best_params = params
    kl_beta = float(cfg.kl_coef) if cfg.method == "kl_prior" else 0.0
    last_prior_kl: float | None = None
    start = time.monotonic()

    for it in range(int(cfg.iters)):
        elapsed = time.monotonic() - start
        if cfg.target_train_seconds is not None and it > 0 and elapsed >= cfg.target_train_seconds:
            break
        env_steps = (it + 1) * int(cfg.envs) * int(cfg.fragment_steps)
        if cfg.max_env_steps is not None and it > 0 and env_steps >= int(cfg.max_env_steps):
            break

        proposal_p, warmup_frac = _knob_values(cfg, it)
        kl_beta = _kl_coef_step(cfg, it, kl_beta, last_prior_kl)
        knobs = jp.asarray([proposal_p], dtype=jp.float32)
        fragment_index = it % fragments_per_episode
        step_offset = jp.int32(fragment_index * int(cfg.fragment_steps))

        key, ck2, uk = jax.random.split(key, 3)
        state, cursor, traj, last_value, eval_summary = collect(
            params, state, cursor, warmup_target, step_offset, knobs, ck2, action_prior_weights)
        data = _alt_data_from_traj(traj, last_value, cfg.gamma, cfg.lam)
        params, opt_state, metrics = update(params, opt_state, data, uk, jp.float32(kl_beta))
        jax.block_until_ready(params)
        last_prior_kl = float(metrics["prior_kl"])

        if fragment_index == fragments_per_episode - 1:
            key, rk, wk = jax.random.split(key, 3)
            state = reset(jax.random.split(rk, cfg.envs))
            cursor = jp.zeros((cfg.envs,), dtype=jp.int32)
            warmup_target = _draw_warmup_target(wk, cfg, int(env.horizon), warmup_frac, cfg.envs)

        row = _training_row(it=it, env_steps=env_steps, fragment_index=fragment_index, task=task,
                            traj=traj, eval_summary=eval_summary, metrics=metrics,
                            elapsed=time.monotonic() - start, proposal_p=proposal_p,
                            warmup_frac=warmup_frac, kl_beta=kl_beta, cfg=cfg)
        rows.append(row)

        if cfg.eval_every > 0 and (it == 0 or (it + 1) % int(cfg.eval_every) == 0):
            ev = evaluate_alt_policy(env=env, params=params, bias=bias, program=program,
                                     task=task, seed=seed + 10_000 + it, cfg=cfg)
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


def evaluate_alt_policy(
    *,
    env: Any,
    params: Any,
    bias: CompiledBias,
    program: dict[str, Any],
    task: str,
    seed: int,
    cfg: AltPPOConfig,
) -> dict[str, Any]:
    """Full-episode deterministic evaluation of the PURE NEURAL policy.

    All prior influence is off here (no proposals, no warmup): this measures what the trained
    policy does on its own, scored with the SAME eval_summary columns and task_graded_objective
    as the prior_only/short_ppo arbiter and the ppo_bias baseline.
    """
    eval_cfg = AltPPOConfig(**{**cfg.__dict__, "envs": int(cfg.eval_envs)})
    actor = Actor(action_dim=int(env.action_size), hidden=cfg.hidden)
    critic = Critic(hidden=cfg.hidden,
                    aux_head=(cfg.method == "value_shaping" and float(cfg.aux_coef) > 0.0))
    reset = jax.jit(lambda keys: jax.vmap(env.reset)(keys))
    collect = make_alt_collect(env=env, actor=actor, critic=critic, bias=bias, program=program,
                               cfg=eval_cfg, deterministic=True)
    action_prior_weights = bias.default_action_prior_weights()
    state = reset(jax.random.split(jax.random.PRNGKey(seed), eval_cfg.envs))
    cursor = jp.zeros((eval_cfg.envs,), dtype=jp.int32)
    warmup_target = jp.zeros((eval_cfg.envs,), dtype=jp.int32)
    knobs = jp.asarray([0.0], dtype=jp.float32)
    traj_parts = []
    summary_parts = []
    for i in range(int(env.horizon) // int(eval_cfg.fragment_steps)):
        step_offset = jp.int32(i * int(eval_cfg.fragment_steps))
        state, cursor, traj, _lv, eval_summary = collect(
            params, state, cursor, warmup_target, step_offset, knobs,
            jax.random.PRNGKey(seed + 1 + i), action_prior_weights)
        traj_parts.append(jax.device_get(traj))
        summary_parts.append(jax.device_get(eval_summary))
    merged = _merge_eval_summaries(summary_parts)
    es = jp.asarray(merged)
    success_rate = float(task_success(task, es).mean())
    fitness = float(task_fitness(task, es).mean())
    base_ret = sum(float(t[6].sum(axis=0).mean()) for t in traj_parts)
    shaping_ret = sum(float(t[7].sum(axis=0).mean()) for t in traj_parts)
    train_ret = sum(float(t[5].sum(axis=0).mean()) for t in traj_parts)
    disagreement = float(np.mean([np.asarray(t[11]).mean() for t in traj_parts]))
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
        "eval_prior_disagreement": round(disagreement, 6),
        "eval_summary": [round(float(x), 6) for x in es.mean(axis=0)],
    }
    out["eval_graded_objective"] = round(task_graded_objective(task, out), 6)
    return out


def _training_row(*, it, env_steps, fragment_index, task, traj, eval_summary, metrics, elapsed,
                  proposal_p, warmup_frac, kl_beta, cfg: AltPPOConfig) -> dict[str, Any]:
    train_reward, base_reward, shaping = traj[5], traj[6], traj[7]
    mask, prior_used, disagreement = traj[8], traj[10], traj[11]
    success = task_success(task, eval_summary).mean()
    row = {
        "iter": int(it),
        "env_steps": int(env_steps),
        "fragment_index": int(fragment_index),
        "task": task,
        "method": cfg.method,
        "fragment_return": round(float(train_reward.sum(axis=0).mean()), 6),
        "base_return": round(float(base_reward.sum(axis=0).mean()), 6),
        "shaping_return": round(float(shaping.sum(axis=0).mean()), 6),
        "success": round(float(success), 6),
        "train_mask_frac": round(float(mask.mean()), 6),
        "prior_used_frac": round(float(prior_used.mean()), 6),
        "prior_disagreement": round(float(disagreement.mean()), 6),
        "eval_summary": [round(float(x), 6) for x in eval_summary.mean(axis=0)],
        "pg_loss": round(float(metrics["pg_loss"]), 6),
        "v_loss": round(float(metrics["v_loss"]), 6),
        "aux_loss": round(float(metrics["aux_loss"]), 6),
        "entropy": round(float(metrics["entropy"]), 6),
        "approx_kl": round(float(metrics["approx_kl"]), 6),
        "elapsed_seconds": round(float(elapsed), 3),
    }
    row["prior_kl"] = round(float(metrics["prior_kl"]), 6)
    if cfg.method == "proposal":
        row["proposal_prob"] = round(float(proposal_p), 6)
    if cfg.method == "curriculum":
        row["warmup_frac"] = round(float(warmup_frac), 6)
    if cfg.method == "kl_prior":
        row["kl_coef"] = round(float(kl_beta), 6)
    return row

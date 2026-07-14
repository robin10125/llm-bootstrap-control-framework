#!/usr/bin/env python3
"""Compact PPO for the MJX manipulation envs (flax + optax).

A continuous-action Gaussian actor-critic, GAE, and a clipped-objective minibatch update.
Kept deliberately small and dependency-light so the LLM-supervision pieces (a behavior-
cloning auxiliary loss, a demo buffer) can be bolted on without fighting a framework. See
`rl_redesign.md`.

Actions are unsquashed Gaussian samples; the env clips them to actuator range when it
executes (standard "clip action" PPO). The behavior-cloning loss in `train.py` reuses
`gaussian_logp` / `evaluate_actions` from here.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jp
import optax
from flax import linen as nn

LOG2PI = float(jp.log(2.0 * jp.pi))


class ActorCritic(nn.Module):
    action_dim: int
    hidden: tuple[int, ...] = (256, 256)

    @nn.compact
    def __call__(self, x):
        a = x
        for h in self.hidden:
            a = nn.tanh(nn.Dense(h, kernel_init=nn.initializers.orthogonal(jp.sqrt(2.0)))(a))
        mean = nn.Dense(self.action_dim, kernel_init=nn.initializers.orthogonal(0.01))(a)
        log_std = self.param("log_std", nn.initializers.constant(-0.5), (self.action_dim,))

        v = x
        for h in self.hidden:
            v = nn.tanh(nn.Dense(h, kernel_init=nn.initializers.orthogonal(jp.sqrt(2.0)))(v))
        value = nn.Dense(1, kernel_init=nn.initializers.orthogonal(1.0))(v)
        return mean, log_std, jp.squeeze(value, -1)


def init_params(key, obs_dim: int, action_dim: int, hidden=(256, 256)):
    net = ActorCritic(action_dim=action_dim, hidden=hidden)
    params = net.init(key, jp.zeros((1, obs_dim)))
    return net, params


def gaussian_logp(action, mean, log_std):
    std = jp.exp(log_std)
    return jp.sum(-0.5 * ((action - mean) / std) ** 2 - log_std - 0.5 * LOG2PI, axis=-1)


def gaussian_entropy(log_std):
    return jp.sum(log_std + 0.5 * (LOG2PI + 1.0), axis=-1)


def sample_action(net, params, obs, key):
    mean, log_std, value = net.apply(params, obs)
    std = jp.exp(log_std)
    action = mean + std * jax.random.normal(key, mean.shape)
    logp = gaussian_logp(action, mean, log_std)
    return action, logp, value


def evaluate_actions(net, params, obs, action):
    mean, log_std, value = net.apply(params, obs)
    logp = gaussian_logp(action, mean, log_std)
    ent = gaussian_entropy(log_std)
    return logp, ent, value


def compute_gae(rewards, values, last_value, gamma: float, lam: float):
    """rewards/values: [T, N]; last_value: [N]. Returns advantages, returns: [T, N].

    Time-limit episodes (no early termination): the final step bootstraps with last_value.
    """
    def body(carry, t):
        adv = carry
        next_v = jp.where(t == rewards.shape[0] - 1, last_value, values[jp.minimum(t + 1, rewards.shape[0] - 1)])
        delta = rewards[t] + gamma * next_v - values[t]
        adv = delta + gamma * lam * adv
        return adv, adv

    T = rewards.shape[0]
    init = jp.zeros_like(last_value)
    _, advs = jax.lax.scan(body, init, jp.arange(T - 1, -1, -1))
    advs = advs[::-1]
    returns = advs + values
    return advs, returns


class PPOConfig(NamedTuple):
    clip: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.0
    max_grad_norm: float = 0.5
    epochs: int = 4
    num_minibatches: int = 8
    target_kl: float = 0.03
    normalize_adv: bool = True
    bc_weight_temp: float = 1.0  # advantage-weighted BC temperature; <=0 = uniform BC


def ppo_loss(net, params, batch, cfg: PPOConfig, bc_batch=None, bc_coef: float = 0.0):
    obs, action, old_logp, adv, ret = batch
    if cfg.normalize_adv:
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    logp, ent, value = evaluate_actions(net, params, obs, action)
    ratio = jp.exp(logp - old_logp)
    pg1 = ratio * adv
    pg2 = jp.clip(ratio, 1.0 - cfg.clip, 1.0 + cfg.clip) * adv
    pg_loss = -jp.minimum(pg1, pg2).mean()
    v_loss = 0.5 * ((value - ret) ** 2).mean()
    ent_loss = -ent.mean()

    loss = pg_loss + cfg.vf_coef * v_loss + cfg.ent_coef * ent_loss

    # Behavior-cloning auxiliary loss: pull the policy mean toward LLM demo actions.
    # Gated structurally on whether a demo batch exists, so bc_coef may be a traced value.
    # Reward-based weighting (advantage-weighted regression): demos that beat the policy by
    # more get a larger weight, exp(adv/temp) on the batch-normalized margin. temp<=0 falls
    # back to uniform BC.
    bc_loss = jp.float32(0.0)
    if bc_batch is not None:
        bc_obs, bc_act, bc_adv = bc_batch
        bc_logp, _, _ = evaluate_actions(net, params, bc_obs, bc_act)
        if cfg.bc_weight_temp > 0:
            a = (bc_adv - bc_adv.mean()) / (bc_adv.std() + 1e-8)
            w = jp.exp(jp.clip(a / cfg.bc_weight_temp, -5.0, 5.0))
            w = w / (w.mean() + 1e-8)
        else:
            w = jp.ones_like(bc_logp)
        bc_loss = -(w * bc_logp).mean()
        loss = loss + bc_coef * bc_loss

    approx_kl = jp.mean(old_logp - logp)
    metrics = {
        "pg_loss": pg_loss, "v_loss": v_loss, "entropy": ent.mean(),
        "approx_kl": approx_kl, "bc_loss": bc_loss,
    }
    return loss, metrics


def make_update(net, optimizer, cfg: PPOConfig):
    """Return a jitted PPO update over one rollout's flattened transitions."""

    grad_fn = jax.value_and_grad(
        lambda p, b, bcb, bcc: ppo_loss(net, p, b, cfg, bcb, bcc), has_aux=True)

    def update(params, opt_state, data, key, bc_data=None, bc_coef=0.0):
        # data: tuple of [B, ...] arrays. Shuffle, split into minibatches, run epochs.
        B = data[0].shape[0]
        mb = B // cfg.num_minibatches

        def epoch(carry, _):
            params, opt_state, key = carry
            key, pk = jax.random.split(key)
            perm = jax.random.permutation(pk, B)

            def minibatch(carry, idx):
                params, opt_state = carry
                sl = jax.lax.dynamic_slice_in_dim(perm, idx * mb, mb)
                batch = tuple(jp.take(x, sl, axis=0) for x in data)
                if bc_data is not None:
                    bc_sl = sl % bc_data[0].shape[0]
                    bcb = tuple(jp.take(x, bc_sl, axis=0) for x in bc_data)
                else:
                    bcb = None
                (loss, m), grads = grad_fn(params, batch, bcb, bc_coef)
                updates, opt_state = optimizer.update(grads, opt_state, params)
                params = optax.apply_updates(params, updates)
                return (params, opt_state), m

            (params, opt_state), ms = jax.lax.scan(
                minibatch, (params, opt_state), jp.arange(cfg.num_minibatches))
            return (params, opt_state, key), ms

        (params, opt_state, key), ms = jax.lax.scan(
            epoch, (params, opt_state, key), None, length=cfg.epochs)
        metrics = jax.tree_util.tree_map(lambda x: x.mean(), ms)
        return params, opt_state, metrics

    return jax.jit(update, static_argnames=())

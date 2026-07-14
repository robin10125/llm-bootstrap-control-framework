"""Experimental PPO for long episodes using short rollout fragments.

This module is intentionally separate from ``policy_bias_lab.training.fragmented_ppo``.  It prototypes three
changes that are useful for 10-20s episodes but still need empirical validation before becoming the
default path:

* PPO updates on short fragments while carrying simulator state across fragments.
* Dense stage-objective rewards authored by the prior program's per-stage ``success`` expressions.
* Learned scale outputs in the policy head that gate how much of the action prior is applied.

The dense stage rewards stay task-agnostic: the only "objective" content is the LLM-authored
``stage.success`` expression evaluated over the same raw/authored signal vocabulary as the prior.
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import jax
import jax.numpy as jp
import numpy as np
import optax


from experiment_runtime import ppo

from policy_bias_lab.bias import CompiledBias
from policy_bias_lab.freeform_priors import (
    _semantic_group,
    all_actuator_names,
    compile_expr,
    make_stage_weight_fn,
    program_signal_fn,
)
from policy_bias_lab.training.fragmented_ppo import squashed_gaussian_logp
from policy_bias_lab.schema import FIELD_INDEX
from policy_bias_lab.tasks import task_fitness, task_graded_objective, task_success


@dataclass(frozen=True)
class FragmentedStagePPOConfig:
    iters: int = 2000
    envs: int = 256
    eval_envs: int = 256
    fragment_steps: int = 64
    lr: float = 3e-4
    gamma: float = 0.99
    lam: float = 0.95
    hidden: tuple[int, ...] = (256, 256)
    ent_coef: float = 0.0
    target_train_seconds: float | None = None
    max_env_steps: int | None = None
    checkpoint_every: int = 0
    eval_every: int = 25
    residual_action_scale: float = 1.0
    use_action_prior: bool = True
    learn_prior_scale: bool = True
    # How finely the controller may tune the prior STRENGTH. "group": one knob per semantic actuator
    # group (robot-structure-derived, e.g. base/wrist/digit groups for this robot class) so the
    # learned residual can release a problematic group without disabling the whole prior. "scalar"
    # is kept as an explicit compatibility option; "per_joint" is most expressive and weakest-prior.
    # All modes start at exactly 1.0 (full prior) at init.
    prior_scale_mode: str = "group"
    prior_scale_bias: float = 1.0
    prior_scale_gain: float = 1.0
    stage_reward_weight: float = 1.0
    stage_progress_weight: float = 1.0
    stage_completion_bonus: float = 0.05
    stage_success_temperature: float = 1.0
    stage_reward_clip: float = 0.5
    base_reward_weight: float = 1.0
    action_transform: str = "tanh"
    warmup_compile: bool = True


def _scale_group_spec(env: Any, mode: str) -> tuple[int, jp.ndarray, list[str]]:
    """Map a small set of learned scale outputs onto the action channels they scale.

    Returns ``(n_outputs, expand, names)`` where ``expand`` is an ``[n_outputs, action_size]`` 0/1
    matrix (a disjoint partition of channels for group/per_joint) so the per-channel prior strength
    is ``scale_signal @ expand``. The groups are the robot's SEMANTIC groups -- generic robot
    structure, no task knowledge -- so this stays task-agnostic.
    """
    a = int(env.action_size)
    if mode == "scalar":
        return 1, jp.ones((1, a), dtype=jp.float32), ["all"]
    if mode == "per_joint":
        return a, jp.eye(a, dtype=jp.float32), list(all_actuator_names(env))
    if mode == "group":
        names = all_actuator_names(env)
        groups: list[str] = []
        for n in names:
            g = _semantic_group(n)
            if g not in groups:
                groups.append(g)  # first-appearance order -> deterministic head layout
        expand = np.zeros((len(groups), a), dtype=np.float32)
        for i, n in enumerate(names):
            expand[groups.index(_semantic_group(n)), i] = 1.0
        return len(groups), jp.asarray(expand), groups
    raise ValueError(f"unknown prior_scale_mode={mode!r} (expected scalar|group|per_joint)")


def _policy_dim(env: Any, cfg: FragmentedStagePPOConfig) -> int:
    n_scale = _scale_group_spec(env, cfg.prior_scale_mode)[0] if cfg.learn_prior_scale else 0
    return int(env.action_size) + n_scale


def _group_scale_means(prior_scale: jp.ndarray, expand: jp.ndarray, names: list[str]) -> dict:
    """Per-group mean of a per-channel prior_scale array (any leading dims), for reporting."""
    ps = np.asarray(prior_scale)
    if ps.ndim < 1 or ps.shape[-1] != expand.shape[1]:
        return {}
    per_ch = ps.reshape(-1, ps.shape[-1]).mean(axis=0)  # [A]
    ex = np.asarray(expand)
    return {names[g]: round(float((per_ch * ex[g]).sum() / max(ex[g].sum(), 1.0)), 6)
            for g in range(len(names))}


def train_fragmented_stage_ppo(
    *,
    env: Any,
    bias: CompiledBias,
    program: dict[str, Any],
    task: str,
    seed: int,
    cfg: FragmentedStagePPOConfig,
    out_dir: Path | None = None,
    action_prior_weights: jp.ndarray | None = None,
) -> tuple[Any, list[dict[str, Any]], Any, float, int]:
    """Train PPO on fixed-length fragments, carrying env state across fragments."""
    if int(env.horizon) % int(cfg.fragment_steps) != 0:
        raise ValueError(
            f"fragment_steps={cfg.fragment_steps} must divide env.horizon={env.horizon}; "
            "pick a divisor so fragments tile each long episode exactly."
        )
    out_dir.mkdir(parents=True, exist_ok=True) if out_dir is not None else None
    key = jax.random.PRNGKey(seed)
    key, nk = jax.random.split(key)
    policy_dim = _policy_dim(env, cfg)
    _n_scale, scale_expand, scale_names = _scale_group_spec(env, cfg.prior_scale_mode)
    net, params = ppo.init_params(nk, env.obs_size, policy_dim, cfg.hidden)
    optimizer = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(cfg.lr))
    opt_state = optimizer.init(params)
    reset = jax.jit(lambda keys: jax.vmap(env.reset)(keys))
    collect = make_fragment_collect(env=env, net=net, bias=bias, program=program, task=task,
                                    cfg=cfg, deterministic=False)
    update = make_fragment_update(net=net, optimizer=optimizer, cfg=cfg)
    action_prior_weights = (
        bias.default_action_prior_weights()
        if action_prior_weights is None
        else jp.asarray(action_prior_weights, dtype=jp.float32)
    )
    base_reward_weight = jp.asarray(float(cfg.base_reward_weight), dtype=jp.float32)

    if cfg.warmup_compile:
        state0 = reset(jax.random.split(jax.random.PRNGKey(seed + 99), cfg.envs))
        _state, traj, last_value, _summary, _eval_traj = collect(
            params, state0, jax.random.PRNGKey(seed + 100), base_reward_weight, action_prior_weights)
        last_value.block_until_ready()
        data = _ppo_data_from_traj(traj, last_value, cfg.gamma, cfg.lam)
        p2, _os2, _m2 = update(params, opt_state, data, jax.random.PRNGKey(seed + 101))
        jax.block_until_ready(p2)

    key, rk = jax.random.split(key)
    state = reset(jax.random.split(rk, cfg.envs))
    fragments_per_episode = int(env.horizon) // int(cfg.fragment_steps)
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

        key, ck, uk = jax.random.split(key, 3)
        state, traj, last_value, eval_summary, _eval_traj = collect(
            params, state, ck, base_reward_weight, action_prior_weights)
        data = _ppo_data_from_traj(traj, last_value, cfg.gamma, cfg.lam)
        params, opt_state, metrics = update(params, opt_state, data, uk)
        jax.block_until_ready(params)

        fragment_index = it % fragments_per_episode
        if fragment_index == fragments_per_episode - 1:
            key, rk = jax.random.split(key)
            state = reset(jax.random.split(rk, cfg.envs))

        row = _training_row(
            it=it, env_steps=env_steps, fragment_index=fragment_index, task=task, traj=traj,
            eval_summary=eval_summary, metrics=metrics, elapsed=time.monotonic() - start,
            scale_expand=scale_expand, scale_names=scale_names)
        rows.append(row)
        if out_dir is not None:
            with (out_dir / "metrics.jsonl").open("a") as f:
                f.write(json.dumps(row) + "\n")

        if cfg.eval_every > 0 and (it == 0 or (it + 1) % int(cfg.eval_every) == 0):
            ev = evaluate_fragmented_policy(
                env=env, params=params, bias=bias, program=program, task=task,
                seed=seed + 10_000 + it, cfg=cfg, action_prior_weights=action_prior_weights)
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

        if out_dir is not None and cfg.checkpoint_every > 0 and (it + 1) % cfg.checkpoint_every == 0:
            with (out_dir / f"params_iter{it + 1:05d}.pkl").open("wb") as f:
                pickle.dump(jax.device_get(params), f)

    return params, rows, best_params, best_score, best_iter


def evaluate_fragmented_policy(
    *,
    env: Any,
    params: Any,
    bias: CompiledBias,
    program: dict[str, Any],
    task: str,
    seed: int,
    cfg: FragmentedStagePPOConfig,
    action_prior_weights: jp.ndarray | None = None,
) -> dict[str, Any]:
    """Full-episode deterministic evaluation using the same fragmented collector."""
    eval_cfg = FragmentedStagePPOConfig(**{**cfg.__dict__, "envs": int(cfg.eval_envs)})
    policy_dim = _policy_dim(env, cfg)
    _n_scale, scale_expand, scale_names = _scale_group_spec(env, cfg.prior_scale_mode)
    net = ppo.ActorCritic(action_dim=policy_dim, hidden=cfg.hidden)
    reset = jax.jit(lambda keys: jax.vmap(env.reset)(keys))
    collect = make_fragment_collect(env=env, net=net, bias=bias, program=program, task=task,
                                    cfg=eval_cfg, deterministic=True)
    action_prior_weights = (
        bias.default_action_prior_weights()
        if action_prior_weights is None
        else jp.asarray(action_prior_weights, dtype=jp.float32)
    )
    state = reset(jax.random.split(jax.random.PRNGKey(seed), eval_cfg.envs))
    base_reward_weight = jp.asarray(float(eval_cfg.base_reward_weight), dtype=jp.float32)
    traj_parts = []
    summary_parts = []
    for i in range(int(env.horizon) // int(eval_cfg.fragment_steps)):
        state, traj, _last_value, eval_summary, _eval_traj = collect(
            params, state, jax.random.PRNGKey(seed + 1 + i), base_reward_weight,
            action_prior_weights)
        traj_parts.append(jax.device_get(traj))
        summary_parts.append(jax.device_get(eval_summary))
    merged = _merge_eval_summaries(summary_parts)
    es = jp.asarray(merged)  # [n_episodes, 6]: [palm_min, finger_min, n_contacts_max, closure_max, lift_max, xy_max]
    success_rate = float(task_success(task, es).mean())
    fitness = float(task_fitness(task, es).mean())
    base_ret = sum(float(t[5].sum(axis=0).mean()) for t in traj_parts)
    stage_ret = sum(float(t[6].sum(axis=0).mean()) for t in traj_parts)
    train_ret = sum(float(t[4].sum(axis=0).mean()) for t in traj_parts)
    prior_scale = np_mean([t[8] for t in traj_parts])
    action_abs = np_mean([t[7] for t in traj_parts])
    # Arbiter-parity rate fields: SAME eval_summary-column logic as evaluate_ppo_policy, so the
    # trained fragmented policy is scored by the SAME task_graded_objective as the prior_only /
    # short_ppo arbiter -- keeping the objective comparable across the baseline and experimental arms.
    lift_thresh = 0.05
    reach_rate = float((es[:, 2] >= 1.0).mean())
    grasp_rate = float(((es[:, 2] >= 1.0) & (es[:, 3] >= 0.5)).mean())
    grasp_lift_rate = float(((es[:, 2] >= 1.0) & (es[:, 3] >= 0.5) & (es[:, 4] >= lift_thresh)).mean())
    lift_reached_rate = float((es[:, 4] >= lift_thresh).mean())
    lift_max = float(es[:, 4].mean())
    out = {
        "eval_success_rate": round(success_rate, 6),
        "eval_task_fitness": round(fitness, 6),
        "eval_reach_rate": round(reach_rate, 6),
        "eval_grasp_rate": round(grasp_rate, 6),
        "eval_grasp_lift_rate": round(grasp_lift_rate, 6),
        "eval_lift_reached_rate": round(lift_reached_rate, 6),
        "eval_lift_max": round(lift_max, 6),
        "eval_train_return": round(train_ret, 6),
        "eval_base_return": round(base_ret, 6),
        "eval_stage_return": round(stage_ret, 6),
        "eval_prior_scale_mean": round(prior_scale, 6),
        "eval_prior_scale_group_means": _group_scale_means(
            np.concatenate([np.asarray(t[8]).reshape(-1, np.asarray(t[8]).shape[-1])
                            for t in traj_parts if np.asarray(t[8]).ndim == 3], axis=0)
            if any(np.asarray(t[8]).ndim == 3 for t in traj_parts) else np.zeros((1, 1)),
            scale_expand, scale_names),
        "eval_action_abs_mean": round(action_abs, 6),
        "eval_summary": [round(float(x), 6) for x in es.mean(axis=0)],
    }
    out["eval_graded_objective"] = round(task_graded_objective(task, out), 6)
    return out


def make_fragment_collect(
    *,
    env: Any,
    net: Any,
    bias: CompiledBias,
    program: dict[str, Any],
    task: str,
    cfg: FragmentedStagePPOConfig,
    deterministic: bool,
):
    step_fn = jax.vmap(env.step)
    stage_reward_fn, stage_count = make_stage_reward_fn(env, program, cfg)
    n_scale, scale_expand, _scale_names = _scale_group_spec(env, cfg.prior_scale_mode)

    def policy_step(params, obs, key, action_prior_weights):
        mean, log_std, value = net.apply(params, obs)
        log_std = jp.clip(log_std, -5.0, 2.0)
        if deterministic:
            pre_action = mean
        else:
            pre_action = mean + jp.exp(log_std) * jax.random.normal(key, mean.shape)
        if cfg.action_transform == "tanh":
            policy_action = jp.tanh(pre_action)
            logp = squashed_gaussian_logp(policy_action, mean, log_std)
        else:
            policy_action = pre_action
            logp = ppo.gaussian_logp(policy_action, mean, log_std)
        residual = policy_action[:, :env.action_size] * float(cfg.residual_action_scale)
        if cfg.learn_prior_scale:
            # Controller-tuned prior STRENGTH only: per-channel scalars in [0, 1] that multiply the
            # compiled prior's action output. It does NOT touch the prior's gates/limits/conditions
            # (stage selection and done/success thresholds are computed inside weighted_action_prior,
            # independent of this scale). The policy emits n_scale outputs (1 for scalar mode, one per
            # semantic group for group mode, one per actuator for per_joint); each is broadcast to the
            # channels it owns via scale_expand. With bias=gain=1 and the tanh-bounded outputs in
            # (-1, 1), clip(bias + gain*signal, 0, 1) spans the FULL [0, 1] and starts at exactly
            # 1.0 (full prior) when the outputs are ~0 at init; PPO can then scale each group to 0.
            scale_signal = policy_action[:, env.action_size:env.action_size + n_scale]  # [B, n_scale]
            per_channel = scale_signal @ scale_expand  # [B, A]
            prior_scale = jp.clip(float(cfg.prior_scale_bias)
                                  + float(cfg.prior_scale_gain) * per_channel, 0.0, 1.0)
        else:
            prior_scale = jp.ones((obs.shape[0], env.action_size), dtype=jp.float32)
        if cfg.use_action_prior and bias.prior_fn is not None:
            prior = jax.vmap(lambda o: bias.weighted_action_prior(o, action_prior_weights, task))(obs)
            env_action = jp.clip(residual + prior_scale * prior, -1.0, 1.0)
        else:
            env_action = jp.clip(residual, -1.0, 1.0)
        return policy_action, logp, value, env_action, prior_scale

    def collect(params, state, key, base_reward_weight, action_prior_weights):
        def body(carry, _):
            state, key = carry
            key, ak = jax.random.split(key)
            policy_action, logp, value, env_action, prior_scale = policy_step(
                params, state.obs, ak, action_prior_weights)
            nstate = step_fn(state, env_action)
            stage_reward, stage_contrib = jax.vmap(stage_reward_fn)(state.obs, nstate.obs)
            train_reward = base_reward_weight * nstate.reward + stage_reward
            return (nstate, key), (
                state.obs,
                policy_action,
                logp,
                value,
                train_reward,
                nstate.reward,
                stage_reward,
                jp.mean(jp.abs(env_action), axis=-1),
                prior_scale,
                stage_contrib,
                nstate.metrics["eval"],
            )

        (state, key), traj = jax.lax.scan(
            body, (state, key), None, length=int(cfg.fragment_steps))
        _mean, _log_std, last_value = net.apply(params, state.obs)
        eval_traj = traj[10]
        eval_summary = _eval_summary(eval_traj)
        return state, traj, last_value, eval_summary, eval_traj

    return jax.jit(collect)


def make_fragment_update(*, net: Any, optimizer: Any, cfg: FragmentedStagePPOConfig):
    def loss_fn(params, batch):
        obs, action, old_logp, adv, ret = batch
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        mean, log_std, value = net.apply(params, obs)
        log_std = jp.clip(log_std, -5.0, 2.0)
        if cfg.action_transform == "tanh":
            logp = squashed_gaussian_logp(action, mean, log_std)
        else:
            logp = ppo.gaussian_logp(action, mean, log_std)
        ent = ppo.gaussian_entropy(log_std)
        ratio = jp.exp(logp - old_logp)
        pg_loss = -jp.minimum(ratio * adv, jp.clip(ratio, 0.8, 1.2) * adv).mean()
        v_loss = 0.5 * ((value - ret) ** 2).mean()
        entropy = ent.mean()
        loss = pg_loss + 0.5 * v_loss - float(cfg.ent_coef) * entropy
        return loss, {
            "pg_loss": pg_loss,
            "v_loss": v_loss,
            "entropy": entropy,
            "approx_kl": (old_logp - logp).mean(),
        }

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    def update(params, opt_state, data, key):
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
                (_loss, metrics), grads = grad_fn(params, batch)
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


def make_stage_reward_fn(
    env: Any,
    program: dict[str, Any],
    cfg: FragmentedStagePPOConfig,
) -> tuple[Callable[[jp.ndarray, jp.ndarray], tuple[jp.ndarray, jp.ndarray]], int]:
    stages = list(program.get("stages", [])) if program.get("mode") == "freeform_staged" else []
    stage_count = len(stages)
    if not stages or float(cfg.stage_reward_weight) == 0.0:
        def zero_fn(prev_obs, obs):
            return jp.float32(0.0), jp.zeros((max(stage_count, 1),), dtype=jp.float32)
        return zero_fn, stage_count

    signals, signal_names = program_signal_fn(env, program)
    weight_fn, _names = make_stage_weight_fn(env, program)
    compiled = []
    has_success = []
    for st in stages:
        sx = st.get("success")
        if sx is None or str(sx).strip() == "":
            compiled.append(None)
            has_success.append(0.0)
        else:
            compiled.append(compile_expr(str(sx), signal_names))
            has_success.append(1.0)
    has = jp.asarray(has_success, dtype=jp.float32)

    def values(obs):
        sig = signals(obs)
        vals = []
        for ev in compiled:
            vals.append(jp.float32(0.0) if ev is None else jp.asarray(ev(sig), dtype=jp.float32))
        return jp.stack(vals) if vals else jp.zeros((0,), dtype=jp.float32)

    def reward_fn(prev_obs, obs):
        prev_v = values(prev_obs)
        cur_v = values(obs)
        temp = max(float(cfg.stage_success_temperature), 1e-6)
        prev_phi = jax.nn.sigmoid(prev_v / temp)
        cur_phi = jax.nn.sigmoid(cur_v / temp)
        crossed = ((cur_v > 0.0) & (prev_v <= 0.0)).astype(jp.float32)
        active = weight_fn(prev_obs) * has
        contrib = active * (
            float(cfg.stage_progress_weight) * (cur_phi - prev_phi)
            + float(cfg.stage_completion_bonus) * crossed
        )
        reward = float(cfg.stage_reward_weight) * jp.sum(contrib)
        reward = jp.clip(reward, -float(cfg.stage_reward_clip), float(cfg.stage_reward_clip))
        return reward, contrib

    return reward_fn, stage_count


def _ppo_data_from_traj(traj, last_value, gamma: float, lam: float):
    obs, action, logp, value, train_reward = traj[:5]
    adv, ret = ppo.compute_gae(train_reward, value, last_value, gamma, lam)
    flat = lambda x: x.reshape((-1,) + x.shape[2:])
    return flat(obs), flat(action), flat(logp), flat(adv), flat(ret)


def _training_row(
    *,
    it: int,
    env_steps: int,
    fragment_index: int,
    task: str,
    traj: tuple,
    eval_summary: jp.ndarray,
    metrics: dict[str, Any],
    elapsed: float,
    scale_expand: jp.ndarray | None = None,
    scale_names: list[str] | None = None,
) -> dict[str, Any]:
    train_reward, base_reward, stage_reward = traj[4], traj[5], traj[6]
    action_abs, prior_scale, stage_contrib = traj[7], traj[8], traj[9]
    success = task_success(task, eval_summary).mean()
    row = {
        "iter": int(it),
        "env_steps": int(env_steps),
        "fragment_index": int(fragment_index),
        "task": task,
        "fragment_return": round(float(train_reward.sum(axis=0).mean()), 6),
        "base_return": round(float(base_reward.sum(axis=0).mean()), 6),
        "stage_return": round(float(stage_reward.sum(axis=0).mean()), 6),
        "stage_contrib_returns": [
            round(float(x), 6) for x in stage_contrib.sum(axis=0).mean(axis=0)
        ],
        "success": round(float(success), 6),
        "prior_scale_mean": round(float(prior_scale.mean()), 6),
        "prior_scale_min": round(float(prior_scale.min()), 6),
        "prior_scale_max": round(float(prior_scale.max()), 6),
        "action_abs_mean": round(float(action_abs.mean()), 6),
        # Per-group prior strength -- shows e.g. base_z released while fingers held (only when the
        # controller has >1 scale output; a scalar-mode row reports one "all" entry).
        "prior_scale_group_means": (
            _group_scale_means(prior_scale, scale_expand, scale_names)
            if scale_expand is not None and scale_names is not None
            and getattr(prior_scale, "ndim", 0) == 3 else {}),
        "eval_summary": [round(float(x), 6) for x in eval_summary.mean(axis=0)],
        "pg_loss": round(float(metrics["pg_loss"]), 6),
        "v_loss": round(float(metrics["v_loss"]), 6),
        "entropy": round(float(metrics["entropy"]), 6),
        "approx_kl": round(float(metrics["approx_kl"]), 6),
        "elapsed_seconds": round(float(elapsed), 3),
    }
    return row


def _eval_summary(eval_traj: jp.ndarray) -> jp.ndarray:
    return jax.vmap(lambda x: jp.asarray([
        x[:, FIELD_INDEX["palm_obj_dist"]].min(),
        x[:, FIELD_INDEX["min_finger_dist"]].min(),
        x[:, FIELD_INDEX["n_contacts"]].max(),
        x[:, FIELD_INDEX["closure"]].max(),
        x[:, FIELD_INDEX["lift"]].max(),
        x[:, FIELD_INDEX["obj_xy_disp"]].max(),
    ]), in_axes=1)(eval_traj)


def _merge_eval_summaries(parts: list[Any]) -> jp.ndarray:
    arr = jp.stack([jp.asarray(p) for p in parts], axis=0)
    first = jp.min(arr[:, :, :2], axis=0)
    rest = jp.max(arr[:, :, 2:], axis=0)
    return jp.concatenate([first, rest], axis=-1)


def np_mean(xs: list[Any]) -> float:
    vals = [jp.asarray(x).mean() for x in xs]
    return float(jp.stack(vals).mean()) if vals else 0.0

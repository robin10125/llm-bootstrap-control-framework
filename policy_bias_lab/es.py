from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jp
import numpy as np
import pickle
import time

from policy_bias_lab.bias import CompiledBias
from policy_bias_lab.policy import Params, apply_policy, flatten_params, unflatten_params
from policy_bias_lab.schema import EVAL_FIELDS
from policy_bias_lab.tasks import task_fitness, task_success


BIAS_ARMS = {
    "baseline": (False, False, False, False),
    "reward": (True, False, False, False),
    "action_prior": (False, True, False, False),
    "exploration": (False, False, True, False),
    "supervised_init": (False, False, False, True),
    "reward_action_prior": (True, True, False, False),
    "full": (True, True, True, True),
}


@dataclass(frozen=True)
class ESConfig:
    generations: int = 80
    population: int = 64
    # Kept for CLI/config compatibility. Shadow/MJX physics is evaluated one candidate
    # at a time over a vmapped env batch to avoid compiling candidate x env physics graphs.
    population_batch: int = 1
    envs: int = 64
    sigma: float = 0.04
    lr: float = 0.03
    elite_frac: float = 0.5
    supervised_steps: int = 80
    supervised_batch: int = 128
    supervised_lr: float = 1e-3
    target_train_seconds: float | None = None


@dataclass(frozen=True)
class RolloutStats:
    fitness: float
    success_rate: float
    eval_summary: dict[str, float]


def train_arm(
    *,
    env: Any,
    init_params: Params,
    bias: CompiledBias,
    task: str,
    arm: str,
    seed: int,
    cfg: ESConfig,
    checkpoint_dir: Path | None = None,
    checkpoint_generations: set[int] | None = None,
    checkpoint_count: int = 5,
) -> tuple[Params, list[dict[str, Any]]]:
    use_reward_bias, use_action_prior, use_exploration_bias, use_supervised_init = BIAS_ARMS[arm]
    reset_fn = jax.jit(jax.vmap(env.reset))
    step_fn = jax.jit(jax.vmap(env.step))
    key = jax.random.PRNGKey(seed)
    params = init_params
    if use_supervised_init:
        key, sk = jax.random.split(key)
        params = supervised_pretrain(env, params, bias, task, sk, cfg, reset_fn=reset_fn)

    flat, shapes = flatten_params(params)
    evaluate_candidate = make_candidate_evaluator(
        env=env,
        bias=bias,
        task=task,
        shapes=shapes,
        n_envs=cfg.envs,
        use_reward_bias=use_reward_bias,
        use_action_prior=use_action_prior,
    )
    if cfg.target_train_seconds is not None:
        warmup_fit, warmup_success = evaluate_candidate(flat, key)
        warmup_fit.block_until_ready()
        warmup_success.block_until_ready()
    metrics: list[dict[str, Any]] = []
    start_time = time.monotonic()
    checkpoint_times = {
        cfg.target_train_seconds * idx / checkpoint_count
        for idx in range(1, checkpoint_count + 1)
        if cfg.target_train_seconds is not None and checkpoint_count > 0
    }
    saved_time_checkpoints: set[float] = set()
    gen = 0
    while gen < cfg.generations:
        elapsed = time.monotonic() - start_time
        if cfg.target_train_seconds is not None and elapsed >= cfg.target_train_seconds and gen > 0:
            break
        key, nk, rk = jax.random.split(key, 3)
        noise = jax.random.normal(nk, (cfg.population, flat.shape[0]))
        if use_exploration_bias:
            noise = _apply_group_noise(noise, shapes, bias.noise_scale)
        candidates = flat[None, :] + cfg.sigma * noise
        keys = jax.random.split(rk, cfg.population)
        fit_values = []
        success_values = []
        for i in range(cfg.population):
            candidate_fit, candidate_success = evaluate_candidate(candidates[i], keys[i])
            fit_values.append(candidate_fit)
            success_values.append(candidate_success)
        fit = jp.stack(fit_values)
        success = jp.stack(success_values)
        fit.block_until_ready()
        success.block_until_ready()
        adv = (fit - fit.mean()) / (fit.std() + 1e-6)
        elite_count = max(1, int(round(cfg.population * cfg.elite_frac)))
        elite_ids = jp.argsort(fit)[-elite_count:]
        elite_noise = noise[elite_ids]
        elite_adv = adv[elite_ids]
        step = (elite_adv[:, None] * elite_noise).mean(axis=0) / max(cfg.sigma, 1e-6)
        flat = flat + cfg.lr * step
        generation = gen + 1
        elapsed = time.monotonic() - start_time
        time_checkpoint_due = [
            point for point in checkpoint_times if point not in saved_time_checkpoints and elapsed >= point
        ]
        generation_checkpoint_due = checkpoint_generations is not None and generation in checkpoint_generations
        if checkpoint_dir is not None and (time_checkpoint_due or generation_checkpoint_due):
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            if time_checkpoint_due:
                params_for_disk = jax.device_get(unflatten_params(flat, shapes))
                for point in sorted(time_checkpoint_due):
                    checkpoint_idx = int(round(point / cfg.target_train_seconds * checkpoint_count))
                    with (checkpoint_dir / f"params_t{checkpoint_idx:02d}_gen{generation:04d}.pkl").open("wb") as f:
                        pickle.dump(params_for_disk, f)
                saved_time_checkpoints.update(time_checkpoint_due)
            elif generation_checkpoint_due:
                with (checkpoint_dir / f"params_gen{generation:04d}.pkl").open("wb") as f:
                    pickle.dump(jax.device_get(unflatten_params(flat, shapes)), f)
        metrics.append({
            "generation": gen,
            "arm": arm,
            "task": task,
            "mean_fitness": round(float(fit.mean()), 6),
            "best_fitness": round(float(fit.max()), 6),
            "mean_success": round(float(success.mean()), 6),
            "best_success": round(float(success.max()), 6),
            "elapsed_train_seconds": round(elapsed, 3),
        })
        gen += 1
    return unflatten_params(flat, shapes), metrics


def make_candidate_evaluator(
    *,
    env: Any,
    bias: CompiledBias,
    task: str,
    shapes: dict[str, tuple[int, ...]],
    n_envs: int,
    use_reward_bias: bool,
    use_action_prior: bool,
):
    reset = jax.vmap(env.reset)
    step = jax.vmap(env.step)

    def eval_one(flat_params: jp.ndarray, key: jp.ndarray) -> tuple[jp.ndarray, jp.ndarray]:
        params = unflatten_params(flat_params, shapes)
        state = reset(jax.random.split(key, n_envs))
        eval0 = state.metrics["eval"]
        mins0, maxs0 = _initial_reduce(eval0)
        reward0 = jp.zeros((n_envs,), dtype=jp.float32)

        def body(carry, _):
            cur_state, reward_sum, mins, maxs = carry
            if use_action_prior:
                priors = jax.vmap(lambda obs: bias.action_prior(obs, task))(cur_state.obs)
            else:
                priors = jp.zeros((n_envs, env.action_size), dtype=jp.float32)
            actions = jax.vmap(apply_policy, in_axes=(None, 0, 0))(params, cur_state.obs, priors)
            nxt = step(cur_state, actions)
            eval_vec = nxt.metrics["eval"]
            reward = task_fitness(task, eval_vec)
            if use_reward_bias:
                shaped = jax.vmap(lambda ev: bias.shaped_reward(ev, task))(eval_vec)
                reward = reward + shaped
            mins = jp.minimum(mins, eval_vec)
            maxs = jp.maximum(maxs, eval_vec)
            return (nxt, reward_sum + reward, mins, maxs), None

        (_state, reward_sum, mins, maxs), _ = jax.lax.scan(
            body, (state, reward0, mins0, maxs0), None, length=env.horizon
        )
        summary = _select_reduced(mins, maxs)
        episode_fitness = reward_sum / jp.maximum(float(env.horizon), 1.0)
        success = task_success(task, summary)
        return jp.mean(episode_fitness), jp.mean(success.astype(jp.float32))

    return jax.jit(eval_one)


def rollout_policy(
    *,
    env: Any,
    params: Params,
    bias: CompiledBias,
    task: str,
    key: jp.ndarray,
    n_envs: int,
    use_reward_bias: bool,
    use_action_prior: bool,
    reset_fn: Any | None = None,
    step_fn: Any | None = None,
) -> RolloutStats:
    keys = jax.random.split(key, n_envs)
    reset = reset_fn or jax.jit(jax.vmap(env.reset))
    step = step_fn or jax.jit(jax.vmap(env.step))
    state = reset(keys)
    eval0 = state.metrics["eval"]
    mins, maxs = _initial_reduce(eval0)
    reward_sum = jp.zeros((n_envs,), dtype=jp.float32)
    for _ in range(env.horizon):
        if use_action_prior:
            priors = jax.vmap(lambda obs: bias.action_prior(obs, task))(state.obs)
        else:
            priors = jp.zeros((n_envs, env.action_size), dtype=jp.float32)
        actions = jax.vmap(apply_policy, in_axes=(None, 0, 0))(params, state.obs, priors)
        state = step(state, actions)
        eval_vec = state.metrics["eval"]
        base_reward = task_fitness(task, eval_vec)
        if use_reward_bias:
            shaped = jax.vmap(lambda ev: bias.shaped_reward(ev, task))(eval_vec)
            base_reward = base_reward + shaped
        reward_sum = reward_sum + base_reward
        mins = jp.minimum(mins, eval_vec)
        maxs = jp.maximum(maxs, eval_vec)
    summary = _select_reduced(mins, maxs)
    episode_fitness = reward_sum / max(float(env.horizon), 1.0)
    success = task_success(task, summary)
    stats = RolloutStats(
        fitness=round(float(jp.mean(episode_fitness)), 6),
        success_rate=round(float(jp.mean(success.astype(jp.float32))), 6),
        eval_summary={name: round(float(jp.mean(summary[:, i])), 6) for i, name in enumerate(EVAL_FIELDS)},
    )
    return stats


def supervised_pretrain(
    env: Any,
    params: Params,
    bias: CompiledBias,
    task: str,
    key: jp.ndarray,
    cfg: ESConfig,
    *,
    reset_fn: Any | None = None,
) -> Params:
    reset = reset_fn or jax.jit(jax.vmap(env.reset))
    def loss_fn(p: Params, obs: jp.ndarray) -> jp.ndarray:
        target = jax.vmap(lambda o: bias.supervised_target(o, task))(obs)
        pred = jax.vmap(apply_policy, in_axes=(None, 0, 0))(p, obs, jp.zeros_like(target))
        return jp.mean((pred - target) ** 2)

    grad_fn = jax.value_and_grad(loss_fn)
    p = params
    for step_idx in range(cfg.supervised_steps):
        key, rk = jax.random.split(key)
        state = reset(jax.random.split(rk, cfg.supervised_batch))
        _loss, grads = grad_fn(p, state.obs)
        p = jax.tree_util.tree_map(lambda value, grad: value - cfg.supervised_lr * grad, p, grads)
    return p


def evaluate_policy(env: Any, params: Params, bias: CompiledBias, task: str, arm: str, seed: int, n_envs: int) -> RolloutStats:
    use_reward_bias, use_action_prior, _use_exploration_bias, _use_supervised_init = BIAS_ARMS[arm]
    reset_fn = jax.jit(jax.vmap(env.reset))
    step_fn = jax.jit(jax.vmap(env.step))
    return rollout_policy(
        env=env,
        params=params,
        bias=bias,
        task=task,
        key=jax.random.PRNGKey(seed),
        n_envs=n_envs,
        use_reward_bias=use_reward_bias,
        use_action_prior=use_action_prior,
        reset_fn=reset_fn,
        step_fn=step_fn,
    )


def _initial_reduce(eval_vec: jp.ndarray) -> tuple[jp.ndarray, jp.ndarray]:
    return eval_vec, eval_vec


def _select_reduced(mins: jp.ndarray, maxs: jp.ndarray) -> jp.ndarray:
    is_min = jp.asarray([True, True, False, False, False, False])
    return jp.where(is_min[None, :], mins, maxs)


def _apply_group_noise(noise: jp.ndarray, shapes: dict[str, tuple[int, ...]], action_scale: jp.ndarray) -> jp.ndarray:
    # Scale only final-layer output weights/biases by action group; earlier layers remain global.
    parts = []
    pos = 0
    for name in ("w1", "b1", "w2", "b2"):
        size = int(np.prod(shapes[name]))
        part = noise[:, pos: pos + size]
        if name == "w2":
            hidden, action_dim = shapes[name]
            part = part.reshape((noise.shape[0], hidden, action_dim)) * action_scale[None, None, :]
            part = part.reshape((noise.shape[0], size))
        elif name == "b2":
            part = part * action_scale[None, :]
        parts.append(part)
        pos += size
    return jp.concatenate(parts, axis=1)

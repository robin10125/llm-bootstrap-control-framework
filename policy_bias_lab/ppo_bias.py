from __future__ import annotations

import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jp
import optax

BOOTSTRAPPING = Path(__file__).resolve().parents[2] / "bootstrapping"
if str(BOOTSTRAPPING) not in sys.path:
    sys.path.insert(0, str(BOOTSTRAPPING))

import ppo

from policy_bias_lab.bias import CompiledBias, REWARD_TEMPLATE_COUNT, default_reward_template_weights
from policy_bias_lab.es import BIAS_ARMS


@dataclass(frozen=True)
class PPOBiasConfig:
    iters: int = 300
    envs: int = 1024
    lr: float = 3e-4
    gamma: float = 0.99
    lam: float = 0.95
    hidden: tuple[int, ...] = (256, 256)
    ent_coef: float = 0.0
    supervised_steps: int = 80
    supervised_batch: int = 128
    supervised_lr: float = 1e-3
    checkpoint_count: int = 5
    target_train_seconds: float | None = None
    # Sample-efficiency budget: stop the arm once this many environment steps have been
    # collected (envs * horizon per iteration). Lets arms be compared on equal data rather
    # than equal wall-clock, which is the quantity that is scarce on real robots.
    max_env_steps: int | None = None
    action_transform: str = "tanh"
    saturation_penalty: float = 0.0
    saturation_threshold: float = 0.98
    prior_logit_clip: float = 0.95
    action_target_reward_weight: float = 0.0
    success_hold_seconds: float = 0.5
    success_lift_threshold: float = 0.05
    warmup_compile: bool = True


def train_ppo_arm(
    *,
    env: Any,
    bias: CompiledBias,
    task: str,
    arm: str,
    seed: int,
    cfg: PPOBiasConfig,
    checkpoint_dir: Path | None = None,
    reward_weights: jp.ndarray | None = None,
    base_reward_weight: float = 1.0,
    checkup_interval: int = 0,
    checkup_fn: Any | None = None,
    action_prior_weights: jp.ndarray | None = None,
    action_prior_checkup_interval: int = 0,
    action_prior_checkup_fn: Any | None = None,
    initial_params: Any | None = None,
    iter_offset: int = 0,
) -> tuple[Any, list[dict[str, Any]]]:
    use_reward_bias, use_action_prior, use_exploration_bias, use_supervised_init = BIAS_ARMS[arm]
    key = jax.random.PRNGKey(seed)
    key, nk = jax.random.split(key)
    net, params = ppo.init_params(nk, env.obs_size, env.action_size, cfg.hidden)
    if initial_params is not None:
        params = initial_params
    optimizer = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(cfg.lr))
    opt_state = optimizer.init(params)

    if use_supervised_init and initial_params is None:
        key, sk = jax.random.split(key)
        params = supervised_pretrain(net, env, params, bias, task, sk, cfg)
        opt_state = optimizer.init(params)

    reset = jax.jit(lambda keys: jax.vmap(env.reset)(keys))
    collect = make_collect(
        env=env,
        net=net,
        bias=bias,
        task=task,
        use_reward_bias=use_reward_bias,
        use_action_prior=use_action_prior,
        use_exploration_bias=use_exploration_bias,
        action_transform=cfg.action_transform,
        saturation_penalty=cfg.saturation_penalty,
        saturation_threshold=cfg.saturation_threshold,
        prior_logit_clip=cfg.prior_logit_clip,
        action_target_reward_weight=cfg.action_target_reward_weight,
    )
    update = make_update(
        net=net,
        optimizer=optimizer,
        bias=bias,
        task=task,
        use_action_prior=use_action_prior,
        use_exploration_bias=use_exploration_bias,
        ent_coef=cfg.ent_coef,
        action_transform=cfg.action_transform,
        prior_logit_clip=cfg.prior_logit_clip,
    )
    checkpoint_iters = _checkpoint_iters(cfg.iters, cfg.checkpoint_count)
    checkpoint_times = _checkpoint_times(cfg.target_train_seconds, cfg.checkpoint_count)
    saved_time_checkpoints: set[int] = set()
    rows: list[dict[str, Any]] = []
    reward_weights = default_reward_template_weights(task) if reward_weights is None else jp.asarray(reward_weights, dtype=jp.float32)
    base_reward_weight_arr = jp.asarray(float(base_reward_weight), dtype=jp.float32)
    action_prior_weights = (
        bias.default_action_prior_weights()
        if action_prior_weights is None
        else jp.asarray(action_prior_weights, dtype=jp.float32)
    )
    current_checkup_interval = int(checkup_interval)

    if cfg.target_train_seconds is not None and cfg.warmup_compile:
        print(f"[{arm}] warmup compile start envs={cfg.envs} horizon={env.horizon}", flush=True)
        key, rk, ck, uk = jax.random.split(key, 4)
        state = reset(jax.random.split(rk, cfg.envs))
        warmup_state, warmup_traj, warmup_last_value, _warmup_summary = collect(
            params, state, ck, reward_weights, base_reward_weight_arr, action_prior_weights
        )
        warmup_state.reward.block_until_ready()
        warmup_last_value.block_until_ready()
        print(f"[{arm}] warmup collect ready", flush=True)
        obs, action, logp, value, train_reward, _success, _lift, _base_reward, _shaped_reward, *_ = warmup_traj
        adv, ret = ppo.compute_gae(train_reward, value, warmup_last_value, cfg.gamma, cfg.lam)
        flat = lambda x: x.reshape((-1,) + x.shape[2:])
        warmup_data = (flat(obs), flat(action), flat(logp), flat(adv), flat(ret))
        warmup_params, _warmup_opt_state, _warmup_metrics = update(params, opt_state, warmup_data, uk, action_prior_weights)
        jax.block_until_ready(warmup_params)
        print(f"[{arm}] warmup update ready", flush=True)

    steps_per_iter = int(cfg.envs) * int(env.horizon)
    start = time.monotonic()
    for it in range(cfg.iters):
        if cfg.target_train_seconds is not None and it > 0 and time.monotonic() - start >= cfg.target_train_seconds:
            break
        if cfg.max_env_steps is not None and it > 0 and (iter_offset + it) * steps_per_iter >= cfg.max_env_steps:
            break
        key, rk, ck, uk = jax.random.split(key, 4)
        state = reset(jax.random.split(rk, cfg.envs))
        if it == 0:
            print(f"[{arm}] iter0 collect start", flush=True)
        state, traj, last_value, eval_summary = collect(
            params, state, ck, reward_weights, base_reward_weight_arr, action_prior_weights
        )
        if it == 0:
            state.reward.block_until_ready()
            last_value.block_until_ready()
            print(f"[{arm}] iter0 collect ready", flush=True)
        (obs, action, logp, value, train_reward, success, lift, base_reward,
         shaped_reward, hard_clip_frac, saturation_frac, action_abs_mean, reward_contrib) = traj
        adv, ret = ppo.compute_gae(train_reward, value, last_value, cfg.gamma, cfg.lam)
        flat = lambda x: x.reshape((-1,) + x.shape[2:])
        data = (flat(obs), flat(action), flat(logp), flat(adv), flat(ret))
        if it == 0:
            print(f"[{arm}] iter0 update start", flush=True)
        params, opt_state, metrics = update(params, opt_state, data, uk, action_prior_weights)
        jax.block_until_ready(params)
        if it == 0:
            print(f"[{arm}] iter0 update ready", flush=True)
        elapsed = time.monotonic() - start
        due_time_checkpoints = {
            idx for idx, seconds in checkpoint_times.items()
            if idx not in saved_time_checkpoints and elapsed >= seconds
        }
        if checkpoint_dir is not None and ((it + 1) in checkpoint_iters or due_time_checkpoints):
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            params_for_disk = jax.device_get(params)
            if due_time_checkpoints:
                for idx in sorted(due_time_checkpoints):
                    with (checkpoint_dir / f"params_t{idx:02d}_iter{iter_offset + it + 1:04d}.pkl").open("wb") as f:
                        pickle.dump(params_for_disk, f)
                saved_time_checkpoints.update(due_time_checkpoints)
            elif (it + 1) in checkpoint_iters:
                with (checkpoint_dir / f"params_t{checkpoint_iters[it + 1]:02d}_iter{iter_offset + it + 1:04d}.pkl").open("wb") as f:
                    pickle.dump(params_for_disk, f)
        instant_success_rate = (success.max(axis=0) > 0.5).mean()
        sustained_success_rate = sustained_lift_success(
            lift,
            control_dt=float(env.cfg.control_dt),
            hold_seconds=cfg.success_hold_seconds,
            lift_threshold=cfg.success_lift_threshold,
        )
        rows.append({
            "iter": iter_offset + it,
            "env_steps": (iter_offset + it + 1) * steps_per_iter,
            "arm": arm,
            "task": task,
            "train_return": round(float(train_reward.sum(axis=0).mean()), 6),
            "base_return": round(float(base_reward.sum(axis=0).mean()), 6),
            "base_reward_weight": round(float(base_reward_weight_arr), 6),
            "shaped_return": round(float(shaped_reward.sum(axis=0).mean()), 6),
            "reward_template_returns": [round(float(x), 6) for x in reward_contrib.sum(axis=0).mean(axis=0)],
            "action_prior_weights": [round(float(x), 6) for x in action_prior_weights],
            "success": round(float(sustained_success_rate), 6),
            "instant_success": round(float(instant_success_rate), 6),
            "lift_max": round(float(lift.max(axis=0).mean()), 6),
            "eval_summary": [round(float(x), 6) for x in jp.mean(eval_summary, axis=0)],
            "hard_clip_frac": round(float(hard_clip_frac.mean()), 6),
            "saturation_frac": round(float(saturation_frac.mean()), 6),
            "action_abs_mean": round(float(action_abs_mean.mean()), 6),
            "pg_loss": round(float(metrics["pg_loss"]), 6),
            "v_loss": round(float(metrics["v_loss"]), 6),
            "entropy": round(float(metrics["entropy"]), 6),
            "approx_kl": round(float(metrics["approx_kl"]), 6),
            "elapsed_seconds": round(elapsed, 3),
        })
        if checkup_fn is not None and current_checkup_interval > 0 and (it + 1) % current_checkup_interval == 0:
            result = checkup_fn({
                "iter": it,
                "arm": arm,
                "task": task,
                "elapsed_seconds": elapsed,
                "rows": rows,
                "reward_weights": jax.device_get(reward_weights).tolist(),
                "base_reward_weight": float(base_reward_weight_arr),
            })
            if isinstance(result, dict):
                if "reward_weights" in result:
                    reward_weights = jp.asarray(result["reward_weights"], dtype=jp.float32)
                if "base_reward_weight" in result:
                    base_reward_weight_arr = jp.asarray(float(result["base_reward_weight"]), dtype=jp.float32)
                if "next_checkup_interval" in result:
                    current_checkup_interval = max(1, int(result["next_checkup_interval"]))
            else:
                reward_weights = jp.asarray(result, dtype=jp.float32)
        if (
            action_prior_checkup_fn is not None
            and action_prior_checkup_interval > 0
            and (it + 1) % action_prior_checkup_interval == 0
        ):
            action_prior_weights = jp.asarray(action_prior_checkup_fn({
                "iter": iter_offset + it,
                "arm": arm,
                "task": task,
                "elapsed_seconds": elapsed,
                "rows": rows,
                "action_prior_weights": jax.device_get(action_prior_weights).tolist(),
            }), dtype=jp.float32)
    if checkpoint_dir is not None and cfg.checkpoint_count > 0:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        with (checkpoint_dir / f"params_t{cfg.checkpoint_count:02d}_final.pkl").open("wb") as f:
            pickle.dump(jax.device_get(params), f)
    return params, rows


def evaluate_ppo_policy(
    *,
    env: Any,
    params: Any,
    bias: CompiledBias,
    task: str,
    arm: str,
    seed: int,
    n_envs: int,
    cfg: PPOBiasConfig | None = None,
    reward_weights: jp.ndarray | None = None,
    base_reward_weight: float = 1.0,
    action_prior_weights: jp.ndarray | None = None,
) -> dict[str, Any]:
    use_reward_bias, use_action_prior, use_exploration_bias, _ = BIAS_ARMS[arm]
    net = ppo.ActorCritic(action_dim=env.action_size, hidden=_infer_hidden(params))
    reset = jax.jit(lambda keys: jax.vmap(env.reset)(keys))
    collect = make_collect(
        env=env,
        net=net,
        bias=bias,
        task=task,
        use_reward_bias=use_reward_bias,
        use_action_prior=use_action_prior,
        use_exploration_bias=use_exploration_bias,
        deterministic=True,
        action_transform=cfg.action_transform if cfg is not None else "tanh",
        saturation_threshold=cfg.saturation_threshold if cfg is not None else 0.98,
        prior_logit_clip=cfg.prior_logit_clip if cfg is not None else 0.95,
        action_target_reward_weight=cfg.action_target_reward_weight if cfg is not None else 0.0,
    )
    reward_weights = default_reward_template_weights(task) if reward_weights is None else jp.asarray(reward_weights, dtype=jp.float32)
    base_reward_weight_arr = jp.asarray(float(base_reward_weight), dtype=jp.float32)
    action_prior_weights = (
        bias.default_action_prior_weights()
        if action_prior_weights is None
        else jp.asarray(action_prior_weights, dtype=jp.float32)
    )
    state = reset(jax.random.split(jax.random.PRNGKey(seed), n_envs))
    _state, traj, _last_value, eval_summary = collect(
        params, state, jax.random.PRNGKey(seed + 1), reward_weights, base_reward_weight_arr, action_prior_weights
    )
    (_obs, _action, _logp, _value, train_reward, success, lift, base_reward,
     shaped_reward, hard_clip_frac, saturation_frac, action_abs_mean, reward_contrib) = traj
    instant_success_rate = (success.max(axis=0) > 0.5).mean()
    sustained_success_rate = sustained_lift_success(
        lift,
        control_dt=float(env.cfg.control_dt),
        hold_seconds=cfg.success_hold_seconds if cfg is not None else 0.5,
        lift_threshold=cfg.success_lift_threshold if cfg is not None else 0.05,
    )
    return {
        "eval_base_return": round(float(base_reward.sum(axis=0).mean()), 6),
        "eval_base_reward_weight": round(float(base_reward_weight_arr), 6),
        "eval_shaped_return": round(float(shaped_reward.sum(axis=0).mean()), 6),
        "eval_reward_template_returns": [round(float(x), 6) for x in reward_contrib.sum(axis=0).mean(axis=0)],
        "eval_action_prior_weights": [round(float(x), 6) for x in action_prior_weights],
        "eval_train_return": round(float(train_reward.sum(axis=0).mean()), 6),
        "eval_success_rate": round(float(sustained_success_rate), 6),
        "eval_instant_success_rate": round(float(instant_success_rate), 6),
        "eval_lift_max": round(float(lift.max(axis=0).mean()), 6),
        "eval_hard_clip_frac": round(float(hard_clip_frac.mean()), 6),
        "eval_saturation_frac": round(float(saturation_frac.mean()), 6),
        "eval_action_abs_mean": round(float(action_abs_mean.mean()), 6),
        "eval_summary": [round(float(x), 6) for x in jp.mean(eval_summary, axis=0)],
    }


def make_collect(
    *,
    env: Any,
    net: Any,
    bias: CompiledBias,
    task: str,
    use_reward_bias: bool,
    use_action_prior: bool,
    use_exploration_bias: bool,
    deterministic: bool = False,
    action_transform: str = "tanh",
    saturation_penalty: float = 0.0,
    saturation_threshold: float = 0.98,
    prior_logit_clip: float = 0.95,
    action_target_reward_weight: float = 0.0,
):
    step_fn = jax.vmap(env.step)
    noise_scale = jp.clip(bias.noise_scale, 0.1, 4.0)

    def policy_dist(params, obs, action_prior_weights):
        mean, log_std, value = net.apply(params, obs)
        if use_action_prior:
            prior = jax.vmap(lambda o: bias.weighted_action_prior(o, action_prior_weights, task))(obs)
            if action_transform == "tanh":
                prior = _atanh_clipped(prior, prior_logit_clip)
            mean = mean + prior
        if use_exploration_bias:
            log_std = log_std + jp.log(noise_scale)
        return mean, jp.clip(log_std, -5.0, 2.0), value

    def collect(params, state, key, reward_weights, base_reward_weight, action_prior_weights):
        def body(carry, _):
            state, key = carry
            key, ak = jax.random.split(key)
            mean, log_std, value = policy_dist(params, state.obs, action_prior_weights)
            if deterministic:
                pre_action = mean
            else:
                pre_action = mean + jp.exp(log_std) * jax.random.normal(ak, mean.shape)
            if action_transform == "tanh":
                action = jp.tanh(pre_action)
                logp = squashed_gaussian_logp(action, mean, log_std)
            else:
                action = pre_action
                logp = ppo.gaussian_logp(action, mean, log_std)
            env_action = jp.clip(action, -1.0, 1.0)
            hard_clip_frac = jp.mean((jp.abs(action) > 1.0).astype(jp.float32), axis=-1)
            saturation_frac = jp.mean((jp.abs(env_action) >= saturation_threshold).astype(jp.float32), axis=-1)
            action_abs_mean = jp.mean(jp.abs(env_action), axis=-1)
            prev_eval = state.metrics["eval"]
            nstate = step_fn(state, action)
            base_reward = nstate.reward
            shaped = jp.zeros_like(base_reward)
            reward_contrib = jp.zeros((base_reward.shape[0], REWARD_TEMPLATE_COUNT), dtype=jp.float32)
            if use_reward_bias:
                shaped, reward_contrib = jax.vmap(
                    lambda pe, ev: bias.dynamic_shaped_reward(pe, ev, reward_weights, task)
                )(prev_eval, nstate.metrics["eval"])
                if action_target_reward_weight > 0.0:
                    shaped = shaped + float(action_target_reward_weight) * jax.vmap(
                        lambda o, a: bias.action_target_reward(o, a, task)
                    )(state.obs, action)
            if saturation_penalty > 0.0:
                excess = jp.maximum(jp.abs(env_action) - saturation_threshold, 0.0)
                denom = jp.maximum(1.0 - saturation_threshold, 1e-6)
                shaped = shaped - float(saturation_penalty) * jp.mean(excess / denom, axis=-1)
            train_reward = base_reward_weight * base_reward + shaped
            return (nstate, key), (state.obs, action, logp, value, train_reward,
                                   nstate.metrics["success"], nstate.metrics["lift"],
                                   base_reward, shaped,
                                   hard_clip_frac, saturation_frac, action_abs_mean,
                                   reward_contrib,
                                   nstate.metrics["eval"])

        (state, key), traj = jax.lax.scan(body, (state, key), None, length=env.horizon)
        _, _, last_value = net.apply(params, state.obs)
        eval_traj = traj[13]
        eval_summary = jax.vmap(lambda x: jp.asarray([
            x[:, 0].min(), x[:, 1].min(), x[:, 2].max(), x[:, 3].max(), x[:, 4].max(), x[:, 5].max()
        ]), in_axes=1)(eval_traj)
        return state, traj[:13], last_value, eval_summary

    return jax.jit(collect)


def make_update(
    *,
    net: Any,
    optimizer: Any,
    bias: CompiledBias,
    task: str,
    use_action_prior: bool,
    use_exploration_bias: bool,
    ent_coef: float,
    action_transform: str = "tanh",
    prior_logit_clip: float = 0.95,
):
    noise_scale = jp.clip(bias.noise_scale, 0.1, 4.0)

    def evaluate(params, obs, action, action_prior_weights):
        mean, log_std, value = net.apply(params, obs)
        if use_action_prior:
            prior = jax.vmap(lambda o: bias.weighted_action_prior(o, action_prior_weights, task))(obs)
            if action_transform == "tanh":
                prior = _atanh_clipped(prior, prior_logit_clip)
            mean = mean + prior
        if use_exploration_bias:
            log_std = log_std + jp.log(noise_scale)
        log_std = jp.clip(log_std, -5.0, 2.0)
        if action_transform == "tanh":
            logp = squashed_gaussian_logp(action, mean, log_std)
        else:
            logp = ppo.gaussian_logp(action, mean, log_std)
        return logp, ppo.gaussian_entropy(log_std), value

    def loss_fn(params, batch, action_prior_weights):
        obs, action, old_logp, adv, ret = batch
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        logp, ent, value = evaluate(params, obs, action, action_prior_weights)
        ratio = jp.exp(logp - old_logp)
        pg_loss = -jp.minimum(ratio * adv, jp.clip(ratio, 0.8, 1.2) * adv).mean()
        v_loss = 0.5 * ((value - ret) ** 2).mean()
        loss = pg_loss + 0.5 * v_loss - ent_coef * ent.mean()
        return loss, {
            "pg_loss": pg_loss,
            "v_loss": v_loss,
            "entropy": ent.mean(),
            "approx_kl": jp.mean(old_logp - logp),
        }

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    def update(params, opt_state, data, key, action_prior_weights):
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
                (_loss, metrics), grads = grad_fn(params, batch, action_prior_weights)
                updates, opt_state = optimizer.update(grads, opt_state, params)
                params = optax.apply_updates(params, updates)
                return (params, opt_state), metrics

            (params, opt_state), metrics = jax.lax.scan(minibatch, (params, opt_state), jp.arange(num_minibatches))
            return (params, opt_state, key), metrics

        (params, opt_state, _key), metrics = jax.lax.scan(epoch, (params, opt_state, key), None, length=4)
        return params, opt_state, jax.tree_util.tree_map(lambda x: x.mean(), metrics)

    return jax.jit(update)


def sustained_lift_success(
    lift: jp.ndarray,
    *,
    control_dt: float,
    hold_seconds: float,
    lift_threshold: float,
) -> jp.ndarray:
    """Fraction of episodes with lift held above threshold for a continuous window."""
    hold_steps = max(1, int(round(float(hold_seconds) / max(float(control_dt), 1e-9))))
    above = lift > float(lift_threshold)

    def episode_success(ep_above: jp.ndarray) -> jp.ndarray:
        def body(run_len, is_above):
            return jp.where(is_above, run_len + 1, 0), jp.where(is_above, run_len + 1, 0)

        _last, run_lengths = jax.lax.scan(body, jp.int32(0), ep_above)
        return (run_lengths.max() >= hold_steps).astype(jp.float32)

    return jax.vmap(episode_success, in_axes=1)(above).mean()


def squashed_gaussian_logp(action: jp.ndarray, mean: jp.ndarray, log_std: jp.ndarray) -> jp.ndarray:
    clipped = jp.clip(action, -0.999999, 0.999999)
    raw = _atanh(clipped)
    correction = jp.sum(jp.log(1.0 - clipped * clipped + 1e-6), axis=-1)
    return ppo.gaussian_logp(raw, mean, log_std) - correction


def _atanh_clipped(value: jp.ndarray, limit: float) -> jp.ndarray:
    limit = float(min(max(limit, 0.0), 0.999999))
    return _atanh(jp.clip(value, -limit, limit))


def _atanh(value: jp.ndarray) -> jp.ndarray:
    return 0.5 * (jp.log1p(value) - jp.log1p(-value))


def supervised_pretrain(net, env: Any, params: Any, bias: CompiledBias, task: str, key, cfg: PPOBiasConfig):
    reset = jax.jit(lambda keys: jax.vmap(env.reset)(keys))
    optimizer = optax.adam(cfg.supervised_lr)
    opt_state = optimizer.init(params)

    def loss_fn(p, obs):
        target = jax.vmap(lambda o: bias.supervised_target(o, task))(obs)
        mean, _log_std, _value = net.apply(p, obs)
        return jp.mean((jp.tanh(mean) - target) ** 2)

    grad_fn = jax.value_and_grad(loss_fn)
    for _ in range(cfg.supervised_steps):
        key, rk = jax.random.split(key)
        state = reset(jax.random.split(rk, cfg.supervised_batch))
        loss, grads = grad_fn(params, state.obs)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
    loss.block_until_ready()
    return params


def _checkpoint_iters(iters: int, count: int) -> dict[int, int]:
    if iters <= 0 or count <= 0:
        return {}
    return {max(1, min(iters, round(iters * idx / count))): idx for idx in range(1, count + 1)}


def _checkpoint_times(seconds: float | None, count: int) -> dict[int, float]:
    if seconds is None or seconds <= 0 or count <= 0:
        return {}
    return {idx: seconds * idx / count for idx in range(1, count + 1)}


def _infer_hidden(params: Any) -> tuple[int, ...]:
    action_dim = int(params["params"]["log_std"].shape[0])
    hidden: list[int] = []
    idx = 0
    while f"Dense_{idx}" in params["params"]:
        width = int(params["params"][f"Dense_{idx}"]["kernel"].shape[1])
        if width == action_dim:
            break
        hidden.append(width)
        idx += 1
    return tuple(hidden)

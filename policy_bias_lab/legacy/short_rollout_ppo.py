from __future__ import annotations

import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jp
import numpy as np
import optax

BOOTSTRAPPING = Path(__file__).resolve().parents[3] / "bootstrapping"
if str(BOOTSTRAPPING) not in sys.path:
    sys.path.insert(0, str(BOOTSTRAPPING))

import ppo

from policy_bias_lab.bias import CompiledBias, REWARD_TEMPLATE_COUNT, default_reward_template_weights
from policy_bias_lab.es import BIAS_ARMS
from policy_bias_lab.tasks import task_failure_signal


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
    # Reformatted warm-start (BC->RL handoff): clone on controller-rollout states and pretrain
    # the critic on those returns (so PPO's first advantages aren't garbage), then anchor the
    # actor to the BC policy with a KL penalty that decays over bc_kl_anneal_iters.
    bc_critic_pretrain: bool = True
    bc_rollout_states: bool = True
    bc_kl_coef: float = 0.0
    bc_kl_anneal_iters: int = 200
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
    # Early termination on sustained success: once the per-step env success metric holds for this
    # many consecutive seconds, the episode is treated as DONE for credit assignment -- later steps
    # carry no reward, no value target, and no loss weight, and the GAE bootstrap is dropped for
    # episodes that finished. The fixed-horizon scan still steps the physics (it cannot stop), so
    # this changes what the policy is trained on, not the compute per iteration. None = off.
    success_terminate_seconds: float | None = None
    # FAILURE termination (the mirror): when the task's failure signal (tasks.task_failure_signal,
    # e.g. object knocked beyond a recoverable radius) holds for this many consecutive seconds,
    # the episode's credit ends -- post-mistake steps carry no reward/value/loss weight, so the
    # actions BEFORE the mistake bear its full cost (concentrated credit assignment). Same masking
    # mechanics as success termination; saves no sim compute (fixed scan). None = off.
    failure_terminate_seconds: float | None = None
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
    phase_teacher: Any | None = None,
    initial_opt_state: Any | None = None,
    control_fn: Any | None = None,
    state_out: dict | None = None,
    shaping_fn: Any | None = None,
) -> tuple[Any, list[dict[str, Any]]]:
    use_reward_bias, use_action_prior, use_exploration_bias, use_supervised_init = BIAS_ARMS[arm]
    key = jax.random.PRNGKey(seed)
    key, nk = jax.random.split(key)
    net, params = ppo.init_params(nk, env.obs_size, env.action_size, cfg.hidden)
    if initial_params is not None:
        params = initial_params
    optimizer = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(cfg.lr))
    opt_state = optimizer.init(params)

    reference_params = params
    bc_anchor_active = False
    if use_supervised_init and initial_params is None:
        key, sk = jax.random.split(key)
        params = supervised_pretrain(net, env, params, bias, task, sk, cfg, phase_teacher=phase_teacher)
        opt_state = optimizer.init(params)
        reference_params = params  # frozen BC policy for the decaying KL anchor
        bc_anchor_active = cfg.bc_kl_coef > 0.0 and cfg.bc_kl_anneal_iters > 0
    if initial_opt_state is not None:  # resumed run: keep the Adam moments, don't restart them
        opt_state = initial_opt_state

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
        shaping_fn=shaping_fn,
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
    best_success = -1.0
    best_score = -1.0
    best_iter = -1
    best_params = params
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
        warmup_state, warmup_traj, warmup_last_value, _warmup_summary, _warmup_fail = collect(
            params, state, ck, reward_weights, base_reward_weight_arr, action_prior_weights
        )
        warmup_state.reward.block_until_ready()
        warmup_last_value.block_until_ready()
        print(f"[{arm}] warmup collect ready", flush=True)
        obs, action, logp, value, train_reward, _success, _lift, _base_reward, _shaped_reward, *_ = warmup_traj
        adv, ret = ppo.compute_gae(train_reward, value, warmup_last_value, cfg.gamma, cfg.lam)
        flat = lambda x: x.reshape((-1,) + x.shape[2:])
        warmup_data = (flat(obs), flat(action), flat(logp), flat(adv), flat(ret),
                       flat(jp.ones_like(train_reward)))
        warmup_params, _warmup_opt_state, _warmup_metrics = update(
            params, opt_state, warmup_data, uk, action_prior_weights, reference_params, jp.float32(0.0))
        jax.block_until_ready(warmup_params)
        print(f"[{arm}] warmup update ready", flush=True)

    steps_per_iter = int(cfg.envs) * int(env.horizon)
    term_window = 0
    if cfg.success_terminate_seconds:
        term_window = max(1, round(float(cfg.success_terminate_seconds)
                                   / float(env.cfg.control_dt)))
        print(f"[{arm}] early termination: sustained success for "
              f"{cfg.success_terminate_seconds}s ({term_window} steps) ends the episode's credit",
              flush=True)
    fail_window = 0
    if cfg.failure_terminate_seconds:
        fail_window = max(1, round(float(cfg.failure_terminate_seconds)
                                   / float(env.cfg.control_dt)))
        print(f"[{arm}] failure termination: the task failure signal sustained for "
              f"{cfg.failure_terminate_seconds}s ({fail_window} steps) ends the episode's credit",
              flush=True)
    stop_reason: str | None = None
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
        state, traj, last_value, eval_summary, fail_signal = collect(
            params, state, ck, reward_weights, base_reward_weight_arr, action_prior_weights
        )
        if it == 0:
            state.reward.block_until_ready()
            last_value.block_until_ready()
            print(f"[{arm}] iter0 collect ready", flush=True)
        (obs, action, logp, value, train_reward, success, lift, base_reward,
         shaped_reward, hard_clip_frac, saturation_frac, action_abs_mean, reward_contrib) = traj
        if term_window or fail_window:
            # Early termination on sustained success and/or a sustained task failure signal:
            # post-terminal steps carry no reward, no value target, and (via the valid mask) no
            # loss weight; finished episodes drop the bootstrap value, so the tail after "done"
            # cannot leak credit either way. Failure termination concentrates the mistake's cost
            # on the actions that caused it instead of diluting it over the chase that follows.
            alive = jp.ones_like(train_reward)
            alive_end = jp.ones_like(last_value)
            if term_window:
                a_s, ae_s = success_termination_mask(success, term_window)
                alive, alive_end = alive * a_s, alive_end * ae_s
            if fail_window:
                a_f, ae_f = success_termination_mask(fail_signal.astype(jp.float32), fail_window)
                alive, alive_end = alive * a_f, alive_end * ae_f
            adv, ret = ppo.compute_gae(train_reward * alive, value * alive,
                                       last_value * alive_end, cfg.gamma, cfg.lam)
            valid = alive
        else:
            adv, ret = ppo.compute_gae(train_reward, value, last_value, cfg.gamma, cfg.lam)
            valid = jp.ones_like(train_reward)
        flat = lambda x: x.reshape((-1,) + x.shape[2:])
        data = (flat(obs), flat(action), flat(logp), flat(adv), flat(ret), flat(valid))
        if it == 0:
            print(f"[{arm}] iter0 update start", flush=True)
        kl_coef = jp.float32(
            cfg.bc_kl_coef * max(0.0, 1.0 - it / float(cfg.bc_kl_anneal_iters))
            if bc_anchor_active else 0.0
        )
        params, opt_state, metrics = update(params, opt_state, data, uk, action_prior_weights, reference_params, kl_coef)
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
            gate=success,  # contact-gated env success -> sustained GRASP, not sustained fling
        )
        # Per-stage rates make curriculum promotion criteria observable: eval_summary columns are
        # [palm_min, finger_min, n_contacts_max, closure_max, lift_max, xy_max] per episode.
        reach_rate = float((eval_summary[:, 2] >= 1.0).mean())
        grasp_rate = float(((eval_summary[:, 2] >= 1.0) & (eval_summary[:, 3] >= 0.5)).mean())
        lift_reached_rate = float((eval_summary[:, 4] >= cfg.success_lift_threshold).mean())
        # Contact-gated lifted grip (grasp AND raised): the real bridge from a secure grip to success,
        # and unlike lift_reached it cannot be inflated by flinging.
        grasp_lift_rate = float(((eval_summary[:, 2] >= 1.0) & (eval_summary[:, 3] >= 0.5)
                                 & (eval_summary[:, 4] >= cfg.success_lift_threshold)).mean())
        # Best-checkpoint selection. Sustained contact-gated success is primary; it stays 0 for long
        # stretches (and whole runs), so fall back to a CONTACT-GATED progress score
        # (grasp -> gated-lift). NOTE: lift_reached alone is NOT contact-gated -- flinging inflates it
        # -- so it is only a negligible tie-breaker here. The earlier version weighted lift_reached
        # (1e-3) far ABOVE grasp (1e-6) and so picked early flingy checkpoints over a genuinely learned
        # grip (e.g. it hid reactive_law's sustained 0.79 grasp behind an iter-4 pick). Ordering:
        # success (1.0) >> gated lifted grip (1e-2) >> secure grip (1e-4) >> raw lift tie-break (1e-7).
        best_metric = (float(sustained_success_rate)
                       + 1e-2 * grasp_lift_rate
                       + 1e-4 * grasp_rate
                       + 1e-7 * lift_reached_rate)
        if best_metric > best_score:
            best_score = best_metric
            best_success = float(sustained_success_rate)
            best_iter = iter_offset + it
            best_params = params
        rows.append({
            "iter": iter_offset + it,
            "env_steps": (iter_offset + it + 1) * steps_per_iter,
            "arm": arm,
            "task": task,
            "reach_rate": round(reach_rate, 6),
            "grasp_rate": round(grasp_rate, 6),
            "lift_reached_rate": round(lift_reached_rate, 6),
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
            "kl_anchor": round(float(metrics.get("kl_anchor", 0.0)), 6),
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
        if control_fn is not None:
            # External run control (long-training runner): sees full live state each iteration and
            # may snapshot it (resume checkpoints) or end the run by returning a reason string.
            stop_reason = control_fn({
                "iter": iter_offset + it, "elapsed_seconds": elapsed, "row": rows[-1],
                "params": params, "opt_state": opt_state, "best_params": best_params,
                "best_score": best_score, "best_success": best_success, "best_iter": best_iter,
            })
            if stop_reason:
                break
    if state_out is not None:
        state_out.update(opt_state=opt_state, best_score=best_score, stop_reason=stop_reason)
    if checkpoint_dir is not None and cfg.checkpoint_count > 0:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        with (checkpoint_dir / f"params_t{cfg.checkpoint_count:02d}_final.pkl").open("wb") as f:
            pickle.dump(jax.device_get(params), f)
        with (checkpoint_dir / f"params_best_iter{best_iter:04d}.pkl").open("wb") as f:
            pickle.dump(jax.device_get(best_params), f)
    return params, rows, best_params, best_success, best_iter


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
    return_obs: bool = False,
    n_batches: int = 1,
) -> dict[str, Any]:
    """Deterministic policy eval. n_batches > 1 runs several independently-seeded rollout batches
    through the SAME compiled collect (execution-only after the first) and returns pooled metrics
    plus the per-batch dicts under "eval_batches" -- lets callers measure and average out the
    spawn-randomization variance of a single batch."""
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
    lift_thresh = cfg.success_lift_threshold if cfg is not None else 0.05

    def _batch_metrics(traj, eval_summary) -> dict[str, Any]:
        (_obs, _action, _logp, _value, train_reward, success, lift, base_reward,
         shaped_reward, hard_clip_frac, saturation_frac, action_abs_mean, reward_contrib) = traj
        instant_success_rate = (success.max(axis=0) > 0.5).mean()
        sustained_success_rate = sustained_lift_success(
            lift,
            control_dt=float(env.cfg.control_dt),
            hold_seconds=cfg.success_hold_seconds if cfg is not None else 0.5,
            lift_threshold=lift_thresh,
            gate=success,  # contact-gated env success -> sustained GRASP, not sustained fling
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
            "eval_reach_rate": round(float((eval_summary[:, 2] >= 1.0).mean()), 6),
            "eval_grasp_rate": round(float(((eval_summary[:, 2] >= 1.0) & (eval_summary[:, 3] >= 0.5)).mean()), 6),
            "eval_grasp_lift_rate": round(float(((eval_summary[:, 2] >= 1.0) & (eval_summary[:, 3] >= 0.5)
                                                 & (eval_summary[:, 4] >= lift_thresh)).mean()), 6),
            "eval_lift_reached_rate": round(float((eval_summary[:, 4] >= lift_thresh).mean()), 6),
            "eval_lift_max": round(float(lift.max(axis=0).mean()), 6),
            "eval_hard_clip_frac": round(float(hard_clip_frac.mean()), 6),
            "eval_saturation_frac": round(float(saturation_frac.mean()), 6),
            "eval_action_abs_mean": round(float(action_abs_mean.mean()), 6),
            "eval_summary": [round(float(x), 6) for x in jp.mean(eval_summary, axis=0)],
        }

    outs: list[dict[str, Any]] = []
    obs_list = []
    fail_list = []
    for b in range(max(1, int(n_batches))):
        bs = seed + 7919 * b  # distinct reset keys per batch; the compiled collect is reused
        state = reset(jax.random.split(jax.random.PRNGKey(bs), n_envs))
        _state, traj, _last_value, eval_summary, fail_signal = collect(
            params, state, jax.random.PRNGKey(bs + 1), reward_weights, base_reward_weight_arr,
            action_prior_weights
        )
        bm = _batch_metrics(traj, eval_summary)
        # Task failure signal (tasks.task_failure_signal). The base metrics are never masked by
        # it; alongside them we emit CALM-conditioned aggregates (episodes where the mistake
        # indicator never fired) so a task's graded objective can refuse to pay for progress
        # achieved by making the mistake (e.g. reach via knocking).
        fail_any = np.asarray(fail_signal).any(axis=0)                       # [E]
        es = np.asarray(eval_summary)
        bm["eval_failure_rate"] = round(float(fail_any.mean()), 6)
        bm["eval_calm_frac"] = round(float((~fail_any).mean()), 6)
        bm["eval_reach_rate_calm"] = round(float(((es[:, 2] >= 1.0) & ~fail_any).mean()), 6)
        bm["eval_summary_calm"] = ([round(float(x), 6) for x in es[~fail_any].mean(axis=0)]
                                   if (~fail_any).any() else None)
        outs.append(bm)
        if return_obs:
            obs_list.append(jax.device_get(traj[0]))
            fail_list.append(np.asarray(jax.device_get(fail_signal)))
    if len(outs) == 1:
        out = dict(outs[0])
    else:  # pooled = mean over equal-size batches (all fields are per-episode means)
        out = {}
        for k, v in outs[0].items():
            vals = [o[k] for o in outs]
            if any(x is None for x in vals):  # e.g. eval_summary_calm with zero calm episodes
                vals = [x for x in vals if x is not None]
                if not vals:
                    out[k] = None
                elif isinstance(vals[0], list):
                    out[k] = [round(float(sum(x[i] for x in vals) / len(vals)), 6)
                              for i in range(len(vals[0]))]
                else:
                    out[k] = round(float(sum(vals) / len(vals)), 6)
            elif isinstance(v, list):
                out[k] = [round(float(sum(o[k][i] for o in outs) / len(outs)), 6)
                          for i in range(len(v))]
            else:
                out[k] = round(float(sum(o[k] for o in outs) / len(outs)), 6)
        out["eval_batches"] = outs
    if return_obs:  # [T, n_batches*E, obs_dim] visited states, for stage occupancy
        out["eval_obs"] = (np.concatenate(obs_list, axis=1) if len(obs_list) > 1 else obs_list[0])
        # [T, n_batches*E] task failure signal aligned with eval_obs, for failure attribution.
        out["eval_fail_steps"] = (np.concatenate(fail_list, axis=1) if len(fail_list) > 1
                                  else fail_list[0])
    return out


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
    shaping_fn: Any | None = None,
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
                if shaping_fn is not None:
                    # Experimental reward mode (reward_modes.py): replaces the template shaping;
                    # sees obs so terms can be gated by the prior program's own stage weights.
                    shaped, reward_contrib = jax.vmap(shaping_fn)(
                        prev_eval, nstate.metrics["eval"], state.obs)
                else:
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
        # Task-defined per-step MISTAKE indicator [T, E] (tasks.task_failure_signal -- injected
        # task data; the framework only applies it). Consumed by failure termination in training
        # and by failure-attribution diagnostics in eval.
        fail = task_failure_signal(task, eval_traj)
        return state, traj[:13], last_value, eval_summary, fail

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

    def loss_fn(params, batch, action_prior_weights, reference_params, kl_coef):
        # valid = per-sample loss weight (0 on post-terminal steps under early termination;
        # all-ones otherwise, in which case every masked statistic reduces to the plain mean).
        obs, action, old_logp, adv, ret, valid = batch
        vsum = valid.sum() + 1e-8
        adv_mean = (adv * valid).sum() / vsum
        adv_std = jp.sqrt((((adv - adv_mean) ** 2) * valid).sum() / vsum)
        adv = (adv - adv_mean) / (adv_std + 1e-8)
        logp, ent, value = evaluate(params, obs, action, action_prior_weights)
        ratio = jp.exp(logp - old_logp)
        pg_loss = -(jp.minimum(ratio * adv, jp.clip(ratio, 0.8, 1.2) * adv) * valid).sum() / vsum
        v_loss = 0.5 * (((value - ret) ** 2) * valid).sum() / vsum
        # Decaying KL anchor to the frozen BC policy (raw net outputs), protecting the
        # warm-start during early PPO. kl_coef is 0 for non-warm-start arms (a no-op).
        cur_mean, cur_log_std, _ = net.apply(params, obs)
        ref_mean, ref_log_std, _ = net.apply(reference_params, obs)
        ref_mean = jax.lax.stop_gradient(ref_mean)
        ref_log_std = jax.lax.stop_gradient(ref_log_std)
        kl_anchor = _diag_gaussian_kl(
            cur_mean, jp.clip(cur_log_std, -5.0, 2.0), ref_mean, jp.clip(ref_log_std, -5.0, 2.0)
        ).mean()
        ent_mean = (ent * valid).sum() / vsum
        loss = pg_loss + 0.5 * v_loss - ent_coef * ent_mean + kl_coef * kl_anchor
        return loss, {
            "pg_loss": pg_loss,
            "v_loss": v_loss,
            "entropy": ent_mean,
            "approx_kl": ((old_logp - logp) * valid).sum() / vsum,
            "kl_anchor": kl_anchor,
        }

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    def update(params, opt_state, data, key, action_prior_weights, reference_params, kl_coef):
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
                (_loss, metrics), grads = grad_fn(params, batch, action_prior_weights, reference_params, kl_coef)
                updates, opt_state = optimizer.update(grads, opt_state, params)
                params = optax.apply_updates(params, updates)
                return (params, opt_state), metrics

            (params, opt_state), metrics = jax.lax.scan(minibatch, (params, opt_state), jp.arange(num_minibatches))
            return (params, opt_state, key), metrics

        (params, opt_state, _key), metrics = jax.lax.scan(epoch, (params, opt_state, key), None, length=4)
        return params, opt_state, jax.tree_util.tree_map(lambda x: x.mean(), metrics)

    return jax.jit(update)


def success_termination_mask(success: jp.ndarray, window: int) -> tuple[jp.ndarray, jp.ndarray]:
    """Treat the first completed run of `window` consecutive per-step successes as termination.

    success: [T, E] per-step env success metric. Returns (alive [T, E] float32, alive_end [E]):
    alive is 1.0 up to AND INCLUDING the step that completes the run and 0.0 after; alive_end is
    0.0 for episodes that terminated before the horizon (drops the GAE bootstrap value for them).
    Terminating on a SUSTAINED window -- not the first threshold crossing -- matters: crossing-only
    termination rewards throwing the object through the success region.
    """
    T, E = success.shape
    w = max(1, min(int(window), T))
    s = (success > 0.5).astype(jp.float32)
    c = jp.cumsum(s, axis=0)
    cshift = jp.concatenate([jp.zeros((w, E), jp.float32), c], axis=0)[:T]
    win_done = (c - cshift) >= w                                     # w-run completes AT step t
    seen = jp.cumsum(win_done.astype(jp.int32), axis=0) > 0
    prev_seen = jp.concatenate([jp.zeros((1, E), dtype=bool), seen[:-1]], axis=0)
    alive = (~prev_seen).astype(jp.float32)
    alive_end = (~seen[-1]).astype(jp.float32)
    return alive, alive_end


def sustained_lift_success(
    lift: jp.ndarray,
    *,
    control_dt: float,
    hold_seconds: float,
    lift_threshold: float,
    gate: jp.ndarray | None = None,
) -> jp.ndarray:
    """Fraction of episodes with lift held above threshold for a continuous window.

    If `gate` (a per-step [T, E] mask, e.g. the contact-gated env `success` signal) is given, a
    step only counts when both lift>threshold AND the gate hold -- so sustained success requires a
    sustained *grasp*, not a sustained fling."""
    hold_steps = max(1, int(round(float(hold_seconds) / max(float(control_dt), 1e-9))))
    above = lift > float(lift_threshold)
    if gate is not None:
        above = above & (gate > 0.5)

    def episode_success(ep_above: jp.ndarray) -> jp.ndarray:
        def body(run_len, is_above):
            return jp.where(is_above, run_len + 1, 0), jp.where(is_above, run_len + 1, 0)

        _last, run_lengths = jax.lax.scan(body, jp.int32(0), ep_above)
        return (run_lengths.max() >= hold_steps).astype(jp.float32)

    return jax.vmap(episode_success, in_axes=1)(above).mean()


def _diag_gaussian_kl(mean_p, log_std_p, mean_q, log_std_q) -> jp.ndarray:
    """KL(p || q) for diagonal Gaussians, summed over action dims."""
    var_ratio = jp.exp(2.0 * (log_std_p - log_std_q))
    sq_term = ((mean_p - mean_q) ** 2) * jp.exp(-2.0 * log_std_q)
    return jp.sum((log_std_q - log_std_p) + 0.5 * (var_ratio + sq_term - 1.0), axis=-1)


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


def supervised_pretrain(net, env: Any, params: Any, bias: CompiledBias, task: str, key, cfg: PPOBiasConfig, *, phase_teacher: Any | None = None):
    """Reformatted BC warm-start. Rolls out the controller (the supervised target) to collect
    the states it actually visits and their discounted returns, then jointly fits the actor mean
    to the target action AND the critic to those returns. Cloning on trajectory states (not just
    reset states) keeps the warm-start on-distribution, and pretraining the critic prevents the
    BC policy from being washed out by garbage advantages on PPO's first updates.

    Teacher source: if a `phase_teacher` (closed-loop curriculum PhaseController) is supplied it
    generates the BC dataset -- this is the contact->close->lift teacher. Otherwise the legacy
    static `bias.supervised_target` rule-set is used."""
    reset = jax.jit(lambda keys: jax.vmap(env.reset)(keys))
    step_fn = jax.vmap(env.step)
    optimizer = optax.adam(cfg.supervised_lr)
    opt_state = optimizer.init(params)

    @jax.jit
    def static_target_dataset(rk):
        state = reset(jax.random.split(rk, cfg.supervised_batch))

        def step(carry, _):
            state = carry
            target = jax.vmap(lambda o: bias.supervised_target(o, task))(state.obs)
            if cfg.bc_rollout_states:
                nstate = step_fn(state, target)
            else:
                nstate = state  # stay at reset distribution (legacy behavior)
            return nstate, (state.obs, target, nstate.reward)

        _final, (obs, target, reward) = jax.lax.scan(step, state, None, length=env.horizon)

        # Discounted return-to-go of the env (base) reward under the controller.
        def disc(carry, r):
            ret = r + cfg.gamma * carry
            return ret, ret

        _last, returns = jax.lax.scan(disc, jp.zeros((cfg.supervised_batch,), jp.float32), reward, reverse=True)
        flat = lambda x: x.reshape((-1,) + x.shape[2:])
        return flat(obs), flat(target), returns.reshape((-1,))

    def controller_dataset(rk):
        if phase_teacher is not None:
            return phase_teacher.bc_dataset(rk, envs=cfg.supervised_batch, gamma=cfg.gamma)
        return static_target_dataset(rk)

    def loss_fn(p, obs, target, returns):
        mean, _log_std, value = net.apply(p, obs)
        actor_loss = jp.mean((jp.tanh(mean) - target) ** 2)
        critic_loss = jp.mean((value - returns) ** 2) if cfg.bc_critic_pretrain else 0.0
        return actor_loss + 0.5 * critic_loss

    grad_fn = jax.value_and_grad(loss_fn)
    key, dk = jax.random.split(key)
    obs_all, tgt_all, ret_all = controller_dataset(dk)
    n = obs_all.shape[0]
    mb = max(1, min(cfg.supervised_batch * 4, n))
    for step_i in range(cfg.supervised_steps):
        key, pk = jax.random.split(key)
        idx = jax.random.randint(pk, (mb,), 0, n)
        loss, grads = grad_fn(params, obs_all[idx], tgt_all[idx], ret_all[idx])
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

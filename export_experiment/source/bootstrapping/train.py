#!/usr/bin/env python3
"""Train a closed-loop control policy on an MJX env with PPO.

Phase 1: the pure-RL control arm (no LLM). Each iteration resets a batch of envs, rolls
out one episode per env on the GPU, computes GAE, and runs a clipped PPO update. The
LLM-supervision arms (Phase 2) add a behavior-cloning loss fed from a demo buffer; the
hook (`bc_data`/`bc_coef`) already exists in `ppo.make_update`.

Usage:
    python train.py --smoke                 # quick sanity run
    python train.py --envs 1024 --iters 300 # real run on the 2070
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from pathlib import Path

# Keep allocator behavior friendly to 8 GB GPUs and repeated experiment launches. This
# must run before importing JAX.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")

import jax
import jax.numpy as jp
import numpy as np
import optax

import ppo
import eval_metrics as EM
import reward_spec as RS
from curriculum import CurriculumController
from mjx_env import make_env

HERE = Path(__file__).resolve().parent

# arms: pure RL is --llm none; LLM-supervised arms pass a backend (codex / claude-code).


def make_collect(env: MjxEnv, net, T: int):
    step_fn = jax.vmap(env.step)

    def collect(params, state, key):
        def body(carry, _):
            state, key = carry
            key, ak = jax.random.split(key)
            mean, log_std, value = net.apply(params, state.obs)
            action = mean + jp.exp(log_std) * jax.random.normal(ak, mean.shape)
            logp = ppo.gaussian_logp(action, mean, log_std)
            nstate = step_fn(state, action)
            return (nstate, key), (state.obs, action, logp, value, nstate.reward,
                                   nstate.metrics["success"], nstate.metrics["lift"],
                                   nstate.metrics["eval"])

        (state, key), traj = jax.lax.scan(body, (state, key), None, length=T)
        _, _, last_value = net.apply(params, state.obs)
        eval_summary = jax.vmap(EM.reduce_summary, in_axes=1)(traj[7])  # [N, F] per-env summary
        return state, traj[:7], last_value, eval_summary

    return jax.jit(collect)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="gripper", choices=["gripper", "shadow"],
                   help="embodiment/scene: gripper (cheap, thousands of envs) or shadow "
                        "(23-DOF Shadow Hand; use a small --envs, e.g. 64, on an 8 GB GPU)")
    p.add_argument("--envs", type=int, default=0, help="0 = task default (gripper 1024, shadow 64)")
    p.add_argument("--iters", type=int, default=300)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--hidden", type=int, nargs="+", default=[256, 256])
    p.add_argument("--ent-coef", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--control-dt", type=float, default=0.025)
    p.add_argument("--episode-seconds", type=float, default=2.5)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--smoke", action="store_true", help="tiny run to validate the pipeline")
    # LLM supervision (Phase 2/3). --llm none = pure-RL control arm.
    p.add_argument("--llm", default="none", help="LLM backend: none|codex|claude-code|anthropic|mock")
    p.add_argument("--llm-model", default=None)
    p.add_argument("--llm-budget", type=int, default=100, help="max LLM calls for the run")
    p.add_argument("--bc-coef", type=float, default=0.5, help="behavior-cloning loss weight")
    p.add_argument("--bc-anneal", type=float, default=0.6,
                   help="linearly decay bc_coef to 0 over this fraction of iters (0=off). "
                        "The LLM nudges early acquisition, then fades so PPO can surpass it.")
    p.add_argument("--bc-batch", type=int, default=4096)
    p.add_argument("--bc-weight-temp", type=float, default=1.0,
                   help="advantage-weighted BC temperature (demos that beat policy by more "
                        "weigh more); <=0 = uniform BC")
    p.add_argument("--residual-after", type=float, default=0.3,
                   help="switch LLM scratch->residual once training success exceeds this")
    p.add_argument("--llm-samples", type=int, default=1, help="samples fed to the worker per iter")
    p.add_argument("--reward", default="builtin",
                   help="builtin | default (hand-written curriculum spec) | llm "
                        "(LLM-authored spec) | a Tier-B reward expression string")
    p.add_argument("--curriculum", action="store_true",
                   help="advance reward milestones as the policy clears them (needs a spec)")
    p.add_argument("--advance-frac", type=float, default=0.6)
    p.add_argument("--reward-reflect", type=int, default=0,
                   help="every N iters, let the LLM revise the reward spec from milestone "
                        "stats (Eureka-style reflection); 0 = off. Triggers a recompile.")
    args = p.parse_args()

    if args.envs == 0:
        args.envs = 64 if args.task == "shadow" else 1024
    if args.smoke:
        args.envs, args.iters = (16 if args.task == "shadow" else 64), 5

    reward = None if args.reward == "builtin" else args.reward
    if reward == "llm":
        reward = "default"  # seed with the hand-written spec; replaced by the LLM spec below
    env = make_env(args.task, reward=reward,
                   control_dt=args.control_dt, episode_seconds=args.episode_seconds)
    T = env.horizon

    # LLM-authored reward/curriculum (Phase 2/3). Must happen BEFORE the jit closures below,
    # since env.step closes over env.reward_fn at trace time. Falls back to the default spec.
    emb = "hand" if args.task == "shadow" else "gripper"
    active_spec = RS.DEFAULT_SPECS[args.task] if reward == "default" else None
    if args.reward == "llm":
        if args.llm == "none":
            print("--reward llm needs an --llm backend; falling back to the default spec")
        else:
            from llm_supervision import author_reward_spec
            spec = author_reward_spec(args.llm, args.llm_model, embodiment=emb, log_dir=None)
            if spec is not None:
                compiled = RS.compile_spec(spec)
                env.compiled_reward = compiled
                env.set_reward_fn(compiled.reward_fn)
                active_spec = spec
                print(f"LLM-authored curriculum: {compiled.milestone_names}")
            else:
                print("LLM reward authoring failed/invalid; using the default spec")
    print(f"env: obs={env.obs_size} act={env.action_size} horizon={T} "
          f"frame_skip={env.frame_skip} | envs={args.envs} iters={args.iters}")

    key = jax.random.PRNGKey(args.seed)
    key, nk = jax.random.split(key)
    net, params = ppo.init_params(nk, env.obs_size, env.action_size, tuple(args.hidden))

    optimizer = optax.chain(
        optax.clip_by_global_norm(0.5),
        optax.adam(args.lr),
    )
    opt_state = optimizer.init(params)

    ppo_cfg = ppo.PPOConfig(ent_coef=args.ent_coef, bc_weight_temp=args.bc_weight_temp)
    update = ppo.make_update(net, optimizer, ppo_cfg)

    def build_rollout():
        # env.step closes over env.reward_fn at trace time, so these are rebuilt whenever the
        # reward spec is revised. stage is a traced arg (no recompile when the stage advances).
        reset = jax.jit(lambda keys, stg: jax.vmap(lambda k: env.reset(k, stg))(keys))
        return reset, make_collect(env, net, T)

    reset, collect = build_rollout()

    curric = None
    if env.compiled_reward is not None and args.curriculum:
        curric = CurriculumController(env.compiled_reward, advance_frac=args.advance_frac)
        print(f"curriculum: {env.compiled_reward.milestone_names}")

    arm = "rl_only" if args.llm == "none" else f"rl_{args.llm}"
    out_dir = args.out or (HERE / "runs_rl" / f"{time.strftime('%Y%m%d-%H%M%S')}-{arm}")
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "metrics.jsonl"
    print(f"arm={arm} logging to {out_dir}")

    supervisor = None
    if args.llm != "none":
        from llm_supervision import Supervisor
        supervisor = Supervisor(
            env, net, backend=args.llm, model=args.llm_model, budget=args.llm_budget,
            bc_batch=args.bc_batch, log_dir=out_dir / "llm",
            residual_after=args.residual_after, samples_per_iter=args.llm_samples,
            task=args.task,
        )
        print(f"LLM supervision: backend={args.llm} model={args.llm_model} "
              f"budget={args.llm_budget} bc_coef={args.bc_coef}")

    history = []
    succ = 0.0
    stage = 0
    for it in range(args.iters):
        t0 = time.time()
        key, rk, ck, uk, sk = jax.random.split(key, 5)
        state = reset(jax.random.split(rk, args.envs), jp.int32(stage))
        state, traj, last_value, eval_summary = collect(params, state, ck)
        obs, action, logp, value, reward, success, lift = traj
        succ = float((success.max(axis=0) > 0.5).mean())  # episode lifted-and-held

        # Curriculum: advance the reward milestones as the batch clears the current stage.
        curric_stats = {}
        if curric is not None:
            summ = np.asarray(jax.device_get(eval_summary))
            curric_stats = curric.update(summ)
            stage = curric_stats["stage"]

            # Eureka-style reflection: periodically let the LLM revise the spec from per-
            # milestone achievement + true success. Recompiles the env reward, so it's opt-in.
            if (args.reward_reflect and active_spec is not None and args.llm != "none"
                    and it > 0 and it % args.reward_reflect == 0):
                from llm_supervision import author_reward_spec
                fracs = curric.milestone_fractions(summ)
                refl = (f"per-milestone reached fraction = "
                        f"{dict(zip(curric.compiled.milestone_names, fracs))}; "
                        f"true success rate = {succ:.3f}; current stage = {stage}.")
                new_spec = author_reward_spec(args.llm, args.llm_model, embodiment=emb,
                                              log_dir=out_dir / "llm", prior_spec=active_spec,
                                              reflection=refl)
                if new_spec is not None:
                    active_spec = new_spec
                    compiled = RS.compile_spec(new_spec)
                    env.compiled_reward = compiled
                    env.set_reward_fn(compiled.reward_fn)
                    reset, collect = build_rollout()       # retrace with the revised reward
                    curric = CurriculumController(compiled, advance_frac=args.advance_frac)
                    stage = 0
                    if supervisor is not None:
                        supervisor.compiled = compiled
                    print(f"[reflect@{it}] revised curriculum: {compiled.milestone_names}")

        adv, ret = ppo.compute_gae(reward, value, last_value, args.gamma, args.lam)
        flat = lambda x: x.reshape((-1,) + x.shape[2:])  # [T, N, ...] -> [T*N, ...]
        data = (flat(obs), flat(action), flat(logp), flat(adv), flat(ret))

        # LLM supervision: feed fresh samples, clone any plans that beat the policy, and
        # return a BC minibatch (scored against the *pre-update* policy).
        sup_stats = {}
        if supervisor is not None:
            bc_np, sup_stats = supervisor.step(params, sk, succ, stage=stage)
            frac = max(0.0, 1.0 - it / (args.iters * args.bc_anneal)) if args.bc_anneal > 0 else 1.0
            bc_coef_t = args.bc_coef * frac
            if bc_np is not None and bc_coef_t > 0.0:
                bc_data = (jp.asarray(bc_np[0]), jp.asarray(bc_np[1]), jp.asarray(bc_np[2]))
                params, opt_state, m = update(params, opt_state, data, uk,
                                              bc_data=bc_data, bc_coef=jp.float32(bc_coef_t))
                sup_stats["bc_coef"] = round(bc_coef_t, 4)
            else:
                params, opt_state, m = update(params, opt_state, data, uk)
        else:
            params, opt_state, m = update(params, opt_state, data, uk)
        jax.block_until_ready(params)

        rec = {
            "iter": it,
            "reward": float(reward.sum(axis=0).mean()),  # episodic return per env
            "success": succ,
            "lift_max": float(lift.max(axis=0).mean()),
            "pg_loss": float(m["pg_loss"]),
            "v_loss": float(m["v_loss"]),
            "entropy": float(m["entropy"]),
            "approx_kl": float(m["approx_kl"]),
            "bc_loss": float(m["bc_loss"]),
            "sps": int(args.envs * T / (time.time() - t0)),
            **curric_stats,
            **sup_stats,
        }
        history.append(rec)
        with open(log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        if it % max(1, args.iters // 20) == 0 or it == args.iters - 1:
            extra = (f" | demos {sup_stats.get('demos', 0)} acc {sup_stats.get('accepted', 0)} "
                     f"budget {sup_stats.get('budget_left', '-')}") if supervisor else ""
            cur = (f" | stage {curric_stats['stage']} clr {curric_stats['stage_clear_frac']:.2f} "
                   f"lvl {curric_stats['mean_level']:.2f}") if curric_stats else ""
            print(f"it {it:4d} | return {rec['reward']:7.2f} | success {succ:5.2f} | "
                  f"lift {rec['lift_max']:.3f} | kl {rec['approx_kl']:.4f} | "
                  f"{rec['sps']:,} steps/s{cur}{extra}")

    ckpt = out_dir / "params.pkl"
    with open(ckpt, "wb") as f:
        pickle.dump(jax.device_get(params), f)
    (out_dir / "config.json").write_text(json.dumps({
        "arm": arm, "task": args.task, "hidden": list(args.hidden), "control_dt": args.control_dt,
        "episode_seconds": args.episode_seconds, "obs_size": env.obs_size,
        "action_size": env.action_size, "envs": args.envs, "iters": args.iters,
        "seed": args.seed, "reward": args.reward, "curriculum": args.curriculum,
        "milestones": env.compiled_reward.milestone_names if env.compiled_reward else None,
    }, indent=2) + "\n")
    print(f"saved policy -> {ckpt}")
    if supervisor is not None:
        usage = supervisor.close()
        (out_dir / "llm_usage.json").write_text(json.dumps(usage, indent=2) + "\n")
        print(f"LLM usage: {usage}")
    print(f"final success={history[-1]['success']:.3f} return={history[-1]['reward']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Default PPO runner for prior programs.

Trains one or more arms with the fragmented-stage PPO trainer from ``policy_bias_lab.training.fragmented_ppo``:
short PPO update fragments carry simulator state across a full episode, staged programs can supply
LLM-authored dense rewards through their per-stage ``success`` expressions, and the policy may learn
how strongly to apply the injected prior.

The previous full-horizon short-rollout trainer is preserved under ``policy_bias_lab.legacy``.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import pickle
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_triton_gemm=false")
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", str(Path(".xla_cache").resolve()))
import jax  # noqa: E402 (must follow the XLA env-var configuration above)

from policy_bias_lab.bias import compile_bias
from policy_bias_lab.composed_priors import prior_program_for_arm
from policy_bias_lab.arms import BIAS_ARMS
from policy_bias_lab.training.fragmented_ppo import (
    PPOBiasConfig,
    evaluate_fragmented_policy,
    train_fragmented_stage_ppo,
)
from policy_bias_lab.reporting import summarize, write_csv

EMPTY_PROGRAM = {"mode": "freeform_staged", "signals": {}, "stages": []}


def _default_fragment_steps(horizon: int, preferred: int) -> int:
    if horizon % preferred == 0:
        return preferred
    for steps in (100, 80, 64, 50, 40, 25, 20, 10, 5, 1):
        if horizon % steps == 0:
            return steps
    return 1


def main() -> int:
    args = parse_args()
    from experiment_runtime.environment import make_env

    if args.out is None:
        args.out = Path("runs") / f"prior_ppo_{time.strftime('%Y%m%d-%H%M%S')}"
    args.out.mkdir(parents=True, exist_ok=True)

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    for arm in arms:
        if arm not in BIAS_ARMS:
            raise KeyError(f"unknown arm {arm!r}; choose from {sorted(BIAS_ARMS)}")

    program_files = {}
    for spec in args.prior_program_arm:
        a, _, p = str(spec).partition("=")
        program_files[a.strip()] = p.strip()

    env = make_env("shadow", control_dt=args.control_dt, episode_seconds=args.episode_seconds,
                   physics_dt=args.physics_dt, obj_xy_range=args.obj_xy_range)
    fragment_steps = _default_fragment_steps(int(env.horizon), int(args.fragment_steps))
    if fragment_steps != args.fragment_steps:
        print(f"[ppo] --fragment-steps {args.fragment_steps} does not divide horizon {env.horizon}; "
              f"using {fragment_steps}")

    cfg = PPOBiasConfig(
        iters=args.iters,
        envs=args.envs,
        eval_envs=args.eval_envs,
        fragment_steps=fragment_steps,
        lr=args.lr,
        gamma=args.gamma,
        lam=args.lam,
        hidden=tuple(args.hidden),
        ent_coef=args.ent_coef,
        target_train_seconds=args.target_arm_seconds,
        max_env_steps=args.max_env_steps,
        checkpoint_every=args.checkpoint_every,
        eval_every=args.eval_every,
        residual_action_scale=args.residual_action_scale,
        learn_prior_scale=not args.no_learn_prior_scale,
        prior_scale_mode=args.prior_scale_mode,
        prior_scale_bias=args.prior_scale_bias,
        prior_scale_gain=args.prior_scale_gain,
        stage_reward_weight=args.stage_reward_weight,
        stage_progress_weight=args.stage_progress_weight,
        stage_completion_bonus=args.stage_completion_bonus,
        stage_success_temperature=args.stage_success_temperature,
        stage_reward_clip=args.stage_reward_clip,
        base_reward_weight=args.base_reward_weight,
        action_transform=args.action_transform,
        warmup_compile=args.warmup_compile,
    )
    config = {
        "learner": "fragmented_stage_ppo", "tasks": tasks, "arms": arms, "seeds": seeds,
        "ppo": cfg.__dict__,
        "prior_program_arms": {a: program_files.get(a) or
                               (prior_program_for_arm(a) is not None and "arm-derived")
                               for a in arms},
        "env": {"action_size": env.action_size, "horizon": env.horizon,
                "actuators": [env.model.actuator(i).name for i in range(env.nu)]},
    }
    (args.out / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    eval_rows: list[dict[str, Any]] = []
    metrics_path = args.out / "metrics.jsonl"
    with metrics_path.open("w") as metrics_file:
        for task in tasks:
            for seed in seeds:
                for arm in arms:
                    run_dir = args.out / f"{task}_s{seed}_{arm}"
                    run_dir.mkdir(parents=True, exist_ok=True)
                    use_reward, use_action_prior, _use_exploration, _use_supervised = BIAS_ARMS[arm]

                    program = None
                    if arm in program_files:
                        program = json.loads(Path(program_files[arm]).read_text())
                    elif use_action_prior:
                        program = prior_program_for_arm(arm)
                    program_for_training = program or EMPTY_PROGRAM
                    if program is not None:
                        (run_dir / "prior_program.json").write_text(json.dumps(program, indent=2) + "\n")

                    bias_spec = {"name": "prior_ppo", "action_priors": []}
                    if program is not None:
                        bias_spec["prior_program"] = program
                    bias = compile_bias(bias_spec, env)
                    action_prior_weights = bias.default_action_prior_weights()
                    arm_cfg = replace(
                        cfg,
                        use_action_prior=bool(use_action_prior and program is not None),
                        stage_reward_weight=(float(cfg.stage_reward_weight) if use_reward else 0.0),
                    )

                    params, rows, best_params, best_score, best_iter = train_fragmented_stage_ppo(
                        env=env, bias=bias, program=program_for_training, task=task, seed=seed,
                        cfg=arm_cfg, out_dir=run_dir,
                        action_prior_weights=action_prior_weights,
                    )
                    for row in rows:
                        metrics_file.write(json.dumps(row | {"seed": seed, "arm": arm}) + "\n")
                        metrics_file.flush()
                    with (run_dir / "params_final.pkl").open("wb") as f:
                        pickle.dump(jax.device_get(params), f)

                    ev = evaluate_fragmented_policy(
                        env=env, params=best_params, bias=bias, program=program_for_training,
                        task=task, seed=seed + 10_000, cfg=arm_cfg,
                        action_prior_weights=action_prior_weights)
                    eval_row = {
                        "task": task, "seed": seed, "arm": arm,
                        "best_checkpoint_iter": int(best_iter),
                        "best_train_objective": round(float(best_score), 6),
                        "best_train_success": round(float(ev.get("eval_success_rate", 0.0)), 6),
                        **ev,
                    }
                    (run_dir / "eval.json").write_text(json.dumps(eval_row, indent=2) + "\n")
                    eval_rows.append(eval_row)
                    jax.clear_caches()
                    gc.collect()

    (args.out / "summary.json").write_text(json.dumps(summarize(eval_rows), indent=2) + "\n")
    write_csv(args.out / "eval.csv", eval_rows)
    print(f"[done] {len(eval_rows)} arm-evals -> {args.out}")
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--tasks", default="lift")
    p.add_argument("--arms", required=True, help="comma-separated arms from arms.BIAS_ARMS")
    p.add_argument("--prior-program-arm", action="append", default=[],
                   help="ARM=path.json (repeatable): inject a prior program file for an arm.")
    p.add_argument("--seeds", default="0")
    p.add_argument("--iters", type=int, default=2000)
    p.add_argument("--envs", type=int, default=256)
    p.add_argument("--eval-envs", type=int, default=256)
    p.add_argument("--fragment-steps", type=int, default=100,
                   help="PPO rollout/update length. Must divide the env horizon; auto-adjusted if not.")
    p.add_argument("--target-arm-seconds", type=float, default=1700.0)
    p.add_argument("--max-env-steps", type=int, default=None)
    p.add_argument("--checkpoint-every", type=int, default=0)
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--hidden", type=int, nargs="+", default=[256, 256])
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--ent-coef", type=float, default=0.0)
    p.add_argument("--residual-action-scale", type=float, default=1.0)
    p.add_argument("--no-learn-prior-scale", action="store_true")
    p.add_argument("--prior-scale-mode", choices=["scalar", "group", "per_joint"], default="group",
                   help="Granularity of learned prior-strength control. Default group learns one "
                        "scale per robot semantic actuator group; scalar is available for replay "
                        "compatibility; per_joint learns one scale per actuator.")
    p.add_argument("--prior-scale-bias", type=float, default=1.0)
    p.add_argument("--prior-scale-gain", type=float, default=1.0)
    p.add_argument("--stage-reward-weight", type=float, default=1.0)
    p.add_argument("--stage-progress-weight", type=float, default=1.0)
    p.add_argument("--stage-completion-bonus", type=float, default=0.05)
    p.add_argument("--stage-success-temperature", type=float, default=1.0)
    p.add_argument("--stage-reward-clip", type=float, default=0.5)
    p.add_argument("--base-reward-weight", type=float, default=1.0)
    p.add_argument("--action-transform", choices=["raw", "tanh"], default="tanh")
    p.add_argument("--episode-seconds", type=float, default=20.0)
    p.add_argument("--control-dt", type=float, default=0.025)
    p.add_argument("--physics-dt", type=float, default=0.01)
    p.add_argument("--obj-xy-range", type=float, default=0.04)
    p.add_argument("--no-warmup-compile", dest="warmup_compile", action="store_false", default=True)
    return p.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())

"""Run the experimental fragmented/stage-reward PPO trainer.

Example:
  python -m policy_bias_lab.experimental.run_fragmented_stage_ppo \
      --out runs/frag_stage_test \
      --program runs/agentic1/best_program.json \
      --episode-seconds 20 --fragment-steps 100 --envs 128
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_triton_gemm=false")
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", str(Path(".xla_cache").resolve()))
import jax

ROOT = Path(__file__).resolve().parents[2]

from policy_bias_lab.bias import compile_bias
from policy_bias_lab.training.fragmented_ppo import (
    FragmentedStagePPOConfig,
    evaluate_fragmented_policy,
    train_fragmented_stage_ppo,
)


def main() -> int:
    args = parse_args()
    from experiment_runtime.environment import make_env

    args.out.mkdir(parents=True, exist_ok=True)
    program = json.loads(args.program.read_text())
    env = make_env("shadow", control_dt=args.control_dt, episode_seconds=args.episode_seconds,
                   physics_dt=args.physics_dt, obj_xy_range=args.obj_xy_range)
    if env.horizon % args.fragment_steps != 0:
        raise SystemExit(
            f"--fragment-steps {args.fragment_steps} must divide horizon {env.horizon} "
            f"({args.episode_seconds}s / {args.control_dt}s)."
        )
    bias = compile_bias({"name": "fragmented_stage_ppo", "action_priors": [],
                         "prior_program": program}, env)
    cfg = FragmentedStagePPOConfig(
        iters=args.iters,
        envs=args.envs,
        eval_envs=args.eval_envs,
        fragment_steps=args.fragment_steps,
        lr=args.lr,
        gamma=args.gamma,
        lam=args.lam,
        hidden=tuple(args.hidden),
        ent_coef=args.ent_coef,
        target_train_seconds=args.target_train_seconds,
        max_env_steps=args.max_env_steps,
        checkpoint_every=args.checkpoint_every,
        eval_every=args.eval_every,
        residual_action_scale=args.residual_action_scale,
        use_action_prior=not args.no_action_prior,
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
        warmup_compile=not args.no_warmup_compile,
    )
    (args.out / "config.json").write_text(json.dumps({
        "learner": "experimental_fragmented_stage_ppo",
        "task": args.task,
        "seed": args.seed,
        "program": str(args.program),
        "env": {
            "horizon": int(env.horizon),
            "control_dt": args.control_dt,
            "episode_seconds": args.episode_seconds,
            "fragment_steps": args.fragment_steps,
            "fragments_per_episode": int(env.horizon) // int(args.fragment_steps),
        },
        "ppo": cfg.__dict__,
    }, indent=2, default=str) + "\n")
    (args.out / "prior_program.json").write_text(json.dumps(program, indent=2) + "\n")

    params, rows, best_params, best_score, best_iter = train_fragmented_stage_ppo(
        env=env, bias=bias, program=program, task=args.task, seed=args.seed, cfg=cfg,
        out_dir=args.out)
    with (args.out / "params_final.pkl").open("wb") as f:
        pickle.dump(jax.device_get(params), f)
    with (args.out / "best_params.pkl").open("wb") as f:
        pickle.dump(jax.device_get(best_params), f)
    ev = evaluate_fragmented_policy(
        env=env, params=best_params, bias=bias, program=program, task=args.task,
        seed=args.seed + 20_000, cfg=cfg)
    report = {
        "iters": len(rows),
        "best_iter": int(best_iter),
        "best_objective": round(float(best_score), 6),
        # Headline = the SAME arbiter objective as the prior_only/short_ppo arbiter (task_graded_objective),
        # so this arm is directly comparable to the baseline arm and to prior-generation scores.
        "eval_objective": round(float(ev["eval_graded_objective"]), 6),
        "eval_graded_objective": round(float(ev["eval_graded_objective"]), 6),
        "eval_task_fitness": round(float(ev["eval_task_fitness"]), 6),
        "eval_success_rate": round(float(ev["eval_success_rate"]), 6),
        "eval": ev,
    }
    (args.out / "final_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"[done] {len(rows)} fragments, best_iter={best_iter}, "
          f"graded_objective={report['eval_graded_objective']} "
          f"success={report['eval_success_rate']} -> {args.out}")
    if args.prior_scale_mode != "scalar":
        print(f"[prior_scale/{args.prior_scale_mode}] {ev.get('eval_prior_scale_group_means', {})}")
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--program", type=Path, required=True)
    p.add_argument("--task", default="lift")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--iters", type=int, default=2000)
    p.add_argument("--envs", type=int, default=256)
    p.add_argument("--eval-envs", type=int, default=256)
    p.add_argument("--fragment-steps", type=int, default=100,
                   help="PPO rollout/update length. Must divide the env horizon.")
    p.add_argument("--target-train-seconds", type=float, default=None)
    p.add_argument("--max-env-steps", type=int, default=None)
    p.add_argument("--checkpoint-every", type=int, default=0)
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--hidden", type=int, nargs="+", default=[256, 256])
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--ent-coef", type=float, default=0.0)
    p.add_argument("--residual-action-scale", type=float, default=1.0)
    p.add_argument("--no-action-prior", action="store_true")
    p.add_argument("--no-learn-prior-scale", action="store_true")
    p.add_argument("--prior-scale-mode", choices=["scalar", "group", "per_joint"], default="group",
                   help="Granularity of the learned prior-strength control: one per robot semantic "
                        "actuator group by default, one knob for the whole prior (scalar), or one "
                        "per actuator (per_joint). All start at 1.0.")
    p.add_argument("--prior-scale-bias", type=float, default=1.0,
                   help="Prior strength scalar = clip(bias + gain*out, 0, 1). Defaults bias=gain=1 "
                        "start it at 1.0 (full prior) when the extra policy output is ~0, and the "
                        "controller can scale it down to 0.")
    p.add_argument("--prior-scale-gain", type=float, default=1.0)
    p.add_argument("--stage-reward-weight", type=float, default=1.0)
    p.add_argument("--stage-progress-weight", type=float, default=1.0)
    p.add_argument("--stage-completion-bonus", type=float, default=0.05)
    p.add_argument("--stage-success-temperature", type=float, default=1.0)
    p.add_argument("--stage-reward-clip", type=float, default=0.5)
    p.add_argument("--base-reward-weight", type=float, default=1.0)
    p.add_argument("--episode-seconds", type=float, default=20.0)
    p.add_argument("--control-dt", type=float, default=0.025)
    p.add_argument("--physics-dt", type=float, default=0.01)
    p.add_argument("--obj-xy-range", type=float, default=0.04)
    p.add_argument("--no-warmup-compile", dest="no_warmup_compile", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())

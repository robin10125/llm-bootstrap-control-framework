"""Long-duration fragmented-stage PPO training with a FIXED selected prior program.

This is the default long PPO runner. It uses the same fragmented-stage trainer as the selection
arbiter and ``run_prior_ppo``: PPO updates on short rollout fragments, carries simulator state
across fragments, can use per-stage ``success`` rewards authored by the prior program, and can learn
how strongly to apply the injected prior.

The previous resumable short-rollout long runner is preserved at
``policy_bias_lab.legacy.run_long_short_ppo``.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_triton_gemm=false")
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", str(Path(".xla_cache").resolve()))
import jax  # noqa: E402



def _default_fragment_steps(horizon: int, preferred: int) -> int:
    if horizon % preferred == 0:
        return preferred
    for steps in (100, 80, 64, 50, 40, 25, 20, 10, 5, 1):
        if horizon % steps == 0:
            return steps
    return 1


def _target_seconds(args: argparse.Namespace) -> float | None:
    if args.target_train_seconds is not None:
        return float(args.target_train_seconds)
    if args.max_hours is not None:
        return float(args.max_hours) * 3600.0
    if args.plateau_hours is not None:
        return (float(args.min_hours) + float(args.plateau_hours)) * 3600.0
    return None


def main() -> int:
    args = parse_args()
    if args.resume:
        print("[resume] fragmented-stage run_long_ppo does not consume legacy resume.pkl files; "
              "use python -m policy_bias_lab.legacy.run_long_short_ppo for old resumable runs.",
              file=sys.stderr)
        return 2
    if args.init_params is not None:
        print("[init] --init-params is not compatible with the fragmented-stage policy head; "
              "starting from a fresh policy.", file=sys.stderr)
    from experiment_runtime.environment import make_env

    from policy_bias_lab.bias import compile_bias
    from policy_bias_lab.arms import BIAS_ARMS
    from policy_bias_lab.training.fragmented_ppo import (
        PPOBiasConfig,
        evaluate_fragmented_policy,
        train_fragmented_stage_ppo,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    program = json.loads(Path(args.program).read_text())
    (args.out / "prior_program.json").write_text(json.dumps(program, indent=2) + "\n")

    env = make_env("shadow", control_dt=args.control_dt, episode_seconds=args.episode_seconds,
                   physics_dt=args.physics_dt, obj_xy_range=args.obj_xy_range)
    fragment_steps = _default_fragment_steps(int(env.horizon), int(args.fragment_steps))
    if fragment_steps != args.fragment_steps:
        print(f"[ppo] --fragment-steps {args.fragment_steps} does not divide horizon {env.horizon}; "
              f"using {fragment_steps}")

    use_reward, use_action_prior, _use_exploration, _use_supervised = BIAS_ARMS[args.arm]
    bias = compile_bias({"name": "long_ppo", "action_priors": [], "prior_program": program}, env)
    cfg = PPOBiasConfig(
        iters=args.stop_after_iters or args.iters,
        envs=args.envs,
        eval_envs=args.eval_envs,
        fragment_steps=fragment_steps,
        lr=args.lr,
        gamma=args.gamma,
        lam=args.lam,
        hidden=tuple(args.hidden),
        ent_coef=args.ent_coef,
        target_train_seconds=_target_seconds(args),
        max_env_steps=args.max_env_steps,
        checkpoint_every=args.checkpoint_every,
        eval_every=args.eval_every,
        residual_action_scale=args.residual_action_scale,
        use_action_prior=bool(use_action_prior),
        learn_prior_scale=not args.no_learn_prior_scale,
        prior_scale_mode=args.prior_scale_mode,
        prior_scale_bias=args.prior_scale_bias,
        prior_scale_gain=args.prior_scale_gain,
        stage_reward_weight=(args.stage_reward_weight if use_reward else 0.0),
        stage_progress_weight=args.stage_progress_weight,
        stage_completion_bonus=args.stage_completion_bonus,
        stage_success_temperature=args.stage_success_temperature,
        stage_reward_clip=args.stage_reward_clip,
        base_reward_weight=args.base_reward_weight,
        action_transform=args.action_transform,
        warmup_compile=not args.no_warmup_compile,
    )
    # Keep the saved config exactly aligned with what the trainer receives.
    cfg = replace(cfg, use_action_prior=bool(use_action_prior))
    (args.out / "config.json").write_text(json.dumps({
        "learner": "long_fragmented_stage_ppo",
        "task": args.task,
        "arm": args.arm,
        "seed": args.seed,
        "program": str(args.program),
        "episode_seconds": args.episode_seconds,
        "env": {"horizon": int(env.horizon), "fragment_steps": int(cfg.fragment_steps)},
        "criteria": {
            "target_train_seconds": cfg.target_train_seconds,
            "max_env_steps": cfg.max_env_steps,
            "iters": cfg.iters,
            "legacy_min_hours": args.min_hours,
            "legacy_plateau_hours": args.plateau_hours,
            "legacy_success_stop": args.success_stop,
        },
        "ppo": cfg.__dict__,
    }, indent=2, default=str) + "\n")

    print(f"[long-ppo] fragmented-stage trainer, target_seconds={cfg.target_train_seconds}, "
          f"iters={cfg.iters}, fragment_steps={cfg.fragment_steps}", flush=True)
    params, rows, best_params, best_score, best_iter = train_fragmented_stage_ppo(
        env=env, bias=bias, program=program, task=args.task, seed=args.seed, cfg=cfg,
        out_dir=args.out, action_prior_weights=bias.default_action_prior_weights())
    with (args.out / "params_final.pkl").open("wb") as f:
        pickle.dump(jax.device_get(params), f)
    with (args.out / "best_params.pkl").open("wb") as f:
        pickle.dump(jax.device_get(best_params), f)
    ev = evaluate_fragmented_policy(
        env=env, params=best_params, bias=bias, program=program, task=args.task,
        seed=args.seed + 20_000, cfg=cfg,
        action_prior_weights=bias.default_action_prior_weights())
    report = {
        "stop_reason": "fragmented-stage training budget exhausted",
        "iters": len(rows),
        "best_iter": int(best_iter),
        "best_train_objective": round(float(best_score), 6),
        "eval": ev,
    }
    (args.out / "final_report.json").write_text(json.dumps(report, indent=2, default=float) + "\n")
    print(f"[done] {len(rows)} fragments, best_iter={best_iter}, "
          f"eval objective {ev.get('eval_graded_objective')} success {ev.get('eval_success_rate')} "
          f"-> {args.out / 'final_report.json'}")
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--program", type=Path, required=True,
                   help="selected prior program JSON (e.g. runs/<selection>/best_program.json)")
    p.add_argument("--task", default="lift")
    p.add_argument("--arm", default="freeform_encourage",
                   help="BIAS_ARMS entry; default matches the selection arbiter.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--init-params", type=Path, default=None,
                   help="legacy short-rollout option; ignored by the fragmented-stage runner.")
    p.add_argument("--resume", action="store_true",
                   help="legacy short-rollout option; use policy_bias_lab.legacy.run_long_short_ppo.")
    p.add_argument("--target-train-seconds", type=float, default=None,
                   help="direct active training-time budget. Overrides --max-hours and plateau-derived time.")
    p.add_argument("--min-hours", type=float, default=8.0,
                   help="legacy-compatible input: with --plateau-hours, sets target seconds to min+plateau hours.")
    p.add_argument("--plateau-hours", type=float, default=2.0,
                   help="legacy-compatible input: with --min-hours, sets target seconds to min+plateau hours.")
    p.add_argument("--plateau-eps", type=float, default=1e-5,
                   help="legacy-compatible accepted option; plateau logic is not used by this trainer.")
    p.add_argument("--success-stop", type=float, default=0.8,
                   help="legacy-compatible accepted option; rolling success stop is not used by this trainer.")
    p.add_argument("--success-window", type=int, default=10,
                   help="legacy-compatible accepted option.")
    p.add_argument("--max-hours", type=float, default=None,
                   help="hard active-time budget in hours; overrides min+plateau derived time.")
    p.add_argument("--stop-after-iters", type=int, default=None,
                   help="run this many PPO fragment updates this session.")
    p.add_argument("--save-every-iters", type=int, default=100,
                   help="legacy-compatible accepted option; use --checkpoint-every for checkpoints.")
    p.add_argument("--iters", type=int, default=10**7)
    p.add_argument("--max-env-steps", type=int, default=None)
    p.add_argument("--checkpoint-every", type=int, default=0)
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--envs", type=int, default=256)
    p.add_argument("--eval-envs", type=int, default=256)
    p.add_argument("--fragment-steps", type=int, default=100)
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
    p.add_argument("--no-warmup-compile", action="store_true")
    p.add_argument("--reward-mode", choices=["default", "lift_only", "adjusted", "stage_gated"],
                   default="default", help="legacy accepted option; not used by fragmented-stage PPO.")
    p.add_argument("--terminate-on-success", type=float, default=None, metavar="SECONDS",
                   help="legacy short-rollout option; ignored by fragmented-stage PPO.")
    p.add_argument("--terminate-on-failure", type=float, default=None, metavar="SECONDS",
                   help="legacy short-rollout option; ignored by fragmented-stage PPO.")
    p.add_argument("--prior-logit-clip", type=float, default=0.95,
                   help="legacy accepted option; not used by fragmented-stage PPO.")
    p.add_argument("--success-hold-seconds", type=float, default=0.5,
                   help="legacy accepted option; not used by fragmented-stage PPO.")
    p.add_argument("--success-lift-threshold", type=float, default=0.05,
                   help="legacy accepted option; not used by fragmented-stage PPO.")
    return p.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())

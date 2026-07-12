"""Legacy short-rollout PPO runner for the action-prior system.

Trains one or more arms with PPO on the base contact-gated reward + the single fixed shaping
template (lift_basin_curriculum), applying an injected stateless prior program per arm. NO coach,
phase teacher, pareto selection, or adaptive reward rewriting -- those confounding subsystems are
quarantined under policy_bias_lab.legacy and the old run_dynamic_reward_experiment runner.

Each arm's prior program is either:
  - injected from a file:  --prior-program-arm ARM=path.json  (e.g. the LLM-generated stacked
    priors from run_dsl_vs_freeform), or
  - derived from the arm name (composed_priors.prior_program_for_arm), or
  - none (a baseline arm with use_action_prior=False in es.BIAS_ARMS).
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_triton_gemm=false")
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", str(Path(".xla_cache").resolve()))
import jax  # noqa: E402 (must follow the XLA env-var configuration above)

from policy_bias_lab.bias import compile_bias, default_reward_template_weights
from policy_bias_lab.composed_priors import prior_program_for_arm
from policy_bias_lab.es import BIAS_ARMS
from policy_bias_lab.legacy.short_rollout_ppo import PPOBiasConfig, evaluate_ppo_policy, train_ppo_arm
from policy_bias_lab.report_utils import summarize, write_csv

ROOT = Path(__file__).resolve().parents[2]
BOOTSTRAPPING = ROOT.parent / "bootstrapping"


def _zeros_like_weights(w):
    return [0.0 for _ in w]


def main() -> int:
    args = parse_args()
    if str(BOOTSTRAPPING) not in sys.path:
        sys.path.insert(0, str(BOOTSTRAPPING))
    from mjx_env import make_env

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
    base_spec = {"name": "prior_ppo", "action_priors": []}

    cfg = PPOBiasConfig(
        envs=args.envs, lr=args.lr, gamma=args.gamma, lam=args.lam, hidden=tuple(args.hidden),
        ent_coef=args.ent_coef, checkpoint_count=args.checkpoint_count,
        target_train_seconds=args.target_arm_seconds, action_transform=args.action_transform,
        prior_logit_clip=args.prior_logit_clip, success_hold_seconds=args.success_hold_seconds,
        success_lift_threshold=args.success_lift_threshold, warmup_compile=args.warmup_compile,
    )
    config = {
        "learner": "ppo_prior_slim", "tasks": tasks, "arms": arms, "seeds": seeds,
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
                    use_reward, use_action_prior, _, _ = BIAS_ARMS[arm]

                    # Resolve the prior program: file injection > arm-name-derived > none.
                    program = None
                    if arm in program_files:
                        program = json.loads(Path(program_files[arm]).read_text())
                    elif use_action_prior:
                        program = prior_program_for_arm(arm)
                    arm_spec = dict(base_spec)
                    if program is not None:
                        arm_spec["prior_program"] = program
                        (run_dir / "prior_program.json").write_text(json.dumps(program, indent=2) + "\n")
                    bias = compile_bias(arm_spec, env)

                    reward_weights = default_reward_template_weights(task)
                    if not use_reward:  # base contact-gated reward only (no fixed template shaping)
                        reward_weights = reward_weights * 0.0
                    action_prior_weights = bias.default_action_prior_weights()

                    params, rows, best_params, best_success, best_iter = train_ppo_arm(
                        env=env, bias=bias, task=task, arm=arm, seed=seed, cfg=cfg,
                        checkpoint_dir=run_dir / "checkpoints",
                        reward_weights=reward_weights, base_reward_weight=1.0,
                        action_prior_weights=action_prior_weights,
                    )
                    for row in rows:
                        metrics_file.write(json.dumps(row | {"seed": seed}) + "\n")
                        metrics_file.flush()
                    import pickle
                    with (run_dir / "params.pkl").open("wb") as f:
                        pickle.dump(jax.device_get(params), f)

                    eval_row = {
                        "task": task, "seed": seed, "arm": arm,
                        "best_checkpoint_iter": int(best_iter),
                        "best_train_success": round(float(best_success), 6),
                        **evaluate_ppo_policy(
                            env=env, params=best_params, bias=bias, task=task, arm=arm,
                            seed=seed + 10_000, n_envs=args.eval_envs, cfg=cfg,
                            reward_weights=reward_weights, base_reward_weight=1.0,
                            action_prior_weights=action_prior_weights),
                    }
                    (run_dir / "eval.json").write_text(json.dumps(eval_row, indent=2) + "\n")
                    eval_rows.append(eval_row)
                    # Free this arm's jitted executables; sequential arms otherwise accumulate the
                    # XLA cache and OOM the GPU (~arm 4 on 8 GB). See run_dynamic_reward_experiment.
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
    p.add_argument("--arms", required=True, help="comma-separated arms from es.BIAS_ARMS")
    p.add_argument("--prior-program-arm", action="append", default=[],
                   help="ARM=path.json (repeatable): inject a prior program file for an arm.")
    p.add_argument("--seeds", default="0")
    p.add_argument("--envs", type=int, default=256)
    p.add_argument("--eval-envs", type=int, default=256)
    p.add_argument("--target-arm-seconds", type=float, default=1700.0)
    p.add_argument("--hidden", type=int, nargs="+", default=[128, 128])
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--ent-coef", type=float, default=0.0)
    p.add_argument("--checkpoint-count", type=int, default=8)
    p.add_argument("--action-transform", choices=["raw", "tanh"], default="tanh")
    p.add_argument("--prior-logit-clip", type=float, default=0.95)
    p.add_argument("--success-hold-seconds", type=float, default=0.5)
    p.add_argument("--success-lift-threshold", type=float, default=0.05)
    p.add_argument("--episode-seconds", type=float, default=5.0)
    p.add_argument("--control-dt", type=float, default=0.025)
    p.add_argument("--physics-dt", type=float, default=0.01)
    p.add_argument("--obj-xy-range", type=float, default=0.04)
    p.add_argument("--no-warmup-compile", dest="warmup_compile", action="store_false", default=True)
    return p.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())

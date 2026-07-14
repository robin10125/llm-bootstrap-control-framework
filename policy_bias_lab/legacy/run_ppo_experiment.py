from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")

from policy_bias_lab.legacy.action_priors import ActionPriorCoach, ActionPriorConfig, load_action_prior_rules
from policy_bias_lab.bias import compile_bias
from policy_bias_lab.legacy.dynamic_rewards import load_pre_run_reward_analysis
from policy_bias_lab.arms import BIAS_ARMS
from policy_bias_lab.legacy.llm_bias import load_bias_spec
from policy_bias_lab.legacy.short_rollout_ppo import PPOBiasConfig, evaluate_ppo_policy, train_ppo_arm
from policy_bias_lab.tasks import task_metadata


ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    args = parse_args()
    from experiment_runtime.environment import make_env

    if args.out is None:
        args.out = Path("runs") / f"policy_bias_ppo_{time.strftime('%Y%m%d-%H%M%S')}"
    args.out.mkdir(parents=True, exist_ok=True)

    tasks = [item.strip() for item in args.tasks.split(",") if item.strip()]
    arms = [item.strip() for item in args.arms.split(",") if item.strip()]
    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    for arm in arms:
        if arm not in BIAS_ARMS:
            raise KeyError(f"unknown arm {arm!r}; choose from {sorted(BIAS_ARMS)}")

    env = make_env(
        "shadow",
        control_dt=args.control_dt,
        episode_seconds=args.episode_seconds,
        physics_dt=args.physics_dt,
        obj_xy_range=args.obj_xy_range,
    )
    env_summary = {
        "obs_size": env.obs_size,
        "action_size": env.action_size,
        "horizon": env.horizon,
        "frame_skip": env.frame_skip,
        "actuators": [env.model.actuator(i).name for i in range(env.nu)],
        "tasks": {name: task_metadata(name) for name in tasks},
    }
    if args.bias_spec is not None:
        bias_spec = json.loads(args.bias_spec.read_text())
    else:
        bias_spec = load_bias_spec(
            backend=args.llm_backend,
            model=args.llm_model,
            task="+".join(tasks),
            tasks=tasks,
            env_summary=env_summary,
            log_dir=args.out / "llm",
        )
    pre_run_reward_analysis = {}
    if args.pre_run_reward_analysis:
        pre_run_reward_analysis = load_pre_run_reward_analysis(
            llm_backend=args.llm_backend,
            llm_model=args.llm_model,
            task="+".join(tasks),
            tasks=tasks,
            env_summary=env_summary,
            bias_spec=bias_spec,
            previous_run_context=None,
            log_dir=args.out / "pre_run_reward_analysis",
        )
        (args.out / "pre_run_reward_analysis.json").write_text(json.dumps(pre_run_reward_analysis, indent=2) + "\n")
    if args.llm_action_prior:
        bias_spec = dict(bias_spec)
        bias_spec["action_priors"] = load_action_prior_rules(
            ActionPriorConfig(
                llm_backend=args.llm_backend,
                llm_model=args.llm_model,
                task="+".join(tasks),
                tasks=tasks,
                arm="global",
                env_summary=env_summary,
                log_dir=args.out / "llm_action_prior",
                max_weight=args.max_action_prior_weight,
                max_checkups=args.action_prior_max_checkups,
                pre_run_reward_analysis=pre_run_reward_analysis,
            ),
            bias_spec,
        )
    (args.out / "bias_spec.json").write_text(json.dumps(bias_spec, indent=2))
    compiled_bias = compile_bias(bias_spec, env)
    unit_count = max(1, len(tasks) * len(seeds) * len(arms))
    target_train_seconds = args.target_arm_seconds
    if target_train_seconds is None and args.target_total_seconds is not None:
        target_train_seconds = args.target_total_seconds / unit_count
    cfg = PPOBiasConfig(
        iters=args.iters,
        envs=args.envs,
        lr=args.lr,
        gamma=args.gamma,
        lam=args.lam,
        hidden=tuple(args.hidden),
        ent_coef=args.ent_coef,
        supervised_steps=args.supervised_steps,
        supervised_batch=args.supervised_batch,
        supervised_lr=args.supervised_lr,
        checkpoint_count=args.checkpoint_count,
        target_train_seconds=target_train_seconds,
        action_transform=args.action_transform,
        saturation_penalty=args.saturation_penalty,
        saturation_threshold=args.saturation_threshold,
        prior_logit_clip=args.prior_logit_clip,
        action_target_reward_weight=args.action_target_reward_weight,
        success_hold_seconds=args.success_hold_seconds,
        success_lift_threshold=args.success_lift_threshold,
    )
    config = {
        "learner": "ppo",
        "tasks": tasks,
        "arms": arms,
        "arm_mechanisms": {
            arm: {
                "reward": bool(BIAS_ARMS[arm][0]),
                "action_prior": bool(BIAS_ARMS[arm][1]),
                "exploration": bool(BIAS_ARMS[arm][2]),
                "supervised_init": bool(BIAS_ARMS[arm][3]),
            }
            for arm in arms
        },
        "seeds": seeds,
        "env": env_summary,
        "ppo": cfg.__dict__,
        "dynamic_action_prior": {
            "llm_action_prior": args.llm_action_prior,
            "pre_run_reward_analysis": args.pre_run_reward_analysis,
            "checkup_steps": args.action_prior_checkup_steps,
            "max_checkups": args.action_prior_max_checkups,
            "max_weight": args.max_action_prior_weight,
        },
        "llm_backend": args.llm_backend,
        "llm_model": args.llm_model,
    }
    (args.out / "config.json").write_text(json.dumps(config, indent=2))

    metrics_path = args.out / "metrics.jsonl"
    eval_rows: list[dict[str, Any]] = []
    with metrics_path.open("w") as metrics_file:
        for task in tasks:
            for seed in seeds:
                for arm in arms:
                    run_dir = args.out / f"{task}_s{seed}_{arm}"
                    run_dir.mkdir(parents=True, exist_ok=True)
                    prior_coach = None
                    if bool(BIAS_ARMS[arm][1]) and args.action_prior_checkup_steps > 0:
                        prior_coach = ActionPriorCoach(
                            ActionPriorConfig(
                                llm_backend=args.llm_backend,
                                llm_model=args.llm_model,
                                task=task,
                                tasks=tasks,
                                arm=arm,
                                env_summary=env_summary,
                                log_dir=run_dir / "action_prior_checkups",
                                max_weight=args.max_action_prior_weight,
                                max_checkups=args.action_prior_max_checkups,
                                pre_run_reward_analysis=pre_run_reward_analysis,
                            ),
                            list(compiled_bias.spec.get("action_priors", [])),
                        )
                    params, rows, *_ = train_ppo_arm(
                        env=env,
                        bias=compiled_bias,
                        task=task,
                        arm=arm,
                        seed=seed,
                        cfg=cfg,
                        checkpoint_dir=run_dir / "checkpoints",
                        action_prior_checkup_interval=args.action_prior_checkup_steps,
                        action_prior_checkup_fn=prior_coach,
                    )
                    active_action_prior_weights = (
                        prior_coach.current_weights if prior_coach is not None else compiled_bias.default_action_prior_weights()
                    )
                    (run_dir / "final_action_prior_weights.json").write_text(
                        json.dumps(_action_prior_weight_dict(compiled_bias, active_action_prior_weights), indent=2) + "\n"
                    )
                    for row in rows:
                        row = row | {"seed": seed}
                        metrics_file.write(json.dumps(row) + "\n")
                        metrics_file.flush()
                    with (run_dir / "params.pkl").open("wb") as f:
                        pickle.dump(params, f)
                    eval_row = {
                        "task": task,
                        "seed": seed,
                        "arm": arm,
                        **evaluate_ppo_policy(
                            env=env,
                            params=params,
                            bias=compiled_bias,
                            task=task,
                            arm=arm,
                            seed=seed + 10_000,
                            n_envs=args.eval_envs,
                            cfg=cfg,
                            action_prior_weights=active_action_prior_weights,
                        ),
                    }
                    (run_dir / "eval.json").write_text(json.dumps(eval_row, indent=2))
                    eval_rows.append(eval_row)

    summary = summarize(eval_rows)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    write_csv(args.out / "eval.csv", eval_rows)
    print(f"Wrote {args.out}")
    print(json.dumps(summary, indent=2))
    return 0


def summarize(eval_rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for arm in sorted({row["arm"] for row in eval_rows}):
        rows = [row for row in eval_rows if row["arm"] == arm]
        out[arm] = {
            "eval_success_rate": round(sum(float(row["eval_success_rate"]) for row in rows) / len(rows), 6),
            "eval_instant_success_rate": round(sum(float(row.get("eval_instant_success_rate", row["eval_success_rate"])) for row in rows) / len(rows), 6),
            "eval_base_return": round(sum(float(row["eval_base_return"]) for row in rows) / len(rows), 6),
            "eval_shaped_return": round(sum(float(row["eval_shaped_return"]) for row in rows) / len(rows), 6),
            "eval_train_return": round(sum(float(row["eval_train_return"]) for row in rows) / len(rows), 6),
            "eval_lift_max": round(sum(float(row["eval_lift_max"]) for row in rows) / len(rows), 6),
            "eval_hard_clip_frac": round(sum(float(row["eval_hard_clip_frac"]) for row in rows) / len(rows), 6),
            "eval_saturation_frac": round(sum(float(row["eval_saturation_frac"]) for row in rows) / len(rows), 6),
            "eval_action_abs_mean": round(sum(float(row["eval_action_abs_mean"]) for row in rows) / len(rows), 6),
            "n_eval": len(rows),
        }
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _action_prior_weight_dict(bias: Any, weights: Any) -> dict[str, float]:
    return {
        str(rule.get("name", f"prior_{idx}")): round(float(value), 6)
        for idx, (rule, value) in enumerate(zip(bias.spec.get("action_priors", []), weights))
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm-backend", required=True)
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--bias-spec", type=Path, default=None)
    parser.add_argument("--tasks", default="lift")
    parser.add_argument("--arms", default="baseline,reward,action_prior,exploration,supervised_init")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--iters", type=int, default=300)
    parser.add_argument("--envs", type=int, default=1024)
    parser.add_argument("--eval-envs", type=int, default=1024)
    parser.add_argument("--hidden", type=int, nargs="+", default=[256, 256])
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--supervised-steps", type=int, default=80)
    parser.add_argument("--supervised-batch", type=int, default=128)
    parser.add_argument("--supervised-lr", type=float, default=1e-3)
    parser.add_argument("--episode-seconds", type=float, default=2.5)
    parser.add_argument("--control-dt", type=float, default=0.025)
    parser.add_argument("--physics-dt", type=float, default=0.01)
    parser.add_argument("--obj-xy-range", type=float, default=0.04)
    parser.add_argument("--checkpoint-count", type=int, default=5)
    parser.add_argument("--target-arm-seconds", type=float, default=None, help="Train each task/seed/arm for this many seconds after compile warmup")
    parser.add_argument("--target-total-seconds", type=float, default=None, help="Split this post-compile training budget across all task/seed/arm units")
    parser.add_argument("--action-transform", choices=["raw", "tanh"], default="tanh")
    parser.add_argument("--saturation-penalty", type=float, default=0.0)
    parser.add_argument("--saturation-threshold", type=float, default=0.98)
    parser.add_argument("--prior-logit-clip", type=float, default=0.95)
    parser.add_argument("--action-target-reward-weight", type=float, default=0.0)
    parser.add_argument("--success-hold-seconds", type=float, default=0.5)
    parser.add_argument("--success-lift-threshold", type=float, default=0.05)
    parser.add_argument("--llm-action-prior", action="store_true")
    parser.add_argument("--pre-run-reward-analysis", action="store_true")
    parser.add_argument("--action-prior-checkup-steps", type=int, default=0)
    parser.add_argument("--action-prior-max-checkups", type=int, default=3)
    parser.add_argument("--max-action-prior-weight", type=float, default=0.6)
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())

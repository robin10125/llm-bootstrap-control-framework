from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Any

import jax

from policy_bias_lab.bias import compile_bias
from policy_bias_lab.es import BIAS_ARMS, ESConfig, evaluate_policy, train_arm
from policy_bias_lab.llm_bias import load_bias_spec
from policy_bias_lab.policy import init_params
from policy_bias_lab.tasks import task_metadata


ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAPPING = ROOT.parent / "bootstrapping"


def main() -> int:
    args = parse_args()
    if str(BOOTSTRAPPING) not in sys.path:
        sys.path.insert(0, str(BOOTSTRAPPING))
    from mjx_env import make_env

    if args.out is None:
        args.out = Path("runs") / f"policy_bias_lab_{time.strftime('%Y%m%d-%H%M%S')}"
    args.out.mkdir(parents=True, exist_ok=True)

    tasks = [item.strip() for item in args.tasks.split(",") if item.strip()]
    arms = [item.strip() for item in args.arms.split(",") if item.strip()]
    for arm in arms:
        if arm not in BIAS_ARMS:
            raise KeyError(f"unknown arm {arm!r}; choose from {sorted(BIAS_ARMS)}")
    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]

    env_overrides: dict[str, Any] = {"episode_seconds": args.episode_seconds, "control_dt": args.control_dt}
    if args.obj_xy_range is not None:
        env_overrides["obj_xy_range"] = args.obj_xy_range
    env = make_env("shadow", **env_overrides)
    env_summary = {
        "obs_size": env.obs_size,
        "action_size": env.action_size,
        "horizon": env.horizon,
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
    (args.out / "bias_spec.json").write_text(json.dumps(bias_spec, indent=2))
    compiled_bias = compile_bias(bias_spec, env)

    if args.smoke:
        args.generations = min(args.generations, 2)
        args.population = min(args.population, 4)
        args.envs = min(args.envs, 4)
        args.eval_envs = min(args.eval_envs, 4)
        args.supervised_steps = min(args.supervised_steps, 2)

    cfg = ESConfig(
        generations=args.generations,
        population=args.population,
        population_batch=args.population_batch,
        envs=args.envs,
        sigma=args.sigma,
        lr=args.lr,
        supervised_steps=args.supervised_steps,
        supervised_batch=args.supervised_batch,
        supervised_lr=args.supervised_lr,
        target_train_seconds=args.target_train_seconds,
    )
    config = {
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
        "es": cfg.__dict__,
        "llm_backend": args.llm_backend,
        "llm_model": args.llm_model,
    }
    (args.out / "config.json").write_text(json.dumps(config, indent=2))

    rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    metrics_path = args.out / "metrics.jsonl"
    with metrics_path.open("w") as metrics_file:
        for task in tasks:
            for seed in seeds:
                key = jax.random.PRNGKey(seed)
                params0 = init_params(key, env.obs_size, env.action_size, hidden=args.hidden)
                for arm in arms:
                    run_dir = args.out / f"{task}_s{seed}_{arm}"
                    run_dir.mkdir(parents=True, exist_ok=True)
                    params, metrics = train_arm(
                        env=env,
                        init_params=params0,
                        bias=compiled_bias,
                        task=task,
                        arm=arm,
                        seed=seed,
                        cfg=cfg,
                        checkpoint_dir=run_dir / "checkpoints",
                        checkpoint_generations=checkpoint_generations(args.generations, args.checkpoint_count),
                        checkpoint_count=args.checkpoint_count,
                    )
                    for row in metrics:
                        row = row | {"seed": seed}
                        metrics_file.write(json.dumps(row) + "\n")
                        metrics_file.flush()
                        rows.append(row)
                    with (run_dir / "params.pkl").open("wb") as f:
                        pickle.dump(jax.device_get(params), f)
                    eval_stats = evaluate_policy(env, params, compiled_bias, task, arm, seed + 10_000, args.eval_envs)
                    eval_row = {
                        "task": task,
                        "seed": seed,
                        "arm": arm,
                        "eval_fitness": eval_stats.fitness,
                        "eval_success_rate": eval_stats.success_rate,
                        "eval_summary": eval_stats.eval_summary,
                    }
                    (run_dir / "eval.json").write_text(json.dumps(eval_row, indent=2))
                    eval_rows.append(eval_row)

    summary = summarize(rows, eval_rows)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    write_csv(args.out / "eval.csv", eval_rows)
    (args.out / "report.md").write_text(report_markdown(summary))
    print(f"Wrote {args.out}")
    print(json.dumps(summary, indent=2))
    return 0


def summarize(rows: list[dict[str, Any]], eval_rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    arms = sorted({row["arm"] for row in rows} | {row["arm"] for row in eval_rows})
    for arm in arms:
        arm_rows = [row for row in rows if row["arm"] == arm]
        arm_eval = [row for row in eval_rows if row["arm"] == arm]
        auc = sum(float(row["mean_success"]) for row in arm_rows)
        out[arm] = {
            "train_success_auc": round(auc, 6),
            "final_train_success": None if not arm_rows else arm_rows[-1]["mean_success"],
            "eval_success_rate": round(sum(float(row["eval_success_rate"]) for row in arm_eval) / max(len(arm_eval), 1), 6),
            "eval_fitness": round(sum(float(row["eval_fitness"]) for row in arm_eval) / max(len(arm_eval), 1), 6),
            "n_eval": len(arm_eval),
        }
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["task", "seed", "arm", "eval_fitness", "eval_success_rate", "eval_summary"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def report_markdown(summary: dict[str, Any]) -> str:
    lines = ["# Policy Bias Lab", "", "## Summary", ""]
    for arm, row in summary.items():
        lines.append(
            f"- `{arm}`: auc={row['train_success_auc']} eval_success={row['eval_success_rate']} "
            f"eval_fitness={row['eval_fitness']} n={row['n_eval']}"
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm-backend", required=True, help="LLM backend, or fixture for tests/smoke only")
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--bias-spec", type=Path, default=None, help="Reuse a saved BiasSpec JSON instead of calling the LLM")
    parser.add_argument("--tasks", default="lift,push,stabilize")
    parser.add_argument("--arms", default="baseline,reward,action_prior,exploration,supervised_init")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--generations", type=int, default=80)
    parser.add_argument("--population", type=int, default=64)
    parser.add_argument("--population-batch", type=int, default=1, help="deprecated compatibility option; Shadow physics is evaluated one candidate at a time")
    parser.add_argument("--envs", type=int, default=64)
    parser.add_argument("--eval-envs", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--sigma", type=float, default=0.04)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--supervised-steps", type=int, default=80)
    parser.add_argument("--supervised-batch", type=int, default=128)
    parser.add_argument("--supervised-lr", type=float, default=1e-3)
    parser.add_argument("--target-train-seconds", type=float, default=None, help="Train each task/seed/arm for this many seconds after evaluator compile warmup")
    parser.add_argument("--episode-seconds", type=float, default=2.5)
    parser.add_argument("--control-dt", type=float, default=0.025)
    parser.add_argument("--obj-xy-range", type=float, default=None)
    parser.add_argument("--checkpoint-count", type=int, default=5, help="Save policy checkpoints at this many evenly spaced points in each run")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def checkpoint_generations(generations: int, count: int) -> set[int]:
    if generations <= 0 or count <= 0:
        return set()
    points = {max(1, round(generations * idx / count)) for idx in range(1, count + 1)}
    return {min(generations, int(point)) for point in points}


if __name__ == "__main__":
    raise SystemExit(main())

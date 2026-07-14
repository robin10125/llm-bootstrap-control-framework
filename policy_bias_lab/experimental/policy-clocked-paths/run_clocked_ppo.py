"""Run one policy-clocked-paths PPO arm (see DESIGN.md / clocked_paths_ppo.py).

Accepts either a native clocked_paths program or a freeform_staged program (converted
mechanically: stage channels -> segment paths, next stage's gate -> done_hint, success carried).

Example (from llm-framework/):
  .venv/bin/python policy_bias_lab/experimental/policy-clocked-paths/run_clocked_ppo.py \
      --out runs/clocked_$(date +%m%d) --program runs/agentic_v4_20260704/best_program.json
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for p in (str(HERE), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import jax

from clocked_paths_ppo import (
    PROGRESSIONS,
    ClockedPPOConfig,
    convert_staged_program,
    evaluate_clocked_policy,
    train_clocked_ppo,
    validate_clocked_program,
)


def main() -> int:
    args = parse_args()
    from experiment_runtime.environment import make_env

    args.out.mkdir(parents=True, exist_ok=True)
    program = json.loads(args.program.read_text())
    if program.get("mode") == "freeform_staged":
        program = convert_staged_program(program, default_est_seconds=args.default_est_seconds)
    env = make_env("shadow", control_dt=args.control_dt, episode_seconds=args.episode_seconds,
                   physics_dt=args.physics_dt, obj_xy_range=args.obj_xy_range)
    if env.horizon % args.fragment_steps != 0:
        raise SystemExit(
            f"--fragment-steps {args.fragment_steps} must divide horizon {env.horizon} "
            f"({args.episode_seconds}s / {args.control_dt}s).")
    errors = validate_clocked_program(env, program)
    if errors:
        raise SystemExit("invalid clocked_paths program:\n  " + "\n  ".join(errors))

    cfg = ClockedPPOConfig(
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
        base_reward_weight=args.base_reward_weight,
        warmup_compile=not args.no_warmup_compile,
        control_dt=args.control_dt,
        default_est_seconds=args.default_est_seconds,
        progression=args.progression,
        rate_max=args.rate_max,
        min_dwell_steps=args.min_dwell_steps,
        dwell_slack=args.dwell_slack,
        hint_dwell_steps=args.hint_dwell_steps,
        recover_threshold=args.recover_threshold,
        max_recoveries=args.max_recoveries,
        recover_cooldown_steps=args.recover_cooldown_steps,
        residual_scale=args.residual_scale,
        potential_weight=args.potential_weight,
        imitation_coef=args.imitation_coef,
        imitation_anneal_iters=args.imitation_anneal_iters,
    )
    (args.out / "config.json").write_text(json.dumps({
        "learner": "experimental_clocked_paths_ppo",
        "progression": args.progression,
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
    (args.out / "clocked_program.json").write_text(json.dumps(program, indent=2) + "\n")

    params, rows, best_params, best_score, best_iter = train_clocked_ppo(
        env=env, program=program, task=args.task, seed=args.seed, cfg=cfg, out_dir=args.out)
    with (args.out / "params_final.pkl").open("wb") as f:
        pickle.dump(jax.device_get(params), f)
    with (args.out / "best_params.pkl").open("wb") as f:
        pickle.dump(jax.device_get(best_params), f)
    ev = evaluate_clocked_policy(env=env, params=best_params, program=program, task=args.task,
                                 seed=args.seed + 20_000, cfg=cfg)
    report = {
        "learner": "experimental_clocked_paths_ppo",
        "progression": args.progression,
        "iters": len(rows),
        "best_iter": int(best_iter),
        "best_objective": round(float(best_score), 6),
        # Headline = the SAME arbiter objective (task_graded_objective) as every other arm.
        "eval_objective": round(float(ev["eval_graded_objective"]), 6),
        "eval_graded_objective": round(float(ev["eval_graded_objective"]), 6),
        "eval_task_fitness": round(float(ev["eval_task_fitness"]), 6),
        "eval_success_rate": round(float(ev["eval_success_rate"]), 6),
        "eval": ev,
    }
    (args.out / "final_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"[done] progression={args.progression} {len(rows)} fragments, best_iter={best_iter}, "
          f"graded_objective={report['eval_graded_objective']} "
          f"success={report['eval_success_rate']} -> {args.out}")
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--program", type=Path, required=True,
                   help="clocked_paths program JSON, or a freeform_staged program (auto-converted)")
    p.add_argument("--task", default="lift")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--iters", type=int, default=2000)
    p.add_argument("--envs", type=int, default=256)
    p.add_argument("--eval-envs", type=int, default=256)
    p.add_argument("--fragment-steps", type=int, default=100)
    p.add_argument("--target-train-seconds", type=float, default=None)
    p.add_argument("--max-env-steps", type=int, default=None)
    p.add_argument("--checkpoint-every", type=int, default=0)
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--hidden", type=int, nargs="+", default=[256, 256])
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--ent-coef", type=float, default=0.0)
    p.add_argument("--base-reward-weight", type=float, default=1.0)
    p.add_argument("--episode-seconds", type=float, default=20.0)
    p.add_argument("--control-dt", type=float, default=0.025)
    p.add_argument("--physics-dt", type=float, default=0.01)
    p.add_argument("--obj-xy-range", type=float, default=0.04)
    p.add_argument("--no-warmup-compile", action="store_true")

    g = p.add_argument_group("progression")
    g.add_argument("--progression", choices=list(PROGRESSIONS), default="learned",
                   help="learned: policy rate/recover head owns switching. "
                        "autopilot: dwell-filtered hints + timeouts (hardened-gates ablation).")
    g.add_argument("--rate-max", type=float, default=3.0,
                   help="Max phase speed as a multiple of the authored est_seconds pace.")
    g.add_argument("--min-dwell-steps", type=int, default=5)
    g.add_argument("--dwell-slack", type=float, default=3.0,
                   help="Forced advance after slack * est_seconds in one segment (no stalls).")
    g.add_argument("--hint-dwell-steps", type=int, default=3)
    g.add_argument("--recover-threshold", type=float, default=0.5)
    g.add_argument("--max-recoveries", type=int, default=2)
    g.add_argument("--recover-cooldown-steps", type=int, default=20)
    g.add_argument("--default-est-seconds", type=float, default=2.0,
                   help="est_seconds for converted segments that do not declare one.")

    h = p.add_argument_group("hints / composition")
    h.add_argument("--residual-scale", type=float, default=1.0,
                   help="env_action = clip(segment_action + scale * policy_residual).")
    h.add_argument("--potential-weight", type=float, default=0.5,
                   help="Potential-based shaping on phi = (seg + u) / n_segments.")
    h.add_argument("--imitation-coef", type=float, default=0.5,
                   help="Initial pull of the progression head toward the authored hint decision; "
                        "anneals to zero over --imitation-anneal-iters (default: --iters).")
    h.add_argument("--imitation-anneal-iters", type=int, default=None)
    return p.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())

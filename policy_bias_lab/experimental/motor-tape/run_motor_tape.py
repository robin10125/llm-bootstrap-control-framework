"""Run one motor-tape PPO experiment (see motor_tape_ppo.py / DESIGN.md).

Arms (the ablation ladder):
  1. tape only (autopilot, no training):   --tape-only
  2. tape + residual:                      --no-rate
  3. tape + residual + rate (default):     (no extra flags)
  4. + continuous plan modulation:         --modulation on

Examples (from llm-framework/, using .venv/bin/python):
  .venv/bin/python policy_bias_lab/experimental/motor-tape/run_motor_tape.py \
      --program runs/motor_tape_gen/program.json --out runs/motor_tape_test \
      --target-train-seconds 300
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

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for p in (str(HERE), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import jax

from motor_tape import render_tape_report, score_tape, tape_report, validate_motor_tape
from motor_tape_ppo import (
    MotorTapePPOConfig,
    evaluate_motor_tape_policy,
    train_motor_tape_ppo,
)


def main() -> int:
    args = parse_args()
    from experiment_runtime.environment import EnvConfig, make_env

    args.out.mkdir(parents=True, exist_ok=True)
    cand = json.loads(args.program.read_text())
    env_overrides: dict = {}
    grasp_gated = None
    if args.reward_mode == "grasp_gated":
        # env pays NO lift income; the collect loop re-adds the same terms closure-gated
        # (see MotorTapePPOConfig.grasp_gated_lift). Weights come from the env defaults so
        # gate -> 1 recovers the default reward exactly.
        d = EnvConfig()
        env_overrides = dict(w_lift=0.0, w_lift_pot=0.0, w_lift_hold=0.0, w_success=0.0)
        grasp_gated = dict(
            w_lift=d.w_lift, w_lift_pot=d.w_lift_pot, w_lift_hold=d.w_lift_hold,
            w_success=d.w_success, lift_target=d.lift_target, success_height=d.success_height,
            contact_target=d.contact_target, fling_xy_thresh=d.fling_xy_thresh,
            pbrs_gamma=d.pbrs_gamma, closure_target=args.closure_target)
    env = make_env("shadow", control_dt=args.control_dt, episode_seconds=args.episode_seconds,
                   physics_dt=args.physics_dt, obj_xy_range=args.obj_xy_range, **env_overrides)
    if env.horizon % args.fragment_steps != 0:
        raise SystemExit(
            f"--fragment-steps {args.fragment_steps} must divide horizon {env.horizon}.")
    errors: list[str] = []
    program = validate_motor_tape(env, cand, errors)
    if program is None:
        raise SystemExit("invalid motor_tape program:\n  " + "\n  ".join(errors))
    rep = tape_report(env, program, envs=32, seed=args.seed)
    print(render_tape_report(rep))

    cfg = MotorTapePPOConfig(
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
        use_residual=not args.no_residual,
        use_rate=not args.no_rate,
        use_modulation=(args.modulation == "on"),
        residual_scale=args.residual_scale,
        rate_lo=args.rate_lo,
        rate_hi=args.rate_hi,
        mod_scale=args.mod_scale,
        mod_smooth_coef=args.mod_smooth_coef,
        mod_l2_coef=args.mod_l2_coef,
        lookahead_deltas=tuple(float(x) for x in args.lookahead.split(",") if x.strip()),
        potential_weight=args.potential_weight,
        imitation_coef=args.imitation_coef,
        imitation_anneal_iters=args.imitation_anneal_iters,
        grasp_gated_lift=grasp_gated,
        handoff_lo=args.handoff_lo,
        handoff_hi=args.handoff_hi,
    )
    (args.out / "config.json").write_text(json.dumps({
        "learner": "experimental_motor_tape_ppo",
        "task": args.task,
        "seed": args.seed,
        "program": str(args.program),
        "tape_only": bool(args.tape_only),
        "tape_report": rep,
        "env": {
            "horizon": int(env.horizon),
            "control_dt": args.control_dt,
            "episode_seconds": args.episode_seconds,
            "fragment_steps": args.fragment_steps,
            "fragments_per_episode": int(env.horizon) // int(args.fragment_steps),
        },
        "ppo": cfg.__dict__,
    }, indent=2, default=str) + "\n")
    (args.out / "program.json").write_text(json.dumps(program, indent=2) + "\n")

    # The tape's own performance (pure playback, rate = 1, no networks): the plan-alone floor
    # every trained arm is compared against. The tape is static, so this is measured ONCE.
    autopilot = score_tape(env, program, envs=int(args.eval_envs), seed=args.seed + 30_000,
                           task=args.task)
    print(f"[autopilot] graded={autopilot['tape_eval']['eval_graded_objective']} "
          f"success={autopilot['tape_eval']['eval_success_rate']} "
          f"contact_engagement={autopilot['contact_engagement']:.3f}")
    if args.tape_only:
        report = {"arm": "tape_only", "tape_autopilot": autopilot,
                  "eval_graded_objective": autopilot["tape_eval"]["eval_graded_objective"],
                  "eval_success_rate": autopilot["tape_eval"]["eval_success_rate"]}
        (args.out / "final_report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"[done] tape_only -> {args.out}")
        return 0

    params, rows, best_params, best_score, best_iter = train_motor_tape_ppo(
        env=env, program=program, task=args.task, seed=args.seed, cfg=cfg, out_dir=args.out)
    with (args.out / "params_final.pkl").open("wb") as f:
        pickle.dump(jax.device_get(params), f)
    with (args.out / "best_params.pkl").open("wb") as f:
        pickle.dump(jax.device_get(best_params), f)
    ev = evaluate_motor_tape_policy(env=env, params=best_params, program=program, task=args.task,
                                    seed=args.seed + 20_000, cfg=cfg)
    report = {
        "arm": ("tape+residual" if args.no_rate else
                "tape+residual+rate+mod" if args.modulation == "on" else
                "tape+residual+rate") if not args.no_residual else "tape+heads",
        "iters": len(rows),
        "best_iter": int(best_iter),
        "best_objective": round(float(best_score), 6),
        "eval_objective": round(float(ev["eval_graded_objective"]), 6),
        "eval_graded_objective": round(float(ev["eval_graded_objective"]), 6),
        "eval_task_fitness": round(float(ev["eval_task_fitness"]), 6),
        "eval_success_rate": round(float(ev["eval_success_rate"]), 6),
        "eval": ev,
        "tape_autopilot": autopilot,
    }
    (args.out / "final_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"[done] {report['arm']} {len(rows)} fragments, best_iter={best_iter}, "
          f"graded_objective={report['eval_graded_objective']} "
          f"success={report['eval_success_rate']} "
          f"(tape alone: {autopilot['tape_eval']['eval_graded_objective']}) -> {args.out}")
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--program", type=Path, required=True, help="motor_tape program JSON")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--task", default="lift")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--iters", type=int, default=2000)
    p.add_argument("--envs", type=int, default=256)
    p.add_argument("--eval-envs", type=int, default=128)
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

    g = p.add_argument_group("corrective heads / arms")
    g.add_argument("--tape-only", action="store_true",
                   help="no training: score pure tape playback (autopilot) and exit")
    g.add_argument("--no-residual", action="store_true")
    g.add_argument("--no-rate", action="store_true",
                   help="remove the time-warp head (playback locked to the authored pace)")
    g.add_argument("--modulation", choices=["on", "off"], default="off",
                   help="continuous plan-modulation head (bends the tape command online)")
    g.add_argument("--residual-scale", type=float, default=1.0)
    g.add_argument("--rate-lo", type=float, default=0.0)
    g.add_argument("--rate-hi", type=float, default=2.0)
    g.add_argument("--mod-scale", type=float, default=0.15)
    g.add_argument("--mod-smooth-coef", type=float, default=1e-2)
    g.add_argument("--mod-l2-coef", type=float, default=1e-3)
    g.add_argument("--lookahead", default="0.1,0.25,0.5,1.0",
                   help="comma-separated efference-copy lookahead offsets (seconds of plan time)")
    g.add_argument("--potential-weight", type=float, default=0.5,
                   help="potential-based shaping on phase progress phi = s/T")
    g.add_argument("--imitation-coef", type=float, default=0.5,
                   help="rate head pulled toward the authored pace (1.0), annealed to zero")
    g.add_argument("--imitation-anneal-iters", type=int, default=None)

    m = p.add_argument_group("contact-harm mitigations")
    m.add_argument("--reward-mode", choices=["default", "grasp_gated"], default="default",
                   help="grasp_gated: env lift income zeroed and re-added in the collect loop "
                        "multiplied by clip(closure/closure_target, 0, 1) -- lift only pays in "
                        "proportion to a formed grasp")
    m.add_argument("--closure-target", type=float, default=0.5,
                   help="closure at which the grasp gate saturates (matches the graded "
                        "objective's grasp threshold)")
    m.add_argument("--handoff-lo", type=float, default=0.0,
                   help="||obj_rel|| (m) at/below which the tape feedforward is fully masked")
    m.add_argument("--handoff-hi", type=float, default=0.0,
                   help="||obj_rel|| (m) above which the tape feedforward is untouched; 0 = "
                        "handoff disabled. Fades linearly between hi and lo.")
    return p.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())

"""Run one alternative prior-influence PPO experiment (see alt_methods_ppo.py).

Examples (from llm-framework/, using .venv/bin/python):
  # 1. prior as exploration proposal
  .venv/bin/python policy_bias_lab/experimental/alternative-methods/run_alt_method.py \
      --method proposal --out runs/alt_proposal_test \
      --program runs/agentic_v4_20260704/best_program.json --episode-seconds 20

  # 2. prior as curriculum / state-visitation shaper
  ... --method curriculum --warmup-frac 0.5 --warmup-anneal-iters 1500

  # 3. prior as value shaping (no action prior at all)
  ... --method value_shaping --potential-weight 0.5 --aux-coef 0.1

  # 4. prior diagnostics as critic features
  ... --method critic_features
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
BOOTSTRAPPING = ROOT.parent / "bootstrapping"
if str(BOOTSTRAPPING) not in sys.path:
    sys.path.insert(0, str(BOOTSTRAPPING))

import jax

from alt_methods_ppo import METHODS, AltPPOConfig, evaluate_alt_policy, train_alt_ppo
from policy_bias_lab.bias import compile_bias


def main() -> int:
    args = parse_args()
    from mjx_env import make_env

    args.out.mkdir(parents=True, exist_ok=True)
    program = json.loads(args.program.read_text())
    env = make_env("shadow", control_dt=args.control_dt, episode_seconds=args.episode_seconds,
                   physics_dt=args.physics_dt, obj_xy_range=args.obj_xy_range)
    if env.horizon % args.fragment_steps != 0:
        raise SystemExit(
            f"--fragment-steps {args.fragment_steps} must divide horizon {env.horizon} "
            f"({args.episode_seconds}s / {args.control_dt}s).")
    bias = compile_bias({"name": f"alt_{args.method}", "action_priors": [],
                         "prior_program": program}, env)
    cfg = AltPPOConfig(
        method=args.method,
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
        proposal_prob=args.proposal_prob,
        proposal_prob_final=args.proposal_prob_final,
        proposal_anneal_iters=args.proposal_anneal_iters,
        proposal_sigma=args.proposal_sigma,
        proposal_gate=args.proposal_gate,
        proposal_offpolicy=args.proposal_offpolicy,
        warmup_frac=args.warmup_frac,
        warmup_frac_final=args.warmup_frac_final,
        warmup_anneal_iters=args.warmup_anneal_iters,
        warmup_mode=args.warmup_mode,
        stage_reward_weight=args.stage_reward_weight,
        stage_progress_weight=args.stage_progress_weight,
        stage_completion_bonus=args.stage_completion_bonus,
        stage_success_temperature=args.stage_success_temperature,
        stage_reward_clip=args.stage_reward_clip,
        potential_weight=args.potential_weight,
        potential_temp=args.potential_temp,
        aux_coef=args.aux_coef,
        critic_stage_onehot=not args.no_critic_stage_onehot,
        critic_prior_action=not args.no_critic_prior_action,
        critic_prior_norms=not args.no_critic_prior_norms,
        critic_success_margins=not args.no_critic_success_margins,
        kl_coef=args.kl_coef,
        kl_coef_final=args.kl_coef_final,
        kl_anneal_iters=args.kl_anneal_iters,
        kl_sigma_ref=args.kl_sigma_ref,
        kl_ref_clip=args.kl_ref_clip,
        kl_target=args.kl_target,
        kl_target_final=args.kl_target_final,
    )
    (args.out / "config.json").write_text(json.dumps({
        "learner": "experimental_alt_method_ppo",
        "method": args.method,
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

    params, rows, best_params, best_score, best_iter = train_alt_ppo(
        env=env, bias=bias, program=program, task=args.task, seed=args.seed, cfg=cfg,
        out_dir=args.out)
    with (args.out / "params_final.pkl").open("wb") as f:
        pickle.dump(jax.device_get(params), f)
    with (args.out / "best_params.pkl").open("wb") as f:
        pickle.dump(jax.device_get(best_params), f)
    ev = evaluate_alt_policy(env=env, params=best_params, bias=bias, program=program,
                             task=args.task, seed=args.seed + 20_000, cfg=cfg)
    report = {
        "method": args.method,
        "iters": len(rows),
        "best_iter": int(best_iter),
        "best_objective": round(float(best_score), 6),
        # Headline = the SAME arbiter objective (task_graded_objective) as the baseline arm and
        # prior generation, so all methods are directly comparable.
        "eval_objective": round(float(ev["eval_graded_objective"]), 6),
        "eval_graded_objective": round(float(ev["eval_graded_objective"]), 6),
        "eval_task_fitness": round(float(ev["eval_task_fitness"]), 6),
        "eval_success_rate": round(float(ev["eval_success_rate"]), 6),
        "eval": ev,
    }
    (args.out / "final_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"[done] method={args.method} {len(rows)} fragments, best_iter={best_iter}, "
          f"graded_objective={report['eval_graded_objective']} "
          f"success={report['eval_success_rate']} -> {args.out}")
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--method", choices=list(METHODS), required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--program", type=Path, required=True)
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

    g1 = p.add_argument_group("method=proposal")
    g1.add_argument("--proposal-prob", type=float, default=0.3,
                    help="Initial per-step probability of executing a prior-centered proposal.")
    g1.add_argument("--proposal-prob-final", type=float, default=0.0)
    g1.add_argument("--proposal-anneal-iters", type=int, default=None,
                    help="Iterations to anneal proposal-prob to its final value (default: iters).")
    g1.add_argument("--proposal-sigma", type=float, default=0.10,
                    help="Gaussian exploration noise around the prior suggestion.")
    g1.add_argument("--proposal-gate", choices=["none", "low_value"], default="none",
                    help="low_value: only propose in states whose critic value is below the "
                         "batch median (a low-confidence proxy).")
    g1.add_argument("--proposal-offpolicy", choices=["ratio", "mask"], default="ratio",
                    help="ratio: keep proposal steps, scored with the executed action's log-prob "
                         "under the neural policy (PPO importance ratio handles them); "
                         "mask: exclude proposal steps from the PPO loss entirely.")

    g2 = p.add_argument_group("method=curriculum")
    g2.add_argument("--warmup-frac", type=float, default=0.5,
                    help="Initial max prior-driven warmup length as a fraction of the horizon.")
    g2.add_argument("--warmup-frac-final", type=float, default=0.0)
    g2.add_argument("--warmup-anneal-iters", type=int, default=None)
    g2.add_argument("--warmup-mode", choices=["uniform", "fixed"], default="uniform",
                    help="uniform: per-env warmup length U{0..max} so training sees every stage "
                         "depth; fixed: all envs warm up exactly max steps.")

    g3 = p.add_argument_group("method=value_shaping")
    g3.add_argument("--stage-reward-weight", type=float, default=1.0)
    g3.add_argument("--stage-progress-weight", type=float, default=1.0)
    g3.add_argument("--stage-completion-bonus", type=float, default=0.05)
    g3.add_argument("--stage-success-temperature", type=float, default=1.0)
    g3.add_argument("--stage-reward-clip", type=float, default=0.5)
    g3.add_argument("--potential-weight", type=float, default=0.5,
                    help="Weight of the potential-based term on ladder progress "
                         "phi(s) = mean_k sigmoid(success_k / temp).")
    g3.add_argument("--potential-temp", type=float, default=1.0)
    g3.add_argument("--aux-coef", type=float, default=0.0,
                    help=">0 adds an auxiliary critic head regressing phi (prior-defined progress).")

    g4 = p.add_argument_group("method=critic_features")
    g4.add_argument("--no-critic-stage-onehot", action="store_true")
    g4.add_argument("--no-critic-prior-action", action="store_true")
    g4.add_argument("--no-critic-prior-norms", action="store_true")
    g4.add_argument("--no-critic-success-margins", action="store_true")

    g5 = p.add_argument_group("method=kl_prior")
    g5.add_argument("--kl-coef", type=float, default=1.0,
                    help="Initial beta: the per-state price of diverging from the instruction.")
    g5.add_argument("--kl-coef-final", type=float, default=0.0,
                    help="Annealed-to beta (ignored when --kl-target is set).")
    g5.add_argument("--kl-anneal-iters", type=int, default=None,
                    help="Anneal window for beta or the KL budget (default: --iters).")
    g5.add_argument("--kl-sigma-ref", type=float, default=0.3,
                    help="Pre-tanh width of the instruction cone the policy explores within.")
    g5.add_argument("--kl-ref-clip", type=float, default=3.0)
    g5.add_argument("--kl-target", type=float, default=None,
                    help="If set, beta is servo-controlled so measured KL tracks this budget, "
                         "annealed toward --kl-target-final (divergence budget grows over "
                         "training).")
    g5.add_argument("--kl-target-final", type=float, default=None)
    return p.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())

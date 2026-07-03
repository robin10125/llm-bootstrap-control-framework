"""CLI entrypoint for the agentic action-prior selection orchestrator.

Runs Stage-0 context -> diverse seeds -> explore/refine under a rollout budget (see
AGENTIC_PRIOR_SELECTION.md and agentic_orchestrator.py), writes the winning prior program, and
prints how to hand it to the slim PPO runner.

Example:
  python -m policy_bias_lab.run_agentic_selection --out runs/agentic1 --rep freeform --budget 20
  # then PPO-train the discovered prior:
  python -m policy_bias_lab.run_prior_ppo --arms freeform_encourage \
      --prior-program-arm freeform_encourage=runs/agentic1/best_program.json
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAPPING = ROOT.parent / "bootstrapping"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--task", default="grasp and lift a 5cm cube off the table")
    ap.add_argument("--rep", choices=["dsl", "freeform", "freeform_staged"], default="freeform",
                    help="output representation. freeform_staged = free-form prior with model-authored "
                    "stages (each its own gate + channels), generalizing the fixed 3-phase gate.")
    ap.add_argument("--dof-mode", choices=["consider", "encourage"], default="encourage")
    ap.add_argument("--llm-backend", default="codex")
    ap.add_argument("--llm-model", default=None)
    ap.add_argument("--budget", type=int, default=10,
                    help="B: max candidate evaluations (default 10 = 3 explore + ~7 refine).")
    ap.add_argument("--n-seeds", type=int, default=3,
                    help="explore cap; breadth saturates ~3 seeds (marginal-value run).")
    ap.add_argument("--patience", type=int, default=3, help="w: plateau/degradation window.")
    ap.add_argument("--per-stage-iters", type=int, default=None,
                    help="guarantee each authored stage this many improvement attempts if it "
                    "becomes the failing frontier: sets patience to this and resizes the refine "
                    "budget to n_stages x this per chosen chain (overrides --budget/--patience "
                    "for freeform_staged runs with a stage report).")
    ap.add_argument("--eps", type=float, default=1e-3, help="improvement tolerance.")
    ap.add_argument("--human-analogy", action="store_true", help="enable the optional C4 context.")
    ap.add_argument("--arbiter", choices=["short_ppo", "open_loop"], default="short_ppo",
                    help="short_ppo (default): rank on TRAINED contact-gated success (the real "
                    "objective); open_loop: the cheap blind rollout score (prefilter-grade).")
    ap.add_argument("--ppo-task", default="lift", help="task key for the PPO arbiter.")
    ap.add_argument("--ppo-train-seconds", type=float, default=180.0,
                    help="short-PPO budget per candidate (the per-iteration cost).")
    ap.add_argument("--ppo-train-envs", type=int, default=256)
    ap.add_argument("--ppo-eval-envs", type=int, default=256)
    ap.add_argument("--score-envs", type=int, default=128, help="open_loop arbiter only.")
    ap.add_argument("--score-seed", type=int, default=0)
    args = ap.parse_args()

    if str(BOOTSTRAPPING) not in sys.path:
        sys.path.insert(0, str(BOOTSTRAPPING))
    from mjx_env import make_env

    from policy_bias_lab.agentic_orchestrator import AgenticOrchestrator

    env = make_env("shadow")
    orch = AgenticOrchestrator(
        env=env, task=args.task, rep=args.rep, dof_mode=args.dof_mode,
        llm_backend=args.llm_backend, llm_model=args.llm_model, out_dir=args.out,
        budget=args.budget, n_seeds=args.n_seeds, patience=args.patience,
        per_stage_iters=args.per_stage_iters, eps=args.eps,
        use_human_analogy=args.human_analogy, arbiter=args.arbiter, ppo_task=args.ppo_task,
        ppo_train_seconds=args.ppo_train_seconds, ppo_train_envs=args.ppo_train_envs,
        ppo_eval_envs=args.ppo_eval_envs, score_envs=args.score_envs, score_seed=args.score_seed,
    )
    report = orch.run()
    print(f"\nBest objective {report['best']['objective']:+.3f} "
          f"(wrist_driven={report['best']['accounting'].get('wrist_driven')}, "
          f"{report['best']['accounting'].get('n_driven')} DOF driven).")
    print(f"Program: {args.out / 'best_program.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

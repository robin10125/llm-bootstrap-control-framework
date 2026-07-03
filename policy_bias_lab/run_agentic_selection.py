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
    ap.add_argument("--resume", action="store_true",
                    help="resume from <out>/checkpoint.pkl (written after every evaluation). "
                    "Config is restored from the checkpoint; any flag you pass EXPLICITLY on the "
                    "resume command line overrides the saved value (e.g. --budget to extend a run).")
    ap.add_argument("--stop-after", type=int, default=None,
                    help="pause after this many evaluations THIS SESSION: checkpoint and exit "
                    "cleanly. The run can also be paused at any time with SIGINT/SIGTERM (the "
                    "in-flight evaluation finishes, then state is saved).")
    ap.add_argument("--plateau-hours", type=float, default=None,
                    help="WALL-CLOCK MODE: terminate only when the best objective has not improved "
                    "for this many hours of run time (paused time excluded). Disables all "
                    "iteration-count plateau stops and chain deactivation; --budget defaults to "
                    "unlimited unless passed explicitly.")
    ap.add_argument("--min-hours", type=float, default=None,
                    help="no wall-clock plateau stop before this much accumulated run time.")
    ap.add_argument("--success-stop", type=float, default=None,
                    help="terminate when a trained policy's sustained contact-gated hold success "
                    "rate (trained_success) reaches this value.")
    ap.add_argument("--dashboard", action="store_true",
                    help="write/refresh dashboard.html on every evaluation (off by default; "
                    "regenerate manually anytime: python -m policy_bias_lab.run_dashboard <out>).")
    args = ap.parse_args()

    if args.plateau_hours is not None and not any(o in sys.argv for o in ("--budget",)):
        args.budget = 10 ** 6  # wall-clock criteria govern; iteration budget out of the way

    if str(BOOTSTRAPPING) not in sys.path:
        sys.path.insert(0, str(BOOTSTRAPPING))
    from mjx_env import make_env

    from policy_bias_lab.agentic_orchestrator import AgenticOrchestrator

    cfg = dict(
        task=args.task, rep=args.rep, dof_mode=args.dof_mode,
        llm_backend=args.llm_backend, llm_model=args.llm_model,
        budget=args.budget, n_seeds=args.n_seeds, patience=args.patience,
        per_stage_iters=args.per_stage_iters, eps=args.eps,
        use_human_analogy=args.human_analogy, arbiter=args.arbiter, ppo_task=args.ppo_task,
        ppo_train_seconds=args.ppo_train_seconds, ppo_train_envs=args.ppo_train_envs,
        ppo_eval_envs=args.ppo_eval_envs, score_envs=args.score_envs, score_seed=args.score_seed,
        min_hours=args.min_hours, plateau_hours=args.plateau_hours,
        success_stop=args.success_stop, write_dash=args.dashboard,
    )
    saved_state = None
    if args.resume:
        ckpt_file = args.out / "checkpoint.pkl"
        if not ckpt_file.exists():
            print(f"[resume] no checkpoint at {ckpt_file}", file=sys.stderr)
            return 1
        payload = AgenticOrchestrator.load_checkpoint(args.out)
        saved_cfg, saved_state = payload["config"], payload["state"]
        # Saved config wins; a flag typed explicitly on the resume command line overrides it.
        explicit = {a.dest for a in ap._actions
                    if a.option_strings and any(o in sys.argv for o in a.option_strings)}
        arg_dest = {"use_human_analogy": "human_analogy",  # cfg key -> argparse dest where they differ
                    "write_dash": "dashboard"}
        for k, v in saved_cfg.items():
            if k in cfg and arg_dest.get(k, k) not in explicit:
                cfg[k] = v
        print(f"[resume] {ckpt_file}: {saved_state['iters']}/{saved_state['budget']} iterations "
              f"done, picking up from there")

    env = make_env("shadow")
    orch = AgenticOrchestrator(env=env, out_dir=args.out, stop_after=args.stop_after, **cfg)
    if saved_state is not None:
        orch.restore(saved_state)
        # An explicit override of a value that is also persisted state (budget/patience) must win
        # over the restored value, e.g. --budget to extend a finished-budget run.
        if "patience" in explicit:
            orch.patience = cfg["patience"]
        if "budget" in explicit:
            if cfg["budget"] > saved_state.get("budget", 0):
                orch.extend_budget(cfg["budget"])
            else:
                orch.budget = cfg["budget"]
    report = orch.run()
    if report.get("paused"):
        return 0
    print(f"\nBest objective {report['best']['objective']:+.3f} "
          f"(wrist_driven={report['best']['accounting'].get('wrist_driven')}, "
          f"{report['best']['accounting'].get('n_driven')} DOF driven).")
    print(f"Program: {args.out / 'best_program.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Marginal-value ablation: is budget better spent on MORE initial candidates (explore breadth) or
on IMPROVING the best one (refine depth)?

For each rep, with a fixed Stage-0 context:
  BREADTH: generate K seeds, evaluate all, record best-of-first-k for k=1..K. The marginal value of
           the k-th seed is best(k) - best(k-1).
  DEPTH:   take the best seed and run R refinement iterations (NO early-stop), record best-so-far
           after r revisions. The marginal value of the r-th revision is best(r) - best(r-1).
Repeat over `reps` independent samplings and average; the spread shows how reliable each lever is.

Each evaluation is one arbiter call (default PPO == one fragmented-stage training run), so total cost ~
reps * (K + R) PPO runs. Keep K/R/reps modest. Uses the orchestrator's own primitives so the
ablation matches the real selection pipeline exactly.

  python -m policy_bias_lab.run_marginal_value --out runs/mv1 --k-seeds 6 --r-depth 6 --reps 3 \
      --arbiter ppo --ppo-train-seconds 180
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAPPING = ROOT.parent / "bootstrapping"


def _cummax(xs: list[float]) -> list[float]:
    out, m = [], -1e18
    for x in xs:
        m = max(m, x)
        out.append(m)
    return out


def _mean_sd(rows: list[list[float]]) -> tuple[list[float], list[float]]:
    import numpy as np
    a = np.asarray(rows, dtype=float)
    return a.mean(0).tolist(), a.std(0).tolist()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--task", default="grasp and lift a 5cm cube off the table")
    ap.add_argument("--rep", choices=["freeform"], default="freeform")
    ap.add_argument("--dof-mode", choices=["consider", "encourage"], default="encourage")
    ap.add_argument("--llm-backend", default="codex")
    ap.add_argument("--llm-model", default=None)
    ap.add_argument("--k-seeds", type=int, default=6, help="breadth: seeds generated/evaluated.")
    ap.add_argument("--r-depth", type=int, default=6, help="depth: refinement iters on the best seed.")
    ap.add_argument("--reps", type=int, default=3, help="independent samplings to average over.")
    ap.add_argument("--arbiter", choices=["ppo", "short_ppo", "open_loop"], default="ppo")
    ap.add_argument("--ppo-task", default="lift")
    ap.add_argument("--ppo-train-seconds", type=float, default=180.0)
    ap.add_argument("--ppo-train-envs", type=int, default=256)
    ap.add_argument("--ppo-eval-envs", type=int, default=256)
    ap.add_argument("--score-envs", type=int, default=128)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if str(BOOTSTRAPPING) not in sys.path:
        sys.path.insert(0, str(BOOTSTRAPPING))
    from mjx_env import make_env
    from policy_bias_lab.agentic_orchestrator import AgenticOrchestrator, _Chain

    env = make_env("shadow")
    orch = AgenticOrchestrator(
        env=env, task=args.task, rep=args.rep, dof_mode=args.dof_mode,
        llm_backend=args.llm_backend, llm_model=args.llm_model, out_dir=args.out,
        n_seeds=args.k_seeds, budget=10**9, patience=10**9, arbiter=args.arbiter,
        ppo_task=args.ppo_task, ppo_train_seconds=args.ppo_train_seconds,
        ppo_train_envs=args.ppo_train_envs, ppo_eval_envs=args.ppo_eval_envs,
        score_envs=args.score_envs,
    )
    context_block = orch.gather_context()  # Stage-0 once; shared across reps

    breadth_rows, depth_rows, per_rep = [], [], []
    for rep_i in range(args.reps):
        print(f"\n=== rep {rep_i + 1}/{args.reps} ===")
        seeds = orch.generate_seeds(context_block)  # fresh sampling each rep
        # BREADTH: evaluate up to K seeds in returned order.
        seed_objs, seed_recs = [], []
        for c in seeds[: args.k_seeds]:
            rec = orch._eval(c, "breadth")
            if rec is None:
                continue
            seed_objs.append(rec["objective"])
            seed_recs.append((c, rec))
        if not seed_recs:
            print("  [skip rep] no seed compiled")
            continue
        breadth_curve = _cummax(seed_objs)
        # DEPTH: refine the best seed for R steps, no early stop.
        bi = max(range(len(seed_recs)), key=lambda i: seed_recs[i][1]["objective"])
        cand0, rec0 = seed_recs[bi]
        chain = _Chain(name=str(cand0.get("name") or "best"), raw_cand=cand0,
                       program=rec0["program"], diagnostics=rec0["diagnostics"],
                       best_obj=rec0["objective"], history=[rec0["objective"]])
        depth_curve = [chain.best_obj]
        for _r in range(args.r_depth):
            cand = orch.revise(chain, context_block)
            if cand is None:
                depth_curve.append(chain.best_obj)
                continue
            rec = orch._eval(cand, f"depth:{chain.name}")
            if rec is None:
                depth_curve.append(chain.best_obj)
                continue
            if rec["objective"] > chain.best_obj:
                chain.best_obj = rec["objective"]
                chain.program = rec["program"]
                chain.raw_cand = cand
                chain.diagnostics = rec["diagnostics"]
            depth_curve.append(chain.best_obj)
        breadth_rows.append(breadth_curve[: args.k_seeds] + [breadth_curve[-1]] * (args.k_seeds - len(breadth_curve)))
        depth_rows.append(depth_curve[: args.r_depth + 1])
        per_rep.append({"rep": rep_i, "breadth_curve": breadth_curve, "depth_curve": depth_curve})
        (args.out / "marginal_value_partial.json").write_text(json.dumps(per_rep, indent=2) + "\n")

    if not breadth_rows:
        print("[abort] no successful reps")
        return 1
    bmean, bsd = _mean_sd(breadth_rows)
    dmean, dsd = _mean_sd(depth_rows)
    breadth_marginal = [bmean[0]] + [bmean[k] - bmean[k - 1] for k in range(1, len(bmean))]
    depth_marginal = [0.0] + [dmean[r] - dmean[r - 1] for r in range(1, len(dmean))]
    summary = {
        "arbiter": args.arbiter, "reps": len(breadth_rows), "k_seeds": args.k_seeds,
        "r_depth": args.r_depth, "ppo_train_seconds": args.ppo_train_seconds,
        "breadth_best_of_first_k_mean": bmean, "breadth_best_of_first_k_sd": bsd,
        "breadth_marginal_gain_per_seed": breadth_marginal,
        "depth_best_so_far_mean": dmean, "depth_best_so_far_sd": dsd,
        "depth_marginal_gain_per_revision": depth_marginal,
        "per_rep": per_rep,
    }
    (args.out / "marginal_value.json").write_text(json.dumps(summary, indent=2) + "\n")
    _maybe_plot(args.out, bmean, bsd, dmean, dsd)

    print("\n--- MARGINAL VALUE ---")
    print("breadth best-of-first-k (mean):", [round(x, 4) for x in bmean])
    print("  marginal gain per added seed:", [round(x, 4) for x in breadth_marginal])
    print("depth best-so-far (mean):      ", [round(x, 4) for x in dmean])
    print("  marginal gain per revision:  ", [round(x, 4) for x in depth_marginal])
    print(f"\nWrote {args.out / 'marginal_value.json'}")
    return 0


def _maybe_plot(out: Path, bmean, bsd, dmean, dsd) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:  # noqa: BLE001
        return
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    k = np.arange(1, len(bmean) + 1)
    ax[0].errorbar(k, bmean, yerr=bsd, marker="o", capsize=3)
    ax[0].set_title("Breadth: best objective vs # initial candidates")
    ax[0].set_xlabel("seeds evaluated (k)"); ax[0].set_ylabel("best objective")
    r = np.arange(0, len(dmean))
    ax[1].errorbar(r, dmean, yerr=dsd, marker="o", color="tab:orange", capsize=3)
    ax[1].set_title("Depth: best objective vs # revisions of the best seed")
    ax[1].set_xlabel("revisions (r)"); ax[1].set_ylabel("best objective")
    fig.tight_layout()
    fig.savefig(out / "marginal_value.png", dpi=120)


if __name__ == "__main__":
    raise SystemExit(main())

#!/bin/bash
# 3-seed head-to-head: critic_features (best arm of the 4h comparison) vs a true
# no-prior baseline (run_long_ppo --arm baseline -- the same trainer/invocation that
# produced the 100%-success 6h baseline in full_long_prior_vs_baseline_20260708),
# with --no-learn-prior-scale so both arms have structurally identical actor heads
# (the scale head is inert with no prior but still enters the PPO ratio).
#
# One shared generated prior (kl_nouse, generation-study phase-1 winner).
# 4h per run x 2 arms x 3 seeds = 24h sequential. Runs are interleaved by seed so an
# early interruption still leaves balanced pairs.
#
# NOT started automatically -- launch from anywhere with:
#   mkdir -p <repo>/runs/critic_vs_baseline_3seed
#   nohup bash <repo>/policy_bias_lab/experimental/prior-generation-study/run_critic_vs_baseline_3seed.sh \
#     > <repo>/runs/critic_vs_baseline_3seed/driver.log 2>&1 &
# Portable: the llm-framework root is derived from this script's own location
# (script lives at policy_bias_lab/experimental/prior-generation-study/).
set -u
cd "$(dirname "$(readlink -f "$0")")/../../.."
OUT=runs/critic_vs_baseline_3seed
PROG=runs/prior_gen_study_20260711/kl_nouse/program.json
PY=.venv/bin/python
ALT=policy_bias_lab/experimental/alternative-methods/run_alt_method.py
T=14400
SEEDS="0 1 2"
mkdir -p "$OUT"

wait_free() { while pgrep -f "run_clocked_ppo.py|run_alt_method.py|run_long_ppo|run_fragmented" | grep -v $$ > /dev/null; do sleep 60; done; }
mark() { echo "=== $1 exit=$2 report=$(test -f "$OUT/$1/final_report.json" && echo YES || echo NO) $(date +%F\ %T) ==="; }

echo "=== critic_features vs baseline, 3 seeds x ${T}s start $(date +%F\ %T) prog=$PROG ==="
wait_free

for S in $SEEDS; do
  B=baseline_s$S
  C=critic_features_s$S
  if [ ! -f "$OUT/$B/final_report.json" ]; then
    $PY -m policy_bias_lab.cli.long_ppo --out $OUT/$B --program $PROG --task lift \
      --arm baseline --seed $S --target-train-seconds $T --iters 100000 \
      --no-learn-prior-scale \
      --checkpoint-every 100 --eval-every 25 > $OUT/$B.log 2>&1
    mark $B $?
  fi
  if [ ! -f "$OUT/$C/final_report.json" ]; then
    $PY $ALT --method critic_features --program $PROG --out $OUT/$C --seed $S \
      --target-train-seconds $T --iters 100000 --eval-every 25 > $OUT/$C.log 2>&1
    mark $C $?
  fi
done

echo "=== ALL RUNS DONE $(date +%F\ %T) ==="

$PY - "$OUT" <<'EOF'
import json, sys
from pathlib import Path

out = Path(sys.argv[1])
rows = []
for d in sorted(out.iterdir()):
    rep = d / "final_report.json"
    if not rep.is_file():
        continue
    r = json.loads(rep.read_text())
    ev = r.get("eval", {})
    rows.append({
        "run": d.name,
        "arm": "critic_features" if d.name.startswith("critic") else "baseline",
        "graded": ev.get("eval_graded_objective"),
        "success": ev.get("eval_success_rate"),
        "fitness": ev.get("eval_task_fitness"),
        "lift_max": ev.get("eval_lift_max"),
        "best_iter": r.get("best_iter"),
        "iters": r.get("iters"),
    })

lines = ["# critic_features vs baseline (3 seeds, 4h each)", "",
         "| run | graded | success | fitness | lift_max | best_iter/iters |", "|---|---:|---:|---:|---:|---:|"]
for r in rows:
    lines.append(f"| {r['run']} | {r['graded']} | {r['success']} | {r['fitness']} | "
                 f"{r['lift_max']} | {r['best_iter']}/{r['iters']} |")
for arm in ("baseline", "critic_features"):
    vals = [r for r in rows if r["arm"] == arm and r["graded"] is not None]
    if vals:
        g = [r["graded"] for r in vals]
        s = [r["success"] for r in vals]
        lines += ["", f"**{arm}** (n={len(vals)}): graded mean {sum(g)/len(g):.3f} "
                      f"(min {min(g):.3f}, max {max(g):.3f}); success mean {sum(s)/len(s):.3f} "
                      f"(min {min(s):.3f}, max {max(s):.3f})"]
(out / "summary.md").write_text("\n".join(lines) + "\n")
print((out / "summary.md").read_text())
EOF

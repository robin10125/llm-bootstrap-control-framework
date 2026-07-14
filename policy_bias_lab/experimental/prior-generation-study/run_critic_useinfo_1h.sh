#!/bin/bash
# Does the prior GENERATOR benefit from knowing its program feeds critic features?
# Two priors, same base prompt + shared context, differing ONLY in the usage paragraph:
#   nouse -> runs/prior_gen_study_20260711/kl_nouse/program.json   (base prompt, no use info;
#            the same program used for critic_features in the 4h comparison)
#   use   -> runs/prior_gen_study_20260711/critic_use/program.json (critic_features use info)
# Both trained with the critic_features method, 1h per run, PPO seeds 0/1/2, interleaved.
set -u
cd /home/robin/Documents/agent-mini-script-control/llm-framework
OUT=runs/critic_useinfo_1h_20260713
PY=.venv/bin/python
ALT=policy_bias_lab/experimental/alternative-methods/run_alt_method.py
NOUSE=runs/prior_gen_study_20260711/kl_nouse/program.json
USE=runs/prior_gen_study_20260711/critic_use/program.json
T=3600
mkdir -p "$OUT"

wait_free() { while pgrep -f "run_clocked_ppo.py|run_alt_method.py|run_long_ppo|run_fragmented" | grep -v $$ > /dev/null; do sleep 60; done; }
mark() { echo "=== $1 exit=$2 report=$(test -f "$OUT/$1/final_report.json" && echo YES || echo NO) $(date +%F\ %T) ==="; }

echo "=== critic_features use-info study start $(date +%F\ %T) ==="
wait_free

for S in 0 1 2; do
  for ARM in nouse use; do
    PROG=$NOUSE; [ "$ARM" = use ] && PROG=$USE
    R=${ARM}_s$S
    if [ ! -f "$OUT/$R/final_report.json" ]; then
      $PY $ALT --method critic_features --program $PROG --out $OUT/$R --seed $S \
        --target-train-seconds $T --iters 100000 --eval-every 25 > $OUT/$R.log 2>&1
      mark $R $?
    fi
  done
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
    rows.append({"run": d.name, "arm": d.name.rsplit("_s", 1)[0],
                 "graded": ev.get("eval_graded_objective"),
                 "success": ev.get("eval_success_rate"),
                 "fitness": ev.get("eval_task_fitness"),
                 "best_iter": r.get("best_iter"), "iters": r.get("iters")})

lines = ["# critic_features prior: use-info vs no-use-info (1h x 3 seeds)", "",
         "| run | graded | success | fitness | best_iter/iters |", "|---|---:|---:|---:|---:|"]
for r in rows:
    lines.append(f"| {r['run']} | {r['graded']} | {r['success']} | {r['fitness']} | "
                 f"{r['best_iter']}/{r['iters']} |")
for arm in ("nouse", "use"):
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

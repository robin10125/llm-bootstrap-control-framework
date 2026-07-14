#!/bin/bash
# Critic-feature diversity, 4h version: 2 arms x 1 seed x 2h each.
# All arms: method=critic_features, seed 0. ALL programs are USE-INFO generated
# (the generation prompt tells the LLM its prior feeds the critic as measurement
# instruments) -- no kl_nouse-family programs anywhere.
#
#   union    -> program COMPLEXITY diversity: feature union of critic_use (8-stage, default
#               emphasis), critic_simple (few-stage), critic_complex (fine-grained);
#               default feature blocks per program, one cursor each.
#   parallel -> critic_use + PARALLEL GATES: all stages' clip(gate,0,1) fed cursor-free
#               (--critic-gate-values only; no critical-stage actions).
#
# Earlier kl_nouse-based run preserved in runs/critic_diversity_4h/ (superseded).
# Launch:
#   cd /home/robin/Documents/agent-mini-script-control/llm-framework
#   nohup bash policy_bias_lab/experimental/alternative-methods/run_critic_diversity_4h.sh \
#     > runs/critic_diversity_4h_useinfo/driver.log 2>&1 &
set -u
cd /home/robin/Documents/agent-mini-script-control/llm-framework
OUT=runs/critic_diversity_4h_useinfo
PY=.venv/bin/python
ALT=policy_bias_lab/experimental/alternative-methods/run_alt_method.py
P=runs/prior_gen_study_20260711
USE=$P/critic_use/program.json
SIMPLE=$P/critic_simple/program.json
COMPLEX=$P/critic_complex/program.json
T=7200
mkdir -p "$OUT"

for f in $USE $SIMPLE $COMPLEX; do
  [ -f "$f" ] || { echo "missing program: $f"; exit 1; }
done

wait_free() { while pgrep -f "run_clocked_ppo.py|run_alt_method.py|run_long_ppo|run_fragmented" | grep -v $$ > /dev/null; do sleep 60; done; }
mark() { echo "=== $1 exit=$2 report=$(test -f "$OUT/$1/final_report.json" && echo YES || echo NO) $(date +%F\ %T) ==="; }

run_arm() {  # run_arm <name> <extra args...>
  local R=$1; shift
  if [ ! -f "$OUT/$R/final_report.json" ]; then
    $PY $ALT --method critic_features --out $OUT/$R \
      --target-train-seconds $T --iters 100000 --eval-every 25 "$@" > $OUT/$R.log 2>&1
    mark $R $?
  fi
}

echo "=== critic-feature diversity 4h study (use-info programs) start $(date +%F\ %T) ==="
wait_free

run_arm union_s0 --seed 0 --program $USE $SIMPLE $COMPLEX
run_arm parallel_s0 --seed 0 --program $USE --critic-gate-values

echo "=== ALL RUNS DONE $(date +%F\ %T) ==="

$PY - "$OUT" <<'EOF'
import json, sys
from pathlib import Path

out = Path(sys.argv[1])
lines = ["# critic-feature diversity study, use-info programs (2 arms x 1 seed x 2h)", "",
         "union = critic_use+critic_simple+critic_complex feature union (complexity diversity)",
         "parallel = critic_use + parallel gate values (cursor-free)", "",
         "| run | graded | success | fitness | best_iter/iters |", "|---|---:|---:|---:|---:|"]
for d in sorted(out.iterdir()):
    rep = d / "final_report.json"
    if not rep.is_file():
        continue
    r = json.loads(rep.read_text())
    ev = r.get("eval", {})
    lines.append(f"| {d.name} | {ev.get('eval_graded_objective')} | {ev.get('eval_success_rate')} | "
                 f"{ev.get('eval_task_fitness')} | {r.get('best_iter')}/{r.get('iters')} |")
(out / "summary.md").write_text("\n".join(lines) + "\n")
print((out / "summary.md").read_text())
EOF

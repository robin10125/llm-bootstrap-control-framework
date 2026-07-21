#!/bin/bash
# Prior COMPOSITION x COMPLEXITY study: how does critic_features performance scale with what
# is composed into the critic's feature union? 4 arms x 1 seed x 3h, method=critic_features.
# All programs are USE-INFO generated. Nested design (each arm adds to the previous):
#
#   noise           -> CONTROL: N(0,1) features, width matched to the LARGEST arm (the same
#                      3 programs set the width via --critic-noise; zero information).
#   simple          -> one simple prior (critic_simple_dual candidate 1).
#   simple_complex  -> simple + complex prior (critic_complex, 9-stage).
#   dual_complex    -> 2 DIVERSE simple priors (dual-generation prompt) + complex prior.
#
# 12h sequential. Launch:
#   cd /home/robin/Documents/agent-mini-script-control/llm-framework
#   nohup bash policy_bias_lab/experimental/alternative-methods/run_critic_composition_12h.sh \
#     > runs/critic_composition_12h/driver.log 2>&1 &
set -u
cd /home/robin/Documents/agent-mini-script-control/llm-framework
OUT=runs/critic_composition_12h
PY=.venv/bin/python
ALT=policy_bias_lab/experimental/alternative-methods/run_alt_method.py
P=runs/prior_gen_study_20260711
SIMPLE_A=$P/critic_simple_dual/program.json
SIMPLE_B=$P/critic_simple_dual/program_2.json
COMPLEX=$P/critic_complex/program.json
T=10800
mkdir -p "$OUT"

for f in $SIMPLE_A $SIMPLE_B $COMPLEX; do
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

echo "=== prior composition/complexity study start $(date +%F\ %T) ==="
wait_free

run_arm noise_s0 --seed 0 --program $SIMPLE_A $SIMPLE_B $COMPLEX --critic-noise
run_arm simple_s0 --seed 0 --program $SIMPLE_A
run_arm simple_complex_s0 --seed 0 --program $SIMPLE_A $COMPLEX
run_arm dual_complex_s0 --seed 0 --program $SIMPLE_A $SIMPLE_B $COMPLEX

echo "=== ALL RUNS DONE $(date +%F\ %T) ==="

$PY - "$OUT" <<'EOF'
import json, sys
from pathlib import Path

out = Path(sys.argv[1])
lines = ["# prior composition x complexity study (4 arms x 1 seed x 3h)", "",
         "noise = N(0,1) control (width of largest arm); simple = 1 simple prior;",
         "simple_complex = simple+complex union; dual_complex = 2 diverse simple + complex", "",
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

#!/bin/bash
# Phase 2: simple vs complex generation, winning scheme from phase 1 (kl_prior, NO use-info).
set -u
cd /home/robin/Documents/agent-mini-script-control/llm-framework
STUDY=runs/prior_gen_study_20260711
GEN=policy_bias_lab/experimental/prior-generation-study/generate_candidate.py
PY=.venv/bin/python

echo "=== phase2 generations start $(date +%F\ %T) ==="
$PY $GEN --study-dir $STUDY --arm kl_simple  --framework kl_prior --complexity simple \
  && echo OK_kl_simple || echo FAIL_kl_simple
$PY $GEN --study-dir $STUDY --arm kl_complex --framework kl_prior --complexity complex \
  && echo OK_kl_complex || echo FAIL_kl_complex

while pgrep -f "run_clocked_ppo.py|run_alt_method.py|run_long_ppo" > /dev/null; do sleep 60; done

run_kl() {
  $PY policy_bias_lab/experimental/alternative-methods/run_alt_method.py \
    --method kl_prior --program "$STUDY/$1/program.json" --out "$STUDY/run_$1" \
    --target-train-seconds 5400 --iters 100000 --kl-anneal-iters 350 --eval-every 25 \
    > "$STUDY/run_$1.log" 2>&1
  echo "=== run_$1 exit=$? report=$(test -f "$STUDY/run_$1/final_report.json" && echo YES || echo NO) $(date +%T) ==="
}
echo "=== phase2 PPO runs start $(date +%F\ %T) ==="
[ -f "$STUDY/kl_simple/program.json" ]  && run_kl kl_simple
[ -f "$STUDY/kl_complex/program.json" ] && run_kl kl_complex
echo "=== PHASE2 DONE $(date +%F\ %T) ==="

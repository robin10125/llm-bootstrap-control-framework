#!/bin/bash
# Phase 1 of the prior-generation study: use-info experiment.
# 4 arms: {kl_prior, clocked} x {use-info, no-use-info}; 1 seed, 0 revisions;
# one shared context response; 90-min PPO run per arm, sequential (RAM limit).
set -u
cd /home/robin/Documents/agent-mini-script-control/llm-framework
STUDY=runs/prior_gen_study_20260711
GEN=policy_bias_lab/experimental/prior-generation-study/generate_candidate.py
PY=.venv/bin/python
mkdir -p "$STUDY"

echo "=== phase1 generations start $(date +%F\ %T) ==="
$PY $GEN --study-dir $STUDY --arm kl_use        --framework kl_prior --use-info \
  && echo OK_kl_use || echo FAIL_kl_use
$PY $GEN --study-dir $STUDY --arm kl_nouse      --framework kl_prior \
  && echo OK_kl_nouse || echo FAIL_kl_nouse
$PY $GEN --study-dir $STUDY --arm clocked_use   --framework clocked --use-info \
  && echo OK_clocked_use || echo FAIL_clocked_use
$PY $GEN --study-dir $STUDY --arm clocked_nouse --framework clocked \
  && echo OK_clocked_nouse || echo FAIL_clocked_nouse

# wait for any lingering jax job (e.g. smoke tests) before heavy runs
while pgrep -f "run_clocked_ppo.py|run_alt_method.py|run_long_ppo" > /dev/null; do sleep 60; done

run_kl() {
  $PY policy_bias_lab/experimental/alternative-methods/run_alt_method.py \
    --method kl_prior --program "$STUDY/$1/program.json" --out "$STUDY/run_$1" \
    --target-train-seconds 5400 --iters 100000 --kl-anneal-iters 350 --eval-every 25 \
    > "$STUDY/run_$1.log" 2>&1
  echo "=== run_$1 exit=$? report=$(test -f "$STUDY/run_$1/final_report.json" && echo YES || echo NO) $(date +%T) ==="
}
run_clocked() {
  $PY policy_bias_lab/experimental/policy-clocked-paths/run_clocked_ppo.py \
    --program "$STUDY/$1/program.json" --out "$STUDY/run_$1" --progression learned \
    --target-train-seconds 5400 --iters 100000 --imitation-anneal-iters 350 --eval-every 25 \
    > "$STUDY/run_$1.log" 2>&1
  echo "=== run_$1 exit=$? report=$(test -f "$STUDY/run_$1/final_report.json" && echo YES || echo NO) $(date +%T) ==="
}

echo "=== phase1 PPO runs start $(date +%F\ %T) ==="
[ -f "$STUDY/kl_use/program.json" ]        && run_kl kl_use
[ -f "$STUDY/kl_nouse/program.json" ]      && run_kl kl_nouse
[ -f "$STUDY/clocked_use/program.json" ]   && run_clocked clocked_use
[ -f "$STUDY/clocked_nouse/program.json" ] && run_clocked clocked_nouse
echo "=== PHASE1 DONE $(date +%F\ %T) ==="

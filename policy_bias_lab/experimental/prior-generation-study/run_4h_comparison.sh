#!/bin/bash
# 4-hour-budget comparison of all experimental frameworks, seed-matched (seed 0),
# one shared prior program (best arm of the generation study: kl_nouse, graded 0.878).
set -u
cd /home/robin/Documents/agent-mini-script-control/llm-framework
CMP=runs/framework_cmp_4h_20260711
PROG=runs/prior_gen_study_20260711/kl_nouse/program.json
PY=.venv/bin/python
ALT=policy_bias_lab/experimental/alternative-methods/run_alt_method.py
CLK=policy_bias_lab/experimental/policy-clocked-paths/run_clocked_ppo.py
BASE=policy_bias_lab/experimental/run_fragmented_stage_ppo.py
T=14400
mkdir -p "$CMP"

wait_free() { while pgrep -f "run_clocked_ppo.py|run_alt_method.py|run_long_ppo|run_fragmented" | grep -v $$ > /dev/null; do sleep 60; done; }
mark() { echo "=== $1 exit=$2 report=$(test -f "$CMP/$1/final_report.json" && echo YES || echo NO) $(date +%F\ %T) ==="; }

echo "=== 4h comparison start $(date +%F\ %T) prog=$PROG ==="
wait_free
$PY $BASE --out $CMP/baseline --program $PROG --target-train-seconds $T --iters 100000 --eval-every 25 > $CMP/baseline.log 2>&1; mark baseline $?
$PY $ALT --method proposal        --program $PROG --out $CMP/proposal   --target-train-seconds $T --iters 100000 --eval-every 25 --proposal-anneal-iters 800 > $CMP/proposal.log 2>&1; mark proposal $?
$PY $ALT --method curriculum      --program $PROG --out $CMP/curriculum --target-train-seconds $T --iters 100000 --eval-every 25 --warmup-anneal-iters 800 > $CMP/curriculum.log 2>&1; mark curriculum $?
$PY $ALT --method value_shaping   --program $PROG --out $CMP/value_shaping --target-train-seconds $T --iters 100000 --eval-every 25 --aux-coef 0.1 > $CMP/value_shaping.log 2>&1; mark value_shaping $?
$PY $ALT --method critic_features --program $PROG --out $CMP/critic_features --target-train-seconds $T --iters 100000 --eval-every 25 > $CMP/critic_features.log 2>&1; mark critic_features $?
$PY $ALT --method kl_prior        --program $PROG --out $CMP/kl_prior   --target-train-seconds $T --iters 100000 --eval-every 25 --kl-anneal-iters 250 > $CMP/kl_prior.log 2>&1; mark kl_prior $?
$PY $CLK --program $PROG --out $CMP/clocked_learned   --progression learned   --target-train-seconds $T --iters 100000 --eval-every 25 --imitation-anneal-iters 250 > $CMP/clocked_learned.log 2>&1; mark clocked_learned $?
$PY $CLK --program $PROG --out $CMP/clocked_autopilot --progression autopilot --target-train-seconds $T --iters 100000 --eval-every 25 > $CMP/clocked_autopilot.log 2>&1; mark clocked_autopilot $?
$PY $ALT --method value_shaping   --program $PROG --out $CMP/no_prior_control --target-train-seconds $T --iters 100000 --eval-every 25 --stage-reward-weight 0 --potential-weight 0 > $CMP/no_prior_control.log 2>&1; mark no_prior_control $?
echo "=== 4H COMPARISON DONE $(date +%F\ %T) ==="

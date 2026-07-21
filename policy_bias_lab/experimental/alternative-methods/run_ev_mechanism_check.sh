#!/bin/bash
# Mechanism-validation pilot for critic_features (see policy_bias_lab/docs/
# critic_features_and_motor_tape_findings.md §2.5): does the union program's win come from
# feature CONTENT, or just added dimensionality? Same program set (critic_use+simple+complex,
# the diversity study's union arm) run twice at IDENTICAL feature width: real content vs.
# --critic-noise (N(0,1), same width, zero information). New this run: ev_pre/ev_post
# (explained variance of the value fn against GAE-bootstrap returns) logged in alt_methods_ppo.py.
# 1h GPU budget pilot -- short (~22min/arm), single seed. Not a substitute for the full Tier-2
# ablation (>=3 seeds) scoped in §5.E; this is the cheap first look.
# Launch:
#   cd /home/robin/Documents/agent-mini-script-control/llm-framework
#   T=1320 nohup bash policy_bias_lab/experimental/alternative-methods/run_ev_mechanism_check.sh \
#     > runs/critic_ev_mechanism_check_20260721/driver.log 2>&1 &
set -u
cd /home/robin/Documents/agent-mini-script-control/llm-framework
OUT=${OUT:-runs/critic_ev_mechanism_check_20260721}
PY=.venv/bin/python
ALT=policy_bias_lab/experimental/alternative-methods/run_alt_method.py
P=runs/prior_gen_study_20260711
USE=$P/critic_use/program.json
SIMPLE=$P/critic_simple/program.json
COMPLEX=$P/critic_complex/program.json
T=${T:-1320}
mkdir -p "$OUT"

for f in $USE $SIMPLE $COMPLEX; do
  [ -f "$f" ] || { echo "missing program: $f"; exit 1; }
done

wait_free() { while pgrep -f "run_clocked_ppo.py|run_alt_method.py|run_long_ppo|run_fragmented|run_motor_tape.py" | grep -v $$ > /dev/null; do sleep 20; done; }
mark() { echo "=== $1 exit=$2 report=$(test -f "$OUT/$1/final_report.json" && echo YES || echo NO) $(date +%F\ %T) ==="; }

run_arm() {  # run_arm <name> <extra args...>
  local R=$1; shift
  if [ ! -f "$OUT/$R/final_report.json" ]; then
    echo "=== $R start $(date +%F\ %T) ==="
    $PY $ALT --method critic_features --out $OUT/$R \
      --target-train-seconds $T --iters 100000 --eval-every 25 "$@" > $OUT/$R.log 2>&1
    mark $R $?
  fi
}

echo "=== ev-mechanism check start $(date +%F\ %T) T=$T ==="
wait_free

run_arm real_s0 --seed 0 --program $USE $SIMPLE $COMPLEX
run_arm noise_s0 --seed 0 --program $USE $SIMPLE $COMPLEX --critic-noise

echo "=== ALL RUNS DONE $(date +%F\ %T) ==="

$PY - "$OUT" <<'EOF'
import json, sys
from pathlib import Path

out = Path(sys.argv[1])
lines = ["# EV-mechanism pilot: real content vs. matched-width noise (same union program set)", "",
         "1 seed/arm, ~22min/arm -- pilot only, not the full Tier-2 ablation (see doc §2.5/§5.E).", "",
         "| run | iters | ev_pre early/mid/late | ev_post early/mid/late | final v_loss | "
         "eval_graded(best) | eval_success(best) |",
         "|---|---:|---|---|---:|---:|---:|"]
for name in ("real_s0", "noise_s0"):
    d = out / name
    mfile = d / "metrics.jsonl"
    if not mfile.is_file():
        lines.append(f"| {name} | MISSING |  |  |  |  |  |")
        continue
    rows = [json.loads(ln) for ln in mfile.open()]
    n = len(rows)
    thirds = [rows[:n // 3] or rows[:1], rows[n // 3: 2 * n // 3] or rows[:1], rows[2 * n // 3:] or rows[-1:]]
    def avg(key, chunk):
        vals = [r[key] for r in chunk if key in r]
        return round(sum(vals) / len(vals), 4) if vals else None
    ev_pre_s = "/".join(str(avg("ev_pre", c)) for c in thirds)
    ev_post_s = "/".join(str(avg("ev_post", c)) for c in thirds)
    rep = d / "final_report.json"
    graded = success = None
    if rep.is_file():
        r = json.loads(rep.read_text())
        ev = r.get("eval", {})
        graded, success = ev.get("eval_graded_objective"), ev.get("eval_success_rate")
    lines.append(f"| {name} | {n} | {ev_pre_s} | {ev_post_s} | {round(rows[-1]['v_loss'], 5)} | "
                 f"{graded} | {success} |")
(out / "summary.md").write_text("\n".join(lines) + "\n")
print((out / "summary.md").read_text())
EOF

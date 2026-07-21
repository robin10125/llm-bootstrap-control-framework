#!/bin/bash
# Phase B of the generation-condition study: 1h PPO x 2 seeds on each arm's best candidate
# (default motor-tape arm: tape + residual + rate). The exact arm trains/evals at
# obj_xy_range=0 (its plan is position-specific by design).
# Usage:
#   STUDY=runs/motor_tape_genstudy_20260716 nohup bash <this script> \
#     > runs/motor_tape_genstudy_20260716/ppo_driver.log 2>&1 &
set -u
cd "$(dirname "$(readlink -f "$0")")/../../.."
STUDY=${STUDY:?set STUDY=<phase-A study dir>}
PY=.venv/bin/python
RUN=policy_bias_lab/experimental/motor-tape/run_motor_tape.py
T=${T:-3600}
SEEDS=${SEEDS:-"0 1"}
ARMS=${ARMS:-"base exact samples fb orient orient_fb"}
OUT="$STUDY/ppo"
mkdir -p "$OUT"

wait_free() { while pgrep -f "run_clocked_ppo.py|run_alt_method.py|run_long_ppo|run_fragmented|run_motor_tape.py" | grep -v $$ > /dev/null; do sleep 60; done; }
mark() { echo "=== $1 exit=$2 report=$(test -f "$OUT/$1/final_report.json" && echo YES || echo NO) $(date +%F\ %T) ==="; }

echo "=== gen-condition PPO start $(date +%F\ %T) study=$STUDY ==="
wait_free

for S in $SEEDS; do
  for ARM in $ARMS; do
    R=${ARM}_s$S
    PROG="$STUDY/$ARM/best_program.json"
    [ -f "$PROG" ] || { echo "=== $R SKIP (no $PROG) ==="; continue; }
    [ -f "$OUT/$R/final_report.json" ] && continue
    EXTRA=""
    [ "$ARM" = exact ] && EXTRA="--obj-xy-range 0"
    $PY $RUN --program "$PROG" --out "$OUT/$R" --seed "$S" \
      --target-train-seconds "$T" --iters 100000 --eval-every 25 $EXTRA \
      > "$OUT/$R.log" 2>&1
    mark "$R" $?
  done
done

echo "=== ALL RUNS DONE $(date +%F\ %T) ==="

$PY - "$STUDY" <<'EOF'
import json, sys
from pathlib import Path

study = Path(sys.argv[1])
out = study / "ppo"
rows = []
for d in sorted(out.iterdir()):
    rep = d / "final_report.json"
    if not rep.is_file():
        continue
    r = json.loads(rep.read_text())
    ev = r.get("eval", {})
    ap = (r.get("tape_autopilot") or {}).get("tape_eval", {})
    rows.append({"run": d.name, "arm": d.name.rsplit("_s", 1)[0],
                 "graded": ev.get("eval_graded_objective"),
                 "success": ev.get("eval_success_rate"),
                 "floor": ap.get("eval_graded_objective"),
                 "best_iter": r.get("best_iter"), "iters": r.get("iters")})

lines = ["# generation-condition study: trained results (1h x 2 seeds, best candidate per arm)",
         "", "| run | graded | success | autopilot floor | best_iter/iters |",
         "|---|---:|---:|---:|---:|"]
for r in rows:
    lines.append(f"| {r['run']} | {r['graded']} | {r['success']} | {r['floor']} | "
                 f"{r['best_iter']}/{r['iters']} |")
lines.append("")
for arm in ("base", "exact", "samples", "fb", "orient", "orient_fb"):
    vals = [r for r in rows if r["arm"] == arm and r["graded"] is not None]
    if vals:
        g = [r["graded"] for r in vals]
        s = [r["success"] for r in vals]
        lines.append(f"**{arm}** (n={len(vals)}): graded mean {sum(g)/len(g):.3f}, "
                     f"success mean {sum(s)/len(s):.3f}, floor {vals[0]['floor']}")
brittle = study / "exact" / "autopilot_brittle.json"
if brittle.is_file():
    bt = json.loads(brittle.read_text())["tape_eval"]
    lines.append(f"\n**exact-arm brittleness** (its plan under +-4cm spawns, autopilot): "
                 f"graded {bt['eval_graded_objective']}, success {bt['eval_success_rate']} "
                 f"-- NOTE the exact arm's trained rows are on the easier fixed-spawn task.")
lines.append("\nContrasts: position info = exact/samples vs base; feedback = fb vs base; "
             "redundancy = orient_fb vs fb and vs orient.")
(study / "summary.md").write_text("\n".join(lines) + "\n")
print((study / "summary.md").read_text())
EOF

#!/bin/bash
# Motor-tape ablation ladder, 1h per trained run x 3 seeds, interleaved by seed:
#   tape_only        -- autopilot floor (no training; one run, seed 0)
#   residual_sS      -- tape + efference-copy residual        (--no-rate)
#   rate_sS          -- tape + residual + time-warp           (default)
#   mod_sS           -- + continuous plan modulation          (--modulation on)
# Usage:
#   mkdir -p <repo>/runs/motor_tape_1h
#   PROG=<program.json> nohup bash <this script> > <repo>/runs/motor_tape_1h/driver.log 2>&1 &
set -u
cd "$(dirname "$(readlink -f "$0")")/../../.."
OUT=${OUT:-runs/motor_tape_1h}
PROG=${PROG:?set PROG=<motor_tape program.json>}
PY=.venv/bin/python
RUN=policy_bias_lab/experimental/motor-tape/run_motor_tape.py
T=${T:-3600}
SEEDS=${SEEDS:-"0 1 2"}
mkdir -p "$OUT"

wait_free() { while pgrep -f "run_clocked_ppo.py|run_alt_method.py|run_long_ppo|run_fragmented|run_motor_tape.py" | grep -v $$ > /dev/null; do sleep 60; done; }
mark() { echo "=== $1 exit=$2 report=$(test -f "$OUT/$1/final_report.json" && echo YES || echo NO) $(date +%F\ %T) ==="; }

echo "=== motor-tape ablation ladder start $(date +%F\ %T) prog=$PROG ==="
wait_free

if [ ! -f "$OUT/tape_only/final_report.json" ]; then
  $PY $RUN --program "$PROG" --out "$OUT/tape_only" --tape-only --eval-envs 128 \
    > "$OUT/tape_only.log" 2>&1
  mark tape_only $?
fi

for S in $SEEDS; do
  for ARM in residual rate mod; do
    R=${ARM}_s$S
    [ -f "$OUT/$R/final_report.json" ] && continue
    EXTRA=""
    [ "$ARM" = residual ] && EXTRA="--no-rate"
    [ "$ARM" = mod ] && EXTRA="--modulation on"
    $PY $RUN --program "$PROG" --out "$OUT/$R" --seed "$S" \
      --target-train-seconds "$T" --iters 100000 --eval-every 25 $EXTRA \
      > "$OUT/$R.log" 2>&1
    mark "$R" $?
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
    ap = (r.get("tape_autopilot") or {}).get("tape_eval", {})
    rows.append({"run": d.name, "arm": d.name.rsplit("_s", 1)[0],
                 "graded": ev.get("eval_graded_objective", r.get("eval_graded_objective")),
                 "success": ev.get("eval_success_rate", r.get("eval_success_rate")),
                 "autopilot_graded": ap.get("eval_graded_objective"),
                 "best_iter": r.get("best_iter"), "iters": r.get("iters")})

lines = ["# motor-tape ablation ladder (1h x 3 seeds)", "",
         "| run | graded | success | autopilot graded | best_iter/iters |",
         "|---|---:|---:|---:|---:|"]
for r in rows:
    lines.append(f"| {r['run']} | {r['graded']} | {r['success']} | {r['autopilot_graded']} | "
                 f"{r['best_iter']}/{r['iters']} |")
for arm in ("residual", "rate", "mod"):
    vals = [r for r in rows if r["arm"] == arm and r["graded"] is not None]
    if vals:
        g = [r["graded"] for r in vals]
        s = [r["success"] for r in vals]
        lines += ["", f"**{arm}** (n={len(vals)}): graded mean {sum(g)/len(g):.3f} "
                      f"(min {min(g):.3f}, max {max(g):.3f}); success mean {sum(s)/len(s):.3f}"]
(out / "summary.md").write_text("\n".join(lines) + "\n")
print((out / "summary.md").read_text())
EOF

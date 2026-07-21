#!/bin/bash
# Revision-depth study: does iterative revision monotonically harm trainability, and is the
# selector part of the harm?
#   Phase A: 3 chains x 4 revision rounds (standard graded revise template), keep_rule=always
#            so chain dynamics are UNCONFOUNDED by selection; every round's program +
#            autopilot saved. Per-round open-loop dose-response comes free (3 chains).
#   Phase B: train chain 1's r0..r4, 1h x 2 seeds each -> trained-performance vs revision depth.
#            The selector question is answered analytically afterwards: map each keep-rule's
#            per-round pick onto the measured trained outcomes.
#   Phase C ("after"): longer ho run -- fb tape + handoff at 3h, plus matched fb-tape 3h
#            baseline (no 3h fb-tape run exists), seed 0 each.
# Usage:
#   STUDY=runs/motor_tape_revdepth_20260718 nohup bash <this script> \
#     > runs/motor_tape_revdepth_20260718/driver.log 2>&1 &
set -u
cd "$(dirname "$(readlink -f "$0")")/../../.."
STUDY=${STUDY:-runs/motor_tape_revdepth_20260718}
BASE=${BASE:-runs/motor_tape_genstudy_20260716}
CTX=${CTX:-runs/prior_gen_study_20260711/context}
PY=.venv/bin/python
DIR=policy_bias_lab/experimental/motor-tape
T=${T:-3600}
T_LONG=${T_LONG:-10800}
SEEDS=${SEEDS:-"0 1"}
DEPTHS=${DEPTHS:-"0 1 2 3 4"}
CHAINS=${CHAINS:-"1 2 3"}
FBPROG="$BASE/fb/best_program.json"
mkdir -p "$STUDY/ppo"

wait_free() { while pgrep -f "run_clocked_ppo.py|run_alt_method.py|run_long_ppo|run_fragmented|run_motor_tape.py" | grep -v $$ > /dev/null; do sleep 60; done; }
mark() { echo "=== $1 exit=$2 report=$(test -f "$STUDY/ppo/$1/final_report.json" && echo YES || echo NO) $(date +%F\ %T) ==="; }

echo "=== revision-depth start $(date +%F\ %T) study=$STUDY ==="
wait_free

# ---- Phase A: chains ---------------------------------------------------------------------
for C in $CHAINS; do
  [ -f "$STUDY/chain$C/meta.json" ] && continue
  mkdir -p "$STUDY/chain$C"
  echo "=== chain$C gen $(date +%F\ %T) ==="
  $PY $DIR/generate_motor_tape.py --out "$STUDY/chain$C" --context-dir "$CTX" \
    --feedback --feedback-rounds 4 --keep-rule always \
    > "$STUDY/chain$C.log" 2>&1 || echo "=== chain$C FAILED (see log) ==="
done
$PY - "$STUDY" <<'EOF'
import json, sys
from pathlib import Path
study = Path(sys.argv[1])
lines = ["# revision chains: per-round open-loop dose-response", "",
         "| chain | round | graded | min_dist | engagement | nearmiss |", "|---|---|---:|---:|---:|---:|"]
for c in sorted(study.glob("chain*/meta.json")):
    m = json.loads(c.read_text())
    for i, r in enumerate(m["rounds"]):
        lines.append(f"| {c.parent.name} | r{i} | {r.get('graded')} | {r.get('min_dist')} | "
                     f"{r.get('engagement')} | {r.get('nearmiss')} |")
(study / "screen.md").write_text("\n".join(lines) + "\n")
print((study / "screen.md").read_text())
EOF

# pick the training chain: the first one with all 5 rounds present, else the deepest
TRAIN_CHAIN=$($PY - "$STUDY" <<'EOF'
import json, sys
from pathlib import Path
study = Path(sys.argv[1])
best, depth = None, -1
for c in sorted(study.glob("chain*/meta.json")):
    n = len(json.loads(c.read_text())["rounds"])
    if n == 5:
        print(c.parent.name); sys.exit()
    if n > depth:
        best, depth = c.parent.name, n
print(best or "")
EOF
)
echo "=== training chain: $TRAIN_CHAIN ==="
[ -n "$TRAIN_CHAIN" ] || { echo "no usable chain"; exit 1; }

# ---- Phase B: train per depth ------------------------------------------------------------
for K in $DEPTHS; do
  for S in $SEEDS; do
    R=d${K}_s$S
    PROG="$STUDY/$TRAIN_CHAIN/program_r$K.json"
    [ -f "$PROG" ] || { echo "=== $R SKIP (no $PROG) ==="; continue; }
    [ -f "$STUDY/ppo/$R/final_report.json" ] && continue
    wait_free
    $PY $DIR/run_motor_tape.py --program "$PROG" --out "$STUDY/ppo/$R" --seed "$S" \
      --target-train-seconds "$T" --iters 100000 --eval-every 25 \
      > "$STUDY/ppo/$R.log" 2>&1
    mark "$R" $?
  done
done

# ---- Phase C: longer ho + matched fb baseline (3h, seed 0) -------------------------------
for RUN in ho3h_s0 fb3h_s0; do
  [ -f "$STUDY/ppo/$RUN/final_report.json" ] && continue
  EXTRA=""
  [ "$RUN" = ho3h_s0 ] && EXTRA="--handoff-lo 0.05 --handoff-hi 0.08"
  wait_free
  $PY $DIR/run_motor_tape.py --program "$FBPROG" --out "$STUDY/ppo/$RUN" --seed 0 \
    --target-train-seconds "$T_LONG" --iters 100000 --eval-every 25 $EXTRA \
    > "$STUDY/ppo/$RUN.log" 2>&1
  mark "$RUN" $?
done

echo "=== ALL RUNS DONE $(date +%F\ %T) ==="

# ---- summary -----------------------------------------------------------------------------
$PY - "$STUDY" "$TRAIN_CHAIN" <<'EOF'
import json, sys
from pathlib import Path
study, chain = Path(sys.argv[1]), sys.argv[2]
lines = ["# revision-depth: trained results (chain %s, 1h x 2 seeds per depth)" % chain, "",
         "| run | graded(final report) | late3 graded | success | grasp | floor |",
         "|---|---:|---:|---:|---:|---:|"]
for d in sorted((study / "ppo").iterdir()):
    rep = d / "final_report.json"
    if not rep.is_file():
        continue
    r = json.loads(rep.read_text())
    ev = r.get("eval", {})
    late = []
    mfile = d / "metrics.jsonl"
    if mfile.is_file():
        evals = [json.loads(ln) for ln in mfile.open()]
        late = [e["eval_graded"] for e in evals if e.get("eval_graded") is not None][-3:]
    late3 = round(sum(late) / len(late), 3) if late else None
    lines.append(f"| {d.name} | {ev.get('eval_graded_objective')} | {late3} | "
                 f"{ev.get('eval_success_rate')} | {ev.get('eval_grasp_rate')} | "
                 f"{((r.get('tape_autopilot') or {}).get('tape_eval') or {}).get('eval_graded_objective')} |")
lines.append("\nSelector analysis: graded-selector pick = argmax r-graded in screen.md; compare "
             "its depth's trained row against the best trained depth.")
(study / "summary.md").write_text("\n".join(lines) + "\n")
print((study / "summary.md").read_text())
EOF

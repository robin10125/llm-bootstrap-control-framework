#!/bin/bash
# Contact-harm mitigation experiments (three tests vs the gen-condition study's fb baseline):
#   gg -- test 1: grasp-gated lift reward, trained ON the known-hacking fb best_program
#         (its trained runs hit grasp_rate = 0.000 -- the graze-and-scoop exploit)
#   nm -- test 2: near-miss retargeted feedback (revise toward aligned near-miss, keep by
#         proximity-without-contact), fresh codex generations, then standard training
#   ho -- test 3: feedforward handoff (a_ff fades to 0 inside 8->5 cm of the object; plan
#         features stay visible to actor+critic), trained ON the fb best_program
# Baselines (runs/motor_tape_genstudy_20260716/ppo): fb_s0/s1 graded 0.38/0.35 grasp 0.000;
# samples_s0/s1 graded 1.06/1.02 grasp 1.000.
# Usage:
#   STUDY=runs/motor_tape_mitigation_20260717 nohup bash <this script> \
#     > runs/motor_tape_mitigation_20260717/driver.log 2>&1 &
# Phase A (nm generation, codex) runs first and only if nm/best_program.json is absent.
set -u
cd "$(dirname "$(readlink -f "$0")")/../../.."
STUDY=${STUDY:-runs/motor_tape_mitigation_20260717}
BASE=${BASE:-runs/motor_tape_genstudy_20260716}
CTX=${CTX:-runs/prior_gen_study_20260711/context}
PY=.venv/bin/python
DIR=policy_bias_lab/experimental/motor-tape
T=${T:-3600}
SEEDS=${SEEDS:-"0 1"}
TESTS=${TESTS:-"gg nm ho"}
FBPROG="$BASE/fb/best_program.json"
mkdir -p "$STUDY/ppo" "$STUDY/nm"

wait_free() { while pgrep -f "run_clocked_ppo.py|run_alt_method.py|run_long_ppo|run_fragmented|run_motor_tape.py" | grep -v $$ > /dev/null; do sleep 60; done; }
mark() { echo "=== $1 exit=$2 report=$(test -f "$STUDY/ppo/$1/final_report.json" && echo YES || echo NO) $(date +%F\ %T) ==="; }

echo "=== contact-mitigation start $(date +%F\ %T) study=$STUDY ==="
wait_free

# ---- Phase A: near-miss generations (3 gens, best by proximity-without-contact) ----------
if [[ " $TESTS " == *" nm "* ]] && [ ! -f "$STUDY/nm/best_program.json" ]; then
  for G in 1 2 3; do
    [ -f "$STUDY/nm/g$G/program.json" ] && continue
    echo "=== nm gen g$G $(date +%F\ %T) ==="
    $PY $DIR/generate_motor_tape.py --out "$STUDY/nm/g$G" --context-dir "$CTX" \
      --feedback --feedback-target nearmiss \
      > "$STUDY/nm/g$G.log" 2>&1 || echo "=== nm g$G FAILED (see log) ==="
  done
  $PY - "$STUDY" <<'EOF'
import json, shutil, sys
from pathlib import Path
nm = Path(sys.argv[1]) / "nm"
best, best_s = None, -1.0
rows = []
for g in sorted(nm.glob("g*/meta.json")):
    m = json.loads(g.read_text())
    rows.append((g.parent.name, m.get("nearmiss"), m.get("autopilot_graded"),
                 m.get("palm_obj_dist_min"), m.get("contact_engagement"), m.get("kept_round")))
    if m.get("nearmiss", -1) > best_s:
        best_s, best = m["nearmiss"], g.parent
print("gen | nearmiss | graded | min_dist | engagement | kept_round")
for r in rows:
    print(" | ".join(str(x) for x in r))
if best is None:
    sys.exit("no nm generation succeeded")
shutil.copy(best / "program.json", nm / "best_program.json")
(nm / "best_meta.json").write_text((best / "meta.json").read_text())
print(f"best: {best.name} nearmiss={best_s}")
EOF
fi

# ---- Phase B: 1h PPO x seeds, sequential ------------------------------------------------
for TEST in $TESTS; do
  for S in $SEEDS; do
    R=${TEST}_s$S
    [ -f "$STUDY/ppo/$R/final_report.json" ] && continue
    case $TEST in
      gg) PROG=$FBPROG; EXTRA="--reward-mode grasp_gated" ;;
      ho) PROG=$FBPROG; EXTRA="--handoff-lo 0.05 --handoff-hi 0.08" ;;
      nm) PROG="$STUDY/nm/best_program.json"; EXTRA="" ;;
      *) echo "unknown test $TEST"; continue ;;
    esac
    [ -f "$PROG" ] || { echo "=== $R SKIP (no $PROG) ==="; continue; }
    wait_free
    $PY $DIR/run_motor_tape.py --program "$PROG" --out "$STUDY/ppo/$R" --seed "$S" \
      --target-train-seconds "$T" --iters 100000 --eval-every 25 $EXTRA \
      > "$STUDY/ppo/$R.log" 2>&1
    mark "$R" $?
  done
done

echo "=== ALL RUNS DONE $(date +%F\ %T) ==="

# ---- summary ----------------------------------------------------------------------------
$PY - "$STUDY" "$BASE" <<'EOF'
import json, sys
from pathlib import Path
study, base = Path(sys.argv[1]), Path(sys.argv[2])

def rows_from(ppo):
    out = []
    for d in sorted(ppo.iterdir()) if ppo.is_dir() else []:
        rep = d / "final_report.json"
        if not rep.is_file():
            continue
        r = json.loads(rep.read_text())
        ev = r.get("eval", {})
        out.append({"run": d.name,
                    "graded": ev.get("eval_graded_objective"),
                    "success": ev.get("eval_success_rate"),
                    "grasp": ev.get("eval_grasp_rate"),
                    "lift_reached": ev.get("eval_lift_reached_rate"),
                    "floor": ((r.get("tape_autopilot") or {}).get("tape_eval") or {})
                             .get("eval_graded_objective"),
                    "best_iter": r.get("best_iter"), "iters": r.get("iters")})
    return out

lines = ["# contact-harm mitigation: trained results (1h x 2 seeds)", "",
         "| run | graded | success | grasp | lift_reached | floor | best_iter/iters |",
         "|---|---:|---:|---:|---:|---:|---:|"]
for r in rows_from(study / "ppo"):
    lines.append(f"| {r['run']} | {r['graded']} | {r['success']} | {r['grasp']} | "
                 f"{r['lift_reached']} | {r['floor']} | {r['best_iter']}/{r['iters']} |")
lines += ["", "Baselines (gen-condition study, same fb tape / default reward / no handoff):", ""]
for r in rows_from(base / "ppo"):
    if r["run"].rsplit("_s", 1)[0] in ("fb", "samples"):
        lines.append(f"- {r['run']}: graded {r['graded']}, success {r['success']}, "
                     f"grasp {r['grasp']}")
lines += ["", "Signals: gg/ho vs fb baseline on the SAME tape (does the mitigation restore "
          "grasp formation? grasp_rate is the diagnostic); nm vs fb (does near-miss revision "
          "produce a samples-class scaffold?) and nm floor/engagement vs fb floor 0.243."]
(study / "summary.md").write_text("\n".join(lines) + "\n")
print((study / "summary.md").read_text())
EOF

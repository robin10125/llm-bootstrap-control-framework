#!/bin/bash
# Critic-feature DIVERSITY study: does the critic benefit from more/diverse instruments?
# All arms: method=critic_features, 1h per run, PPO seeds 0/1/2, interleaved by seed.
#
#   control        -> NOT rerun; reuses runs/critic_useinfo_1h_20260713/nouse_s{0,1,2}
#                     (kl_nouse program, default feature blocks, same trainer/budget).
#   parallel       -> kl_nouse + parallel stage evidence: all gate values (cursor-free) and
#                     always-on critical-stage actions (grasp/lift/acquire).
#   union          -> feature union of 3 diverse programs (kl_nouse 8-stage winner,
#                     kl_simple 3-stage coarse, critic_use 8-stage critic-tailored),
#                     default blocks per program, one cursor each.
#   union_parallel -> union + parallel stage evidence.
#
# 9h sequential. NOT started automatically -- launch with:
#   cd /home/robin/Documents/agent-mini-script-control/llm-framework
#   nohup bash policy_bias_lab/experimental/alternative-methods/run_critic_diversity_1h.sh \
#     > runs/critic_diversity_1h/driver.log 2>&1 &
set -u
cd /home/robin/Documents/agent-mini-script-control/llm-framework
OUT=runs/critic_diversity_1h
CONTROL=runs/critic_useinfo_1h_20260713
PY=.venv/bin/python
ALT=policy_bias_lab/experimental/alternative-methods/run_alt_method.py
P=runs/prior_gen_study_20260711
NOUSE=$P/kl_nouse/program.json
SIMPLE=$P/kl_simple/program.json
CRITIC=$P/critic_use/program.json
T=3600
CRIT_KW=grasp,lift,acquire
mkdir -p "$OUT"

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

echo "=== critic-feature diversity study start $(date +%F\ %T) ==="
wait_free

for S in 0 1 2; do
  run_arm parallel_s$S --seed $S --program $NOUSE \
    --critic-gate-values --critic-critical-actions --critical-stages $CRIT_KW
  run_arm union_s$S --seed $S --program $NOUSE $SIMPLE $CRITIC
  run_arm union_parallel_s$S --seed $S --program $NOUSE $SIMPLE $CRITIC \
    --critic-gate-values --critic-critical-actions --critical-stages $CRIT_KW
done

echo "=== ALL RUNS DONE $(date +%F\ %T) ==="

$PY - "$OUT" "$CONTROL" <<'EOF'
import json, sys
from pathlib import Path

out, control = Path(sys.argv[1]), Path(sys.argv[2])
rows = []
for d in sorted(out.iterdir()):
    if (d / "final_report.json").is_file():
        rows.append((d.name.rsplit("_s", 1)[0], d.name, d / "final_report.json"))
for d in sorted(control.glob("nouse_s*")):
    if (d / "final_report.json").is_file():
        rows.append(("control", f"control({d.name})", d / "final_report.json"))

recs = []
for arm, run, rep in rows:
    r = json.loads(rep.read_text())
    ev = r.get("eval", {})
    recs.append({"arm": arm, "run": run,
                 "graded": ev.get("eval_graded_objective"),
                 "success": ev.get("eval_success_rate"),
                 "fitness": ev.get("eval_task_fitness"),
                 "best_iter": r.get("best_iter"), "iters": r.get("iters")})

lines = ["# critic-feature diversity study (1h x 3 seeds)", "",
         "control = kl_nouse/default blocks (from critic_useinfo_1h_20260713 nouse arms)", "",
         "| run | graded | success | fitness | best_iter/iters |", "|---|---:|---:|---:|---:|"]
for r in sorted(recs, key=lambda r: (r["arm"], r["run"])):
    lines.append(f"| {r['run']} | {r['graded']} | {r['success']} | {r['fitness']} | "
                 f"{r['best_iter']}/{r['iters']} |")
for arm in ("control", "parallel", "union", "union_parallel"):
    vals = [r for r in recs if r["arm"] == arm and r["graded"] is not None]
    if vals:
        g = [r["graded"] for r in vals]
        s = [r["success"] for r in vals]
        lines += ["", f"**{arm}** (n={len(vals)}): graded mean {sum(g)/len(g):.3f} "
                      f"(min {min(g):.3f}, max {max(g):.3f}); success mean {sum(s)/len(s):.3f} "
                      f"(min {min(s):.3f}, max {max(s):.3f})"]
(out / "summary.md").write_text("\n".join(lines) + "\n")
print((out / "summary.md").read_text())
EOF

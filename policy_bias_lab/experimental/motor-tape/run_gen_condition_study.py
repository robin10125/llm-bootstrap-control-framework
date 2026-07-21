"""Phase A of the generation-condition study: generate + autopilot-screen all arms.

Arms (3 candidates each, codex backend, shared context block):
  base       status quo prompt
  exact      exact fixed object position injected; env obj_xy_range = 0
  samples    4 concrete sampled reset observations injected
  fb         one measured-evidence revision round
  orient     probed sign-calibration table injected
  orient_fb  calibration table + revision round (is the table redundant given feedback?)

Screening = score_tape autopilot graded objective (what the interventions target: grounding).
Writes <study>/screen.md, copies each arm's best candidate to <study>/<arm>/best_program.json,
and for the exact arm also scores its best plan under the RANDOMIZED env
(autopilot_brittle.json -- how brittle is a position-specific plan).

Resumable: a candidate dir with program.json is skipped.

Usage:
  nohup .venv/bin/python policy_bias_lab/experimental/motor-tape/run_gen_condition_study.py \
      --study-dir runs/motor_tape_genstudy_20260716 \
      --context-dir runs/prior_gen_study_20260711/context \
      > runs/motor_tape_genstudy_20260716.log 2>&1 &
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_triton_gemm=false")
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", str(Path(".xla_cache").resolve()))

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for p in (str(HERE), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from generate_motor_tape import generate_one  # noqa: E402
from motor_tape import score_tape  # noqa: E402

ARMS = {
    #  name        position_info  feedback  calibration  fixed_env
    "base":       ("none",        False,    False,       False),
    "exact":      ("exact",       False,    False,       True),
    "samples":    ("samples",     False,    False,       False),
    "fb":         ("none",        True,     False,       False),
    "orient":     ("none",        False,    True,        False),
    "orient_fb":  ("none",        True,     True,        False),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--study-dir", type=Path, required=True)
    ap.add_argument("--context-dir", type=Path, required=True,
                    help="shared context block dir (reused across every arm)")
    ap.add_argument("--task", default="grasp and lift a 5cm cube off the table")
    ap.add_argument("--llm-backend", default="codex")
    ap.add_argument("--llm-model", default=None)
    ap.add_argument("--n-gens", type=int, default=3)
    ap.add_argument("--arms", default=",".join(ARMS),
                    help="comma-separated subset of arms to run")
    ap.add_argument("--dry-run", action="store_true",
                    help="assemble prompts only (one per arm), no LLM calls")
    args = ap.parse_args()
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    for a in arms:
        if a not in ARMS:
            raise SystemExit(f"unknown arm {a!r} (known: {list(ARMS)})")

    from experiment_runtime.environment import make_env
    env_rand = make_env("shadow", control_dt=0.025, episode_seconds=20.0, physics_dt=0.01,
                        obj_xy_range=0.04)
    env_fixed = None
    if any(ARMS[a][3] for a in arms):
        env_fixed = make_env("shadow", control_dt=0.025, episode_seconds=20.0, physics_dt=0.01,
                             obj_xy_range=0.0)

    args.study_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, list[dict]] = {}
    for arm in arms:
        position_info, feedback, calibration, fixed = ARMS[arm]
        env = env_fixed if fixed else env_rand
        results[arm] = []
        n = 1 if args.dry_run else args.n_gens
        for g in range(1, n + 1):
            out = args.study_dir / arm / f"g{g}"
            meta_file = out / "meta.json"
            if (out / "program.json").exists() and meta_file.exists():
                print(f"[study] {arm}/g{g}: exists, skipping")
                results[arm].append(json.loads(meta_file.read_text()))
                continue
            print(f"[study] === {arm}/g{g} (pos={position_info} fb={feedback} "
                  f"cal={calibration} fixed_env={fixed}) ===")
            try:
                meta = generate_one(
                    env, out_dir=out, task=args.task, backend=args.llm_backend,
                    model=args.llm_model, context_dir=args.context_dir,
                    position_info=position_info, feedback=feedback, calibration=calibration,
                    dry_run=args.dry_run)
            except Exception as e:  # noqa: BLE001 - one failed gen must not kill the study
                print(f"[study] {arm}/g{g} FAILED: {e}")
                meta = {"failed": str(e)}
            meta["arm"] = arm
            meta["gen"] = g
            results[arm].append(meta)
    if args.dry_run:
        print("[study] dry-run complete; inspect <study>/<arm>/g1/prompt.md")
        return 0

    # ---- screen: best per arm by autopilot graded ----
    lines = ["# generation-condition study: autopilot screen", "",
             "| arm | gen | graded | success | engagement | keyframes | kept_round | warnings |",
             "|---|---:|---:|---:|---:|---:|---:|---|"]
    for arm in arms:
        best_g, best_score = None, -1e30
        for meta in results[arm]:
            if meta.get("failed"):
                lines.append(f"| {arm} | {meta.get('gen')} | FAILED | | | | | {meta['failed'][:40]} |")
                continue
            g = float(meta["autopilot_graded"])
            w = meta.get("warnings", {})
            lines.append(
                f"| {arm} | {meta['gen']} | {g:.3f} | {meta['autopilot_success']:.3f} | "
                f"{meta['contact_engagement']:.3f} | {meta['n_keyframes']} | "
                f"{meta.get('kept_round', 0)} | oor={w.get('out_of_range', 0)} "
                f"slew={w.get('over_slew', 0)} |")
            if g > best_score:
                best_score, best_g = g, meta["gen"]
        if best_g is not None:
            src = args.study_dir / arm / f"g{best_g}" / "program.json"
            dst = args.study_dir / arm / "best_program.json"
            shutil.copyfile(src, dst)
            lines.append(f"| **{arm} best** | g{best_g} | **{best_score:.3f}** | | | | | |")
            if ARMS[arm][3]:  # exact arm: brittleness under the randomized env
                prog = json.loads(src.read_text())
                brittle = score_tape(env_rand, prog, envs=128, seed=7, task="lift")
                (args.study_dir / arm / "autopilot_brittle.json").write_text(
                    json.dumps(brittle, indent=2) + "\n")
                bt = brittle["tape_eval"]
                lines.append(
                    f"| {arm} best under +-4cm | g{best_g} | "
                    f"{bt['eval_graded_objective']:.3f} | {bt['eval_success_rate']:.3f} | "
                    f"{brittle['contact_engagement']:.3f} | | | brittleness diagnostic |")
    (args.study_dir / "screen.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

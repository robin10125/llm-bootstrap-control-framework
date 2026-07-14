#!/usr/bin/env python3
"""Compare Codex-bootstrapped primitive-policy training against normal RL search.

Both arms use the same policy parameterization, MuJoCo environment, scoring function,
population size, number of generations, and worker count.

- `normal_rl_cem`: starts from a zero/random latent policy and improves by CEM.
- `codex_bootstrap_cem`: starts from the Codex-supervised primitive policy and uses the
  same CEM update budget.

This is not a full deep-RL benchmark; it is a controlled test of whether the proposed
LLM/Codex primitive-policy bootstrap gives the optimizer a measurable head start under
the same downstream exploration budget.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from shadow_bootstrap_experiment import (
    Rollout,
    decode_policy,
    encode_policy,
    evaluate_many,
    evaluate_policy,
    obs_from_setup,
    safe_worker_count,
    save_rollout,
)
from shadow_inspect import render_policy_video
from shadow_policy_runner import ShadowPolicyRunner


DEFAULT_CODEX_POLICY = Path("policies/codex_supervised_lift_cube.json")


@dataclass
class ArmState:
    name: str
    mean: np.ndarray
    std: np.ndarray
    best: Rollout | None = None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("runs") / f"{time.strftime('%Y%m%d-%H%M%S')}-shadow_compare")
    parser.add_argument("--codex-policy", type=Path, default=DEFAULT_CODEX_POLICY)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--generations", type=int, default=6)
    parser.add_argument("--population", type=int, default=10)
    parser.add_argument("--elite-frac", type=float, default=0.25)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--sample-videos-per-arm", type=int, default=6)
    parser.add_argument("--video-fps", type=int, default=20)
    args = parser.parse_args()

    workers = args.workers or safe_worker_count()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    setup = {"object_pos": [0.0, 0.0, 0.025]}
    obs = obs_from_setup(setup)
    codex_policy = json.loads(args.codex_policy.read_text())
    codex_latent = encode_policy(codex_policy)

    all_rows: list[dict[str, Any]] = []
    manifest: dict[str, Any] = {
        "workers": workers,
        "seeds": args.seeds,
        "generations": args.generations,
        "population": args.population,
        "elite_frac": args.elite_frac,
        "codex_policy": str(args.codex_policy),
        "arms": {},
    }

    for seed in range(args.seeds):
        rng = np.random.default_rng(seed)
        seed_dir = out_dir / f"seed_{seed:02d}"
        arms = [
            ArmState(
                name="normal_rl_cem",
                mean=np.zeros_like(codex_latent),
                std=np.full_like(codex_latent, 1.3),
            ),
            ArmState(
                name="codex_bootstrap_cem",
                mean=codex_latent.copy(),
                std=np.full_like(codex_latent, 0.45),
            ),
        ]

        for arm in arms:
            arm_dir = seed_dir / arm.name
            arm_dir.mkdir(parents=True, exist_ok=True)

            if arm.name == "codex_bootstrap_cem":
                initial = evaluate_policy(codex_policy, kind="codex_bootstrap_seed")
                save_rollout(arm_dir / "generation_00_codex_seed", initial)
                arm.best = initial
                all_rows.append(row(seed, arm.name, 0, "seed", initial, 0))

            for gen in range(1, args.generations + 1):
                jobs: list[tuple[str, dict[str, Any]]] = []
                latents: list[np.ndarray] = []
                for i in range(args.population):
                    latent = arm.mean + rng.normal(0.0, arm.std)
                    latents.append(latent)
                    policy = decode_policy(latent, setup, name=f"{arm.name}_s{seed:02d}_g{gen:02d}_{i:02d}")
                    jobs.append((arm.name, policy))

                rollouts = evaluate_many(jobs, workers=workers)
                ranked = sorted(enumerate(rollouts), key=lambda t: t[1].score, reverse=True)
                elite_count = max(1, int(round(args.population * args.elite_frac)))
                elite_indices = [idx for idx, _ in ranked[:elite_count]]
                elite_latents = np.vstack([latents[i] for i in elite_indices])
                arm.mean = elite_latents.mean(axis=0)
                arm.std = np.maximum(elite_latents.std(axis=0), 0.08)

                best_gen = ranked[0][1]
                if arm.best is None or best_gen.score > arm.best.score:
                    arm.best = best_gen

                gen_dir = arm_dir / f"generation_{gen:02d}"
                save_rollout(gen_dir / "best", best_gen)
                (gen_dir / "population_summary.json").write_text(json.dumps([
                    {
                        "rank": rank,
                        "policy": rollout.policy["name"],
                        "score": rollout.score,
                        "success": rollout.result.get("success"),
                        "lifted": rollout.result.get("final_state", {}).get("lifted"),
                        "object_z": rollout.result.get("final_state", {}).get("object", {}).get("z"),
                        "contacts": rollout.result.get("final_state", {}).get("contacts", {}).get("hand_object_count"),
                    }
                    for rank, (_, rollout) in enumerate(ranked)
                ], indent=2) + "\n")

                all_rows.append(row(seed, arm.name, gen, "best_generation", best_gen, 0))
                all_rows.append(row(seed, arm.name, gen, "best_so_far", arm.best, 0))

            if arm.best is not None:
                save_rollout(arm_dir / "best_overall", arm.best)
                manifest["arms"].setdefault(arm.name, []).append({
                    "seed": seed,
                    "best_score": arm.best.score,
                    "best_policy": arm.best.policy["name"],
                    "success": arm.best.result.get("success"),
                    "lifted": arm.best.result.get("final_state", {}).get("lifted"),
                    "object_z": arm.best.result.get("final_state", {}).get("object", {}).get("z"),
                    "contacts": arm.best.result.get("final_state", {}).get("contacts", {}).get("hand_object_count"),
                })

    write_metrics(out_dir / "metrics.csv", all_rows)
    summary = summarize(all_rows)
    manifest["summary"] = summary
    videos = render_progression_videos(out_dir, sample_per_arm=args.sample_videos_per_arm, fps=args.video_fps)
    manifest["videos"] = [str(v) for v in videos]
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (out_dir / "README.md").write_text(report_markdown(manifest))

    print(json.dumps({"out_dir": str(out_dir), "summary": summary, "videos": manifest["videos"]}, indent=2))
    return 0


def row(seed: int, arm: str, generation: int, phase: str, rollout: Rollout, eval_index: int) -> dict[str, Any]:
    final = rollout.result.get("final_state", {})
    return {
        "seed": seed,
        "arm": arm,
        "generation": generation,
        "phase": phase,
        "eval_index": eval_index,
        "policy": rollout.policy["name"],
        "score": rollout.score,
        "success": bool(rollout.result.get("success")),
        "grasped": bool(final.get("grasped")),
        "lifted": bool(final.get("lifted")),
        "object_z": final.get("object", {}).get("z"),
        "contacts": final.get("contacts", {}).get("hand_object_count"),
    }


def write_metrics(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    final_rows = [r for r in rows if r["phase"] == "best_so_far"]
    max_gen = max((int(r["generation"]) for r in final_rows), default=0)
    for arm in sorted({r["arm"] for r in final_rows}):
        arm_rows = [r for r in final_rows if r["arm"] == arm and int(r["generation"]) == max_gen]
        scores = np.array([float(r["score"]) for r in arm_rows], dtype=float)
        lifts = np.array([1.0 if r["lifted"] else 0.0 for r in arm_rows], dtype=float)
        object_z = np.array([float(r["object_z"]) for r in arm_rows], dtype=float)
        out[arm] = {
            "n": int(len(arm_rows)),
            "score_mean": round(float(scores.mean()), 4) if len(scores) else None,
            "score_std": round(float(scores.std()), 4) if len(scores) else None,
            "lift_rate": round(float(lifts.mean()), 4) if len(lifts) else None,
            "object_z_mean": round(float(object_z.mean()), 4) if len(object_z) else None,
            "best_score": round(float(scores.max()), 4) if len(scores) else None,
        }
    return out


def render_progression_videos(out_dir: Path, *, sample_per_arm: int, fps: int) -> list[Path]:
    videos: list[Path] = []
    for arm_dir in sorted(out_dir.glob("seed_00/*")):
        if not arm_dir.is_dir():
            continue
        generation_dirs = sorted(arm_dir.glob("generation_*"))
        selected = evenly_spaced(generation_dirs, sample_per_arm)
        best_overall = arm_dir / "best_overall"
        if best_overall.exists() and best_overall not in selected:
            selected.append(best_overall)
        for d in selected:
            policy_path = d / "best" / "policy.json" if (d / "best" / "policy.json").exists() else d / "policy.json"
            if not policy_path.exists():
                continue
            policy = json.loads(policy_path.read_text())
            video_path = policy_path.parent / "replay.mp4"
            try:
                render_policy_video(policy, ShadowPolicyRunner(), video_path, fps=fps, width=640, height=480)
            except Exception as exc:  # noqa: BLE001
                (policy_path.parent / "replay_error.txt").write_text(str(exc) + "\n")
                continue
            videos.append(video_path)
    return videos


def evenly_spaced(items: list[Path], n: int) -> list[Path]:
    if len(items) <= n:
        return items
    if n <= 1:
        return [items[-1]]
    idxs = np.linspace(0, len(items) - 1, n).round().astype(int)
    return [items[int(i)] for i in idxs]


def report_markdown(manifest: dict[str, Any]) -> str:
    lines = [
        "# Shadow Training Comparison",
        "",
        "## Setup",
        "",
        f"- Seeds: `{manifest['seeds']}`",
        f"- Generations: `{manifest['generations']}`",
        f"- Population: `{manifest['population']}`",
        f"- Workers: `{manifest['workers']}`",
        f"- Codex bootstrap policy: `{manifest['codex_policy']}`",
        "",
        "## Summary",
        "",
        "```json",
        json.dumps(manifest["summary"], indent=2),
        "```",
        "",
        "## Videos",
        "",
    ]
    lines.extend(f"- `{v}`" for v in manifest.get("videos", []))
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())

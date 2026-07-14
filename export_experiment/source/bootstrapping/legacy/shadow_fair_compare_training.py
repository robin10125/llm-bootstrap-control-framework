#!/usr/bin/env python3
"""Fair comparison: search baselines vs LLM bootstrap variants on manipulation tasks.

Core arms:

1. `normal_rl_cem`: zero-init CEM search.
2. `demo_init_cem`: initialized from a demonstration policy (LLM-authored when available,
   `--init-policy`), then CEM only.
3. `demo_init_residual_cem`: same initialization, with LLM residual candidate schedules
   injected inside the same population budget.

`--ablation` adds init-mean/init-std controls:

4. `demo_init_wide_std_cem`, 5. `zero_init_narrow_std_cem`, 6. `random_best_init_cem`
   (generation 1 samples wide, then restarts CEM from the best random candidate).

`--baselines` adds:

7. `reinforce`: Gaussian REINFORCE/ES over the same latent space and budget (the
   standard policy-gradient choice for this contextual-bandit setting; the episode is a
   single schedule, so PPO/SAC-style multistep machinery does not apply).
8. `normal_rl_bonus_cem`: zero-init CEM with extra generations matching the rollout
   value of the residual arm's LLM calls.
9. `scripted_expert`: the hand-written expert evaluated directly on the test setups.
10. `llm_only`: a fresh LLM scratch schedule per test setup, no learning (with
    `--llm mock` this degenerates to `scripted_expert`).

Design properties:

- randomized train and held-out test setups, shared across arms within a seed,
- each arm draws from its own identically-derived rng stream,
- equal rollout budget per arm and generation; residual candidates consume population slots,
- residual prompts are built from the arm's best and worst per-setup rollouts, conditioned
  on each rollout's own setup,
- with `--llm claude-code|codex|anthropic` residual/scratch candidates come from real
  model calls (prompt, completion, cost logged per call); `--llm mock` uses deterministic
  stand-ins for offline plumbing tests,
- `--policy-space sequence` searches variable-length primitive schedules so structure is
  discovered, not given; `template` keeps the original 12-parameter fixed schedule,
- `--task` selects lift / lift_perturbed / place / push; `--object-set varied` randomizes
  object shape, size, mass, and friction,
- success_rate is the task's own binary success (held lift for lift tasks),
- test metrics get bootstrap 95% CIs over seeds and paired comparisons vs `normal_rl_cem`.
"""
from __future__ import annotations

import argparse
import concurrent.futures as futures
import csv
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from llm_backend import LLMUsageTracker, call_llm, extract_json_policy
from policy_spaces import expert_for, get_space, jitter_residual
from shadow_bootstrap_experiment import (
    Rollout,
    evaluate_policy,
    safe_worker_count,
    save_rollout,
)
from shadow_inspect import render_policy_video
from shadow_policy_runner import runner_for_policy
from shadow_residual_prompt import build_residual_prompt
from tasks import get_task


DEFAULT_INIT_POLICY = Path("policies/codex_supervised_lift_cube.json")
PRIMITIVES_PATH = Path("shadow_primitives.json")


@dataclass
class Candidate:
    latent: np.ndarray
    kind: str
    policy_name: str


@dataclass
class Arm:
    name: str
    mean: np.ndarray
    std: np.ndarray
    use_residual: bool
    algo: str = "cem"  # "cem" | "reinforce"
    reset_std_after_first: float | None = None
    extra_generations: int = 0
    best_train: dict[str, Any] | None = None


@dataclass
class LLMContext:
    backend: str
    model: str | None
    primitives: dict[str, Any]
    tracker: LLMUsageTracker = field(default_factory=LLMUsageTracker)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("runs") / f"{time.strftime('%Y%m%d-%H%M%S')}-shadow_fair_compare")
    parser.add_argument("--task", choices=["lift", "lift_perturbed", "place", "push"], default="lift")
    parser.add_argument("--object-set", choices=["cube", "varied"], default="cube")
    parser.add_argument("--embodiment", choices=["shadow", "gripper"], default="shadow")
    parser.add_argument("--policy-space", choices=["template", "sequence"], default="template")
    parser.add_argument("--init-policy", type=Path,
                        help="demonstration policy JSON used to initialize the demo arms "
                             f"(default {DEFAULT_INIT_POLICY} for lift tasks, scripted expert otherwise)")
    parser.add_argument("--llm", choices=["mock", "claude-code", "codex", "anthropic"], default="mock",
                        help="backend for residual/scratch candidate schedules; mock is the offline stand-in")
    parser.add_argument("--llm-model", help="model override for the LLM backend")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--generations", type=int, default=5)
    parser.add_argument("--population", type=int, default=8)
    parser.add_argument("--elite-frac", type=float, default=0.25)
    parser.add_argument("--train-positions", type=int, default=3)
    parser.add_argument("--test-positions", type=int, default=8)
    parser.add_argument("--position-radius", type=float, default=0.025)
    parser.add_argument("--ablation", action="store_true", help="add init-mean/init-std control arms")
    parser.add_argument("--baselines", action="store_true",
                        help="add reinforce, bonus-budget CEM, scripted-expert, and llm-only baselines")
    parser.add_argument("--bonus-generations", type=int,
                        help="extra generations for normal_rl_bonus_cem (default: residual arm's LLM calls / population)")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--sample-videos-per-arm", type=int, default=5)
    parser.add_argument("--video-fps", type=int, default=20)
    args = parser.parse_args()

    task = get_task(args.task)
    space = get_space(args.policy_space)
    if space.name == "template" and task.name not in {"lift", "lift_perturbed"}:
        parser.error("--policy-space template only supports lift tasks; use --policy-space sequence")
    if space.name == "template" and args.embodiment != "shadow":
        parser.error("--policy-space template is Shadow-tuned; use --policy-space sequence for other embodiments")

    workers = args.workers or safe_worker_count()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    init_policy, init_source = load_init_policy(args.init_policy, task, space, args.embodiment)
    init_latent = space.encode(init_policy)
    llm_ctx = LLMContext(
        backend=args.llm,
        model=args.llm_model,
        primitives=json.loads(PRIMITIVES_PATH.read_text()),
    )
    residual_calls = 2 * max(0, args.generations - 1)
    bonus_generations = args.bonus_generations
    if bonus_generations is None:
        bonus_generations = int(np.ceil(residual_calls / args.population))

    all_rows: list[dict[str, Any]] = []
    manifest: dict[str, Any] = {
        "workers": workers,
        "task": task.name,
        "object_set": args.object_set,
        "embodiment": args.embodiment,
        "policy_space": space.name,
        "llm": {"backend": args.llm, "model": args.llm_model},
        "seeds": args.seeds,
        "generations": args.generations,
        "population": args.population,
        "elite_frac": args.elite_frac,
        "train_positions": args.train_positions,
        "test_positions": args.test_positions,
        "position_radius": args.position_radius,
        "ablation": args.ablation,
        "baselines": args.baselines,
        "bonus_generations": bonus_generations,
        "init_policy": init_source,
        "arms": {},
    }

    for seed in range(args.seeds):
        train_setups = [task.sample_setup(np.random.default_rng([seed, i]), args.position_radius, args.object_set)
                        for i in range(args.train_positions)]
        test_setups = [task.sample_setup(np.random.default_rng([1000 + seed, i]), args.position_radius, args.object_set)
                       for i in range(args.test_positions)]
        if args.embodiment != "shadow":
            for setup in train_setups + test_setups:
                setup["embodiment"] = args.embodiment
        seed_dir = out_dir / f"seed_{seed:02d}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        (seed_dir / "setups.json").write_text(json.dumps({"train": train_setups, "test": test_setups}, indent=2) + "\n")

        arms = make_arms(init_latent, ablation=args.ablation, baselines=args.baselines, bonus_generations=bonus_generations)
        for arm_idx, arm in enumerate(arms):
            # Independent but identically-derived stream per (seed, arm) so arms do not
            # consume each other's randomness.
            arm_rng = np.random.default_rng([seed, arm_idx])
            arm_dir = seed_dir / arm.name
            arm_dir.mkdir(parents=True, exist_ok=True)
            for gen in range(1, args.generations + arm.extra_generations + 1):
                gen_dir = arm_dir / f"generation_{gen:02d}"
                gen_dir.mkdir(parents=True, exist_ok=True)
                candidates = make_candidates(arm, arm_rng, args.population, gen, seed, space, task, llm_ctx, gen_dir)
                evaluated = evaluate_candidates(candidates, train_setups, space.name, task.name, workers=workers)
                evaluated.sort(key=lambda x: x["mean_score"], reverse=True)

                update_arm(arm, evaluated, gen, args.population, args.elite_frac)

                best_gen = evaluated[0]
                if arm.best_train is None or best_gen["mean_score"] > arm.best_train["mean_score"]:
                    arm.best_train = best_gen

                save_evaluated_candidate(gen_dir / "best_train", best_gen)
                (gen_dir / "population_summary.json").write_text(json.dumps(summarize_population(evaluated), indent=2) + "\n")
                all_rows.append(metric_row(seed, arm.name, gen, "train_best_generation", best_gen))
                all_rows.append(metric_row(seed, arm.name, gen, "train_best_so_far", arm.best_train))

            assert arm.best_train is not None
            test_eval = evaluate_latent_on_setups(
                arm.best_train["latent"],
                test_setups,
                f"{arm.name}_s{seed:02d}_test_best",
                f"{arm.name}_test",
                space.name,
                task.name,
            )
            save_evaluated_candidate(arm_dir / "test_best", test_eval)
            all_rows.append(metric_row(seed, arm.name, args.generations, "heldout_test", test_eval))
            manifest["arms"].setdefault(arm.name, []).append({
                "seed": seed,
                "test_mean_score": test_eval["mean_score"],
                "test_success_rate": test_eval["success_rate"],
                "best_train_score": arm.best_train["mean_score"],
            })

        if args.baselines:
            for name, rollouts in direct_baselines(test_setups, task, space, llm_ctx, seed, seed_dir):
                summary = summarize_rollouts(rollouts, policy_name=f"{name}_s{seed:02d}", kind=name)
                save_evaluated_candidate(seed_dir / name, summary)
                all_rows.append(metric_row(seed, name, args.generations, "heldout_test", summary))
                manifest["arms"].setdefault(name, []).append({
                    "seed": seed,
                    "test_mean_score": summary["mean_score"],
                    "test_success_rate": summary["success_rate"],
                })

    write_metrics(out_dir / "metrics.csv", all_rows)
    manifest["llm"]["usage"] = llm_ctx.tracker.summary()
    manifest["summary"] = summarize_metrics(all_rows)
    manifest["comparisons"] = pairwise_comparisons(all_rows, baseline="normal_rl_cem")
    manifest["videos"] = [str(p) for p in render_progression_videos(out_dir, args.sample_videos_per_arm, args.video_fps)]
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (out_dir / "README.md").write_text(report_markdown(manifest))

    print(json.dumps({
        "out_dir": str(out_dir),
        "task": task.name,
        "policy_space": space.name,
        "llm_usage": manifest["llm"]["usage"],
        "summary": manifest["summary"],
        "comparisons": manifest["comparisons"],
    }, indent=2))
    return 0


def load_init_policy(path: Path | None, task, space, embodiment: str) -> tuple[dict[str, Any], str]:
    """The demonstration policy seeding the demo arms. For Shadow lift tasks this defaults
    to the stored LLM-authored policy; otherwise the scripted expert at a nominal centered
    setup stands in until a real LLM-authored demo is provided via --init-policy."""
    if path is not None:
        return json.loads(path.read_text()), str(path)
    if embodiment == "shadow" and task.name in {"lift", "lift_perturbed"} and DEFAULT_INIT_POLICY.exists():
        return json.loads(DEFAULT_INIT_POLICY.read_text()), str(DEFAULT_INIT_POLICY)
    nominal = task.sample_setup(np.random.default_rng(0), 0.0, "cube")
    if embodiment != "shadow":
        nominal["embodiment"] = embodiment
    return expert_for(space, task, f"scripted_init_{task.name}", nominal), f"scripted_expert:{task.name}:{embodiment}"


def make_arms(init_latent: np.ndarray, *, ablation: bool, baselines: bool, bonus_generations: int) -> list[Arm]:
    zeros = np.zeros_like(init_latent)
    wide = np.full_like(init_latent, 1.35)
    narrow = np.full_like(init_latent, 0.45)
    arms = [
        Arm("normal_rl_cem", zeros.copy(), wide.copy(), False),
        Arm("demo_init_cem", init_latent.copy(), narrow.copy(), False),
        Arm("demo_init_residual_cem", init_latent.copy(), narrow.copy(), True),
    ]
    if ablation:
        arms += [
            Arm("demo_init_wide_std_cem", init_latent.copy(), wide.copy(), False),
            Arm("zero_init_narrow_std_cem", zeros.copy(), narrow.copy(), False),
            Arm("random_best_init_cem", zeros.copy(), wide.copy(), False, reset_std_after_first=0.45),
        ]
    if baselines:
        arms += [
            Arm("reinforce", zeros.copy(), wide.copy(), False, algo="reinforce"),
            Arm("normal_rl_bonus_cem", zeros.copy(), wide.copy(), False, extra_generations=bonus_generations),
        ]
    return arms


def update_arm(arm: Arm, evaluated: list[dict[str, Any]], gen: int, population: int, elite_frac: float) -> None:
    if gen == 1 and arm.reset_std_after_first is not None:
        arm.mean = evaluated[0]["latent"].copy()
        arm.std = np.full_like(arm.mean, arm.reset_std_after_first)
        return
    if arm.algo == "reinforce":
        scores = np.array([item["mean_score"] for item in evaluated], dtype=float)
        adv = (scores - scores.mean()) / (scores.std() + 1e-8)
        latents = np.vstack([item["latent"] for item in evaluated])
        # ES-style REINFORCE step: advantage-weighted recombination, fixed std.
        arm.mean = arm.mean + 0.5 * (adv[:, None] * (latents - arm.mean)).mean(axis=0)
        return
    elite_count = max(1, int(round(population * elite_frac)))
    elite_latents = np.vstack([item["latent"] for item in evaluated[:elite_count]])
    arm.mean = elite_latents.mean(axis=0)
    arm.std = np.maximum(elite_latents.std(axis=0), 0.08)


def make_candidates(
    arm: Arm,
    rng: np.random.Generator,
    population: int,
    gen: int,
    seed: int,
    space,
    task,
    llm_ctx: LLMContext,
    log_dir: Path,
) -> list[Candidate]:
    candidates: list[Candidate] = []
    if arm.use_residual and arm.best_train is not None:
        # Edit both the strongest and weakest per-setup rollouts, each conditioned on its
        # own setup: one call refines what works, the other repairs what does not.
        rollouts = sorted(arm.best_train["rollouts"], key=lambda r: r.score, reverse=True)
        references = [rollouts[0]]
        if len(rollouts) > 1:
            references.append(rollouts[-1])
        for attempt, reference in enumerate(references[: min(2, population)], start=1):
            name = f"{arm.name}_s{seed:02d}_g{gen:02d}_residual{attempt:02d}"
            candidates.append(residual_candidate(name, reference, space, task, llm_ctx, log_dir, attempt))

    for i in range(population - len(candidates)):
        latent = arm.mean + rng.normal(0.0, arm.std)
        candidates.append(Candidate(latent, "cem_sample", f"{arm.name}_s{seed:02d}_g{gen:02d}_{i:02d}"))
    return candidates


def residual_candidate(
    name: str,
    reference: Rollout,
    space,
    task,
    llm_ctx: LLMContext,
    log_dir: Path,
    attempt: int,
) -> Candidate:
    setup = reference.policy.get("setup", {})
    if llm_ctx.backend != "mock":
        prompt = build_residual_prompt(
            reference.policy, reference.result, llm_ctx.primitives,
            f"Act as a residual policy. Task: {task.goal}. Edit this rollout's primitive "
            "schedule so the next rollout measurably improves. Prefer small parameter and "
            "timing edits.",
        )
        response = call_llm(llm_ctx.backend, prompt, model=llm_ctx.model, log_dir=log_dir, tag=f"residual{attempt:02d}")
        llm_ctx.tracker.add(response, tag=name)
        policy = extract_json_policy(response.text) if response.ok else None
        if policy is not None:
            policy["setup"] = setup
            policy["name"] = name
            try:
                return Candidate(space.encode(policy), "llm_residual", name)
            except (KeyError, IndexError, TypeError, ValueError) as exc:
                (log_dir / f"residual{attempt:02d}_encode_error.txt").write_text(f"{exc!r}\n")
        # A failed call or unencodable policy falls back to the local heuristic so the
        # rollout budget stays identical; the kind records that this slot is degraded.
        policy = jitter_residual(space, reference.policy, setup, task, name=name, attempt=attempt)
        return Candidate(space.encode(policy), "llm_residual_fallback", name)

    policy = jitter_residual(space, reference.policy, setup, task, name=name, attempt=attempt)
    return Candidate(space.encode(policy), "mock_residual", name)


def direct_baselines(test_setups, task, space, llm_ctx: LLMContext, seed: int, seed_dir: Path):
    """Baselines that involve no search: the scripted expert, and (with a real backend)
    one fresh LLM scratch schedule per test setup."""
    scripted = []
    for i, setup in enumerate(test_setups):
        policy = task.scripted_expert(f"scripted_expert_s{seed:02d}_p{i:02d}", setup)
        scripted.append(evaluate_policy(policy, kind="scripted_expert", task_name=task.name))
    yield "scripted_expert", scripted

    llm_only = []
    for i, setup in enumerate(test_setups):
        name = f"llm_only_s{seed:02d}_p{i:02d}"
        if llm_ctx.backend != "mock":
            reference = scripted[i]
            prompt = build_residual_prompt(
                reference.policy, reference.result, llm_ctx.primitives,
                f"Task: {task.goal}. Write the best primitive schedule you can for this "
                "setup from scratch.",
            )
            response = call_llm(llm_ctx.backend, prompt, model=llm_ctx.model,
                                log_dir=seed_dir / "llm_only", tag=f"scratch_p{i:02d}")
            llm_ctx.tracker.add(response, tag=name)
            policy = extract_json_policy(response.text) if response.ok else None
            if policy is not None:
                policy["setup"] = setup
                policy["name"] = name
                policy["success"] = task.success_checks(setup)
                llm_only.append(evaluate_policy(policy, kind="llm_only", task_name=task.name))
                continue
        policy = task.scripted_expert(name, setup)
        llm_only.append(evaluate_policy(policy, kind="llm_only_fallback", task_name=task.name))
    yield "llm_only", llm_only


def evaluate_candidates(
    candidates: list[Candidate],
    setups: list[dict[str, Any]],
    space_name: str,
    task_name: str,
    workers: int,
) -> list[dict[str, Any]]:
    jobs = [(c.latent, setups, c.policy_name, c.kind, space_name, task_name) for c in candidates]
    if workers <= 1 or len(jobs) <= 1:
        return [evaluate_latent_on_setups(*job) for job in jobs]
    with futures.ProcessPoolExecutor(max_workers=workers) as pool:
        fs = [pool.submit(evaluate_latent_on_setups, *job) for job in jobs]
        return [f.result() for f in fs]


def evaluate_latent_on_setups(
    latent: np.ndarray,
    setups: list[dict[str, Any]],
    policy_name: str,
    kind: str,
    space_name: str,
    task_name: str,
) -> dict[str, Any]:
    space = get_space(space_name)
    task = get_task(task_name)
    rollouts = []
    for i, setup in enumerate(setups):
        policy = space.decode(latent, setup, f"{policy_name}_p{i:02d}", task)
        rollouts.append(evaluate_policy(policy, kind=kind, task_name=task_name))
    summary = summarize_rollouts(rollouts, policy_name=policy_name, kind=kind)
    summary["latent"] = latent
    return summary


def summarize_rollouts(rollouts: list[Rollout], *, policy_name: str, kind: str) -> dict[str, Any]:
    scores = np.array([r.score for r in rollouts], dtype=float)
    successes = np.array([1.0 if r.result.get("success") else 0.0 for r in rollouts], dtype=float)
    object_z = np.array([float(r.result["final_state"]["object"]["z"]) for r in rollouts], dtype=float)
    contacts = np.array([float(r.result["final_state"]["contacts"]["hand_object_count"]) for r in rollouts], dtype=float)
    return {
        "policy_name": policy_name,
        "kind": kind,
        "mean_score": round(float(scores.mean()), 4),
        "score_std": round(float(scores.std()), 4),
        "success_rate": round(float(successes.mean()), 4),
        "object_z_mean": round(float(object_z.mean()), 4),
        "contact_mean": round(float(contacts.mean()), 4),
        "rollouts": rollouts,
    }


def save_evaluated_candidate(path: Path, candidate: dict[str, Any]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    summary = {k: v for k, v in candidate.items() if k not in {"latent", "rollouts"}}
    if "latent" in candidate:
        summary["latent"] = candidate["latent"].tolist()
    (path / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for i, rollout in enumerate(candidate["rollouts"]):
        save_rollout(path / f"rollout_{i:02d}", rollout)


def summarize_population(evaluated: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "rank": i,
            "policy_name": c["policy_name"],
            "kind": c["kind"],
            "mean_score": c["mean_score"],
            "success_rate": c["success_rate"],
            "object_z_mean": c["object_z_mean"],
            "contact_mean": c["contact_mean"],
        }
        for i, c in enumerate(evaluated)
    ]


def metric_row(seed: int, arm: str, generation: int, phase: str, candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "seed": seed,
        "arm": arm,
        "generation": generation,
        "phase": phase,
        "policy_name": candidate["policy_name"],
        "kind": candidate["kind"],
        "mean_score": candidate["mean_score"],
        "score_std": candidate["score_std"],
        "success_rate": candidate["success_rate"],
        "object_z_mean": candidate["object_z_mean"],
        "contact_mean": candidate["contact_mean"],
    }


def write_metrics(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def bootstrap_ci(values: np.ndarray, *, n_resamples: int = 10000, seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_resamples, len(values)))
    means = values[idx].mean(axis=1)
    return [round(float(np.percentile(means, 2.5)), 4), round(float(np.percentile(means, 97.5)), 4)]


def summarize_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out = {}
    test_rows = [r for r in rows if r["phase"] == "heldout_test"]
    for arm in sorted({r["arm"] for r in test_rows}):
        arm_rows = [r for r in test_rows if r["arm"] == arm]
        scores = np.array([float(r["mean_score"]) for r in arm_rows])
        successes = np.array([float(r["success_rate"]) for r in arm_rows])
        out[arm] = {
            "n_seeds": len(arm_rows),
            "test_score_mean": round(float(scores.mean()), 4),
            "test_score_ci95": bootstrap_ci(scores),
            "test_success_rate_mean": round(float(successes.mean()), 4),
            "test_success_rate_ci95": bootstrap_ci(successes),
            "best_test_score": round(float(scores.max()), 4),
        }
    return out


def pairwise_comparisons(rows: list[dict[str, Any]], *, baseline: str) -> dict[str, Any]:
    """Paired (per-seed) bootstrap CIs of each arm's test metrics minus the baseline's."""
    test_rows = [r for r in rows if r["phase"] == "heldout_test"]
    by_arm: dict[str, dict[int, dict[str, float]]] = {}
    for r in test_rows:
        by_arm.setdefault(r["arm"], {})[int(r["seed"])] = {
            "score": float(r["mean_score"]),
            "success_rate": float(r["success_rate"]),
        }
    base = by_arm.get(baseline)
    if not base:
        return {}
    out = {}
    for arm, per_seed in by_arm.items():
        if arm == baseline:
            continue
        seeds = sorted(set(per_seed) & set(base))
        if not seeds:
            continue
        score_diff = np.array([per_seed[s]["score"] - base[s]["score"] for s in seeds])
        success_diff = np.array([per_seed[s]["success_rate"] - base[s]["success_rate"] for s in seeds])
        out[f"{arm}_vs_{baseline}"] = {
            "n_seeds": len(seeds),
            "score_diff_mean": round(float(score_diff.mean()), 4),
            "score_diff_ci95": bootstrap_ci(score_diff),
            "success_rate_diff_mean": round(float(success_diff.mean()), 4),
            "success_rate_diff_ci95": bootstrap_ci(success_diff),
        }
    return out


def render_progression_videos(out_dir: Path, sample_per_arm: int, fps: int) -> list[Path]:
    if sample_per_arm <= 0:
        return []
    videos = []
    for arm_dir in sorted((out_dir / "seed_00").glob("*")):
        if not arm_dir.is_dir():
            continue
        generation_dirs = sorted(arm_dir.glob("generation_*"))
        selected = evenly_spaced(generation_dirs, sample_per_arm)
        selected.append(arm_dir / "test_best")
        for gen_dir in selected:
            rollout_dirs = sorted(gen_dir.glob("rollout_*"))
            if not rollout_dirs and (gen_dir / "best_train").exists():
                rollout_dirs = sorted((gen_dir / "best_train").glob("rollout_*"))
            if not rollout_dirs:
                continue
            rollout_dir = rollout_dirs[0]
            policy_path = rollout_dir / "policy.json"
            if not policy_path.exists():
                continue
            policy = json.loads(policy_path.read_text())
            video_path = rollout_dir / "replay.mp4"
            try:
                render_policy_video(policy, runner_for_policy(policy), video_path, fps=fps, width=640, height=480)
            except Exception as exc:  # noqa: BLE001
                (rollout_dir / "replay_error.txt").write_text(str(exc) + "\n")
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
        "# Fair Primitive Training Comparison",
        "",
        "## Design",
        "",
        f"- Task: `{manifest['task']}` (objects: `{manifest['object_set']}`), policy space: `{manifest['policy_space']}`.",
        f"- Demo init: `{manifest['init_policy']}`.",
        "- Randomized train setups and held-out test setups, shared across arms within a seed.",
        "- Independent rng stream per (seed, arm); equal rollout budget per arm and generation.",
        "- Residual candidates consume population slots, so they do not get extra rollout budget.",
        f"- Residual backend: `{manifest['llm']['backend']}` (model: `{manifest['llm'].get('model')}`).",
        "- `success_rate` is the task's binary success (held lift for lift tasks).",
        "",
        "## LLM usage",
        "",
        "```json",
        json.dumps(manifest["llm"].get("usage", {}), indent=2),
        "```",
        "",
        "## Summary (held-out test, bootstrap 95% CIs over seeds)",
        "",
        "```json",
        json.dumps(manifest["summary"], indent=2),
        "```",
        "",
        "## Paired comparisons vs normal_rl_cem",
        "",
        "```json",
        json.dumps(manifest.get("comparisons", {}), indent=2),
        "```",
        "",
        "## Videos",
        "",
    ]
    lines.extend(f"- `{v}`" for v in manifest.get("videos", []))
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())

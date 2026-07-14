#!/usr/bin/env python3
"""Bootstrap a primitive policy with LLM/residual supervision.

Loop per cycle and training setup:

1. Roll out the neural policy (template MLP or autoregressive sequence net).
2. While it is weak, request expert primitive schedules (LLM scratch mode) and train the
   policy to imitate the ones that beat the current best.
3. Once the policy beats the scratch baseline, switch the LLM to residual mode: it sees
   the policy's rollout trace and edits it rather than replacing it.
4. Save exact primitive instructions, traces, prompts, training data, and sampled videos.

The expert is pluggable via `--llm`: `claude-code` / `codex` / `anthropic` make real model
calls (prompt, completion, cost logged per call); `mock` / `none` use the deterministic
scripted expert and jitter-residual so the loop runs offline. Every supervision policy
records its source so heuristic data can never pass as LLM supervision.

`--policy-space template` keeps the original 12-parameter fixed schedule (lift tasks
only). `--policy-space sequence` searches variable-length primitive sequences and trains
an autoregressive policy net, so schedule structure is learned, not baked in.
"""
from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from llm_backend import LLMUsageTracker, call_llm, extract_json_policy
from policy_spaces import (
    PolicySpace,
    decode_template,
    encode_template,
    expert_for,
    get_space,
    jitter_residual,
    make_policy,
)
from sequence_policy import SeqPolicyNet, schedule_to_tokens, tokens_to_schedule
from shadow_inspect import render_policy_video
from shadow_policy_runner import runner_for_policy
from shadow_residual_prompt import build_residual_prompt
from tasks import OBS_DIM, Task, get_task, obs_from_setup


HIDDEN = 32
OUT_DIM = 12


# Backwards-compatible aliases (shadow_compare_training.py and archived runs import these).
def decode_policy(vec: np.ndarray, setup: dict[str, Any], name: str) -> dict[str, Any]:
    return decode_template(vec, setup, name)


encode_policy = encode_template


@dataclass
class Rollout:
    policy: dict[str, Any]
    result: dict[str, Any]
    score: float
    kind: str
    path: Path | None = None


class TinyPolicyNet:
    """Small NumPy MLP trained by behavior cloning on template schedule parameters."""

    def __init__(self, seed: int):
        rng = np.random.default_rng(seed)
        self.w1 = rng.normal(0.0, 0.15, (OBS_DIM, HIDDEN))
        self.b1 = np.zeros(HIDDEN)
        self.w2 = rng.normal(0.0, 0.15, (HIDDEN, OUT_DIM))
        self.b2 = np.zeros(OUT_DIM)

    def forward(self, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h = np.tanh(obs @ self.w1 + self.b1)
        y = h @ self.w2 + self.b2
        return y, h

    def predict(self, obs: np.ndarray, noise: float, rng: np.random.Generator) -> np.ndarray:
        y, _ = self.forward(obs)
        if noise > 0:
            y = y + rng.normal(0.0, noise, y.shape)
        return y

    def train_supervised(self, xs: np.ndarray, ys: np.ndarray, *, epochs: int, lr: float) -> list[float]:
        losses = []
        if len(xs) == 0:
            return losses
        for _ in range(epochs):
            yhat, h = self.forward(xs)
            diff = yhat - ys
            loss = float(np.mean(diff * diff))
            losses.append(loss)

            grad_y = (2.0 / len(xs)) * diff / OUT_DIM
            grad_w2 = h.T @ grad_y
            grad_b2 = grad_y.sum(axis=0)
            grad_h = grad_y @ self.w2.T
            grad_z1 = grad_h * (1.0 - h * h)
            grad_w1 = xs.T @ grad_z1
            grad_b1 = grad_z1.sum(axis=0)

            self.w2 -= lr * grad_w2
            self.b2 -= lr * grad_b2
            self.w1 -= lr * grad_w1
            self.b1 -= lr * grad_b1
        return losses

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, w1=self.w1, b1=self.b1, w2=self.w2, b2=self.b2)


class SupervisionDataset:
    """Accumulates accepted expert schedules in the active policy space's format."""

    def __init__(self, space: PolicySpace):
        self.space = space
        self.template_x: list[np.ndarray] = []
        self.template_y: list[np.ndarray] = []
        self.sequences: list[tuple[np.ndarray, np.ndarray]] = []

    def add(self, setup: dict[str, Any], policy: dict[str, Any]) -> bool:
        obs = obs_from_setup(setup)
        try:
            if self.space.name == "template":
                self.template_x.append(obs)
                self.template_y.append(self.space.encode(policy))
            else:
                self.sequences.append((obs, schedule_to_tokens(policy)))
        except (KeyError, IndexError, TypeError, ValueError):
            return False
        return True

    def __len__(self) -> int:
        return len(self.template_x) if self.space.name == "template" else len(self.sequences)

    def train(self, net, *, epochs: int, lr: float) -> list[float]:
        if len(self) == 0:
            return []
        if self.space.name == "template":
            return net.train_supervised(np.vstack(self.template_x), np.vstack(self.template_y), epochs=epochs, lr=lr)
        return net.train_supervised(self.sequences, epochs=epochs, lr=lr)

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            if self.space.name == "template":
                for x, y in zip(self.template_x, self.template_y):
                    f.write(json.dumps({"obs": x.tolist(), "target": y.tolist()}, separators=(",", ":")) + "\n")
            else:
                for obs, tokens in self.sequences:
                    f.write(json.dumps({"obs": obs.tolist(), "tokens": tokens.tolist()}, separators=(",", ":")) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("runs") / f"{time.strftime('%Y%m%d-%H%M%S')}-shadow_bootstrap")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--cycles", type=int, default=4)
    parser.add_argument("--task", choices=["lift", "lift_perturbed", "place", "push"], default="lift")
    parser.add_argument("--object-set", choices=["cube", "varied"], default="cube")
    parser.add_argument("--embodiment", choices=["shadow", "gripper"], default="shadow")
    parser.add_argument("--policy-space", choices=["template", "sequence"], default="template")
    parser.add_argument("--explore-rollouts", type=int, default=8, help="exploration rollouts per setup per cycle")
    parser.add_argument("--explore-noise", type=float, default=0.55)
    parser.add_argument("--bc-epochs", type=int, default=120)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--llm", choices=["mock", "claude-code", "codex", "anthropic", "none"], default="mock")
    parser.add_argument("--llm-model", help="model override for the LLM backend")
    parser.add_argument("--expert-mode", choices=["auto", "scratch", "residual"], default="auto",
                        help="ablation: force the expert request mode instead of switching automatically")
    parser.add_argument("--expert-attempts", type=int, default=2)
    parser.add_argument("--train-setups", type=int, default=4)
    parser.add_argument("--test-setups", type=int, default=6)
    parser.add_argument("--position-radius", type=float, default=0.025)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--sample-videos", type=int, default=4)
    parser.add_argument("--video-fps", type=int, default=20)
    args = parser.parse_args()

    task = get_task(args.task)
    space = get_space(args.policy_space)
    if space.name == "template" and task.name not in {"lift", "lift_perturbed"}:
        parser.error("--policy-space template only supports lift tasks; use --policy-space sequence")
    if space.name == "template" and args.embodiment != "shadow":
        parser.error("--policy-space template is Shadow-tuned; use --policy-space sequence for other embodiments")

    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    workers = args.workers or safe_worker_count()
    tracker = LLMUsageTracker()
    train_setups = [task.sample_setup(np.random.default_rng([args.seed, i]), args.position_radius, args.object_set) for i in range(args.train_setups)]
    test_setups = [task.sample_setup(np.random.default_rng([1000 + args.seed, i]), args.position_radius, args.object_set) for i in range(args.test_setups)]
    if args.embodiment != "shadow":
        for setup in train_setups + test_setups:
            setup["embodiment"] = args.embodiment
    manifest: dict[str, Any] = {
        "seed": args.seed,
        "cycles": args.cycles,
        "task": task.name,
        "object_set": args.object_set,
        "embodiment": args.embodiment,
        "policy_space": space.name,
        "expert_mode": args.expert_mode,
        "explore_rollouts": args.explore_rollouts,
        "workers": workers,
        "llm": {"backend": args.llm, "model": args.llm_model},
        "train_setups": train_setups,
        "test_setups": test_setups,
        "events": [],
    }

    net = TinyPolicyNet(args.seed) if space.name == "template" else SeqPolicyNet(args.seed)
    dataset = SupervisionDataset(space)
    originals: list[Rollout] = []
    bests: list[Rollout] = []
    best_robots: list[Rollout | None] = [None] * len(train_setups)

    for si, setup in enumerate(train_setups):
        original_expert = expert_for(space, task, f"scripted_{task.name}_t{si:02d}", setup)
        original_rollout = evaluate_policy(original_expert, kind="scripted_expert", task_name=task.name)
        save_rollout(out_dir / "rollouts" / f"original_scripted_t{si:02d}", original_rollout)
        dataset.add(setup, original_expert)
        originals.append(original_rollout)
        bests.append(original_rollout)
        manifest["events"].append({"setup": si, **event("original_scripted_expert", original_rollout)})

    for cycle in range(1, args.cycles + 1):
        cycle_dir = out_dir / f"cycle_{cycle:02d}"
        cycle_dir.mkdir(parents=True, exist_ok=True)
        any_accepted = False

        for si, setup in enumerate(train_setups):
            setup_dir = cycle_dir / f"setup_{si:02d}"
            jobs = []
            for i in range(args.explore_rollouts):
                policy = policy_from_net(net, space, task, setup, noise=args.explore_noise, rng=rng,
                                         name=f"robot_c{cycle:02d}_t{si:02d}_{i:02d}")
                jobs.append(("robot_policy", policy, task.name))

            robot_candidates = evaluate_many(jobs, workers=workers)
            robot_candidates.sort(key=lambda r: r.score, reverse=True)
            for i, rollout in enumerate(robot_candidates):
                save_rollout(setup_dir / "robot" / f"{i:02d}_{rollout.policy['name']}", rollout)

            if robot_candidates and (best_robots[si] is None or robot_candidates[0].score > best_robots[si].score):
                best_robots[si] = robot_candidates[0]
            if robot_candidates and robot_candidates[0].score > bests[si].score:
                bests[si] = robot_candidates[0]

            if args.expert_mode == "auto":
                mode = choose_expert_mode(best_robots[si], originals[si])
            else:
                mode = args.expert_mode
            expert_rollout: Rollout | None = None
            expert_policy: dict[str, Any] | None = None
            expert_source = None
            expert_attempts = []
            for attempt in range(1, max(1, args.expert_attempts) + 1):
                candidate_policy, source = request_expert_policy(
                    mode=mode,
                    llm=args.llm,
                    llm_model=args.llm_model,
                    tracker=tracker,
                    task=task,
                    space=space,
                    setup=setup,
                    reference=best_robots[si] or originals[si],
                    original=originals[si],
                    log_dir=setup_dir,
                    attempt=attempt,
                )
                candidate = evaluate_policy(candidate_policy, kind=f"llm_{mode}", task_name=task.name)
                save_rollout(setup_dir / f"llm_{mode}_attempt{attempt:02d}", candidate)
                expert_attempts.append({
                    "attempt": attempt,
                    "policy": candidate_policy["name"],
                    "score": candidate.score,
                    "source": source,
                })
                if expert_rollout is None or candidate.score > expert_rollout.score:
                    expert_rollout = candidate
                    expert_policy = candidate_policy
                    expert_source = source
                if candidate.score > bests[si].score + 0.03:
                    break
            assert expert_rollout is not None and expert_policy is not None

            accepted = False
            threshold = max(0.03, 0.05 * max(1.0, bests[si].score))
            if expert_rollout.score > bests[si].score + threshold:
                bests[si] = expert_rollout
                accepted = True
            elif best_robots[si] is None or expert_rollout.score >= best_robots[si].score + threshold:
                accepted = True
            encodable = dataset.add(setup, expert_policy) if accepted else None
            if accepted and encodable:
                any_accepted = True

            manifest["events"].append({
                "cycle": cycle,
                "setup": si,
                "mode": mode,
                "best_robot_score": robot_candidates[0].score if robot_candidates else None,
                "best_robot_policy": robot_candidates[0].policy["name"] if robot_candidates else None,
                "expert_score": expert_rollout.score,
                "expert_policy": expert_rollout.policy["name"],
                "expert_source": expert_source,
                "expert_attempts": expert_attempts,
                "accepted_as_supervision": accepted,
                "encodable": encodable,
                "best_score": bests[si].score,
            })

        epochs = args.bc_epochs if any_accepted else max(10, args.bc_epochs // 4)
        losses = dataset.train(net, epochs=epochs, lr=args.lr)
        net.save(cycle_dir / "policy_net.npz")
        manifest["events"].append({
            "cycle": cycle,
            "phase": "train",
            "dataset_size": len(dataset),
            "epochs": epochs,
            "loss_final": losses[-1] if losses else None,
        })

    # Held-out evaluation: greedy (noise-free) policy on fresh setups.
    heldout_rows = []
    for si, setup in enumerate(test_setups):
        policy = policy_from_net(net, space, task, setup, noise=0.0, rng=rng, name=f"net_heldout_{si:02d}")
        rollout = evaluate_policy(policy, kind="net_heldout", task_name=task.name)
        save_rollout(out_dir / "heldout" / f"setup_{si:02d}", rollout)
        heldout_rows.append({
            "setup": si,
            "object_pos": setup["object_pos"],
            "score": rollout.score,
            "success": bool(rollout.result.get("success")),
        })
    manifest["heldout"] = {
        "mean_score": round(float(np.mean([r["score"] for r in heldout_rows])), 4) if heldout_rows else None,
        "success_rate": round(float(np.mean([1.0 if r["success"] else 0.0 for r in heldout_rows])), 4) if heldout_rows else None,
        "rows": heldout_rows,
    }

    best_overall = max(bests, key=lambda r: r.score, default=None)
    if best_overall is not None:
        save_rollout(out_dir / "best", best_overall)
    dataset.write(out_dir / "supervision.jsonl")
    net.save(out_dir / "policy_net_final.npz")

    videos = sample_replay_videos(out_dir, best_path=(best_overall.path if best_overall else None), limit=args.sample_videos, fps=args.video_fps)
    manifest["sample_videos"] = [str(v) for v in videos]
    manifest["best"] = {
        "policy": best_overall.policy["name"] if best_overall else None,
        "score": best_overall.score if best_overall else None,
        "success": best_overall.result.get("success") if best_overall else None,
    }
    manifest["llm"]["usage"] = tracker.summary()
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (out_dir / "README.md").write_text(report_markdown(manifest))

    print(json.dumps({
        "out_dir": str(out_dir),
        "task": task.name,
        "policy_space": space.name,
        "best": manifest["best"],
        "heldout": {k: v for k, v in manifest["heldout"].items() if k != "rows"},
        "llm_usage": manifest["llm"]["usage"],
        "videos": manifest["sample_videos"],
    }, indent=2))
    return 0


def policy_from_net(
    net,
    space: PolicySpace,
    task: Task,
    setup: dict[str, Any],
    *,
    noise: float,
    rng: np.random.Generator,
    name: str,
) -> dict[str, Any]:
    obs = obs_from_setup(setup)
    if space.name == "template":
        vec = net.predict(obs, noise=noise, rng=rng)
        return space.decode(vec, setup, name, task)
    tokens = net.generate_tokens(obs, noise=noise, rng=rng)
    return tokens_to_schedule(tokens, setup, name=name, goal=task.goal, success=task.success_checks(setup))


def request_expert_policy(
    *,
    mode: str,
    llm: str,
    llm_model: str | None,
    tracker: LLMUsageTracker,
    task: Task,
    space: PolicySpace,
    setup: dict[str, Any],
    reference: Rollout,
    original: Rollout,
    log_dir: Path,
    attempt: int = 1,
) -> tuple[dict[str, Any], str]:
    """Return (policy, source). Source records whether a real LLM authored the policy or
    the deterministic local heuristic supplied it (mock backend or fallback after a failed
    call), so runs cannot silently pass off heuristic data as LLM supervision."""
    primitives = json.loads(Path("shadow_primitives.json").read_text())
    if mode == "scratch":
        prompt = build_residual_prompt(
            original.policy,
            original.result,
            primitives,
            f"The current robot policy appears weak or random. Task: {task.goal}. "
            "Provide a better primitive policy from scratch, in multiple explicit steps if needed.",
        )
    else:
        prompt = build_residual_prompt(
            reference.policy,
            reference.result,
            primitives,
            f"Act as a residual policy. Task: {task.goal}. Refine the best robot-policy rollout "
            "so the next rollout measurably improves and remains a clean imitation target.",
        )

    tag = f"{mode}_attempt{attempt:02d}"
    if llm in {"claude-code", "codex", "anthropic"}:
        response = call_llm(llm, prompt, model=llm_model, log_dir=log_dir, tag=tag)
        tracker.add(response, tag=tag)
        parsed = extract_json_policy(response.text) if response.ok else None
        if parsed is not None:
            parsed["setup"] = setup
            parsed["name"] = f"llm_{mode}_a{attempt:02d}_{int(time.time())}"
            # The LLM does not get to weaken its own acceptance criteria.
            parsed["success"] = task.success_checks(setup)
            return parsed, llm
        fallback_source = "mock_fallback"
    else:
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / f"{tag}_prompt.md").write_text(prompt)
        fallback_source = "mock"

    if mode == "residual":
        policy = jitter_residual(
            space, reference.policy, setup, task,
            name=f"mock_residual_a{attempt}_{int(time.time())}", attempt=attempt,
        )
    else:
        policy = expert_for(space, task, f"mock_scratch_a{attempt}_{int(time.time())}", setup)
    return policy, fallback_source


def choose_expert_mode(best_robot: Rollout | None, original: Rollout) -> str:
    if best_robot is None:
        return "scratch"
    # Better than random/contact-only: ask for residual refinements to the robot rollout.
    if best_robot.score >= max(0.6, original.score * 0.8):
        return "residual"
    return "scratch"


def evaluate_many(jobs: list[tuple[str, dict[str, Any], str]], workers: int) -> list[Rollout]:
    if workers <= 1 or len(jobs) <= 1:
        return [evaluate_policy(policy, kind=kind, task_name=task_name) for kind, policy, task_name in jobs]
    with futures.ProcessPoolExecutor(max_workers=workers) as pool:
        fs = [pool.submit(evaluate_policy, policy, kind, task_name) for kind, policy, task_name in jobs]
        return [f.result() for f in fs]


def evaluate_policy(policy: dict[str, Any], kind: str, task_name: str = "lift") -> Rollout:
    result = runner_for_policy(policy).run(policy)
    score = get_task(task_name).score(result, policy.get("setup", {}))
    return Rollout(policy=policy, result=result, score=score, kind=kind)


def score_result(result: dict[str, Any]) -> float:
    """Backwards-compatible lift scoring used by archived comparison scripts."""
    return get_task("lift").score(result, result.get("setup", {}) or {"object_pos": [0, 0, 0.025]})


def save_rollout(path: Path, rollout: Rollout) -> None:
    path.mkdir(parents=True, exist_ok=True)
    rollout.path = path
    (path / "policy.json").write_text(json.dumps(rollout.policy, indent=2) + "\n")
    data = dict(rollout.result)
    data["score"] = rollout.score
    data["kind"] = rollout.kind
    (path / "result.json").write_text(json.dumps(data, indent=2) + "\n")
    (path / "control_sequence.md").write_text(control_sequence_md(rollout))


def control_sequence_md(rollout: Rollout) -> str:
    lines = [
        f"# {rollout.policy['name']}",
        "",
        f"- Kind: `{rollout.kind}`",
        f"- Score: `{rollout.score}`",
        f"- Success: `{rollout.result.get('success')}`",
        "",
        "| Step | Primitive | Params | Duration | Until |",
        "| ---: | --- | --- | ---: | --- |",
    ]
    for i, step in enumerate(rollout.policy.get("steps", []), 1):
        until = json.dumps(step["until"], separators=(",", ":")) if step.get("until") else ""
        lines.append(
            f"| {i} | `{step['primitive']}` | `{json.dumps(step.get('params', {}), separators=(',', ':'))}` "
            f"| {step['duration_s']}s | `{until}` |"
        )
    return "\n".join(lines) + "\n"


def event(name: str, rollout: Rollout) -> dict[str, Any]:
    return {
        "name": name,
        "policy": rollout.policy["name"],
        "kind": rollout.kind,
        "score": rollout.score,
        "success": rollout.result.get("success"),
        "final_state": rollout.result.get("final_state"),
    }


def sample_replay_videos(out_dir: Path, *, best_path: Path | None, limit: int, fps: int) -> list[Path]:
    if limit <= 0:
        return []
    candidates = sorted(out_dir.glob("**/result.json"), key=lambda p: p.stat().st_mtime)
    if best_path is not None:
        best_result = best_path / "result.json"
        candidates = [best_result] + [p for p in candidates if p != best_result]
    selected = []
    for result_file in candidates:
        if len(selected) >= limit:
            break
        policy_file = result_file.parent / "policy.json"
        if not policy_file.exists():
            continue
        policy = json.loads(policy_file.read_text())
        video = result_file.parent / "replay.mp4"
        try:
            render_policy_video(policy, runner_for_policy(policy), video, fps=fps, width=640, height=480)
        except Exception as exc:  # noqa: BLE001
            (result_file.parent / "replay_error.txt").write_text(str(exc) + "\n")
            continue
        selected.append(video)
    return selected


def safe_worker_count() -> int:
    cpus = os.cpu_count() or 2
    available_kb = mem_available_kb()
    # MuJoCo workers are not huge here, but leave headroom because this machine is already
    # memory loaded. Rough cap: one worker per 750 MiB available, never more than cpus-2.
    mem_workers = max(1, int(available_kb / (750 * 1024))) if available_kb else 2
    return max(1, min(max(1, cpus - 2), mem_workers, 8))


def mem_available_kb() -> int:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1])
    except OSError:
        return 0
    return 0


def report_markdown(manifest: dict[str, Any]) -> str:
    lines = [
        "# Shadow Bootstrap Experiment",
        "",
        f"- Task: `{manifest['task']}` (objects: `{manifest['object_set']}`)",
        f"- Policy space: `{manifest['policy_space']}`",
        f"- Expert mode: `{manifest['expert_mode']}`",
        f"- LLM backend: `{manifest['llm']['backend']}` (model: `{manifest['llm'].get('model')}`)",
        f"- Workers: `{manifest['workers']}`",
        f"- Cycles: `{manifest['cycles']}`",
        f"- Explore rollouts/setup/cycle: `{manifest['explore_rollouts']}`",
        f"- Train setups: `{len(manifest.get('train_setups', []))}`, held-out test setups: `{len(manifest.get('test_setups', []))}`",
        "",
        "## LLM usage",
        "",
        "```json",
        json.dumps(manifest["llm"].get("usage", {}), indent=2),
        "```",
        "",
        "## Held-out evaluation",
        "",
        "```json",
        json.dumps({k: v for k, v in manifest.get("heldout", {}).items() if k != "rows"}, indent=2),
        "```",
        "",
        "## Best",
        "",
        "```json",
        json.dumps(manifest.get("best", {}), indent=2),
        "```",
        "",
        "## Events",
        "",
        "```json",
        json.dumps(manifest.get("events", []), indent=2),
        "```",
        "",
        "## Sample Videos",
        "",
    ]
    lines.extend(f"- `{path}`" for path in manifest.get("sample_videos", []))
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())

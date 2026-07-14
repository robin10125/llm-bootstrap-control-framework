"""The code-as-action control loop.

A goal is handled by `run_node`: generate a response for it, which is either a LEAF
SCRIPT (executed via the bridge) or a PLAN (an ordered list of subgoals, each handled by
`run_node` again — recursion is the `achieve()` mechanism). After acting, the node's own
`check` is evaluated; on failure the node repairs up to max_iterations. This loop is
domain-agnostic: it talks to a Bridge, a SuccessCheck, and an llm with `generate`/`judge`.
This file is byte-for-byte the same idea as the Minecraft controller — swapping Minecraft
for the MuJoCo hand meant swapping the bridge and the prompt's primitives, not this loop.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

from . import prompts
from .bridge import Bridge
from .llm import extract_envelope
from .skills import Skill, SkillLibrary
from .tasks import SuccessCheck, Task


@dataclass
class NodeResult:
    goal: str
    success: bool
    iterations: int
    obs: dict           # observation after this node finished
    summary: str        # short human-readable outcome


@dataclass
class EpisodeResult:
    task: str
    success: bool
    iterations: int
    run_dir: str


def _slug(text: str, maxlen: int = 40) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return (s[:maxlen] or "goal")


class Controller:
    def __init__(self, bridge: Bridge, library: SkillLibrary, llm,
                 runs_root: str | Path, gen_mode: str = "scratch",
                 max_depth: int = 3, exec_timeout_ms: int = 0,
                 record: bool = True):
        self.bridge = bridge
        self.library = library
        self.llm = llm
        self.runs_root = Path(runs_root)
        self.gen_mode = gen_mode
        self.max_depth = max_depth
        self.exec_timeout_ms = exec_timeout_ms
        self.record = record
        self._run_dir: Path | None = None  # set per episode, used by nested nodes

    # -- public entry ---------------------------------------------------------
    def run_episode(self, task: Task, exec_timeout_ms: int | None = None) -> EpisodeResult:
        if exec_timeout_ms is not None:
            self.exec_timeout_ms = exec_timeout_ms
        run_dir = self.runs_root / f"{time.strftime('%Y%m%d-%H%M%S')}-{task.name}"
        run_dir.mkdir(parents=True, exist_ok=True)
        self._run_dir = run_dir

        self.bridge.wait_until_spawned()
        self.bridge.reset(**task.setup)
        if self.record:
            self.bridge.enable_recording()
        obs = self.bridge.observe()

        node = self.run_node(task.goal, task.success, obs, depth=0, path="0",
                             save_as=task.name, max_iterations=task.max_iterations)
        if self.record:
            n = self.bridge.save_trajectory(run_dir / "trajectory.npz")
            print(f"recorded {n} physics steps -> {run_dir / 'trajectory.npz'}")
        return EpisodeResult(task.name, node.success, node.iterations, str(run_dir))

    # -- the recursive unit ---------------------------------------------------
    def run_node(self, goal: str, check: SuccessCheck, obs: dict, depth: int, path: str,
                 save_as: str | None = None, max_iterations: int = 5) -> NodeResult:
        """Achieve `goal` from `obs`, returning when `check` passes or attempts run out."""
        indent = "  " * depth
        obs_before = obs
        can_decompose = depth < self.max_depth
        starting_skill = None
        if self.gen_mode == "edit-nearest":
            hits = self.library.retrieve(goal, k=1)
            starting_skill = hits[0] if hits else None

        last = None
        for i in range(1, max_iterations + 1):
            user = prompts.build_user_prompt(
                goal, obs, gen_mode=self.gen_mode, starting_skill=starting_skill,
                can_decompose=can_decompose, last_attempt=last)
            raw = self.llm.generate(prompts.SYSTEM, user)
            env = extract_envelope(raw)
            kind = env["type"]
            print(f"{indent}[{path} iter {i}/{max_iterations}] goal={goal!r} -> {kind}")

            if kind == "plan" and can_decompose:
                exec_log, child_fail = self._run_plan(env["subgoals"], obs, depth, path)
                obs = exec_log["obs"]
            else:
                # leaf (or a plan returned at the depth limit — treat its code, if any)
                code = env.get("code", "")
                feedback = self.bridge.execute(code, timeout_ms=self.exec_timeout_ms)
                obs = feedback.get("observationAfter", obs)
                exec_log = {"kind": "script", "code": code, "feedback": feedback, "obs": obs}
                child_fail = None if feedback.get("success") else "leaf script errored"

            passed, unmet = check.evaluate(obs, obs_before=obs_before, judge=self._judge)
            self._log(path, i, goal, user, exec_log, passed, unmet, depth)
            print(f"{indent}           goal {'MET' if passed else 'unmet'}"
                  + ("" if passed else f" ({'; '.join(unmet) or child_fail or '...'})"))

            if passed:
                if exec_log["kind"] == "script" and exec_log.get("code"):
                    self._save_skill(save_as or _slug(goal), goal, exec_log["code"])
                return NodeResult(goal, True, i, obs, _summary(exec_log))

            last = self._feedback_for_repair(exec_log, unmet, child_fail)

        return NodeResult(goal, False, max_iterations, obs, f"failed: {goal}")

    # -- helpers --------------------------------------------------------------
    def _run_plan(self, subgoals: list[dict], obs: dict, depth: int, path: str):
        """Run subgoals in order; stop at the first failure and report it upward."""
        results = []
        child_fail = None
        for j, sg in enumerate(subgoals, 1):
            child_check = SuccessCheck.from_dict(sg.get("check"))
            child = self.run_node(sg["goal"], child_check, obs, depth=depth + 1,
                                  path=f"{path}.{j}", max_iterations=5)
            obs = child.obs
            results.append({"goal": sg["goal"], "success": child.success,
                            "summary": child.summary})
            if not child.success:
                child_fail = f"subgoal {j} failed: {sg['goal']}"
                break
        return {"kind": "plan", "subgoals": results, "obs": obs}, child_fail

    def _judge(self, criterion: str, obs_before: dict, obs_after: dict):
        judge = getattr(self.llm, "judge", None)
        if judge is None:
            return (False, "no judge available")
        return judge(criterion, obs_before, obs_after)

    def _feedback_for_repair(self, exec_log: dict, unmet: list[str], child_fail) -> dict:
        if exec_log["kind"] == "plan":
            done = [r["goal"] for r in exec_log["subgoals"] if r["success"]]
            return {"success": False, "logs": [f"completed subgoals: {done}"] if done else [],
                    "unmet": unmet, "planError": child_fail}
        fb = dict(exec_log["feedback"])
        fb["unmet"] = unmet
        return fb

    def _log(self, path, i, goal, user, exec_log, passed, unmet, depth):
        if self._run_dir is None:
            return
        kind = exec_log["kind"]
        rec = {"path": path, "iteration": i, "depth": depth, "goal": goal,
               "kind": kind, "passed": passed, "unmet": unmet, "prompt": user}
        if kind == "plan":
            rec["subgoals"] = exec_log["subgoals"]
        else:
            rec["skill"] = exec_log.get("code")
            rec["feedback"] = exec_log.get("feedback")
        (self._run_dir / f"n{path.replace('.', '-')}_iter{i:02d}.json").write_text(
            json.dumps(rec, indent=2))

    def _save_skill(self, name: str, description: str, code: str) -> None:
        existing = self.library.get(name)
        skill = Skill(
            name=name, description=description, code=code, task=name,
            uses=(existing.uses + 1 if existing else 1),
            successes=(existing.successes + 1 if existing else 1),
        )
        self.library.save(skill)
        print(f"           saved skill {name!r} to library")


def _summary(exec_log: dict) -> str:
    if exec_log["kind"] == "plan":
        return "plan: " + ", ".join(r["goal"] for r in exec_log["subgoals"] if r["success"])
    fb = exec_log.get("feedback", {})
    return str(fb.get("result") or (fb.get("logs") or ["ok"])[-1])

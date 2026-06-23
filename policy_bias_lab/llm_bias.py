from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

from policy_bias_lab.schema import ACTION_GROUPS, EVAL_FIELDS, PRIOR_DIRECTIONS
from policy_bias_lab.schema import default_bias_spec, validate_bias_spec


ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAPPING = ROOT.parent / "bootstrapping"


def build_bias_prompt(task: str, tasks: list[str], env_summary: dict[str, Any]) -> str:
    return (
        "You are authoring inductive biases for a closed-loop robot policy model.\n"
        "The policy is trained by actor-critic PPO and maps observation vectors directly "
        "to actuator actions. Do not emit actions, scripts, primitives, or Python. Emit only "
        "a structured JSON BiasSpec that deterministic code will compile into reward shaping, "
        "action priors, exploration scaling, and supervised initialization targets.\n\n"
        "Use only the task description, robot/environment summary, observable schema, and "
        "allowed action abstractions injected below. Do not assume fixed start states, fixed "
        "coordinates, hidden reward terms, or task-specific facts not present in the injected "
        "context. If an allowed direction has runtime analytic semantics, treat it as a symbolic "
        "operator over current observations rather than a waypoint.\n\n"
        "Before proposing biases, reason about likely failure modes, reward-hacking routes, "
        "ways to avoid those problems, and the optimal kinematic forms or approaches implied "
        "by the task and robot. Reward shaping must remain auxiliary and bounded: it should "
        "guide exploration into task-valid basins without replacing held-out environment success.\n\n"
        f"Use only these eval observables: {json.dumps(EVAL_FIELDS)}.\n"
        f"Use only these action groups: {json.dumps(ACTION_GROUPS)}.\n"
        f"Use only these action directions: {json.dumps(PRIOR_DIRECTIONS)}.\n\n"
        "Return ONLY JSON with fields: name, description, reasoning, reward_terms, action_priors, "
        "exploration_groups, supervised_targets, curriculum.\n"
        "reasoning should include failure_modes, reward_hacking_risks, avoidance_plan, and "
        "optimal_kinematic_approach.\n"
        "Each reward term: name, observable, direction=minimize|maximize, weight, scale, "
        "optional tasks=[...]. The compiler will bound these as potential-based progress "
        "shaping and discard task-contradictory or overlapping terms where it can detect them.\n"
        "Each prior/target: name, group, direction, weight.\n"
        "Each exploration group: group, scale.\n\n"
        f"Primary task: {task}\n"
        f"Task suite: {json.dumps(tasks)}\n"
        f"Environment summary: {json.dumps(env_summary, indent=2)}\n"
    )


def load_bias_spec(
    *,
    backend: str,
    model: str | None,
    task: str,
    tasks: list[str],
    env_summary: dict[str, Any],
    log_dir: Path,
) -> dict[str, Any]:
    if backend == "fixture":
        spec = default_bias_spec(task)
    else:
        prompt = build_bias_prompt(task, tasks, env_summary)
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / "bias_prompt.md").write_text(prompt)
        result = _call_llm(backend, prompt, model=model, log_dir=log_dir)
        (log_dir / "bias_completion.txt").write_text(result)
        spec = _parse_json(result)
    validation = validate_bias_spec(spec)
    if not validation.ok:
        raise ValueError("invalid LLM bias spec: " + "; ".join(validation.errors))
    return spec


def _call_llm(backend: str, prompt: str, *, model: str | None, log_dir: Path) -> str:
    path = str(BOOTSTRAPPING)
    if path not in sys.path:
        sys.path.insert(0, path)
    from llm_backend import call_llm

    response = call_llm(backend, prompt, model=model, log_dir=log_dir, tag="bias_spec")
    if not response.ok:
        raise RuntimeError(response.error or "LLM call failed")
    return response.text


def _parse_json(text: str) -> dict[str, Any]:
    candidates = [m.group(1) for m in re.finditer(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S)]
    m = re.search(r"\{.*\}", text, flags=re.S)
    if m:
        candidates.append(m.group(0))
    for raw in candidates:
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    raise ValueError("no JSON object found in LLM response")

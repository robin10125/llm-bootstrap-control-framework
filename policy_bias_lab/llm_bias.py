from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

from policy_bias_lab.schema import default_bias_spec, validate_bias_spec


ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAPPING = ROOT.parent / "bootstrapping"


def build_bias_prompt(task: str, tasks: list[str], env_summary: dict[str, Any]) -> str:
    return (
        "You are authoring inductive biases for a closed-loop Shadow Hand policy model.\n"
        "The policy is trained by evolution strategies and maps observation vectors directly "
        "to actuator actions. Do not emit actions, scripts, primitives, or Python. Emit only "
        "a structured JSON BiasSpec that deterministic code will compile into reward shaping, "
        "action priors, exploration scaling, and supervised initialization targets.\n\n"
        "Use only these eval observables: palm_obj_dist, min_finger_dist, n_contacts, "
        "closure, lift, obj_xy_disp.\n"
        "Use only these action groups: base_xy, base_z, hand, thumb, index, middle, ring, "
        "little, all.\n"
        "Use only these action directions: toward_object_xy, lower_base, raise_base, "
        "close_hand, open_hand, stabilize.\n\n"
        "Return ONLY JSON with fields: name, description, reward_terms, action_priors, "
        "exploration_groups, supervised_targets, curriculum.\n"
        "Each reward term: name, observable, direction=minimize|maximize, weight, scale.\n"
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

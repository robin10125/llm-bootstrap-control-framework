"""Small reused LLM helpers, kept out of the quarantined action_priors module.

- call_llm: invoke the project LLM backend (codex->claude-code fallback) for a prompt.
- candidate_score: the contact-gated, real-world-observable objective used to rank action-prior
  candidates (rewards sustained contact-grasp-lift, penalizes flinging).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any



def call_llm(backend: str, prompt: str, *, model: str | None, log_dir: Path, tag: str) -> str:
    from experiment_runtime.llm_backend import call_llm as _backend_call

    response = _backend_call(backend, prompt, model=model, timeout_s=600.0, log_dir=log_dir, tag=tag)
    return response.text if response.ok else ""


def candidate_score(score: dict[str, Any]) -> float:
    """Contact-gated candidate objective from real-world-observable signals (flinging penalized)."""
    contact_gated = float(score.get("contact_gated_success", 0.0))
    contact_lift = float(score.get("contact_conditioned_lift", 0.0))
    engagement = float(score.get("contact_engagement", 0.0))
    contacts_mean = float(score.get("contacts_mean", 0.0))
    fling = float(score.get("fling_fraction", 0.0))
    palm_dist = float(score.get("palm_obj_dist_min", 0.0))
    saturation = float(score.get("saturation_frac", 0.0))
    action_abs = float(score.get("action_abs_mean", 0.0))
    return (
        120.0 * contact_gated + 30.0 * contact_lift + 6.0 * engagement + 1.0 * contacts_mean
        - 25.0 * fling - 4.0 * palm_dist - 6.0 * saturation - 0.5 * max(action_abs - 0.55, 0.0)
    )

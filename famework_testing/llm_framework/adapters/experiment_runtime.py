"""Adapters for the repository-local experiment runtime."""
from __future__ import annotations

from pathlib import Path
from typing import Any


def make_runtime_env(name: str = "shadow", **overrides: Any):
    if name != "shadow":
        raise ValueError("llm-framework only supports the Shadow Hand embodiment; use name='shadow'")
    from experiment_runtime.environment import make_env

    return make_env("shadow", **overrides)


def call_runtime_llm(
    backend: str,
    prompt: str,
    *,
    model: str | None = None,
    log_dir: Path | None = None,
    tag: str = "call",
):
    from experiment_runtime.llm_backend import call_llm

    return call_llm(backend, prompt, model=model, log_dir=log_dir, tag=tag)


def waypoint_compiler(env: Any):
    from experiment_runtime.waypoints import WaypointCompiler

    return WaypointCompiler(env)

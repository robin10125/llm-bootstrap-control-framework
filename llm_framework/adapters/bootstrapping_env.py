from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
BOOTSTRAPPING = ROOT.parent / "bootstrapping"


def ensure_bootstrapping_path() -> None:
    path = str(BOOTSTRAPPING)
    if path not in sys.path:
        sys.path.insert(0, path)


def make_bootstrapping_env(name: str = "shadow", **overrides: Any):
    ensure_bootstrapping_path()
    from mjx_env import make_env

    return make_env(name, **overrides)


def call_bootstrapping_llm(
    backend: str,
    prompt: str,
    *,
    model: str | None = None,
    log_dir: Path | None = None,
    tag: str = "call",
):
    ensure_bootstrapping_path()
    from llm_backend import call_llm

    return call_llm(backend, prompt, model=model, log_dir=log_dir, tag=tag)


def waypoint_compiler(env: Any):
    ensure_bootstrapping_path()
    from waypoints import WaypointCompiler

    return WaypointCompiler(env)


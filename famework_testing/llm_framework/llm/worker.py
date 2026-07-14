from __future__ import annotations

from pathlib import Path

from llm_framework.adapters.experiment_runtime import call_runtime_llm
from llm_framework.core.state import WorldState
from llm_framework.core.tasks import TaskContext
from llm_framework.interfaces.base import ControlInterface
from llm_framework.llm.types import LLMCallResult


def complete_for_interface(
    interface: ControlInterface,
    ctx: TaskContext,
    state: WorldState,
    *,
    backend: str,
    model: str | None = None,
    log_dir: Path | None = None,
    tag: str = "call",
) -> LLMCallResult:
    complete = getattr(interface, "complete_with_llm", None)
    if complete is not None:
        return complete(ctx, state, backend=backend, model=model, log_dir=log_dir, tag=tag)
    if backend == "mock":
        return LLMCallResult(
            text=interface.mock_response(ctx, state),
            ok=True,
            error=None,
            source="mock",
        )
    prompt = interface.build_prompt(ctx, state)
    response = call_runtime_llm(backend, prompt, model=model, log_dir=log_dir, tag=tag)
    return LLMCallResult(
        text=response.text,
        ok=response.ok,
        error=response.error,
        source=response.source,
    )

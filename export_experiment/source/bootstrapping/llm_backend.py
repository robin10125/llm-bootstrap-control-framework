#!/usr/bin/env python3
"""LLM backends for primitive-policy supervision.

Backends:

- `claude-code`: headless Claude Code (`claude -p --output-format json`). Bills through
  the local Claude login, reports cost and token usage per call.
- `codex`: headless Codex (`codex exec`). Final message captured via a temp file.
- `anthropic`: raw Anthropic SDK, needs `ANTHROPIC_API_KEY`.
- `mock`: never calls out; callers supply their own deterministic policy instead.

Every call can log prompt, raw completion, and metadata to a directory so completed
runs carry the full supervision record.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

BACKENDS = ("mock", "claude-code", "codex", "anthropic", "none")
DEFAULT_TIMEOUT_S = 360.0

# Substrings that indicate a provider declined the call because a usage/rate limit
# (rather than a genuine prompt or transport error) was reached. Matched case-insensitively
# against the recorded error so codex can transparently fall back to another reasoner.
RATE_LIMIT_MARKERS = (
    "rate limit",
    "ratelimit",
    "rate_limit",
    "429",
    "too many requests",
    "usage limit",
    "quota",
    "resource_exhausted",
    "exhausted your",
    "exceeded",
    "overloaded",
)


def _looks_rate_limited(response: "LLMResponse") -> bool:
    blob = " ".join(str(part) for part in (response.error, response.text) if part).lower()
    return any(marker in blob for marker in RATE_LIMIT_MARKERS)


@dataclass
class LLMResponse:
    text: str
    source: str
    model: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    cost_usd: float | None = None
    duration_s: float | None = None
    error: str | None = None
    notes: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.text.strip())


class LLMUsageTracker:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def add(self, response: LLMResponse, *, tag: str = "") -> None:
        self.calls.append({
            "tag": tag,
            "source": response.source,
            "model": response.model,
            "usage": response.usage,
            "cost_usd": response.cost_usd,
            "duration_s": response.duration_s,
            "error": response.error,
        })

    def summary(self) -> dict[str, Any]:
        costs = [c["cost_usd"] for c in self.calls if c["cost_usd"] is not None]
        errors = [c for c in self.calls if c["error"]]
        return {
            "calls": len(self.calls),
            "errors": len(errors),
            "total_cost_usd": round(sum(costs), 4) if costs else None,
            "input_tokens": _sum_usage(self.calls, ("input_tokens",)),
            "output_tokens": _sum_usage(self.calls, ("output_tokens",)),
        }


def _sum_usage(calls: list[dict[str, Any]], keys: tuple[str, ...]) -> int | None:
    total = 0
    seen = False
    for call in calls:
        usage = call.get("usage") or {}
        for key in keys:
            if key in usage and isinstance(usage[key], (int, float)):
                total += int(usage[key])
                seen = True
    return total if seen else None


def call_llm(
    backend: str,
    prompt: str,
    *,
    model: str | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    log_dir: Path | None = None,
    tag: str = "call",
) -> LLMResponse:
    start = time.monotonic()
    if backend == "claude-code":
        response = _call_claude_code(prompt, model=model, timeout_s=timeout_s)
    elif backend == "codex":
        response = _call_codex(prompt, model=model, timeout_s=timeout_s)
        if not response.ok and _looks_rate_limited(response):
            # Codex hit a usage/rate limit: transparently retry on claude-code so long
            # runs are not derailed. Use claude-code's default model, not the codex id.
            primary_error = response.error
            fallback = _call_claude_code(prompt, model=None, timeout_s=timeout_s)
            fallback.notes = f"codex_rate_limited_fell_back_to_claude_code; codex_error={primary_error}"
            response = fallback
    elif backend == "anthropic":
        response = _call_anthropic(prompt, model=model, timeout_s=timeout_s)
    else:
        response = LLMResponse(text="", source=backend, error=f"backend {backend!r} makes no calls")
    response.duration_s = round(time.monotonic() - start, 2)
    if log_dir is not None:
        _log_call(log_dir, tag, prompt, response)
    return response


def _call_claude_code(prompt: str, *, model: str | None, timeout_s: float) -> LLMResponse:
    cmd = ["claude", "-p", "--output-format", "json", "--no-session-persistence"]
    if model:
        cmd += ["--model", model]
    try:
        proc = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=timeout_s)
    except (subprocess.TimeoutExpired, OSError) as exc:
        return LLMResponse(text="", source="claude-code", model=model, error=f"claude subprocess failed: {exc}")
    if proc.returncode != 0:
        return LLMResponse(
            text="", source="claude-code", model=model,
            error=f"claude exited {proc.returncode}: {proc.stderr.strip()[:500]}",
        )
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        return LLMResponse(text=proc.stdout, source="claude-code", model=model, error=f"unparseable claude output: {exc}")
    if payload.get("is_error"):
        return LLMResponse(text="", source="claude-code", model=model, error=str(payload.get("result", "unknown claude error"))[:500])
    # modelUsage lists every model the CLI touched (utility/subagent models included) in arbitrary
    # order; taking the first key mislabeled fable-5 calls as haiku. Prefer the requested model
    # when it appears, else the heaviest-usage entry.
    mu = payload.get("modelUsage") or {}
    if model and model in mu:
        used = model
    elif mu:
        used = max(mu, key=lambda k: (mu[k] or {}).get("outputTokens", 0) if isinstance(mu[k], dict) else 0)
    else:
        used = model
    return LLMResponse(
        text=str(payload.get("result", "")),
        source="claude-code",
        model=used,
        usage=payload.get("usage", {}),
        cost_usd=payload.get("total_cost_usd"),
    )


def _call_codex(prompt: str, *, model: str | None, timeout_s: float) -> LLMResponse:
    with tempfile.NamedTemporaryFile(mode="r", suffix=".txt", delete=False) as f:
        out_path = Path(f.name)
    cmd = ["codex", "exec", "--skip-git-repo-check", "-o", str(out_path)]
    if model:
        cmd += ["-c", f'model="{model}"']
    cmd.append("-")
    try:
        proc = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=timeout_s)
        text = out_path.read_text() if out_path.exists() else ""
    except (subprocess.TimeoutExpired, OSError) as exc:
        return LLMResponse(text="", source="codex", model=model, error=f"codex subprocess failed: {exc}")
    finally:
        out_path.unlink(missing_ok=True)
    if proc.returncode != 0:
        # Codex prints a startup banner first and the actual failure (e.g. "usage limit")
        # last, so surface ERROR lines if present and otherwise the *tail* of stderr — taking
        # the head would truncate the real reason and defeat rate-limit fallback detection.
        stderr = proc.stderr.strip()
        error_lines = [ln for ln in stderr.splitlines() if "error" in ln.lower()]
        detail = " | ".join(error_lines)[-800:] if error_lines else stderr[-800:]
        return LLMResponse(text="", source="codex", model=model, error=f"codex exited {proc.returncode}: {detail}")
    return LLMResponse(text=text, source="codex", model=model)


def _call_anthropic(prompt: str, *, model: str | None, timeout_s: float) -> LLMResponse:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return LLMResponse(text="", source="anthropic", model=model, error="ANTHROPIC_API_KEY not set")
    try:
        import anthropic
    except ImportError as exc:
        return LLMResponse(text="", source="anthropic", model=model, error=f"anthropic SDK unavailable: {exc}")
    resolved = model or os.environ.get("SHADOW_BOOTSTRAP_MODEL", "claude-opus-4-8")
    try:
        client = anthropic.Anthropic(timeout=timeout_s)
        resp = client.messages.create(
            model=resolved,
            max_tokens=2500,
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception as exc:  # noqa: BLE001 - any SDK failure becomes a recorded error
        return LLMResponse(text="", source="anthropic", model=resolved, error=str(exc)[:500])
    text = "".join(block.text for block in resp.content if block.type == "text")
    usage = {"input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens}
    return LLMResponse(text=text, source="anthropic", model=resolved, usage=usage)


def extract_json_policy(text: str) -> dict[str, Any] | None:
    """Pull the first plausible primitive-policy JSON object out of a completion."""
    candidates: list[str] = []
    for match in re.finditer(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S):
        candidates.append(match.group(1))
    match = re.search(r"\{.*\}", text, flags=re.S)
    if match:
        candidates.append(match.group(0))
    for raw in candidates:
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and isinstance(obj.get("steps"), list):
            return obj
    return None


def _log_call(log_dir: Path, tag: str, prompt: str, response: LLMResponse) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / f"{tag}_prompt.md").write_text(prompt)
    (log_dir / f"{tag}_completion.txt").write_text(response.text)
    meta = {
        "source": response.source,
        "model": response.model,
        "usage": response.usage,
        "cost_usd": response.cost_usd,
        "duration_s": response.duration_s,
        "error": response.error,
        "notes": response.notes,
    }
    (log_dir / f"{tag}_meta.json").write_text(json.dumps(meta, indent=2) + "\n")

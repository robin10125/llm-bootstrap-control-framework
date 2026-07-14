from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class LLMCallResult:
    text: str
    ok: bool
    error: str | None
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)


from __future__ import annotations

import json
import re
from typing import Any

from llm_framework.core.state import CandidateProgram


class ParseError(ValueError):
    pass


def parse_json_program(text: str, *, interface: str) -> CandidateProgram:
    for raw in _json_candidates(text):
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return CandidateProgram(interface=interface, source=obj, raw_text=text)
    raise ParseError("no JSON object found")


def _json_candidates(text: str) -> list[str]:
    candidates = []
    for match in re.finditer(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S):
        candidates.append(match.group(1))
    match = re.search(r"\{.*\}", text, flags=re.S)
    if match:
        candidates.append(match.group(0))
    return candidates


def require_blocks(program: dict[str, Any]) -> list[dict[str, Any]]:
    blocks = program.get("blocks")
    if not isinstance(blocks, list) or not blocks:
        raise ParseError("program must contain a non-empty 'blocks' list")
    for block in blocks:
        if not isinstance(block, dict):
            raise ParseError("each block must be an object")
    return blocks


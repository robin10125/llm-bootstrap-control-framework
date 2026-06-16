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
        except json.JSONDecodeError as exc:
            obj = _repair_single_extra_close_before_field(raw, exc)
            if obj is None:
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


def _repair_single_extra_close_before_field(raw: str, exc: json.JSONDecodeError) -> Any | None:
    if exc.msg != "Extra data":
        return None
    pos = exc.pos
    if pos <= 0 or pos >= len(raw):
        return None
    if raw[pos] != "," or raw[pos - 1] != "}":
        return None
    try:
        return json.loads(raw[: pos - 1] + raw[pos:])
    except json.JSONDecodeError:
        return None


def require_blocks(program: dict[str, Any]) -> list[dict[str, Any]]:
    blocks = program.get("blocks")
    if not isinstance(blocks, list) or not blocks:
        raise ParseError("program must contain a non-empty 'blocks' list")
    for block in blocks:
        if not isinstance(block, dict):
            raise ParseError("each block must be an object")
    return blocks

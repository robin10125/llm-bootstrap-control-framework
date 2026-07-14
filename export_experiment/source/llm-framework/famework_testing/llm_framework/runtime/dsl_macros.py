from __future__ import annotations

from copy import deepcopy
from typing import Any


def expand_generated_primitives(source: dict[str, Any]) -> dict[str, Any]:
    """Expand LLM-defined primitive macros into executable blocks.

    The LLM may invent any primitive names it wants under `generated_primitives`.
    The safety boundary is that macro bodies must eventually expand to known executable
    blocks; undefined invented ops still fail validation.
    """
    source = dict(source)
    definitions = _definitions(source.get("generated_primitives", {}))
    if not definitions:
        return source
    source["blocks"] = _expand_blocks(source.get("blocks", []), definitions, stack=())
    source["expanded_generated_primitives"] = sorted(definitions)
    return source


def _definitions(raw: Any) -> dict[str, dict[str, Any]]:
    if isinstance(raw, dict):
        return {str(name): body for name, body in raw.items() if isinstance(body, dict)}
    if isinstance(raw, list):
        out = {}
        for item in raw:
            if isinstance(item, dict) and isinstance(item.get("name"), str):
                out[item["name"]] = item
        return out
    return {}


def _expand_blocks(blocks: list[dict[str, Any]], definitions: dict[str, dict[str, Any]], stack: tuple[str, ...]) -> list[dict[str, Any]]:
    expanded: list[dict[str, Any]] = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        op = str(block.get("op", ""))
        macro_name = str(block.get("primitive", op)) if op == "call_generated_primitive" else op
        if macro_name not in definitions:
            expanded.append(block)
            continue
        if macro_name in stack:
            expanded.append({"op": "macro_recursion_error", "name": macro_name, "duration_s": 0.0})
            continue
        body = definitions[macro_name]
        params = dict(body.get("defaults", {}))
        params.update(block.get("params", {}))
        params.update({k: v for k, v in block.items() if k not in {"op", "primitive", "params"}})
        body_blocks = deepcopy(body.get("blocks", []))
        body_blocks = [_substitute_params(b, params) for b in body_blocks]
        expanded.extend(_expand_blocks(body_blocks, definitions, stack + (macro_name,)))
    return expanded


def _substitute_params(value: Any, params: dict[str, Any]) -> Any:
    if isinstance(value, str) and value.startswith("$"):
        return params.get(value[1:], value)
    if isinstance(value, list):
        return [_substitute_params(v, params) for v in value]
    if isinstance(value, dict):
        return {k: _substitute_params(v, params) for k, v in value.items()}
    return value


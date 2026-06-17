from __future__ import annotations

from typing import Any

import numpy as np

from llm_framework.core.state import WorldState


def derived_context(state: WorldState) -> dict[str, Any]:
    obj = np.asarray(state.object_pos, dtype=float)
    out: dict[str, Any] = {
        "object_to_base_distance": _dist(obj[: min(3, len(obj))], state.base_q[: min(3, len(state.base_q))]),
        "object_height": round(float(obj[2]), 4) if len(obj) >= 3 else None,
        "object_speed": round(float(np.linalg.norm(state.object_vel)), 4),
        "appendage_distances_to_object": {},
    }
    for name, pos in state.fingertip_pos.items():
        out["appendage_distances_to_object"][name] = _dist(obj, pos)
    return out


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) != len(b) or len(a) == 0:
        return 0.0
    return round(float(np.linalg.norm(a - b)), 4)


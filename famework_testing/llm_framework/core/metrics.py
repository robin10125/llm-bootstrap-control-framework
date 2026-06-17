from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

from llm_framework.core.state import RolloutResult


def write_metrics_csv(path: Path, rows: Iterable[RolloutResult]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].row().keys()) if rows else [
        "interface", "task", "seed", "success", "score", "total_return",
        "final_object_x", "final_object_y", "final_object_z", "max_object_z", "errors",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in rows:
            writer.writerow(result.row())


def summarize(results: Iterable[RolloutResult]) -> dict:
    out: dict[str, dict] = {}
    for result in results:
        key = result.interface
        bucket = out.setdefault(key, {"n": 0, "successes": 0, "score_sum": 0.0})
        bucket["n"] += 1
        bucket["successes"] += int(result.success)
        bucket["score_sum"] += result.score
    for bucket in out.values():
        n = max(bucket["n"], 1)
        bucket["success_rate"] = round(bucket["successes"] / n, 4)
        bucket["mean_score"] = round(bucket["score_sum"] / n, 4)
        del bucket["score_sum"]
    return out


def write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


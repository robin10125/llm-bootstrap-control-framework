"""Eval-summary and CSV helpers shared by experiment runners (extracted from run_ppo_experiment)."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


def summarize(eval_rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for arm in sorted({row["arm"] for row in eval_rows}):
        rows = [row for row in eval_rows if row["arm"] == arm]
        n = len(rows)
        def avg(key, default=None):
            return round(sum(float(r.get(key, r.get(default) if default else 0)) for r in rows) / n, 6)
        out[arm] = {
            "eval_success_rate": avg("eval_success_rate"),
            "eval_instant_success_rate": avg("eval_instant_success_rate", "eval_success_rate"),
            "eval_base_return": avg("eval_base_return"),
            "eval_shaped_return": avg("eval_shaped_return"),
            "eval_train_return": avg("eval_train_return"),
            "eval_lift_max": avg("eval_lift_max"),
            "eval_hard_clip_frac": avg("eval_hard_clip_frac"),
            "eval_saturation_frac": avg("eval_saturation_frac"),
            "eval_action_abs_mean": avg("eval_action_abs_mean"),
            "n_eval": n,
        }
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

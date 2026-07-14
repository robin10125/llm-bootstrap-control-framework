#!/usr/bin/env python3
"""Curriculum controller + progress scoring (host-side / numpy).

Two jobs, both reading the episode **summary** eval vector (`eval_metrics.reduce_summary`):

- `progress_score(summary)` -> float: a single comparable number = (# milestones reached,
  gated cumulatively) + (fraction of the way to the next one). This is how the supervisor
  judges whether an LLM demo is *more helpful* than the policy — partial credit, not just
  success. A demo that gets 3 fingers on and lifts 0.03 m outranks one that only reaches.

- `CurriculumController`: tracks the training stage. When the batch reliably clears the
  current milestone it unlocks the next (the env's reward shapes up to stage+1); if a stage
  stalls it raises a flag so the LLM can revise the reward/milestones. See
  `llm_reward_curriculum_plan.md`.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from eval_metrics import EVAL_FIELDS, FIELD_INDEX, predicate
from reward_spec import CompiledReward

_RANGE = {f[0]: f[3] for f in EVAL_FIELDS}  # name -> (lo, hi)


def _raw_predicate_matrix(summaries: np.ndarray, specs: list[dict[str, Any]]) -> np.ndarray:
    """[N, F] summaries -> [N, M] bool: does each env satisfy each milestone predicate?"""
    cols = []
    for m in specs:
        p = m["predicate"]
        v = summaries[:, FIELD_INDEX[p["field"]]]
        cols.append(np.asarray(predicate(p["op"], v, float(p["thr"]))))
    return np.stack(cols, axis=1) if cols else np.zeros((summaries.shape[0], 0), bool)


def _gated(reached: np.ndarray) -> np.ndarray:
    """Gate raw per-milestone satisfaction so milestone i counts only if 0..i all hold."""
    return np.cumprod(reached.astype(bool), axis=-1).astype(bool)


def _intra_fraction(value: float, op: str, thr: float, field: str) -> float:
    """How close `value` is to satisfying (op value thr), in [0, 1)."""
    lo, hi = _RANGE[field]
    if op in ("ge", "gt"):
        return float(np.clip(value / thr, 0.0, 1.0)) if thr > 0 else float(value >= thr)
    # lt/le: value must fall from the nominal high end down to thr
    span = max(hi - thr, 1e-6)
    return float(np.clip((hi - value) / span, 0.0, 1.0))


def progress_score(summary: np.ndarray, compiled: CompiledReward) -> float:
    """(# gated milestones reached) + (fraction toward the first unreached one)."""
    specs = compiled.milestone_specs
    reached = _gated(_raw_predicate_matrix(summary[None, :], specs))[0]
    level = int(reached.sum())
    if level >= len(specs):
        return float(level)
    m = specs[level]  # first unreached milestone (its predecessors all hold)
    p = m["predicate"]
    frac = _intra_fraction(float(summary[FIELD_INDEX[p["field"]]]), p["op"],
                           float(p["thr"]), p["field"])
    return level + frac


class CurriculumController:
    def __init__(self, compiled: CompiledReward, *, advance_frac: float = 0.6,
                 patience: int = 3, stall_iters: int = 40):
        self.compiled = compiled
        self.n = compiled.n_milestones
        self.advance_frac = advance_frac
        self.patience = patience
        self.stall_iters = stall_iters
        self.stage = 0
        self._streak = 0          # consecutive iters the current stage was cleared
        self._since_advance = 0

    def milestone_fractions(self, summaries: np.ndarray) -> list[float]:
        """Fraction of envs reaching each milestone (gated) — for LLM reward reflection."""
        reached = _gated(_raw_predicate_matrix(summaries, self.compiled.milestone_specs))
        return [round(float(reached[:, i].mean()), 3) for i in range(self.n)]

    def update(self, summaries: np.ndarray) -> dict[str, Any]:
        """Feed one batch of [N, F] episode summaries; maybe advance the stage."""
        reached = _gated(_raw_predicate_matrix(summaries, self.compiled.milestone_specs))
        frac = float(reached[:, self.stage].mean()) if self.n else 1.0
        advanced = False
        self._streak = self._streak + 1 if frac >= self.advance_frac else 0
        if self.stage < self.n - 1 and self._streak >= self.patience:
            self.stage += 1
            self._streak = 0
            self._since_advance = 0
            advanced = True
        else:
            self._since_advance += 1
        stalled = (self._since_advance >= self.stall_iters) and (self.stage < self.n - 1)
        # mean gated milestones reached across the batch (a smooth curriculum-progress metric)
        mean_level = float(reached.sum(axis=1).mean()) if self.n else 0.0
        return {"stage": self.stage, "stage_clear_frac": round(frac, 3),
                "mean_level": round(mean_level, 2), "advanced": advanced, "stalled": stalled}

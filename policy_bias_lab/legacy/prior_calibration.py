"""Generic parameter calibration for authored prior candidates.

The LLM authors structure and declares tunable scalar parameters.  This module explores those
parameters in simulation with a caller-provided scorer.  It is deliberately task-agnostic: the
objective comes from the scorer, and the search only rewrites numeric parameter values.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any, Callable

from policy_bias_lab.prior_eval import validate_program


ScoreFn = Callable[[dict[str, Any]], float]


@dataclass(frozen=True)
class CalibrationResult:
    candidate: dict[str, Any]
    program: dict[str, Any]
    objective: float
    parameters: dict[str, float]
    trials: list[dict[str, Any]]


def calibrate_candidate(
    env: Any,
    candidate: dict[str, Any],
    rep: str,
    score_program_fn: ScoreFn,
    *,
    max_trials: int = 25,
    levels: int = 3,
) -> CalibrationResult | None:
    """Search a candidate's declared parameter ranges and return the best compiled program.

    ``score_program_fn`` receives a compiled prior program and returns a scalar objective.  Use a
    cheap simulator exploration scorer here; short-PPO arbitration can remain the later expensive
    selection step.
    """
    params = _parameter_specs(candidate)
    if not params:
        errors: list[str] = []
        program = validate_program(env, candidate, rep, errors=errors)
        if program is None:
            return None
        objective = float(score_program_fn(program))
        return CalibrationResult(candidate, program, objective, {}, [{
            "parameters": {}, "objective": objective,
        }])

    names = list(params)
    values = [_values_for(params[n], levels) for n in names]
    combos = list(itertools.product(*values))[:max(1, int(max_trials))]
    trials: list[dict[str, Any]] = []
    best: tuple[float, dict[str, float], dict[str, Any], dict[str, Any]] | None = None
    for combo in combos:
        chosen = dict(zip(names, combo, strict=True))
        cand_i = with_parameter_values(candidate, chosen)
        errors: list[str] = []
        program = validate_program(env, cand_i, rep, errors=errors)
        if program is None:
            trials.append({"parameters": chosen, "objective": None, "errors": errors})
            continue
        objective = float(score_program_fn(program))
        trials.append({"parameters": chosen, "objective": objective})
        if best is None or objective > best[0]:
            best = (objective, chosen, cand_i, program)
    if best is None:
        return None
    objective, chosen, cand_best, program_best = best
    return CalibrationResult(cand_best, program_best, objective, chosen, trials)


def with_parameter_values(candidate: dict[str, Any], values: dict[str, float]) -> dict[str, Any]:
    """Return a copy with parameter ``value`` fields set, preserving init/range metadata."""
    out = dict(candidate)
    params = dict(candidate.get("parameters") or {})
    new_params: dict[str, Any] = {}
    for name, spec in params.items():
        if isinstance(spec, dict):
            item = dict(spec)
            item["value"] = float(values.get(str(name), item.get("value", item.get("init", 0.0))))
            new_params[str(name)] = item
        else:
            new_params[str(name)] = float(values.get(str(name), spec))
    out["parameters"] = new_params
    return out


def _parameter_specs(candidate: dict[str, Any]) -> dict[str, tuple[float, float, float]]:
    raw = candidate.get("parameters")
    if not isinstance(raw, dict):
        return {}
    out: dict[str, tuple[float, float, float]] = {}
    for name, spec in raw.items():
        if isinstance(spec, (int, float)):
            v = float(spec)
            out[str(name)] = (v, v, v)
        elif isinstance(spec, dict):
            init = float(spec.get("value", spec.get("init", 0.0)))
            rng = spec.get("range", spec.get("bounds", [init, init]))
            if isinstance(rng, (list, tuple)) and len(rng) == 2:
                lo, hi = float(rng[0]), float(rng[1])
                out[str(name)] = (init, min(lo, hi), max(lo, hi))
    return out


def _values_for(spec: tuple[float, float, float], levels: int) -> list[float]:
    init, lo, hi = spec
    levels = max(1, int(levels))
    if levels == 1 or lo == hi:
        return [init]
    vals = [lo + (hi - lo) * i / (levels - 1) for i in range(levels)]
    if init not in vals:
        vals.append(init)
    return sorted(set(float(v) for v in vals))

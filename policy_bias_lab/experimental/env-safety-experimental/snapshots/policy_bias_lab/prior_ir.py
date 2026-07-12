"""Typed normalization and structural checks for authored prior candidates.

The LLM-facing JSON is intentionally flexible, but the compiler should see a small,
versioned intermediate representation.  This module is task-agnostic: it validates only
mechanics of observability, stateless staged structure, parameter shape, and actuator
accounting.  It never interprets what a signal or stage is supposed to mean for a task.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


IR_VERSION = 1


@dataclass(frozen=True)
class Issue:
    severity: str
    message: str


@dataclass
class ChannelIR:
    actuators: list[str]
    expr: str

    def to_json(self) -> dict[str, Any]:
        return {"actuators": self.actuators, "expr": self.expr}


@dataclass
class StageIR:
    name: str
    gate: str
    success: str | None
    channels: list[ChannelIR] = field(default_factory=list)
    constraints: list[dict[str, Any]] = field(default_factory=list)
    est_seconds: float | None = None  # author's estimate of how long the stage should take (budget check)

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "name": self.name,
            "gate": self.gate,
            "channels": [c.to_json() for c in self.channels],
        }
        if self.success is not None:
            out["success"] = self.success
        if self.constraints:
            out["constraints"] = self.constraints
        if self.est_seconds is not None:
            out["est_seconds"] = self.est_seconds
        return out


@dataclass
class PriorIR:
    name: str
    rationale: str
    signals: dict[str, str]
    parameters: dict[str, dict[str, Any]]
    stages: list[StageIR]
    probes: list[dict[str, Any]]
    evals: list[dict[str, Any]]
    unused_dofs: list[Any]
    temperature: float | None = None

    def to_candidate(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "ir_version": IR_VERSION,
            "name": self.name,
            "rationale": self.rationale,
            "signals": self.signals,
            "stages": [s.to_json() for s in self.stages],
            "unused_dofs": self.unused_dofs,
        }
        if self.parameters:
            out["parameters"] = self.parameters
        if self.probes:
            out["probes"] = self.probes
        if self.evals:
            out["evals"] = self.evals
        if self.temperature is not None:
            out["temperature"] = self.temperature
        return out


_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def normalize_candidate(raw: dict[str, Any], rep: str) -> tuple[dict[str, Any], list[Issue]]:
    """Normalize an LLM candidate into the active prior candidate shape.

    Current callers still pass dictionaries to ``validate_program``.  Returning a dict keeps the
    existing compiler contract while giving the generation layer a typed boundary.
    """
    if rep != "freeform_staged":
        return dict(raw), []

    src = raw.get("prior_ir") if isinstance(raw.get("prior_ir"), dict) else raw
    issues: list[Issue] = []
    stages_raw = src.get("stages")
    stages: list[StageIR] = []
    if not isinstance(stages_raw, list) or not stages_raw:
        issues.append(Issue("error", "freeform_staged candidate has no nonempty 'stages' list"))
        stages_raw = []
    for i, st in enumerate(stages_raw):
        if not isinstance(st, dict):
            issues.append(Issue("error", f"stage {i} is not an object"))
            continue
        channels = []
        for j, ch in enumerate(st.get("channels") or []):
            if not isinstance(ch, dict):
                issues.append(Issue("error", f"stage {i} channel {j} is not an object"))
                continue
            acts = [str(a) for a in (ch.get("actuators") or [])]
            channels.append(ChannelIR(actuators=acts, expr=str(ch.get("expr", "0"))))
            if not acts:
                issues.append(Issue("error", f"stage {i} channel {j} has no actuators"))
        constraints = []
        for j, c in enumerate(st.get("constraints") or []):
            if not isinstance(c, dict):
                issues.append(Issue("error", f"stage {i} constraint {j} is not an object"))
                continue
            cc = {
                "name": str(c.get("name") or f"constraint_{j}"),
                "mode": str(c.get("mode", "replace")),
                "priority": float(c.get("priority", j)) if isinstance(c.get("priority", j), (int, float)) else float(j),
                "channels": [],
            }
            if c.get("active") is not None:
                cc["active"] = str(c.get("active"))
            elif c.get("violation") is not None:
                cc["violation"] = str(c.get("violation"))
            else:
                issues.append(Issue("error", f"stage {i} constraint {j} has no active/violation expression"))
                cc["active"] = "0"
            for h, ch in enumerate(c.get("channels") or []):
                if not isinstance(ch, dict):
                    issues.append(Issue("error", f"stage {i} constraint {j} channel {h} is not an object"))
                    continue
                acts = [str(a) for a in (ch.get("actuators") or [])]
                if not acts:
                    issues.append(Issue("error", f"stage {i} constraint {j} channel {h} has no actuators"))
                cc["channels"].append({"actuators": acts, "expr": str(ch.get("expr", "0"))})
            constraints.append(cc)
        success = st.get("success")
        if success is None:
            issues.append(Issue("warning", f"stage {i} has no success/exit measurement"))
        est = st.get("est_seconds", st.get("expected_seconds"))
        stages.append(StageIR(
            name=str(st.get("name") or f"stage_{i}"),
            gate=str(st.get("gate", "1")),
            success=(None if success is None else str(success)),
            channels=channels,
            constraints=constraints,
            est_seconds=(float(est) if isinstance(est, (int, float)) and not isinstance(est, bool) else None),
        ))

    ir = PriorIR(
        name=str(src.get("name") or raw.get("name") or "candidate"),
        rationale=str(src.get("rationale") or raw.get("rationale") or ""),
        signals={str(k): str(v) for k, v in (src.get("signals") or {}).items()}
        if isinstance(src.get("signals"), dict) else {},
        parameters=_normalize_parameters(src.get("parameters"), issues),
        stages=stages,
        probes=list(src.get("probes") or [])[:8] if isinstance(src.get("probes"), list) else [],
        evals=list(src.get("evals") or [])[:8] if isinstance(src.get("evals"), list) else [],
        unused_dofs=list(src.get("unused_dofs") or []),
        temperature=(float(src["temperature"]) if src.get("temperature") is not None else None),
    )
    return ir.to_candidate(), issues


def structural_issues(env: Any, cand: dict[str, Any], rep: str) -> list[Issue]:
    """Compile-independent checks that can fail before a rollout is spent."""
    if rep != "freeform_staged":
        return []

    issues: list[Issue] = []
    try:
        from policy_bias_lab.freeform_priors import (
            all_actuator_names,
            compile_expr,
            raw_signal_fn,
        )
    except Exception as e:  # pragma: no cover - import failure is surfaced to caller.
        return [Issue("error", f"could not import prior validators: {e}")]

    _raw_fn, raw_names, _doc = raw_signal_fn(env)
    available = set(raw_names) | set(cand.get("parameters") or {})
    signals = cand.get("signals") or {}
    if not isinstance(signals, dict) or not signals:
        issues.append(Issue("warning", "candidate defines no authored signals"))
    elif not _names_are_valid(signals, issues, "signal"):
        pass
    for name, expr in signals.items():
        try:
            compile_expr(str(expr), available)
        except Exception as e:  # noqa: BLE001
            issues.append(Issue("error", f"signal {name!r} does not compile: {e}"))
        available.add(str(name))

    all_actuators = set(all_actuator_names(env))
    touched: set[str] = set()
    stages = cand.get("stages") or []
    for i, st in enumerate(stages):
        if not isinstance(st, dict):
            continue
        for field_name in ("gate", "success"):
            if st.get(field_name) is None:
                if field_name == "success":
                    issues.append(Issue("warning", f"stage {i} has no success expression"))
                continue
            try:
                compile_expr(str(st.get(field_name)), available)
            except Exception as e:  # noqa: BLE001
                issues.append(Issue("error", f"stage {i} {field_name} does not compile: {e}"))
        for j, ch in enumerate(st.get("channels") or []):
            if not isinstance(ch, dict):
                continue
            acts = [str(a) for a in (ch.get("actuators") or [])]
            touched.update(a for a in acts if a in all_actuators)
            try:
                compile_expr(str(ch.get("expr", "0")),
                             available | {"ctrl_self", "q_self", "v_self", "c_self", "env_c_self"})
            except Exception as e:  # noqa: BLE001
                issues.append(Issue("error", f"stage {i} channel {j} expr does not compile: {e}"))
        for j, c in enumerate(st.get("constraints") or []):
            if not isinstance(c, dict):
                continue
            src = c.get("active", c.get("violation"))
            if src is None:
                issues.append(Issue("error", f"stage {i} constraint {j} has no active/violation expression"))
            else:
                try:
                    compile_expr(str(src), available)
                except Exception as e:  # noqa: BLE001
                    issues.append(Issue("error", f"stage {i} constraint {j} active/violation does not compile: {e}"))
            mode = str(c.get("mode", "replace")).lower()
            if mode not in {"replace", "add"}:
                issues.append(Issue("error", f"stage {i} constraint {j} mode must be 'replace' or 'add'"))
            for h, ch in enumerate(c.get("channels") or []):
                if not isinstance(ch, dict):
                    continue
                acts = [str(a) for a in (ch.get("actuators") or [])]
                touched.update(a for a in acts if a in all_actuators)
                try:
                    compile_expr(str(ch.get("expr", "0")),
                                 available | {"ctrl_self", "q_self", "v_self", "c_self", "env_c_self"})
                except Exception as e:  # noqa: BLE001
                    issues.append(Issue("error", f"stage {i} constraint {j} channel {h} expr does not compile: {e}"))
    if not stages:
        issues.append(Issue("error", "candidate has no stages"))

    listed_unused = set()
    for u in cand.get("unused_dofs") or []:
        listed_unused.add(str(u.get("actuator", u)) if isinstance(u, dict) else str(u))
    unaccounted = sorted(all_actuators - touched - listed_unused)
    if unaccounted:
        issues.append(Issue("warning", "unaccounted actuators: " + ", ".join(unaccounted)))
    return issues


def _normalize_parameters(raw: Any, issues: list[Issue]) -> dict[str, dict[str, float]]:
    if not raw:
        return {}
    if not isinstance(raw, dict):
        issues.append(Issue("error", "'parameters' must be an object"))
        return {}
    out: dict[str, dict[str, float]] = {}
    for name, spec in raw.items():
        n = str(name)
        if not _IDENT.match(n):
            issues.append(Issue("error", f"parameter name {n!r} is not a valid expression name"))
            continue
        if isinstance(spec, (int, float)):
            v = float(spec)
            out[n] = {"init": v, "range": [v, v]}
            continue
        if not isinstance(spec, dict):
            issues.append(Issue("error", f"parameter {n!r} must be a number or object"))
            continue
        init = float(spec.get("init", 0.0))
        rng = spec.get("range", spec.get("bounds", [init, init]))
        if not isinstance(rng, (list, tuple)) or len(rng) != 2:
            issues.append(Issue("error", f"parameter {n!r} range must have two numbers"))
            continue
        lo, hi = float(rng[0]), float(rng[1])
        if lo > hi:
            lo, hi = hi, lo
        if not (lo <= init <= hi):
            issues.append(Issue("warning", f"parameter {n!r} init is outside its range"))
        out[n] = {"init": init, "range": [lo, hi]}
    return out


def _names_are_valid(values: dict[str, Any], issues: list[Issue], kind: str) -> bool:
    ok = True
    for name in values:
        if not _IDENT.match(str(name)):
            issues.append(Issue("error", f"{kind} name {name!r} is not a valid expression name"))
            ok = False
    return ok

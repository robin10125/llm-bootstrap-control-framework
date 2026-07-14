#!/usr/bin/env python3
"""Reward / curriculum spec -> a jit-safe reward function.

The LLM authors a **spec** (Tier A: structured JSON of milestones + reward terms drawn from
the audited operator library below) or, opt-in, a **reward expression** (Tier B: a single
arithmetic expression over eval-field names, AST-allowlisted). Neither lets the LLM run
arbitrary code in the jitted training loop. The compiler turns a spec into:

  reward_fn(eval_vec, stage) -> scalar        # used inside the jitted env step
  milestone predicates + names                # used host-side to score progress / advance

`eval_vec` is the flat array from `eval_metrics` (per step for reward, episode-summary for
judging). `stage` is a traced int: the curriculum unlocks milestones 0..stage+1, each gated by
the previous milestone's predicate holding this step (cumulative shaping + one-step lookahead).
See `llm_reward_curriculum_plan.md`.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any, Callable

import jax.numpy as jp

from eval_metrics import FIELD_INDEX, FIELD_NAMES, predicate

# --- operator library (Tier A terms) -----------------------------------------
# Each term is a dict {"op", "field"?, "w", ...}. A term maps an eval vector -> jp scalar.

_TERM_OPS = {"const", "linear", "neg_dist", "proximity", "clip", "ge", "gt", "lt", "le"}
_PRED_OPS = {"lt", "le", "gt", "ge"}


class SpecError(ValueError):
    pass


def _field(e, name: str):
    return e[FIELD_INDEX[name]]


def _term_value(term: dict[str, Any], e):
    op = term["op"]
    w = float(term.get("w", 1.0))
    if op == "const":
        return jp.float32(w)
    f = _field(e, term["field"])
    if op == "linear":
        return w * f
    if op == "neg_dist":           # reward that grows as a distance shrinks
        return -w * f
    if op == "proximity":          # bounded proximity bonus, w*exp(-dist/scale)
        scale = float(term.get("scale", 0.05))
        return w * jp.exp(-f / scale)
    if op == "clip":
        lo = float(term.get("lo", 0.0))
        hi = float(term.get("hi", 1.0))
        return w * jp.clip(f, lo, hi)
    if op in _PRED_OPS:            # milestone-style indicator bonus
        thr = float(term["thr"])
        return w * predicate(op, f, thr).astype(jp.float32)
    raise SpecError(f"unknown term op {op!r}")


@dataclass
class CompiledReward:
    reward_fn: Callable[[Any, Any], Any]      # (eval_vec, stage) -> scalar
    milestone_names: list[str]
    milestone_preds: list[Callable[[Any], Any]]  # (summary) -> bool-ish
    milestone_fields: list[str]
    milestone_specs: list[dict[str, Any]]
    n_milestones: int

    def progress_predicates(self, summary) -> list[bool]:
        """Per-milestone satisfied? on an episode summary (gated cumulatively)."""
        out, prev = [], True
        for pred in self.milestone_preds:
            prev = bool(prev) and bool(pred(summary))
            out.append(prev)
        return out


# --- Tier A: JSON spec --------------------------------------------------------

def validate_spec(spec: dict[str, Any]) -> None:
    if not isinstance(spec, dict) or "milestones" not in spec:
        raise SpecError("spec must be a dict with a 'milestones' list")
    ms = spec["milestones"]
    if not isinstance(ms, list) or not ms:
        raise SpecError("'milestones' must be a non-empty list")
    for m in ms:
        if "predicate" not in m or "reward" not in m:
            raise SpecError(f"milestone needs 'predicate' and 'reward': {m}")
        p = m["predicate"]
        if p.get("op") not in _PRED_OPS or p.get("field") not in FIELD_INDEX:
            raise SpecError(f"bad milestone predicate {p}; fields={FIELD_NAMES}")
        float(p["thr"])
        for t in m["reward"] + spec.get("penalties", []):
            if t.get("op") not in _TERM_OPS:
                raise SpecError(f"unknown term op in {t}; ops={sorted(_TERM_OPS)}")
            if t["op"] != "const" and t.get("field") not in FIELD_INDEX:
                raise SpecError(f"term needs a valid 'field': {t}; fields={FIELD_NAMES}")


def compile_spec(spec: dict[str, Any]) -> CompiledReward:
    validate_spec(spec)
    milestones = spec["milestones"]
    penalties = spec.get("penalties", [])

    def reward_fn(e, stage):
        r = jp.float32(0.0)
        for t in penalties:
            r = r + _term_value(t, e)
        prev_met = jp.bool_(True)
        for i, m in enumerate(milestones):
            unlocked = jp.asarray(i) <= (stage + 1)      # traced: unlock 0..stage+1
            active = unlocked & prev_met                 # gate next stage on previous predicate
            bonus = jp.float32(0.0)
            for t in m["reward"]:
                bonus = bonus + _term_value(t, e)
            r = r + jp.where(active, bonus, 0.0)
            p = m["predicate"]
            prev_met = prev_met & predicate(p["op"], _field(e, p["field"]), float(p["thr"]))
        return r

    names = [m.get("name", f"m{i}") for i, m in enumerate(milestones)]
    fields = [m["predicate"]["field"] for m in milestones]
    preds = [(lambda s, p=m["predicate"]: predicate(p["op"], s[FIELD_INDEX[p["field"]]],
                                                     float(p["thr"]))) for m in milestones]
    return CompiledReward(reward_fn, names, preds, fields, milestones, len(milestones))


# --- Tier B: AST-allowlisted reward expression --------------------------------
# The LLM writes ONE arithmetic expression over field names + `stage`, e.g.
#   "-palm_obj_dist + 0.5*n_contacts + 12*clip(lift, 0, 0.05)"
# Only the whitelisted nodes/names below are allowed; everything else raises.

_ALLOWED_FUNCS = {
    "clip": jp.clip, "exp": jp.exp, "abs": jp.abs,
    "min": jp.minimum, "max": jp.maximum, "where": jp.where,
}
_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Load,
    ast.Constant, ast.Compare, ast.BoolOp, ast.IfExp,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.USub, ast.UAdd,
    ast.And, ast.Or, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq,
)


def validate_expr(expr: str) -> None:
    tree = ast.parse(expr, mode="eval")
    allowed_names = set(FIELD_NAMES) | {"stage"} | set(_ALLOWED_FUNCS)
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise SpecError(f"disallowed syntax {type(node).__name__} in reward expr")
        if isinstance(node, ast.Call):
            if not (isinstance(node.func, ast.Name) and node.func.id in _ALLOWED_FUNCS):
                raise SpecError(f"only {sorted(_ALLOWED_FUNCS)} may be called")
        if isinstance(node, ast.Name) and node.id not in allowed_names:
            raise SpecError(f"unknown name {node.id!r}; allowed: {sorted(allowed_names)}")


def compile_expr(expr: str, milestones: list[dict[str, Any]] | None = None) -> CompiledReward:
    """Tier B reward expression. Milestones (for judging/curriculum) still come from a spec;
    pass them through so progress scoring is unchanged."""
    validate_expr(expr)
    code = compile(ast.parse(expr, mode="eval"), "<llm_reward_expr>", "eval")

    def reward_fn(e, stage):
        ns = {n: e[FIELD_INDEX[n]] for n in FIELD_NAMES}
        ns["stage"] = stage
        ns.update(_ALLOWED_FUNCS)
        return jp.float32(eval(code, {"__builtins__": {}}, ns))  # noqa: S307 (AST-validated)

    ms = milestones or []
    names = [m.get("name", f"m{i}") for i, m in enumerate(ms)]
    fields = [m["predicate"]["field"] for m in ms]
    preds = [(lambda s, p=m["predicate"]: predicate(p["op"], s[FIELD_INDEX[p["field"]]],
                                                     float(p["thr"]))) for m in ms]
    return CompiledReward(reward_fn, names, preds, fields, ms, len(ms))


# --- default hand-written curricula (Phase 1; also the LLM-authoring template) -

DEFAULT_SHADOW_SPEC: dict[str, Any] = {
    "milestones": [
        {"name": "reach", "predicate": {"op": "lt", "field": "palm_obj_dist", "thr": 0.06},
         "reward": [{"op": "proximity", "field": "palm_obj_dist", "w": 1.0, "scale": 0.08},
                    {"op": "neg_dist", "field": "palm_obj_dist", "w": 1.0}]},
        {"name": "touch1", "predicate": {"op": "ge", "field": "n_contacts", "thr": 1},
         "reward": [{"op": "proximity", "field": "min_finger_dist", "w": 1.0, "scale": 0.05},
                    {"op": "linear", "field": "n_contacts", "w": 0.3}]},
        {"name": "grip3", "predicate": {"op": "ge", "field": "n_contacts", "thr": 3},
         "reward": [{"op": "linear", "field": "n_contacts", "w": 0.5},
                    {"op": "linear", "field": "closure", "w": 0.5}]},
        {"name": "liftlow", "predicate": {"op": "gt", "field": "lift", "thr": 0.02},
         "reward": [{"op": "clip", "field": "lift", "lo": 0.0, "hi": 0.05, "w": 12.0}]},
        {"name": "success", "predicate": {"op": "gt", "field": "lift", "thr": 0.05},
         "reward": [{"op": "gt", "field": "lift", "thr": 0.05, "w": 5.0}]},
    ],
    "penalties": [{"op": "linear", "field": "obj_xy_disp", "w": -0.5}],
}

DEFAULT_GRIPPER_SPEC: dict[str, Any] = {
    "milestones": [
        {"name": "reach", "predicate": {"op": "lt", "field": "palm_obj_dist", "thr": 0.05},
         "reward": [{"op": "proximity", "field": "palm_obj_dist", "w": 1.0, "scale": 0.06},
                    {"op": "neg_dist", "field": "palm_obj_dist", "w": 1.0}]},
        {"name": "grip", "predicate": {"op": "ge", "field": "n_contacts", "thr": 2},
         "reward": [{"op": "linear", "field": "n_contacts", "w": 0.5},
                    {"op": "linear", "field": "closure", "w": 0.25}]},
        {"name": "liftlow", "predicate": {"op": "gt", "field": "lift", "thr": 0.02},
         "reward": [{"op": "clip", "field": "lift", "lo": 0.0, "hi": 0.05, "w": 12.0}]},
        {"name": "success", "predicate": {"op": "gt", "field": "lift", "thr": 0.05},
         "reward": [{"op": "gt", "field": "lift", "thr": 0.05, "w": 5.0}]},
    ],
    "penalties": [{"op": "linear", "field": "obj_xy_disp", "w": -0.3}],
}

DEFAULT_SPECS = {"gripper": DEFAULT_GRIPPER_SPEC, "shadow": DEFAULT_SHADOW_SPEC}

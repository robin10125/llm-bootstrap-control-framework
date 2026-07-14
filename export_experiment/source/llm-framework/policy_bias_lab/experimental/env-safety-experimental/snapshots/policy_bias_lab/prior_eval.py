"""Shared open-loop evaluation + validation for action-prior candidates.

One place for the contact-gated rollout scorer and the compile/accounting checks, used by both
the preliminary DSL-vs-free-form harness (run_dsl_vs_freeform) and the agentic orchestrator
(agentic_orchestrator). Real-world-observable signals only; flinging penalized.

An "iteration" in the agentic budget == one call to score_program (one rollout + evaluation).
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", str(Path(".xla_cache").resolve()))

import jax
import jax.numpy as jp
import numpy as np

from policy_bias_lab.bias import compile_bias
from policy_bias_lab.composed_priors import make_composed_prior_fn
from policy_bias_lab.freeform_priors import (
    MAX_EVALS, MAX_PROBES, addressed_actuators, all_actuator_names, _semantic_group,
)
from policy_bias_lab.llm_util import candidate_score
from policy_bias_lab.prior_ir import normalize_candidate, structural_issues
from policy_bias_lab.symbolic_control import sustained_bool


def flatten_for_dofcheck(prog: dict, rep: str) -> dict:
    """Flatten a program's phases/stages into a single channels candidate for DOF accounting."""
    chans: list = []
    if rep == "freeform_staged":
        for st in prog.get("stages", []):
            chans += list(st.get("channels", []))
            for c in st.get("constraints", []) or []:
                chans += list(c.get("channels", []))
    else:  # freeform: stacked phase sub-priors, each a set of channels
        for s in prog["subpriors"]:
            chans += list(s.get("channels", []))
    return {"mode": "freeform", "channels": chans}


def validate_program(env: Any, cand: dict, rep: str,
                     errors: list[str] | None = None) -> dict | None:
    """Wrap an LLM candidate's phase sub-priors into a stacked program and compile-check it.

    We do NOT force every DOF to be driven (the vocabulary is basis-complete, so any DOF is
    movable; the model is only required to CONSIDER each -- accounting is reported separately).
    Reject only on a hard compile failure.
    """
    cand, ir_issues = normalize_candidate(cand, rep)
    for issue in ir_issues:
        if issue.severity == "error":
            if errors is not None:
                errors.append(issue.message)
            return None

    static = structural_issues(env, cand, rep)
    for issue in static:
        if issue.severity == "error":
            if errors is not None:
                errors.append(issue.message)
            return None

    if rep == "freeform_staged":
        stages = cand.get("stages")
        if not isinstance(stages, list) or not stages:
            if errors is not None:
                errors.append(f"candidate {cand.get('name')!r}: no 'stages' list")
            return None
        # Blend is NOT author-selectable: HARD one-stage selection (clip to [0,1] + argmax, exactly
        # one stage active) is the only gate semantics -- chosen for strict no-overlap sequencing
        # (orient-before-move). `temperature` is carried but vestigial (no longer softens selection).
        prog = {"mode": "freeform_staged", "stages": stages}
        # LLM-authored derived signals (the only derived vocabulary; raw observables otherwise).
        # A candidate WITH `signals` compiles strictly against raw + its own names; compile
        # failures below reject it.
        if isinstance(cand.get("signals"), dict) and cand["signals"]:
            prog["signals"] = {str(k): str(v) for k, v in cand["signals"].items()}
        if isinstance(cand.get("parameters"), dict) and cand["parameters"]:
            prog["parameters"] = cand["parameters"]
        if cand.get("temperature") is not None:
            prog["temperature"] = float(cand["temperature"])
        # LLM-authored diagnostic probes ride on the program (measured by stage_occupancy; a bad
        # probe is reported there, never a validation failure).
        if isinstance(cand.get("probes"), list) and cand["probes"]:
            prog["probes"] = cand["probes"][:MAX_PROBES]
        # LLM-authored acceptance-test EVALS ride the same way (scored by stage_occupancy as
        # per-episode pass fractions; a bad eval is reported there, never a validation failure).
        if isinstance(cand.get("evals"), list) and cand["evals"]:
            prog["evals"] = cand["evals"][:MAX_EVALS]
    else:
        subs = cand.get("subpriors")
        if not isinstance(subs, list) or len(subs) < 3:
            return None
        prog = {"mode": "stacked", "representation": rep, "gates": {}, "subpriors": subs[:3]}
    try:
        compile_bias({"name": "x", "action_priors": [], "prior_program": prog}, env)
    except Exception as e:  # noqa: BLE001 - report and reject any uncompilable candidate
        print(f"  [reject] compile failed: {e}")
        if errors is not None:
            errors.append(f"candidate {cand.get('name')!r}: {e}")
        return None
    return prog


def accounting(env: Any, prog: dict, rep: str, raw_cand: dict) -> dict:
    """Consideration accounting: every DOF should be DRIVEN or explicitly listed unused.

    Reports n_driven, n_unused_listed, unaccounted (forgotten -> a consideration failure), and
    whether the wrist (the DOF the old vocabulary structurally omitted) is driven.
    """
    driven = addressed_actuators(env, flatten_for_dofcheck(prog, rep))
    listed = set()
    for u in (raw_cand.get("unused_dofs") or []):
        listed.add(str(u.get("actuator", u)) if isinstance(u, dict) else str(u))
    alln = set(all_actuator_names(env))
    wrist = [n for n in alln if _semantic_group(n) == "wrist"]
    return {
        "n_driven": len(driven), "n_unused_listed": len(listed & alln),
        "unaccounted": sorted(alln - driven - listed),
        "wrist_driven": all(w in driven for w in wrist) if wrist else None,
    }


# Cache jitted rollouts so re-scoring an IDENTICAL program (e.g. a different seed, or the same
# candidate revisited during a marginal-value sweep) is execution-only instead of a fresh ~57s XLA
# compile of env.step-under-scan. Keyed on (id(env), program-hash, envs). DISTINCT freeform
# candidates still compile once each -- their AST differs, so the graph genuinely differs.
_ROLLOUT_CACHE: dict[tuple[int, str, int], Callable] = {}


def _program_hash(prog: dict) -> str:
    return hashlib.sha1(json.dumps(prog, sort_keys=True).encode()).hexdigest()


def _get_rollout(env: Any, prog: dict, envs: int) -> Callable:
    key = (id(env), _program_hash(prog), envs)
    fn = _ROLLOUT_CACHE.get(key)
    if fn is not None:
        return fn
    pf, dw, _ = make_composed_prior_fn(env, prog)
    reset = jax.jit(lambda k: jax.vmap(env.reset)(k))
    step = jax.vmap(env.step)

    def rollout(rng):
        st = reset(jax.random.split(rng, envs))

        def body(s, _):
            a = jax.vmap(lambda o: pf(o, dw))(s.obs)
            ns = step(s, a)
            return ns, ns.metrics["eval"]

        _s, ev = jax.lax.scan(body, st, None, length=env.horizon)
        return ev

    fn = jax.jit(rollout)
    _ROLLOUT_CACHE[key] = fn
    return fn


def score_program(env: Any, prog: dict, envs: int = 128, seed: int = 0) -> dict:
    """Open-loop contact-gated rollout score for a compiled prior program. One budget iteration."""
    ev = _get_rollout(env, prog, envs)(jax.random.PRNGKey(seed))  # [T, E, 6]
    ev = np.asarray(ev)
    contacts, lift, xy = ev[:, :, 2], ev[:, :, 4], ev[:, :, 5]
    in_c = contacts >= 1.0
    lifted = lift > 0.05
    notflung = xy <= 0.08
    grasp_lift = jp.asarray(in_c & lifted & notflung)
    fling = jp.asarray(lifted & ~in_c)
    out = {
        "contact_gated_success": float(sustained_bool(grasp_lift, 20)),
        "contact_conditioned_lift": float(np.mean(np.where(in_c & notflung, lift, 0.0))),
        "contact_engagement": float(np.mean(in_c)),
        "contacts_mean": float(np.mean(contacts)),
        "fling_fraction": float(sustained_bool(fling, 20)),
        "palm_obj_dist_min": float(np.mean(np.min(ev[:, :, 0], axis=0))),
        "saturation_frac": 0.0, "action_abs_mean": 0.0,
    }
    out["objective_score"] = float(candidate_score(out))
    return out

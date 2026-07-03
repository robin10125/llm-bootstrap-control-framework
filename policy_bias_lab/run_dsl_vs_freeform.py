"""Preliminary experiment harness: LLM-generate DSL vs free-form stacked-gate action priors,
validate basis/DOF completeness, score open-loop, and write the best program per condition.
The PPO comparison is run separately via run_prior_ppo --prior-program-arm.

See PRELIM_dsl_vs_freeform.md. Framework is held constant (stacked phase gate); only the sub-prior
representation varies. Every DOF must be ACTIVELY movable (basis-complete), not held.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from string import Template
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAPPING = ROOT.parent / "bootstrapping"
if str(BOOTSTRAPPING) not in sys.path:
    sys.path.insert(0, str(BOOTSTRAPPING))

from policy_bias_lab.llm_util import call_llm as _call_llm
from policy_bias_lab.freeform_priors import (
    robot_spec, derive_motion_basis, check_basis_complete,
)
from policy_bias_lab.prior_eval import (
    score_program as _score, validate_program as _validate, accounting as _accounting,
)

PHASES = ["approach", "grasp", "lift"]
OPERATORS = ["toward_object_xy", "lower_base", "raise_base", "close_hand", "open_hand"]
SIGNALS = {
    "palm_obj_dist": "distance from grasp site to object (m); ~0 means at the object",
    "closure": "mean finger closure 0..1",
    "lift": "object height above rest (m); success at > 0.05",
    "obj_rel_x/y/z": "object position minus grasp site, world frame (m)",
    "near": "smooth gate ~1 when palm within ~4cm of object",
    "gripped": "smooth gate ~1 when closure past ~0.6",
}


PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


def _tmpl(name: str) -> Template:
    return Template((PROMPTS_DIR / name).read_text())


def build_prompt(rep: str, rs: dict, task: str, dof_mode: str = "encourage") -> str:
    """Assemble the generation prompt from editable templates in policy_bias_lab/prompts/.

    base + representation_<rep> + dof_requirement_<dof_mode>; only injected data is substituted.
    Edit the .md files to change wording without touching code.
    """
    acts = [a["name"] for a in rs["actuators"]]
    groups = rs["semantic_groups"]
    base_sign = {k: v["note"] for k, v in rs["base_world_sign"].items()}
    rep_doc = _tmpl(f"representation_{rep}.md").substitute(
        groups=json.dumps(list(groups)), operators=json.dumps(OPERATORS))
    requirement = _tmpl(f"dof_requirement_{dof_mode}.md").substitute(actuators=json.dumps(acts))
    field = "rules" if rep == "dsl" else "channels"
    out_item = f"subpriors: [{{name, {field}:[...]}}] -- one per phase, in order: {', '.join(PHASES)}"
    return _tmpl("action_prior_base.md").substitute(
        task=task,
        phases=", ".join(PHASES),
        actuators=json.dumps(rs["actuators"], indent=1),
        semantic_groups=json.dumps(groups),
        base_world_sign=json.dumps(base_sign),
        control=json.dumps(rs["control_law"]),
        signals=json.dumps(SIGNALS, indent=1),
        representation_doc=rep_doc.strip(),
        dof_requirement=requirement.strip(),
        output_item=out_item,
    )


def fallback(rep: str) -> list[dict]:
    if rep == "freeform_staged":
        return [{"name": "staged_fallback", "rationale": "hand seed", "blend": "soft", "stages": [
            {"name": "reach", "gate": "1 - near", "channels": [
                {"actuators": ["base_x"], "expr": "clip(obj_rel_x/0.05,-1,1)"},
                {"actuators": ["base_y"], "expr": "clip(-obj_rel_y/0.05,-1,1)"},
                {"actuators": ["base_z"], "expr": "0.5"}]},
            {"name": "grip", "gate": "near*(1-gripped)", "channels": [
                {"actuators": ["thumb", "index", "middle", "ring", "little"], "expr": "0.45"}]},
            {"name": "lift", "gate": "gripped*(lift<0.06)", "channels": [
                {"actuators": ["base_z"], "expr": "-0.5"},
                {"actuators": ["thumb", "index", "middle", "ring", "little"], "expr": "0.15"}]}]}]
    if rep == "dsl":
        return [{"name": "dsl_fallback", "rationale": "hand seed", "subpriors": [
            {"name": "approach", "rules": [{"kind": "operator", "group": "base_xy", "direction": "toward_object_xy", "weight": 0.5},
                                           {"kind": "operator", "group": "base_z", "direction": "lower_base", "weight": 0.3}]},
            {"name": "grasp", "rules": [{"kind": "operator", "group": "thumb", "direction": "close_hand", "weight": 0.5},
                                        {"kind": "operator", "group": "index", "direction": "close_hand", "weight": 0.45},
                                        {"kind": "operator", "group": "middle", "direction": "close_hand", "weight": 0.4},
                                        {"kind": "basis", "group": "wrist", "sign": 1, "weight": 0.2}]},
            {"name": "lift", "rules": [{"kind": "operator", "group": "base_z", "direction": "raise_base", "weight": 0.4},
                                       {"kind": "basis", "group": "wrist", "sign": 1, "weight": 0.15},
                                       {"kind": "operator", "group": "thumb", "direction": "close_hand", "weight": 0.2}]}]}]
    return [{"name": "free_fallback", "rationale": "hand seed", "subpriors": [
        {"name": "approach", "channels": [{"actuators": ["base_z"], "expr": "0.6*clip(palm_obj_dist-0.04,0,1)"},
                                          {"actuators": ["base_x", "base_y"], "expr": "0"}]},
        {"name": "grasp", "channels": [{"actuators": ["thumb", "index", "middle"], "expr": "0.5*near*(1-gripped)"},
                                       {"actuators": ["wrist"], "expr": "0.2*near"}]},
        {"name": "lift", "channels": [{"actuators": ["base_z"], "expr": "-0.4*gripped"},
                                      {"actuators": ["wrist"], "expr": "0.2*gripped"},
                                      {"actuators": ["ring", "little"], "expr": "0.2*gripped"}]}]}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--llm-backend", default="codex")
    ap.add_argument("--llm-model", default=None)
    ap.add_argument("--task", default="grasp and lift a 5cm cube off the table")
    ap.add_argument("--dof-mode", choices=["consider", "encourage"], default="encourage",
                    help="encourage (default, chosen) = strongly encouraged to use every DOF; "
                    "consider = able to use every DOF + must consider each, free not to drive all.")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    from mjx_env import make_env
    env = make_env("shadow")
    rs = robot_spec(env)
    basis = derive_motion_basis(env, "group")
    print("[basis-complete?]", check_basis_complete(env, basis))
    (args.out / "robot_spec.json").write_text(json.dumps(rs, indent=2))

    chosen = {}
    for rep, arm in [("dsl", "dsl_stacked"), ("freeform", "freeform_stacked")]:
        print(f"\n=== {rep} (dof-mode={args.dof_mode}) ===")
        prompt = build_prompt(rep, rs, args.task, dof_mode=args.dof_mode)
        (args.out / f"{rep}_prompt.md").write_text(prompt)
        cands = []
        if args.llm_backend not in {"fixture", "none"}:
            txt = _call_llm(args.llm_backend, prompt, model=args.llm_model, log_dir=args.out / "llm", tag=rep)
            (args.out / f"{rep}_completion.txt").write_text(txt or "")
            try:
                import re
                m = re.search(r"\{.*\}", txt or "", flags=re.S)
                cands = json.loads(m.group(0)).get("candidates", []) if m else []
            except Exception as e:
                print(f"  [parse fail] {e}")
        if not cands:
            print("  [fallback seeds]")
            cands = fallback(rep)
        scored = []
        for c in cands:
            prog = _validate(env, c, rep)
            if prog is None:
                continue
            s = _score(env, prog)
            acc = _accounting(env, prog, rep, c)
            scored.append({"name": c.get("name"), "score": s, "accounting": acc, "program": prog})
            print(f"  {c.get('name'):28s} obj={s['objective_score']:+.3f} engage={s['contact_engagement']:.3f} "
                  f"driven={acc['n_driven']}/23 unused_listed={acc['n_unused_listed']} unaccounted={len(acc['unaccounted'])} "
                  f"wrist_driven={acc['wrist_driven']}")
        if not scored:
            scored = [{"name": "fallback", "program": _validate(env, fallback(rep)[0], rep), "score": {}, "accounting": {}}]
        best = max(scored, key=lambda r: r["score"].get("objective_score", -1e9))
        (args.out / f"{arm}_program.json").write_text(json.dumps(best["program"], indent=2))
        (args.out / f"{rep}_scored.json").write_text(json.dumps(scored, indent=2))
        chosen[arm] = best
        print(f"  -> chose {best['name']}")
    print("\nWrote programs:", [str(args.out / f"{a}_program.json") for a in chosen])


if __name__ == "__main__":
    raise SystemExit(main())

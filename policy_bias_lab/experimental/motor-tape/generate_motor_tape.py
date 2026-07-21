"""Generate ONE motor-tape candidate, with optional generation-condition interventions.

Conditions (the generation-condition study's factors; all default OFF = the base arm):
  --position-info exact    inject the exact settled object position (use with an env whose
                           obj_xy_range is 0 -- the position must actually be fixed)
  --position-info samples  inject 4 concrete sampled reset observations (positions vary +-range)
  --calibration            inject a mechanically probed sign-calibration table (how each base
                           actuator's ctrl actually moves palm_pos_*/obj_rel_*)
  --feedback               one revision round: play the plan back, feed the measured evidence
                           (score, grounding, tape report) to the LLM, keep the better plan by
                           autopilot graded score

Prompt = framework doc + task line + spec block [+ calibration] [+ position info] +
representation doc + shared context block. Validation gate: validate_motor_tape, up to 3
attempts feeding errors back. Output per candidate dir:
{prompt.md, completion.txt, candidate.json, program.json, autopilot.json, tape_report.json,
 meta.json} plus, with --feedback: {program_r0.json, program_r1.json, feedback.md}.

Example:
  .venv/bin/python policy_bias_lab/experimental/motor-tape/generate_motor_tape.py \
      --out runs/motor_tape_gen_$(date +%Y%m%d) \
      --context-dir runs/prior_gen_study_20260711/context
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
GEN_STUDY_DIR = ROOT / "policy_bias_lab" / "experimental" / "prior-generation-study"
for p in (str(HERE), str(ROOT), str(GEN_STUDY_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from generate_candidate import gather_context  # noqa: E402 (context C0->C1 with caching)
from motor_tape import (  # noqa: E402
    nearmiss_score,
    probe_base_calibration,
    render_calibration_block,
    render_grounding_report,
    render_tape_report,
    score_tape,
    tape_grounding_report,
    tape_report,
    validate_motor_tape,
)
from policy_bias_lab.agentic_orchestrator import build_spec_block, parse_json_obj  # noqa: E402
from policy_bias_lab.freeform_priors import raw_obs_entries, robot_spec  # noqa: E402
from policy_bias_lab.llm_util import call_llm  # noqa: E402
from policy_bias_lab.prompt_utils import prompt_template as _tmpl  # noqa: E402

POSITION_INFO = ("none", "exact", "samples")
FEEDBACK_TARGETS = ("graded", "nearmiss")


def _exact_position_block(env) -> str:
    import jax
    obs_idx = dict(raw_obs_entries(env)[0])
    o = env.reset(jax.random.PRNGKey(0)).obs
    x, y, z = (float(o[obs_idx[f"obj_pos_{a}"]]) for a in "xyz")
    return ("OBJECT POSITION (exact -- IDENTICAL in every episode; the spawn state does not "
            f"vary): the object rests at world ({x:+.4f}, {y:+.4f}, {z:+.4f}) m. You may use "
            "these numbers directly; expressions over the reset observables remain available "
            "and will evaluate to exactly these values.")


def _samples_block(env, n: int = 4, seed: int = 1234) -> str:
    import jax
    obs_idx = dict(raw_obs_entries(env)[0])
    keys = jax.random.split(jax.random.PRNGKey(seed), n)
    cols = [f"obj_pos_{a}" for a in "xyz"] + [f"obj_rel_{a}" for a in "xyz"] + \
           [f"palm_pos_{a}" for a in "xyz"]
    rng = float(getattr(env.cfg, "obj_xy_range", 0.0) or 0.0)
    lines = [f"SAMPLED RESET OBSERVATIONS ({n} real episodes -- the object position varies "
             f"uniformly +-{rng:g} m in xy between episodes; your expressions are evaluated on "
             "each episode's OWN reset observation, so anchor to observables, and use these "
             "samples to CHECK your arithmetic numerically, sign by sign):"]
    for i, k in enumerate(keys):
        o = env.reset(k).obs
        vals = ", ".join(f"{c}={float(o[obs_idx[c]]):+.4f}" for c in cols)
        lines.append(f"  sample {i}: {vals}")
    return "\n".join(lines)


def build_prompt(env, task: str, spec_block: str, context_block: str, *,
                 position_info: str = "none", calibration: bool = False) -> str:
    dt = float(env.cfg.control_dt)
    max_slew = float(env.cfg.action_scale) / dt
    framework = _tmpl("framework_motor_tape.md").template  # static text, no substitution
    representation = _tmpl("representation_motor_tape.md").substitute(
        max_slew=f"{max_slew:g}",
        episode_seconds=f"{float(env.cfg.episode_seconds):g}")
    extra = ""
    if calibration:
        extra += "\n" + render_calibration_block(probe_base_calibration(env)) + "\n"
    if position_info == "exact":
        extra += "\n" + _exact_position_block(env) + "\n"
    elif position_info == "samples":
        extra += "\n" + _samples_block(env) + "\n"
    return (
        f"{framework}\n"
        f"TASK: {task}. Design for the task's success predicate and honor its constraints (see "
        "the spec below); prefer real, transferable behavior and avoid non-transferable or "
        "sim-only exploits.\n\n"
        f"{spec_block}\n{extra}\n"
        f"{representation}\n\n"
        "CONTEXT (generated upstream -- a comprehensive body-action account, then a "
        "moment-by-moment embodied procedure: its phases, interactions, budgets, forbidden "
        "motions/exploits, and a per-phase OBSERVABLE exit condition). USE it to ground your "
        "plan and derive your keyframes from its phases:\n"
        f"{context_block}\n\n"
        "Build the candidate in this order:\n"
        " 1. Define `signals` for every start-of-episode quantity your targets need.\n"
        " 2. Optionally define `parameters` {init, range} for values worth calibrating later.\n"
        " 3. Lay out the whole plan as phases on a timeline, then write the keyframes: for each\n"
        "    phase, which actuators move, to what absolute targets, arriving by when, coordinated\n"
        "    with which other groups. Check every segment against the slew ceiling and the\n"
        "    episode budget, with time to spare after the final keyframe.\n"
        " 4. Give every keyframe a label naming its job.\n"
        "Return JSON ONLY:\n"
        '{"candidates": [{"name": ..., "rationale": ..., "mode": "motor_tape", '
        '"signals": {...}, "parameters": {...}, "defaults": {"interp": "minjerk"}, '
        '"keyframes": [...], "unused_dofs": [{"actuator": ..., "reason": ...}]}]}'
    )


def _call_until_valid(env, prompt: str, *, backend: str, model: str | None, log_dir: Path,
                      tag: str, attempts: int = 3) -> tuple[dict | None, dict | None, str]:
    """(program, candidate, last_completion) after up to `attempts` validation-gated calls."""
    note = ""
    completion = ""
    for attempt in range(attempts):
        txt = call_llm(backend, prompt if attempt == 0 else prompt + "\n" + note,
                       model=model, log_dir=log_dir, tag=f"{tag}_attempt{attempt}")
        completion = txt or ""
        obj = parse_json_obj(txt)
        cands = (obj or {}).get("candidates", []) if obj else []
        if not cands:
            note = "NOTE: previous response had no parseable candidate. Resend JSON only."
            print(f"[gen] {tag} attempt {attempt + 1}/{attempts}: no candidates parsed")
            continue
        errors: list[str] = []
        program = validate_motor_tape(env, cands[0], errors)
        if program is not None:
            return program, cands[0], completion
        note = ("NOTE: your previous candidate failed validation; fix these and resend JSON "
                "only (the full corrected candidate):\n- " + "\n- ".join(errors[:6]))
        print(f"[gen] {tag} attempt {attempt + 1}/{attempts} rejected: {errors[:2]}")
    return None, None, completion


def generate_one(env, *, out_dir: Path, task: str, backend: str = "codex",
                 model: str | None = None, context_dir: Path | None = None,
                 position_info: str = "none", feedback: bool = False,
                 calibration: bool = False, dry_run: bool = False,
                 score_envs: int = 128, grounding_envs: int = 4,
                 feedback_target: str = "graded",
                 nearmiss_band: tuple[float, float] = (0.05, 0.065),
                 feedback_rounds: int = 1, keep_rule: str = "selector") -> dict[str, Any]:
    """Generate (and optionally revise) one candidate into out_dir; returns the meta dict.

    feedback_target selects BOTH the revision objective (prompt template) and the keep rule:
      graded    -- revise to fix grounded errors; keep the round with the higher autopilot
                   graded score (the gen-condition study's rule -- known to anti-predict
                   trainability when the winning plan makes contact).
      nearmiss  -- revise toward aligned near-miss (arrive centered, stop inside
                   nearmiss_band metres short of the object, zero contact); keep the round
                   with the higher proximity-without-contact score (motor_tape.nearmiss_score).

    feedback_rounds runs an iterative chain of up to N revision rounds: each round re-measures
    the CURRENT plan and revises it. The chain always advances on the newest valid revision
    (revision dynamics are not confounded by the selector); the FINAL kept program is then
    chosen by keep_rule: "selector" (argmax of the feedback_target keep-metric over all rounds,
    ties to the earliest -- rounds=1 reproduces the old behavior) or "always" (last valid
    round, i.e. no selection at all). Every round's program and autopilot score is saved
    (program_r<k>.json / autopilot_r<k>.json / feedback_r<k>.md) so any depth can be trained.
    """
    if position_info not in POSITION_INFO:
        raise ValueError(f"position_info must be one of {POSITION_INFO}")
    if feedback_target not in FEEDBACK_TARGETS:
        raise ValueError(f"feedback_target must be one of {FEEDBACK_TARGETS}")
    if keep_rule not in ("selector", "always"):
        raise ValueError("keep_rule must be 'selector' or 'always'")
    out_dir.mkdir(parents=True, exist_ok=True)
    spec_block = build_spec_block(robot_spec(env))
    ctx_dir = context_dir if context_dir is not None else out_dir / "context"
    context_block = gather_context(env, task, spec_block, backend, model, ctx_dir)
    prompt = build_prompt(env, task, spec_block, context_block,
                          position_info=position_info, calibration=calibration)
    (out_dir / "prompt.md").write_text(prompt)
    if dry_run:
        print(f"[gen] dry-run: prompt saved -> {out_dir / 'prompt.md'}")
        return {"dry_run": True, "prompt_chars": len(prompt)}

    program, cand, completion = _call_until_valid(
        env, prompt, backend=backend, model=model, log_dir=out_dir / "llm", tag="tape")
    (out_dir / "completion.txt").write_text(completion)
    if program is None:
        raise RuntimeError(f"[gen] {out_dir.name}: no valid motor_tape candidate after 3 attempts")

    rounds = [{"program": program, "candidate": cand}]
    kept_round = 0
    if feedback and feedback_rounds > 0:
        def _score_summary(score: dict) -> dict:
            te = score["tape_eval"]
            return {"graded": float(te["eval_graded_objective"]),
                    "nearmiss": round(nearmiss_score(score, *nearmiss_band), 6),
                    "min_dist": round(float(score["palm_obj_dist_min"]), 4),
                    "engagement": round(float(score["contact_engagement"]), 4)}

        def _keep_metric(r: dict) -> float:
            return r["nearmiss"] if feedback_target == "nearmiss" else r["graded"]

        def _revise_prompt(cur_cand: dict, score: dict, current: dict) -> str:
            te = score["tape_eval"]
            score_block = ("AUTOPILOT SCORE (your plan alone, no learned corrections): graded "
                           f"objective {te['eval_graded_objective']}, success rate "
                           f"{te['eval_success_rate']}, fraction of steps in real object "
                           f"contact {score['contact_engagement']:.1%}, mean per-episode "
                           "minimum grasp-point-to-object distance "
                           f"{score['palm_obj_dist_min']:.3f} m, max-lift mean "
                           f"{te['eval_lift_max']} m")
            grounding = tape_grounding_report(env, current, envs=grounding_envs, seed=7)
            trep = tape_report(env, current, envs=16, seed=7)
            template = ("revise_motor_tape_nearmiss.md" if feedback_target == "nearmiss"
                        else "revise_motor_tape.md")
            subst = dict(candidate_json=json.dumps(cur_cand, indent=2),
                         score_block=score_block,
                         grounding_block=render_grounding_report(grounding),
                         tape_report_block=render_tape_report(trep))
            if feedback_target == "nearmiss":
                subst.update(nearmiss_lo=f"{nearmiss_band[0]:g}",
                             nearmiss_hi=f"{nearmiss_band[1]:g}")
            return _tmpl(template).substitute(**subst)

        (out_dir / "program_r0.json").write_text(json.dumps(program, indent=2) + "\n")
        score_k = score_tape(env, program, envs=score_envs, seed=7, task="lift")
        (out_dir / "autopilot_r0.json").write_text(json.dumps(score_k, indent=2) + "\n")
        rounds[0].update(_score_summary(score_k))
        current, cur_cand = program, cand
        for k in range(1, int(feedback_rounds) + 1):
            rev_prompt = _revise_prompt(cur_cand, score_k, current)
            (out_dir / f"feedback_r{k}.md").write_text(rev_prompt)
            prog_k, cand_k, _comp = _call_until_valid(
                env, rev_prompt, backend=backend, model=model, log_dir=out_dir / "llm",
                tag=f"revise{k}", attempts=2)
            if prog_k is None:
                print(f"[gen] feedback round {k}: revision failed validation; chain stops")
                break
            score_k = score_tape(env, prog_k, envs=score_envs, seed=7, task="lift")
            (out_dir / f"program_r{k}.json").write_text(json.dumps(prog_k, indent=2) + "\n")
            (out_dir / f"autopilot_r{k}.json").write_text(json.dumps(score_k, indent=2) + "\n")
            rounds.append({"program": prog_k, "candidate": cand_k, **_score_summary(score_k)})
            print(f"[gen] feedback round {k} ({feedback_target}): keep-metric "
                  f"{_keep_metric(rounds[-2]):.3f} -> {_keep_metric(rounds[-1]):.3f}, "
                  f"min_dist {rounds[-1]['min_dist']}, engagement {rounds[-1]['engagement']}")
            current, cur_cand = prog_k, cand_k
        if keep_rule == "always":
            kept_round = len(rounds) - 1
        else:
            kept_round = max(range(len(rounds)), key=lambda i: (_keep_metric(rounds[i]), -i))
        program, cand = rounds[kept_round]["program"], rounds[kept_round]["candidate"]
        print(f"[gen] keep_rule={keep_rule}: keeping r{kept_round} of {len(rounds) - 1} rounds")

    rep = tape_report(env, program, envs=32, seed=0)
    print(render_tape_report(rep))
    autopilot = score_tape(env, program, envs=score_envs, seed=7, task="lift")
    te = autopilot["tape_eval"]
    print(f"[gen] autopilot: graded={te['eval_graded_objective']} "
          f"success={te['eval_success_rate']} engagement={autopilot['contact_engagement']:.3f}")
    (out_dir / "candidate.json").write_text(json.dumps(cand, indent=2) + "\n")
    (out_dir / "program.json").write_text(json.dumps(program, indent=2) + "\n")
    (out_dir / "tape_report.json").write_text(json.dumps(rep, indent=2) + "\n")
    (out_dir / "autopilot.json").write_text(json.dumps(autopilot, indent=2) + "\n")
    meta = {
        "framework": "motor_tape", "task": task, "backend": backend, "model": model,
        "position_info": position_info, "feedback": bool(feedback),
        "feedback_target": feedback_target if feedback else None,
        "feedback_rounds": int(feedback_rounds) if feedback else 0,
        "keep_rule": keep_rule if feedback else None,
        "nearmiss_band": list(nearmiss_band),
        "calibration": bool(calibration),
        "obj_xy_range": float(getattr(env.cfg, "obj_xy_range", 0.0) or 0.0),
        "n_keyframes": len(program.get("keyframes", [])),
        "n_signals": len(program.get("signals", {}) or {}),
        "autopilot_graded": te["eval_graded_objective"],
        "autopilot_success": te["eval_success_rate"],
        "contact_engagement": autopilot["contact_engagement"],
        "palm_obj_dist_min": autopilot["palm_obj_dist_min"],
        "nearmiss": round(nearmiss_score(autopilot, *nearmiss_band), 6),
        "rounds": [{k: r.get(k) for k in ("graded", "nearmiss", "min_dist", "engagement")}
                   for r in rounds],
        "kept_round": kept_round,
        "warnings": {"out_of_range": len(rep["out_of_range_knots"]),
                     "over_slew": len(rep["over_slew_segments"])},
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"[gen] OK ({meta['n_keyframes']} keyframes) -> {out_dir / 'program.json'}")
    return meta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--context-dir", type=Path, default=None,
                    help="existing context dir to reuse (default: <out>/context, generated "
                         "fresh if absent)")
    ap.add_argument("--task", default="grasp and lift a 5cm cube off the table")
    ap.add_argument("--llm-backend", default="codex")
    ap.add_argument("--llm-model", default=None)
    ap.add_argument("--episode-seconds", type=float, default=20.0)
    ap.add_argument("--obj-xy-range", type=float, default=0.04)
    ap.add_argument("--position-info", choices=list(POSITION_INFO), default="none")
    ap.add_argument("--feedback", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--feedback-target", choices=list(FEEDBACK_TARGETS), default="graded",
                    help="revision objective + keep rule: graded (status quo) or nearmiss "
                         "(aligned near-miss, keep by proximity-without-contact)")
    ap.add_argument("--nearmiss-lo", type=float, default=0.05)
    ap.add_argument("--nearmiss-hi", type=float, default=0.065)
    ap.add_argument("--feedback-rounds", type=int, default=1,
                    help="number of iterative revision rounds (chain re-measures each round)")
    ap.add_argument("--keep-rule", choices=["selector", "always"], default="selector",
                    help="final kept program: selector = argmax keep-metric over rounds; "
                         "always = last valid round (no selection)")
    ap.add_argument("--calibration", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--dry-run", action="store_true",
                    help="assemble and save prompt.md only; no LLM call, no scoring")
    args = ap.parse_args()
    if args.position_info == "exact" and args.obj_xy_range != 0.0:
        raise SystemExit("--position-info exact requires --obj-xy-range 0 (the position must "
                         "actually be fixed)")

    from experiment_runtime.environment import make_env
    env = make_env("shadow", control_dt=0.025, episode_seconds=args.episode_seconds,
                   physics_dt=0.01, obj_xy_range=args.obj_xy_range)
    generate_one(env, out_dir=args.out, task=args.task, backend=args.llm_backend,
                 model=args.llm_model, context_dir=args.context_dir,
                 position_info=args.position_info, feedback=args.feedback,
                 calibration=args.calibration, dry_run=args.dry_run,
                 feedback_target=args.feedback_target,
                 nearmiss_band=(args.nearmiss_lo, args.nearmiss_hi),
                 feedback_rounds=args.feedback_rounds, keep_rule=args.keep_rule)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

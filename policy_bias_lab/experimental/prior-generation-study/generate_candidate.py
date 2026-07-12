"""Generate ONE prior candidate for the prior-generation study (1 seed, no revisions).

Controlled comparison of GENERATION SCHEMES, not frameworks: the representation (freeform_staged),
the task, the robot spec, and one shared task-completion context response (C0 body-action account
-> C1 embodied procedure, generated once into <study-dir>/context/ and reused by every arm) are
identical across arms. Only the experiment's manipulated prompt content varies:

  --use-info     replace the seed prompt's (now-inaccurate) "weak per-step mean-shift" usage
                 description with an ACCURATE account of how the consuming framework uses the
                 program (kl_prior: KL reference guidance; clocked: clocked segments with a
                 learned switch). Mechanism only -- no task content.
  --complexity   append a design-emphasis block: 'simple' (fewest viable stages) or 'complex'
                 (fine-grained many-stage decomposition).

Validation is the standard structural gate (validate_program; plus clocked conversion checks for
clocked arms) with up to 2 retries feeding the errors back. Output per arm:
<study-dir>/<arm>/{prompt.md, completion.txt, candidate.json, program.json, meta.json}.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BOOTSTRAPPING = ROOT.parent / "bootstrapping"
CLOCKED_DIR = ROOT / "policy_bias_lab" / "experimental" / "policy-clocked-paths"
for p in (str(ROOT), str(BOOTSTRAPPING), str(CLOCKED_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from policy_bias_lab.agentic_orchestrator import (
    _dof_requirement,
    _framework_doc,
    _output_item,
    _render_body_actions,
    _render_procedure,
    _representation_doc,
    build_spec_block,
    parse_json_obj,
)
from policy_bias_lab.freeform_priors import compile_expr, raw_obs_entries, robot_spec
from policy_bias_lab.llm_util import call_llm
from policy_bias_lab.prior_eval import validate_program
from policy_bias_lab.run_dsl_vs_freeform import _tmpl

# The static usage paragraph at the top of seed_candidates.md (the OLD additive-mean-shift
# account). --use-info replaces exactly this text; if the template changes, fail loudly.
BASE_USE_TEXT = (
    "Design action priors: weak per-step mean-shifts added to a PPO policy's action to bias early\n"
    "exploration. Channels, gates, and success tests are pure expressions over the current "
    "observation and\nyour authored signals; staged priors use the framework's monotone rollout "
    "cursor for stage progress.")

USE_INFO = {
    "kl_prior": (
        "Design action priors as REFERENCE GUIDANCE for a reinforcement-learning policy: your "
        "program will NOT drive the robot. It is compiled into a per-state reference action; a "
        "separate neural policy is trained with PPO and receives a KL penalty pulling its action "
        "distribution toward your suggestion in every state, with a penalty weight that anneals "
        "to zero over training. The policy explores NEAR your suggested behavior early and "
        "diverges wherever the task reward disagrees with you. Design implications: (1) "
        "approximate correctness EVERYWHERE beats precise correctness somewhere -- a locally "
        "wrong suggestion is overridden by learning, not fatal; (2) prefer SMOOTH, GRADED "
        "activations over sharp thresholds -- a graded suggestion gives learning a useful "
        "direction near stage boundaries, a step function gives none; (3) your stage structure "
        "still sequences the reference behavior via the monotone rollout cursor, and your "
        "success tests still drive diagnostics. Channels, gates, and success tests are pure "
        "expressions over the current observation and your authored signals."),
    "clocked": (
        "Design action priors as CLOCKED BEHAVIOR SEGMENTS: each stage you author becomes a "
        "segment of a sequenced path that DOES drive the robot (with a learned neural residual "
        "added on top), but stage SWITCHING is owned by a learned policy head, not by your "
        "gates. The next stage's gate is consumed as the current segment's done-hint: greater "
        "than 0 must mean 'this segment has done its job'. Hints are used as features, as "
        "initialization for the learned switch, and as reward shaping -- NOT as the switch "
        "itself, so exact thresholds are not critical but the SIGN of the margin is. Your "
        "est_seconds sets each segment's pacing budget and a forced-advance timeout (about 3x "
        "est_seconds), so give honest duration estimates. Design implications: (1) make each "
        "stage's channels a CONVERGING feedback law -- servo toward the target state and settle "
        "-- because the learned clock may dwell in a segment longer or shorter than you planned; "
        "(2) make each hand-off condition an informative signed margin around its boundary; (3) "
        "your success tests still drive diagnostics. Channels, gates, and success tests are pure "
        "expressions over the current observation and your authored signals; stage progress is "
        "monotone."),
}

COMPLEXITY = {
    "simple": (
        "\nDESIGN EMPHASIS -- SIMPLICITY: author the SIMPLEST viable prior. Use the FEWEST "
        "stages that can complete the task (2-3), the fewest signals, and one clear job per "
        "stage. Prefer one robust behavior over several specialized ones; omit any stage whose "
        "job could be absorbed by a neighbor. Every element you add must earn its place.\n"),
    "complex": (
        "\nDESIGN EMPHASIS -- FINE-GRAINED STRUCTURE: decompose the task into MANY small stages "
        "(6-9), each with a NARROW job, its own exit measurement, and a short est_seconds. "
        "Separate orientation from transport from contact from verification; give distinct "
        "actuator groups their own preparation stages where sensible. Richness of structure is "
        "the goal -- but every stage must still have a real observable exit condition.\n"),
}


def gather_context(env, task: str, spec_block: str, backend: str, model: str | None,
                   ctx_dir: Path) -> str:
    """C0 -> C1 with the orchestrator's schema/compile gates; cached in ctx_dir and REUSED."""
    block_file = ctx_dir / "context_block.md"
    if block_file.exists():
        print(f"[context] reusing {block_file}")
        return block_file.read_text()
    ctx_dir.mkdir(parents=True, exist_ok=True)
    obs_names = {n for n, _ in raw_obs_entries(env)[0]}

    def validate_procedure(parsed: dict) -> list[str]:
        phases = parsed.get("procedure") or []
        if not phases:
            return ["'procedure' is empty"]
        errs = []
        for i, ph in enumerate(phases):
            ec = ph.get("exit_condition")
            if not ec:
                errs.append(f"phase {i} ('{ph.get('phase')}') has no exit_condition")
                continue
            try:
                compile_expr(str(ec), obs_names)
            except Exception as e:  # noqa: BLE001
                errs.append(f"phase {i} ('{ph.get('phase')}') exit_condition {ec!r} does not "
                            f"compile over the observables: {e}")
        return errs[:6]

    def context_call(tmpl_name: str, tag: str, subs: dict, required: tuple[str, ...],
                     validate=None) -> dict | None:
        base = _tmpl(f"{tmpl_name}.md").substitute(**subs)
        parsed, note = None, ""
        for attempt in range(2):
            txt = call_llm(backend, base if attempt == 0 else base + "\n" + note, model=model,
                           log_dir=ctx_dir / "llm", tag=tag)
            parsed = parse_json_obj(txt)
            errs = (["response was not valid JSON"] if not parsed
                    else [f"missing required key {k!r}" for k in required if k not in parsed])
            if not errs and validate is not None and parsed:
                errs = validate(parsed)
            if not errs:
                break
            note = ("NOTE: your previous response was rejected; fix these and resend JSON only:"
                    "\n- " + "\n- ".join(errs[:6]))
            print(f"  [context] {tag} rejected (attempt {attempt + 1}/2): {errs[0]}")
        (ctx_dir / f"{tag}.json").write_text(json.dumps(parsed, indent=2) if parsed else "")
        return parsed

    body = context_call("context_body_actions", "C0_body_actions",
                        dict(task=task, spec_block=spec_block),
                        required=("execution_account", "body_parts"))
    body_block = _render_body_actions(body) if body else "(no body-action account produced)"
    proc = context_call("context_procedure", "C1_procedure",
                        dict(task=task, spec_block=spec_block, body_account=body_block),
                        required=("procedure",), validate=validate_procedure)
    parts = []
    if body:
        parts.append("[COMPREHENSIVE BODY-ACTION ACCOUNT]\n" + body_block)
    if proc:
        parts.append("[EMBODIED PROCEDURE ACCOUNT (moment-by-moment: phases, contacts, budgets, "
                     "forbidden motions, and a per-phase observable exit condition)]\n"
                     + _render_procedure(proc))
    block = "\n\n".join(parts) if parts else "(no upstream context available)"
    block_file.write_text(block)
    return block


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--study-dir", type=Path, required=True)
    ap.add_argument("--arm", required=True, help="output subdirectory name for this arm")
    ap.add_argument("--framework", choices=["kl_prior", "clocked"], required=True)
    ap.add_argument("--use-info", action="store_true",
                    help="replace the usage paragraph with the consuming framework's real usage")
    ap.add_argument("--complexity", choices=["none", "simple", "complex"], default="none")
    ap.add_argument("--task", default="grasp and lift a 5cm cube off the table")
    ap.add_argument("--llm-backend", default="codex")
    ap.add_argument("--llm-model", default=None)
    ap.add_argument("--episode-seconds", type=float, default=20.0)
    args = ap.parse_args()

    from mjx_env import make_env
    env = make_env("shadow", control_dt=0.025, episode_seconds=args.episode_seconds,
                   physics_dt=0.01, obj_xy_range=0.04)
    rs = robot_spec(env)
    spec_block = build_spec_block(rs)
    rep = "freeform_staged"

    arm_dir = args.study_dir / args.arm
    arm_dir.mkdir(parents=True, exist_ok=True)
    context_block = gather_context(env, args.task, spec_block, args.llm_backend, args.llm_model,
                                   args.study_dir / "context")

    prompt = _tmpl("seed_candidates.md").substitute(
        task=args.task, framework=_framework_doc(rep), spec_block=spec_block,
        representation_doc=_representation_doc(rep, rs),
        dof_requirement=_dof_requirement("consider", rs),
        context_block=context_block, n_seeds="1", output_item=_output_item(rep))
    if args.use_info:
        if BASE_USE_TEXT not in prompt:
            raise SystemExit("seed_candidates.md usage paragraph changed; update BASE_USE_TEXT")
        prompt = prompt.replace(BASE_USE_TEXT, USE_INFO[args.framework])
    if args.complexity != "none":
        prompt += COMPLEXITY[args.complexity]
    (arm_dir / "prompt.md").write_text(prompt)

    program, cand, completion = None, None, ""
    note = ""
    for attempt in range(3):
        txt = call_llm(args.llm_backend, prompt if attempt == 0 else prompt + "\n" + note,
                       model=args.llm_model, log_dir=arm_dir / "llm",
                       tag=f"seed_attempt{attempt}")
        completion = txt or ""
        obj = parse_json_obj(txt)
        cands = (obj or {}).get("candidates", []) if obj else []
        if not cands:
            note = "NOTE: previous response had no parseable candidates JSON; resend JSON only."
            print(f"[gen] attempt {attempt + 1}/3: no candidates")
            continue
        errors: list[str] = []
        program = validate_program(env, cands[0], rep, errors=errors)
        if program is not None and args.framework == "clocked":
            from clocked_paths_ppo import convert_staged_program, validate_clocked_program
            from clocked_paths_ppo import ClockedPPOConfig  # noqa: F401 (import check)
            converted = convert_staged_program(program)
            errors.extend(validate_clocked_program(env, converted))
            if errors:
                program = None
        if program is not None:
            cand = cands[0]
            break
        note = ("NOTE: your previous candidate failed validation; fix these and resend JSON "
                "only:\n- " + "\n- ".join(errors[:6]))
        print(f"[gen] attempt {attempt + 1}/3 rejected: {errors[:2]}")

    (arm_dir / "completion.txt").write_text(completion)
    if program is None:
        raise SystemExit(f"[gen] arm {args.arm}: no valid candidate after 3 attempts")
    (arm_dir / "candidate.json").write_text(json.dumps(cand, indent=2) + "\n")
    (arm_dir / "program.json").write_text(json.dumps(program, indent=2) + "\n")
    (arm_dir / "meta.json").write_text(json.dumps({
        "arm": args.arm, "framework": args.framework, "use_info": bool(args.use_info),
        "complexity": args.complexity, "task": args.task, "backend": args.llm_backend,
        "model": args.llm_model, "n_stages": len(program.get("stages", [])),
        "n_signals": len(program.get("signals", {}) or {}),
    }, indent=2) + "\n")
    print(f"[gen] arm {args.arm}: OK ({len(program.get('stages', []))} stages) -> "
          f"{arm_dir / 'program.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

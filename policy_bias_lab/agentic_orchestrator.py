"""Agentic, robot/task-agnostic action-prior selection (AGENTIC_PRIOR_SELECTION.md).

Pipeline, all grounded in robot-spec-derived vocabulary (no hardcoded ACTION_GROUPS):

  Stage 0  context generation -- parallel single-job LLM calls (C1 execution account, C2 failure
           modes, C3 kinematics, optional C4 human analogy). Outputs injected as CONTEXT; NOT
           counted against the rollout budget.
  Stage 2  diverse seed candidates -- one LLM call returns up to n_seeds genuinely different
           compilable candidates.
  Stage 3  empirical evaluation -- each candidate compiled + rolled out under the contact-gated
           open-loop scorer (prior_eval.score_program). One rollout-eval == one budget ITERATION.
  Stage 4  iterative revision -- revise the best seed(s) against diagnostics + failure modes,
           re-evaluate, under the remaining budget.

Orchestrator = agentic WITHIN a fixed structure: explore (<=8 diverse seeds, 1 iter each) -> refine
(spend remaining budget on the best seed(s)) -> finish (return best by contact-gated objective).
Early stopping is performance-DYNAMICS based (plateau / degradation with patience w, tolerance eps),
not a "promising threshold". Only rollout-evaluations consume the budget B; reasoning/context LLM
calls do not (the budget models the scarce real-robot-rollout resource).
"""
from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import jax

from policy_bias_lab.freeform_priors import robot_spec
from policy_bias_lab.llm_util import call_llm as _call_llm
from policy_bias_lab.ppo_arbiter import evaluate_candidate_ppo
from policy_bias_lab.prior_eval import accounting, score_program, validate_program
from policy_bias_lab.run_dsl_vs_freeform import OPERATORS, PHASES, SIGNALS, fallback, _tmpl


# ----------------------------------------------------------------------------------------------
# Prompt assembly (shared spec block + staged single-job prompts)
# ----------------------------------------------------------------------------------------------

def build_spec_block(rs: dict) -> str:
    """The robot/task/env injection block shared by every staged prompt (data only, no task)."""
    base_sign = {k: v["note"] for k, v in rs["base_world_sign"].items()}
    return (
        "ROBOT ACTUATORS (every one is ACTIVELY MOVABLE by the prior vocabulary -- the motion basis "
        "spans all DOF in both directions):\n"
        f"{json.dumps(rs['actuators'], indent=1)}\n"
        f"SEMANTIC GROUPS: {json.dumps(rs['semantic_groups'])}\n"
        f"BASE WORLD-SIGN: {json.dumps(base_sign)}\n"
        f"CONTROL: {json.dumps(rs['control_law'])}\n"
        f"SIGNALS available to gates/expressions: {json.dumps(SIGNALS, indent=1)}"
    )


def _representation_doc(rep: str, rs: dict) -> str:
    return _tmpl(f"representation_{rep}.md").substitute(
        groups=json.dumps(list(rs["semantic_groups"])), operators=json.dumps(OPERATORS)).strip()


def _dof_requirement(dof_mode: str, rs: dict) -> str:
    acts = [a["name"] for a in rs["actuators"]]
    return _tmpl(f"dof_requirement_{dof_mode}.md").substitute(actuators=json.dumps(acts)).strip()


def _framework_doc(rep: str) -> str:
    """The framework paragraph, per representation (stacked fixed-phase vs. free-form staged)."""
    name = "framework_freeform_staged" if rep == "freeform_staged" else "framework_stacked"
    return _tmpl(f"{name}.md").substitute(phases=", ".join(PHASES)).strip()


def _output_item(rep: str) -> str:
    if rep == "freeform_staged":
        return ("blend:'soft'|'hard', stages:[{name, gate:'<expr>', success:'<expr>' (optional), "
                "channels:[{actuators:[...], expr:'<expr>'}]}]")
    field_name = "rules" if rep == "dsl" else "channels"
    return (f"subpriors: [{{name, {field_name}:[...]}}] -- one per phase, in order: "
            f"{', '.join(PHASES)}")


def parse_json_obj(txt: str) -> dict | None:
    """Extract the first complete top-level JSON object from a model completion.

    Uses JSONDecoder.raw_decode scanning each '{', so it tolerates code fences, leading prose, and
    (critically) TRAILING text/notes after the object -- a greedy `\\{.*\\}` + json.loads breaks on
    trailing data ("Extra data") and silently discarded a valid multi-candidate seed response.
    """
    if not txt:
        return None
    dec = json.JSONDecoder()
    for i, ch in enumerate(txt):
        if ch != "{":
            continue
        try:
            obj, _end = dec.raw_decode(txt, i)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


# ----------------------------------------------------------------------------------------------
# Orchestrator
# ----------------------------------------------------------------------------------------------

@dataclass
class _Chain:
    """A refinement line: one seed and the revisions descended from it."""
    name: str
    raw_cand: dict           # latest raw candidate (subpriors + unused_dofs) used for revision
    program: dict            # compiled-validated program of the current base (best-so-far)
    diagnostics: dict        # behavioral diagnostics of `program` (what the policy DID), for revise
    best_obj: float
    history: list[float] = field(default_factory=list)  # objective of each scored revision
    since_improve: int = 0
    active: bool = True
    frontier: int = -1       # deepest stage progress seen (stall idx; n_stages if terminal reached)
    failed: list = field(default_factory=list)  # rejected revisions since the last accepted base
    focus_side: str = "entry"  # which side of the broken hand-off to revise: "entry" = successor's
                               # gate/channels first (cheap), then roll back to "exit" = the stage
                               # itself, carrying the failed entry attempts as context
    focus_attempts: int = 0    # rejected revisions in the current focus phase


def _frontier(diagnostics: dict) -> int:
    """Stage-progress depth of a candidate: its stall stage index, or n_stages when the trained
    policy reaches the terminal stage. -1 when there is no staged report. Lets refinement value
    UNLOCKING a deeper stage even when the scalar objective momentarily drops -- a newly reached
    stage usually starts out badly and needs its own iterative-improvement window."""
    sr = (diagnostics or {}).get("stage_report") if isinstance(diagnostics, dict) else None
    if not isinstance(sr, dict) or not sr.get("stage_names"):
        return -1
    if sr.get("reaches_terminal"):
        return len(sr["stage_names"])
    s = sr.get("stall_stage")
    return -1 if s is None else int(s)


@dataclass
class AgenticOrchestrator:
    env: Any
    task: str
    rep: str = "freeform"            # settled by PRELIM_dsl_vs_freeform; default to the winner
    dof_mode: str = "encourage"
    llm_backend: str = "codex"
    llm_model: str | None = None
    out_dir: Path = Path("runs/agentic")
    budget: int = 10                 # B: max evaluations (3 explore + ~7 refine); see marginal-value run
    n_seeds: int = 3                 # explore cap -- breadth saturates ~3 seeds (mv_20260630-183954)
    patience: int = 3                # w: plateau/degradation window
    handoff_attempts: int = 2        # revisions on the ENTRY side of a broken hand-off before
                                     # rolling back to the EXIT side (the stage itself)
    per_stage_iters: int | None = None  # if set: patience = this, and the refine budget is resized
                                     # after seed selection to n_stages * this per chosen chain, so
                                     # EVERY authored stage is guaranteed that many improvement
                                     # attempts if it becomes the (poor) frontier. Overrides budget.
    eps: float = 1e-3                # improvement tolerance
    use_human_analogy: bool = False  # C4 flag
    arbiter: str = "short_ppo"       # "short_ppo" (trained-success, the real objective) | "open_loop"
    ppo_task: str = "lift"           # task KEY for the PPO side (NL `task` is for the prompts)
    ppo_train_seconds: float = 180.0
    ppo_train_envs: int = 256
    ppo_eval_envs: int = 256
    score_envs: int = 128            # open-loop arbiter only
    score_seed: int = 0
    log: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.out_dir = Path(self.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.rs = robot_spec(self.env)
        self.spec_block = build_spec_block(self.rs)
        self.rep_doc = _representation_doc(self.rep, self.rs)
        self.dof_req = _dof_requirement(self.dof_mode, self.rs)
        self.framework = _framework_doc(self.rep)
        self.iters = 0
        self.evaluated: list[dict] = []   # every scored candidate (for the report)
        self.since_improve_global = 0
        self.best_obj_global = -1e18

    # -- LLM plumbing ---------------------------------------------------------------------------
    def _llm(self, prompt: str, tag: str) -> str:
        # "none"/"fixture" skip the backend entirely (offline smoke / fallback-seed dry runs).
        if self.llm_backend in {"none", "fixture"}:
            return ""
        return _call_llm(self.llm_backend, prompt, model=self.llm_model,
                         log_dir=self.out_dir / "llm", tag=tag)

    # -- Stage 0: parallel context -------------------------------------------------------------
    def gather_context(self) -> str:
        calls = [
            ("context_execution", "C1_execution"),
            ("context_failure_modes", "C2_failure_modes"),
            ("context_kinematics", "C3_kinematics"),
        ]
        if self.use_human_analogy:
            calls.append(("context_human_analogy", "C4_human_analogy"))

        def run_one(tmpl_name: str, tag: str) -> tuple[str, dict | None, str]:
            # phases is only used by context_execution; extra keys are ignored by Template.substitute.
            prompt = _tmpl(f"{tmpl_name}.md").substitute(
                task=self.task, spec_block=self.spec_block, phases=", ".join(PHASES))
            txt = self._llm(prompt, tag)
            return tag, parse_json_obj(txt), txt

        results: dict[str, dict | None] = {}
        with ThreadPoolExecutor(max_workers=len(calls)) as pool:
            for tag, parsed, txt in pool.map(lambda c: run_one(*c), calls):
                results[tag] = parsed
                (self.out_dir / f"{tag}.json").write_text(json.dumps(parsed, indent=2)
                                                          if parsed else (txt or ""))
        self.log["context"] = results
        # Assemble a readable CONTEXT block; fall back to a stub if a call returned nothing.
        parts = []
        labels = {
            "C1_execution": "PER-ACTUATOR EXECUTION ACCOUNT",
            "C2_failure_modes": "FAILURE MODES / EXPLOITS TO AVOID",
            "C3_kinematics": "KINEMATIC / AFFORDANCE ANALYSIS",
            "C4_human_analogy": "HUMAN ANALOGY (DOF mapping)",
        }
        for tag, label in labels.items():
            if tag in results and results[tag]:
                parts.append(f"[{label}]\n{json.dumps(results[tag], indent=1)}")
        return "\n\n".join(parts) if parts else "(no upstream context available)"

    # -- Stage 2: diverse seeds ----------------------------------------------------------------
    def generate_seeds(self, context_block: str) -> list[dict]:
        prompt = _tmpl("seed_candidates.md").substitute(
            task=self.task, framework=self.framework, spec_block=self.spec_block,
            representation_doc=self.rep_doc, dof_requirement=self.dof_req,
            context_block=context_block, n_seeds=str(self.n_seeds), output_item=_output_item(self.rep))
        (self.out_dir / "seed_prompt.md").write_text(prompt)
        txt = self._llm(prompt, "seeds")
        (self.out_dir / "seed_completion.txt").write_text(txt or "")
        obj = parse_json_obj(txt)
        cands = (obj or {}).get("candidates", []) if obj else []
        if not cands:
            print("  [seeds] LLM produced none -> hand-authored fallback")
            cands = fallback(self.rep)
        return cands[: self.n_seeds]

    # -- Stage 4: one revision -----------------------------------------------------------------
    def revise(self, chain: _Chain, context_block: str) -> dict | None:
        diag = chain.diagnostics
        if isinstance(diag, dict) and chain.failed:
            # Surface rejected attempts since the last accepted base so the model does not
            # re-propose them (names + objectives only; task-agnostic).
            diag = dict(diag)
            diag["prior_failed_revisions"] = chain.failed[-4:]
        prompt = _tmpl("revise_candidate.md").substitute(
            task=self.task, phases=", ".join(PHASES), spec_block=self.spec_block,
            representation_doc=self.rep_doc, dof_requirement=self.dof_req,
            context_block=context_block, candidate=json.dumps(chain.program, indent=1),
            diagnostics=json.dumps(diag, indent=1),
            stage_focus=_stage_focus(diag, focus_side=chain.focus_side),
            output_item=_output_item(self.rep))
        txt = self._llm(prompt, f"revise_{chain.name}")
        obj = parse_json_obj(txt)
        if not obj:
            return None
        return obj.get("candidate") or obj  # accept either {candidate:{...}} or a bare candidate

    # -- evaluation (one budget iteration) -----------------------------------------------------
    def _eval(self, raw_cand: dict, source: str) -> dict | None:
        # Cheap compile-check + DOF accounting BEFORE spending a (short-PPO) iteration on it.
        prog = validate_program(self.env, raw_cand, self.rep)
        if prog is None:
            return None
        acc = accounting(self.env, prog, self.rep, raw_cand)
        best_params = None
        if self.arbiter == "short_ppo":
            # The real arbiter: train a short PPO, rank on TRAINED contact-gated success, and read
            # the policy's actual behavior for the feedback loop. See ppo_arbiter (why open-loop
            # is not trusted as the arbiter).
            res = evaluate_candidate_ppo(
                self.env, prog, task=self.ppo_task, seed=self.score_seed,
                train_seconds=self.ppo_train_seconds, train_envs=self.ppo_train_envs,
                eval_envs=self.ppo_eval_envs,
                checkpoint_dir=self.out_dir / "ppo" / f"iter{self.iters + 1}")
            objective = res["objective_score"]
            diagnostics = res["diagnostics"]
            full = {"eval": res["eval"], "best_train_success": res["best_train_success"],
                    "best_checkpoint_iter": res["best_checkpoint_iter"]}
            best_params = res["best_params"]
        else:  # open_loop: the cheap blind scorer (prefilter-grade only)
            score = score_program(self.env, prog, envs=self.score_envs, seed=self.score_seed)
            objective = score["objective_score"]
            diagnostics = _brief_diag(score)
            full = score
        self.iters += 1
        rec = {"iter": self.iters, "source": source, "name": raw_cand.get("name"),
               "objective": objective, "diagnostics": diagnostics, "full": full,
               "accounting": acc, "program": prog, "raw_cand": raw_cand, "best_params": best_params}
        self.evaluated.append(rec)
        if objective > self.best_obj_global + self.eps:
            self.best_obj_global = objective
            self.since_improve_global = 0
        else:
            self.since_improve_global += 1
        fm = diagnostics.get("likely_failure_modes") if isinstance(diagnostics, dict) else None
        print(f"  [iter {self.iters}/{self.budget}] {source:12s} {str(raw_cand.get('name')):22s} "
              f"obj={objective:+.4f} driven={acc['n_driven']}/{self.rs['n_actuators']} "
              f"wrist={acc['wrist_driven']}" + (f" | {fm}" if fm else ""))
        return rec

    # -- main loop -----------------------------------------------------------------------------
    def run(self) -> dict:
        context_block = self.gather_context()
        seeds = self.generate_seeds(context_block)

        # EXPLORE: up to min(n_seeds, budget) diverse seeds, one iteration each.
        chains: list[_Chain] = []
        explore_cap = min(self.n_seeds, self.budget)
        for cand in seeds:
            if self.iters >= explore_cap:
                break
            rec = self._eval(cand, "explore")
            if rec is None:
                continue
            chains.append(_Chain(name=str(cand.get("name") or f"seed{len(chains)}"),
                                 raw_cand=cand, program=rec["program"], diagnostics=rec["diagnostics"],
                                 best_obj=rec["objective"], history=[rec["objective"]],
                                 frontier=_frontier(rec["diagnostics"])))
        if not chains:
            raise RuntimeError("no seed candidate compiled; cannot refine")

        # REFINE: pick the best seed(s) -- one by default, a second only if competitive -- and
        # spend the remaining budget on revise->eval, with plateau/degradation early-stop.
        chains.sort(key=lambda c: c.best_obj, reverse=True)
        best = chains[0]
        chosen = [best]
        if len(chains) > 1 and chains[1].best_obj >= best.best_obj - 0.2 * abs(best.best_obj):
            chosen.append(chains[1])
        for c in chosen:
            c.active = True
        self.log["explore"] = [{"name": c.name, "obj": c.best_obj} for c in chains]
        self.log["refining"] = [c.name for c in chosen]
        print(f"  [refine] seeds chosen: {[c.name for c in chosen]} "
              f"(best explore obj={best.best_obj:+.3f})")
        # Explore is diversity probing, not hill-climbing: weaker seeds must not pre-charge the
        # global plateau counter (observed: best seed FIRST -> two weaker seeds + one rejected
        # refine tripped the global patience after a single refinement iteration).
        self.since_improve_global = 0

        if self.per_stage_iters:
            # Guarantee every authored stage `per_stage_iters` improvement attempts should it
            # become the (failing) frontier: patience = per_stage_iters, and the refine budget is
            # sized from the ACTUAL stage count of the chosen seed(s) (LLM-authored, so only known
            # now). Falls back to the configured budget when there is no staged report.
            n_st = max((len(((c.diagnostics or {}).get("stage_report") or {})
                            .get("stage_names") or []) for c in chosen), default=0)
            if n_st > 0:
                self.patience = self.per_stage_iters
                self.budget = self.iters + self.per_stage_iters * n_st * len(chosen)
                print(f"  [budget] per_stage_iters={self.per_stage_iters} x {n_st} stages x "
                      f"{len(chosen)} chain(s) -> budget={self.budget} (patience={self.patience})")

        rr = 0
        while self.iters < self.budget and any(c.active for c in chosen):
            if self.since_improve_global >= self.patience:
                print(f"  [stop] global plateau: no gain > eps in {self.patience} iters")
                break
            active = [c for c in chosen if c.active]
            chain = active[rr % len(active)]
            rr += 1
            cand = self.revise(chain, context_block)
            if cand is None:  # reasoning call failed to yield a candidate; not a budget iteration
                chain.since_improve += 1
                if chain.since_improve >= self.patience:
                    chain.active = False
                continue
            rec = self._eval(cand, f"refine:{chain.name}")
            if rec is None:  # uncompilable revision; no rollout consumed
                continue
            chain.history.append(rec["objective"])
            improved = rec["objective"] > chain.best_obj + self.eps
            new_frontier = _frontier(rec["diagnostics"])
            unlocked = new_frontier > chain.frontier
            if improved or unlocked:
                # Adopt as the new base. A frontier UNLOCK is adopted even when the objective
                # momentarily drops: a newly reached stage starts out badly, and the reset patience
                # window below is its chance at iterative improvement. finish() still returns the
                # best candidate by objective, so a failed unlock line cannot win the run.
                if unlocked:
                    print(f"  [frontier] {chain.name}: stage frontier {chain.frontier} -> "
                          f"{new_frontier}" + ("" if improved else
                          f" (adopted despite obj {rec['objective']:+.4f} <= {chain.best_obj:+.4f})"))
                    chain.frontier = new_frontier
                    self.since_improve_global = 0  # structural progress: don't let global patience kill it
                chain.best_obj = max(chain.best_obj, rec["objective"])
                chain.program = rec["program"]
                chain.raw_cand = cand
                chain.diagnostics = rec["diagnostics"]
                chain.since_improve = 0
                chain.failed = []
                chain.focus_side, chain.focus_attempts = "entry", 0  # fresh hand-off, entry first
            else:  # plateau/degradation step
                chain.since_improve += 1
                chain.failed.append({"name": rec["name"], "objective": rec["objective"]})
                chain.focus_attempts += 1
                if chain.focus_side == "entry" and chain.focus_attempts >= self.handoff_attempts:
                    # Entry-side attempts exhausted: roll back to the exit side (the stage itself),
                    # carrying the rejected entry attempts as context (chain.failed -> prompt).
                    chain.focus_side, chain.focus_attempts = "exit", 0
                    print(f"  [focus] {chain.name}: entry-side attempts exhausted -> rolling back "
                          f"to the EXIT side of the hand-off")
            if chain.since_improve >= self.patience:
                chain.active = False
                print(f"  [stop] chain {chain.name} plateaued (best={chain.best_obj:+.3f})")

        return self.finish()

    def finish(self) -> dict:
        best = max(self.evaluated, key=lambda r: r["objective"])
        (self.out_dir / "best_program.json").write_text(json.dumps(best["program"], indent=2) + "\n")
        # Save the winner's trained weights (short-PPO arbiter) so they can be reused / extended.
        if best.get("best_params") is not None:
            import pickle
            with (self.out_dir / "best_params.pkl").open("wb") as f:
                pickle.dump(jax.device_get(best["best_params"]), f)
        report = {
            "task": self.task, "representation": self.rep, "dof_mode": self.dof_mode,
            "arbiter": self.arbiter, "budget": self.budget, "iters_used": self.iters,
            "n_seeds": self.n_seeds,
            "best": {"name": best["name"], "source": best["source"],
                     "objective": best["objective"], "accounting": best["accounting"],
                     "diagnostics": best["diagnostics"], "full": best["full"]},
            "trajectory": [{"iter": r["iter"], "source": r["source"], "name": r["name"],
                            "objective": r["objective"], "wrist_driven": r["accounting"].get("wrist_driven"),
                            "n_driven": r["accounting"].get("n_driven"),
                            "failure_modes": r["diagnostics"].get("likely_failure_modes")
                            if isinstance(r["diagnostics"], dict) else None}
                           for r in self.evaluated],
            "explore": self.log.get("explore"), "refining": self.log.get("refining"),
        }
        (self.out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"[done] best obj={best['objective']:+.4f} ({best['source']} '{best['name']}') "
              f"after {self.iters} iters -> {self.out_dir / 'best_program.json'}")
        return report


def _brief_diag(score: dict) -> dict:
    keys = ("objective_score", "contact_gated_success", "contact_conditioned_lift",
            "contact_engagement", "contacts_mean", "fling_fraction", "palm_obj_dist_min")
    return {k: round(float(score.get(k, 0.0)), 4) for k in keys}


def _stage_focus(diagnostics: dict, focus_side: str | None = None) -> str:
    """Turn the stage-occupancy report into a focused revision directive (task-agnostic).

    Points the LLM at the broken hand-off and asks for a single-stage edit so each refine is a
    controlled ablation. `focus_side` picks which side of the hand-off to revise: "entry" = the
    successor stage's gate/channels (tried first -- the cheap fix), "exit" = the stalling stage
    itself (the rollback, after entry-side attempts were rejected; those attempts appear as
    prior_failed_revisions in the diagnostics). None keeps the neutral either-side directive.
    Falls back to a no-op line when there is no staged report.
    """
    diagnostics = diagnostics if isinstance(diagnostics, dict) else {}
    rep = diagnostics.get("stage_report")
    tr = diagnostics.get("training_report") or {}
    train_note = ""
    if tr.get("verdict") == "undertrained":
        rising = ", ".join(tr.get("still_improving") or [])
        train_note = (f"\nTRAINING BUDGET CAVEAT: the training curve was STILL RISING when the "
                      f"budget ran out ({rising} improving mid->late training), so the low score "
                      f"may reflect UNDER-TRAINING rather than a defect in the prior. Prefer a "
                      f"MINIMAL change (or gate/hand-off polish) over restructuring -- do not "
                      f"'fix' behavior the policy was still in the middle of learning.")
    elif tr.get("verdict") == "converged":
        train_note = ("\nTraining had CONVERGED (no metric still rising at budget end): the prior, "
                      "not the training budget, is the limiter -- a real change is warranted.")
    if not isinstance(rep, dict) or not rep.get("stage_names"):
        return ("No per-stage report available; make one focused change against the diagnostics."
                + train_note)
    names = rep["stage_names"]
    occ = rep.get("occupancy") or []
    reached = rep.get("reached_frac") or []
    table = "; ".join(f"{i}:{names[i]} (occ={occ[i] if i < len(occ) else '?'}, "
                      f"reached={reached[i] if i < len(reached) else '?'})" for i in range(len(names)))
    def _pct(v):
        return "?" if v is None else f"{100.0 * float(v):.0f}%"

    handoff = rep.get("handoff_frac") or []
    entered = rep.get("entered_frac") or []
    reverse = rep.get("reverse_frac") or []
    conv = rep.get("conversion") or []

    def _directive(k: int) -> str:
        nxt = names[k + 1] if k + 1 < len(names) else "?"
        entry_valid = k < len(entered) and entered[k] is not None and entered[k] >= 0.1
        if focus_side == "entry" and entry_valid:
            return (f"FOCUS = ENTRY side of the hand-off: revise ONLY stage {k + 1} ('{nxt}') -- its "
                    f"GATE (this hand-off's entry condition) and/or its channels -- so it takes over "
                    f"in the states stage {k} actually produces (see the signal table below). Return "
                    f"every OTHER stage byte-identical to the input.")
        if focus_side == "exit":
            return (f"FOCUS = EXIT side of the hand-off: the entry side (stage {k + 1}) was already "
                    f"tried and rejected -- see prior_failed_revisions in the diagnostics. Now revise "
                    f"ONLY stage {k} -- change WHAT IT DELIVERS: its channels (move the signals into "
                    f"the next gate's region) and/or its own gate (yield earlier or elsewhere). "
                    f"Return every OTHER stage byte-identical to the input.")
        return (f"Revise ONLY stage {k} -- its gate (its exit/hand-off condition) and/or its "
                f"channels -- so the policy can make the transition. Return every OTHER stage "
                f"byte-identical to the input.")

    if rep.get("reaches_terminal"):
        k = rep.get("weakest_stage")
        if k is None:
            return (f"Stage residence [{table}]. The policy completes the stage chain; refine "
                    f"whichever stage the diagnostics implicate, keep the others UNCHANGED."
                    + train_note)
        rev = reverse[k] if k < len(reverse) else None
        rev_note = (f", and in {_pct(rev)} of episodes the policy FALLS BACK from stage {k + 1} to "
                    f"stage {k} (unstable hand-off)") if rev and rev >= 0.2 else ""
        lines = [
            f"Stage residence [{table}]. The policy COMPLETES the stage chain, but its WEAKEST "
            f"hand-off is stage {k} ('{rep.get('weakest_name')}') -> stage {k + 1}: only "
            f"{_pct(handoff[k] if k < len(handoff) else None)} of episodes make a dwell-qualified "
            f"transition{rev_note}.",
            _directive(k)]
    else:
        k = rep.get("stall_stage")
        if k is None:
            return (f"Stage residence [{table}]. Make one focused change against the diagnostics."
                    + train_note)
        hand_line = ""
        if k < len(entered) and k < len(handoff) and handoff[k] is not None:
            hand_line = (f" Dwell-qualified: stage {k} is ENTERED in {_pct(entered[k])} of episodes "
                         f"but HANDS OFF to stage {k + 1} in only {_pct(handoff[k])} "
                         f"(conversion {_pct(conv[k] if k < len(conv) else None)}).")
        lines = [
            f"Stage residence [{table}]. The policy STALLS in stage {k} ('{rep.get('stall_name')}'): "
            f"it reliably reaches this stage but rarely advances to stage {k + 1}.{hand_line}",
            _directive(k)]

    # Dominance-success discrepancy (authored `success` expr vs the hand-off predicate).
    disc = rep.get("success_discrepancy") or []
    asf = rep.get("authored_success_frac") or []
    if k < len(disc) and disc[k]:
        a = _pct(asf[k] if k < len(asf) else None)
        h = _pct(handoff[k] if k < len(handoff) else None)
        if disc[k] == "handoff_without_success":
            lines.append(
                f"DISCREPANCY: your own success test for stage {k} passes in only {a} of episodes "
                f"even though the hand-off to stage {k + 1} fires in {h} -- either stage {k}'s gate "
                f"hands off BEFORE the stage has done its job, or the success expression is wrong. "
                f"Fix whichever is mistaken.")
        elif disc[k] == "success_without_handoff":
            lines.append(
                f"DISCREPANCY: your success test for stage {k} passes in {a} of episodes but the "
                f"hand-off to stage {k + 1} fires in only {h} -- stage {k + 1}'s entry gate looks "
                f"too strict; align it with the states where the success test holds.")
        elif disc[k] == "entered_without_success":
            lines.append(
                f"DISCREPANCY: the terminal stage {k} is entered in {_pct(entered[k] if k < len(entered) else None)} "
                f"of episodes but your success test passes in only {a} -- the stage runs without "
                f"accomplishing its post-condition; fix its channels (or the success expression).")

    def _fmt(pair):
        if not pair or pair[0] is None or pair[1] is None:
            return "n/a"
        return f"{pair[0]:+.3f} -> {pair[1]:+.3f}"

    trend = rep.get("stall_signal_trend")
    if isinstance(trend, dict) and trend:
        rows = "; ".join(f"{s} {_fmt(p)}" for s, p in trend.items())
        lines.append(f"While stage {k} is active, mean signal values over the FIRST -> SECOND half "
                     f"of the episode: {rows}.")
    ng = rep.get("next_gate")
    if isinstance(ng, dict):
        pair = ng.get("value_early_late")
        lines.append(f"The NEXT stage's gate is `{ng.get('expr')}` (signals it reads: "
                     f"{', '.join(ng.get('signals') or []) or 'none'}); its raw value went "
                     f"{_fmt(pair)} while stage {k} was active.")
        if pair and pair[0] is not None and pair[1] is not None and pair[1] <= pair[0] + 1e-4:
            lines.append(
                "That gate value is NOT rising, so the current stage's channels are moving its "
                "signals AWAY from the hand-off (or not moving them). Find the gate signal trending "
                "the wrong way in the table above and change the stage so that signal reverses -- "
                "when a signal diverges under the current channels, pushing the same direction "
                "HARDER makes it worse; prefer a gentler or decelerating response, or reshape the "
                "hand-off between the two gates, over raising magnitudes.")
    if rep.get("self_lock"):
        lines.append(
            f"HARD-BLEND SELF-LOCK: stage {k} dominates ~100% of steps and stage {k + 1} is almost "
            f"never reached -- with blend='hard' the argmax never yields, so no channel change alone "
            f"can exit the stage. Fix the GATES: lower/narrow stage {k}'s gate or raise stage "
            f"{k + 1}'s gate over the states where the hand-off should occur (see the signal ranges "
            f"above), rather than strengthening stage {k}'s channels.")
    # Edit-menu suggestion (menu defined in revise_candidate.md): smallest edit the evidence supports.
    pair = (rep.get("next_gate") or {}).get("value_early_late") or [None, None]
    rising = (pair[0] is not None and pair[1] is not None
              and pair[1] > pair[0] + max(0.05 * abs(pair[0]), 1e-3))
    if rep.get("self_lock"):
        menu = "(d) restructure the hand-off between the two adjacent gates"
    elif rising:
        menu = "(b) nudge the gate threshold -- the hand-off is already approaching, just not firing"
    elif tr.get("verdict") == "converged":
        menu = ("(c) REWRITE the gate condition -- training converged and the hand-off is not "
                "approaching, so the boundary is likely misplaced")
    else:
        menu = "(a) reshape a channel's response (gentler / decelerating / sign-corrected)"
    lines.append(f"Suggested EDIT MENU entry: {menu}.")
    return "\n".join(lines) + train_note

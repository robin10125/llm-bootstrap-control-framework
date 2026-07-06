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
import os
import pickle
import re
import signal
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import jax

from policy_bias_lab.freeform_priors import robot_spec
from policy_bias_lab.llm_util import call_llm as _call_llm
from policy_bias_lab.ppo_arbiter import evaluate_candidate_ppo, evaluate_candidate_prior_only
from policy_bias_lab.prior_eval import accounting, score_program, validate_program
from policy_bias_lab.run_dsl_vs_freeform import OPERATORS, PHASES, fallback, _tmpl


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
        f"{rs['observables_doc']}"
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
        return ("signals:{'<name>':'<expr over observables>'} (your derived-signal definitions), "
                "stages:[{name, gate:'<expr>', success:'<expr>' (optional), "
                "channels:[{actuators:[...], expr:'<expr>'}]}], "
                "probes:[{name, expr, stage:'<stage name>' (optional)}] (optional, <=8), "
                "evals:[{name, expr, when:'ever'|'end'}] (optional, <=8)")
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


def _eval_battery_delta(old_diag: dict, new_diag: dict,
                        min_gain: float = 0.05) -> tuple[str | None, list[str]]:
    """Compare the authored-eval pass fractions of two evaluations (task 'evals' on the program).

    Only eval names present in BOTH reports are compared (the author may add/replace evals, but a
    revision cannot earn acceptance from tests its parent was never scored on). Returns
    (gain_description | None, solved_names): a gain requires at least one shared eval up by
    >= min_gain with none down by more than min_gain; solved = shared evals whose pass_frac
    crossed 0.5 upward (a failing test now passes -- the branch-store trigger).
    """
    def _report(d):
        sr = (d or {}).get("stage_report") if isinstance(d, dict) else None
        rep = (sr or {}).get("eval_report") or {}
        return {k: float(v["pass_frac"]) for k, v in rep.items()
                if isinstance(v, dict) and v.get("pass_frac") is not None}
    old, new = _report(old_diag), _report(new_diag)
    shared = sorted(set(old) & set(new))
    if not shared:
        return None, []
    deltas = {k: new[k] - old[k] for k in shared}
    if any(dv < -min_gain for dv in deltas.values()):
        return None, []
    gains = {k: dv for k, dv in deltas.items() if dv >= min_gain}
    if not gains:
        return None, []
    solved = [k for k in shared if old[k] < 0.5 <= new[k]]
    desc = ", ".join(f"{k} {old[k]:.2f}->{new[k]:.2f}" for k in sorted(gains))
    return desc, solved


def _frontier(diagnostics: dict) -> int:
    """Stage-progress depth of a candidate: the deepest stage entered through an ORDERED chain of
    dwell-qualified hand-offs (n_stages when the full chain completes). -1 when there is no staged
    report. Lets refinement value UNLOCKING a deeper stage even when the scalar objective
    momentarily drops -- a newly reached stage usually starts out badly and needs its own
    iterative-improvement window.

    ORDERED accomplishment is required (v9 lesson): terminal gates whose situation terms are ~0 at
    spawn can be argmax-dominant from step 0, so "reaches_terminal"/stall alone certify gate
    arithmetic, not progress -- a candidate that idled in its default terminal stage jumped the
    frontier 3->6 and two objective regressions were adopted for it. A hand-off only counts here
    if the ordered i -> i+1 transition (stage_transition_stats) fires in >= 5% of episodes."""
    sr = (diagnostics or {}).get("stage_report") if isinstance(diagnostics, dict) else None
    if not isinstance(sr, dict) or not sr.get("stage_names"):
        return -1
    n = len(sr["stage_names"])
    entered = sr.get("entered_frac")
    handoff = sr.get("handoff_frac")
    if entered and handoff and len(entered) == n:
        start = next((i for i, e in enumerate(entered) if e is not None and e >= 0.1), None)
        if start is None:
            return 0
        d = start
        while d < n - 1 and d < len(handoff) and handoff[d] is not None and handoff[d] >= 0.05:
            d += 1
        if d == start:
            # No ordered hand-off fired anywhere. A deep `start` here is default-dominance
            # (only that stage's gate ever wins, e.g. a terminal gate true at spawn), not depth.
            return 0
        return n if d == n - 1 else d
    # legacy reports without transition stats: the old dominance-based rule
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
    arbiter: str = "short_ppo"       # "short_ppo" (trained-success, the real objective) |
                                     # "prior_only" (untrained, one rollout batch -- pure prior
                                     # quality, no training/reward confound) | "open_loop" (legacy)
    refine_top: int = 2              # max chains kept for refinement (2nd+ only if competitive)
    ppo_task: str = "lift"           # task KEY for the PPO side (NL `task` is for the prompts)
    ppo_train_seconds: float = 180.0
    ppo_train_envs: int = 256
    ppo_eval_envs: int = 256
    ppo_terminate_on_success: float | None = None  # PPOBiasConfig.success_terminate_seconds
    ppo_terminate_on_failure: float | None = None  # PPOBiasConfig.failure_terminate_seconds
    eval_batches: int = 1            # rollout batches per prior_only evaluation (variance control)
    score_envs: int = 128            # open-loop arbiter only
    score_seed: int = 0
    stop_after: int | None = None    # pause (checkpoint + exit) after this many evals THIS SESSION
    # Wall-clock termination (long runs). Setting plateau_hours switches the loop into wall-clock
    # mode: iteration-count plateau stops and chain deactivation are DISABLED, and the run ends
    # only on one of these criteria (or budget/pause). Paused time never counts as run time.
    min_hours: float | None = None      # no plateau-stop before this much accumulated run time
    plateau_hours: float | None = None  # stop when the best objective has not improved for this long
    success_stop: float | None = None   # stop when any eval's trained_success (SUSTAINED
                                        # contact-gated hold rate) reaches this
    write_dash: bool = False            # live dashboard.html on each checkpoint (off = paused;
                                        # regenerate anytime: python -m policy_bias_lab.run_dashboard)
    log: dict = field(default_factory=dict)

    # Everything needed to reconstruct the orchestrator on resume (out_dir/env come from the CLI).
    CONFIG_KEYS = ("task", "rep", "dof_mode", "llm_backend", "llm_model", "budget", "n_seeds",
                   "patience", "handoff_attempts", "per_stage_iters", "eps", "use_human_analogy",
                   "arbiter", "ppo_task", "ppo_train_seconds", "ppo_train_envs", "ppo_eval_envs",
                   "ppo_terminate_on_success", "ppo_terminate_on_failure",
                   "refine_top", "eval_batches", "score_envs",
                   "score_seed", "min_hours", "plateau_hours", "success_stop", "write_dash")
    # Mutable progress restored verbatim on resume.
    STATE_KEYS = ("iters", "evaluated", "since_improve_global", "best_obj_global", "log",
                  "context_block", "seeds", "chains", "chosen_idx", "rr", "next_seed",
                  "budget", "patience", "budget_resized", "wall_elapsed", "t_improve_elapsed",
                  "branches")

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
        # Resumable pipeline position (all persisted by save_checkpoint).
        self.context_block: str | None = None
        self.seeds: list[dict] | None = None
        self.chains: list[_Chain] = []
        self.chosen_idx: list[int] | None = None  # indices into self.chains chosen for refinement
        self.rr = 0                               # round-robin cursor over active chains
        self.next_seed = 0                        # first not-yet-evaluated explore seed
        self.budget_resized = False               # per_stage_iters resize already applied
        # BRANCH STORE: every accepted revision that SOLVED a problem (frontier unlock or a
        # failing authored eval flipped to passing), snapshotted with its performance breakdown.
        self.branches: list[dict] = []
        # Wall-clock bookkeeping (persisted): total ACTIVE run seconds across sessions, and the
        # elapsed-time stamp of the last global-best improvement.
        self.wall_elapsed = 0.0
        self.t_improve_elapsed = 0.0
        # Session-local (never persisted).
        self._ckpt_path = self.out_dir / "checkpoint.pkl"
        self._stop_requested = False
        self._session_evals = 0
        self._session_t0 = time.time()

    @property
    def wall_mode(self) -> bool:
        return self.plateau_hours is not None

    def _elapsed(self) -> float:
        """Accumulated ACTIVE run seconds (pauses excluded): prior sessions + this one."""
        return self.wall_elapsed + (time.time() - self._session_t0)

    def _wall_stop(self) -> str | None:
        """Wall-clock termination criteria; returns the stop reason or None."""
        if self.success_stop is not None:
            # "success_rate" is the current diagnostics key; "trained_success" covers checkpoints
            # written before the diagnostics were de-interpreted.
            s = max((float((r.get("diagnostics") or {}).get("success_rate")
                           or (r.get("diagnostics") or {}).get("trained_success") or 0.0)
                     for r in self.evaluated), default=0.0)
            if s >= self.success_stop:
                return (f"success criterion met: trained sustained-hold success {s:.3f} >= "
                        f"{self.success_stop}")
        if self.plateau_hours is not None:
            el = self._elapsed()
            stall_h = (el - self.t_improve_elapsed) / 3600.0
            if ((self.min_hours is None or el >= self.min_hours * 3600.0)
                    and stall_h >= self.plateau_hours):
                return (f"wall-clock plateau: best objective unimproved for {stall_h:.2f} h "
                        f"(threshold {self.plateau_hours} h) after {el / 3600.0:.2f} h of run time")
        return None

    # -- checkpoint / resume ----------------------------------------------------------------------
    def save_checkpoint(self) -> None:
        """Atomically persist config + full progress. Called after EVERY evaluation, so a hard kill
        (SIGKILL, power loss) loses at most the in-flight eval; SIGINT/SIGTERM lose nothing."""
        state = {k: getattr(self, k) for k in self.STATE_KEYS}
        state["wall_elapsed"] = self._elapsed()  # live total, so resume continues the clock
        # best_params are JAX device arrays until finish(); materialize for pickling.
        state["evaluated"] = [
            {**r, "best_params": (jax.device_get(r["best_params"])
                                  if r.get("best_params") is not None else None)}
            for r in self.evaluated]
        payload = {"version": 1, "config": {k: getattr(self, k) for k in self.CONFIG_KEYS},
                   "state": state}
        tmp = self._ckpt_path.with_suffix(".pkl.tmp")
        with tmp.open("wb") as f:
            pickle.dump(payload, f)
        os.replace(tmp, self._ckpt_path)
        if self.write_dash:
            try:  # live metrics page from the same payload; must never take the run down
                from policy_bias_lab.run_dashboard import write_dashboard
                write_dashboard(payload, self.out_dir / "dashboard.html")
            except Exception as e:
                print(f"  [dashboard] update failed (run unaffected): {e}")

    def restore(self, state: dict) -> None:
        for k in self.STATE_KEYS:
            if k in state:
                setattr(self, k, state[k])

    def extend_budget(self, new_budget: int) -> None:
        """Reopen a resumed run whose chains already plateaued or whose budget ran out: raise the
        budget and give the chosen chains a fresh patience window."""
        self.budget = new_budget
        self.since_improve_global = 0
        for i in (self.chosen_idx or []):
            self.chains[i].active = True
            self.chains[i].since_improve = 0
        print(f"  [resume] budget extended to {new_budget}; refinement chains reactivated")

    @staticmethod
    def load_checkpoint(out_dir: Path) -> dict:
        with (Path(out_dir) / "checkpoint.pkl").open("rb") as f:
            return pickle.load(f)

    def _install_signal_handlers(self) -> None:
        def handler(signum, frame):
            if self._stop_requested:  # second signal: abort now (last checkpoint is already on disk)
                raise KeyboardInterrupt
            self._stop_requested = True
            print(f"\n[pause] got signal {signum}: will checkpoint and exit after the current "
                  f"evaluation finishes (signal again to abort immediately -- everything up to the "
                  f"last finished iteration is already saved)")
        for s in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(s, handler)
            except ValueError:  # not the main thread (e.g. driven from a test harness)
                pass

    def _should_pause(self) -> bool:
        return self._stop_requested or (self.stop_after is not None
                                        and self._session_evals >= self.stop_after)

    def _paused(self) -> dict:
        self.save_checkpoint()
        print(f"[paused] {self.iters}/{self.budget} iterations done, state -> {self._ckpt_path}")
        print(f"         resume with: python -m policy_bias_lab.run_agentic_selection "
              f"--out {self.out_dir} --resume")
        return {"paused": True, "iters_used": self.iters, "budget": self.budget,
                "checkpoint": str(self._ckpt_path)}

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
            ("context_procedure", "C0_procedure"),
            ("context_execution", "C1_execution"),
            ("context_failure_modes", "C2_failure_modes"),
            ("context_kinematics", "C3_kinematics"),
        ]
        if self.use_human_analogy:
            calls.append(("context_human_analogy", "C4_human_analogy"))

        def run_one(tmpl_name: str, tag: str) -> tuple[str, dict | None, str]:
            prompt = _tmpl(f"{tmpl_name}.md").substitute(
                task=self.task, spec_block=self.spec_block)
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
            "C0_procedure": ("EMBODIED PROCEDURE ACCOUNT (moment-by-moment: what moves, what stays "
                             "still, contacts, precision, forbidden motions)"),
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
    def generate_seeds(self, context_block: str, retry_note: str | None = None) -> list[dict]:
        prompt = _tmpl("seed_candidates.md").substitute(
            task=self.task, framework=self.framework, spec_block=self.spec_block,
            representation_doc=self.rep_doc, dof_requirement=self.dof_req,
            context_block=context_block, n_seeds=str(self.n_seeds), output_item=_output_item(self.rep))
        tag = "seeds" if retry_note is None else "seeds_retry"
        if retry_note:
            prompt += "\n" + retry_note
        (self.out_dir / f"{tag}_prompt.md" if retry_note else self.out_dir / "seed_prompt.md"
         ).write_text(prompt)
        txt = self._llm(prompt, tag)
        (self.out_dir / f"{tag}_completion.txt" if retry_note else
         self.out_dir / "seed_completion.txt").write_text(txt or "")
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
            task=self.task, spec_block=self.spec_block,
            representation_doc=self.rep_doc, dof_requirement=self.dof_req,
            context_block=context_block, candidate=json.dumps(chain.program, indent=1),
            diagnostics=json.dumps(diag, indent=1),
            stage_focus=_stage_focus(diag, focus_side=chain.focus_side),
            output_item=_output_item(self.rep))
        # Unique tag per call: revisions of the same chain must not overwrite each other's
        # prompt/completion logs (iters+1 = the evaluation this revision will feed).
        txt = self._llm(prompt, f"revise_i{self.iters + 1:02d}_{chain.name}")
        obj = parse_json_obj(txt)
        if not obj:
            return None
        return obj.get("candidate") or obj  # accept either {candidate:{...}} or a bare candidate

    def _save_branch(self, chain: "_Chain", rec: dict, *, reason: str, solved: list[str]) -> None:
        """Snapshot an accepted, problem-solving revision as a persistent BRANCH of the prior.

        Written to out_dir/branches/ (program + performance breakdown) and indexed in
        branches/index.json; the index also rides the checkpoint. Branch = a program that durably
        solved something (a newly reached stage, or an authored eval flipped to passing) -- the
        substrate an ultra-long refinement can fork from instead of only walking one chain."""
        bdir = self.out_dir / "branches"
        bdir.mkdir(exist_ok=True)
        d = rec.get("diagnostics") or {}
        sr = d.get("stage_report") or {}
        prev = [b for b in self.branches if b.get("chain") == chain.name]
        fname = f"iter{self.iters:03d}_{chain.name}.json"
        entry = {
            "iter": self.iters, "chain": chain.name, "name": rec.get("name"),
            "parent_iter": prev[-1]["iter"] if prev else None,
            "reason": reason, "solved_evals": solved,
            "objective": rec.get("objective"),
            "breakdown": {
                "occupancy": sr.get("occupancy"), "entered_frac": sr.get("entered_frac"),
                "handoff_frac": sr.get("handoff_frac"), "conversion": sr.get("conversion"),
                "stall_stage": sr.get("stall_stage"), "reaches_terminal": sr.get("reaches_terminal"),
                "eval_report": sr.get("eval_report"),
                "rates": {k: d.get(k) for k in ("success_rate", "grasp_rate", "reach_rate",
                                                "lift_reached_rate", "lift_max")},
            },
            "file": f"branches/{fname}",
        }
        (bdir / fname).write_text(json.dumps({**entry, "program": rec["program"]}, indent=1) + "\n")
        self.branches.append(entry)
        (bdir / "index.json").write_text(json.dumps(self.branches, indent=1) + "\n")
        print(f"  [branch] {entry['file']} saved ({reason}"
              + (f"; solved: {', '.join(solved)}" if solved else "") + ")")

    # -- evaluation (one budget iteration) -----------------------------------------------------
    def _eval(self, raw_cand: dict, source: str,
              errors: list[str] | None = None) -> dict | None:
        # Cheap compile-check + DOF accounting BEFORE spending a (short-PPO) iteration on it.
        # A validation reject consumes NO budget iteration; `errors` (when given) collects the
        # reject reason so the caller can feed it back to the LLM instead of losing it.
        prog = validate_program(self.env, raw_cand, self.rep, errors=errors)
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
                checkpoint_dir=self.out_dir / "ppo" / f"iter{self.iters + 1}",
                cfg_overrides=({k: v for k, v in (
                    ("success_terminate_seconds", self.ppo_terminate_on_success),
                    ("failure_terminate_seconds", self.ppo_terminate_on_failure)) if v} or None))
            objective = res["objective_score"]
            diagnostics = res["diagnostics"]
            full = {"eval": res["eval"], "best_train_success": res["best_train_success"],
                    "best_checkpoint_iter": res["best_checkpoint_iter"],
                    "telemetry": res.get("telemetry")}
            best_params = res["best_params"]
        elif self.arbiter == "prior_only":
            # No training: one rollout batch of the prior at PPO iteration 0. Pure prior quality,
            # free of the reward/training confound (and ~100x cheaper than short_ppo).
            res = evaluate_candidate_prior_only(
                self.env, prog, task=self.ppo_task, seed=self.score_seed,
                eval_envs=self.ppo_eval_envs, n_batches=self.eval_batches)
            objective = res["objective_score"]
            diagnostics = res["diagnostics"]
            full = {"eval": res["eval"]}
        else:  # open_loop: the cheap blind scorer (prefilter-grade only)
            score = score_program(self.env, prog, envs=self.score_envs, seed=self.score_seed)
            objective = score["objective_score"]
            diagnostics = _brief_diag(score)
            full = score
        self.iters += 1
        self._session_evals += 1
        rec = {"iter": self.iters, "source": source, "name": raw_cand.get("name"),
               "objective": objective, "diagnostics": diagnostics, "full": full,
               "accounting": acc, "program": prog, "raw_cand": raw_cand,
               "best_params": best_params, "t_wall": time.time()}
        self.evaluated.append(rec)
        if objective > self.best_obj_global + self.eps:
            self.best_obj_global = objective
            self.since_improve_global = 0
            self.t_improve_elapsed = self._elapsed()
        else:
            self.since_improve_global += 1
        print(f"  [iter {self.iters}/{self.budget}] {source:12s} {str(raw_cand.get('name')):22s} "
              f"obj={objective:+.4f} driven={acc['n_driven']}/{self.rs['n_actuators']} "
              f"wrist={acc['wrist_driven']}")
        return rec

    # -- main loop -----------------------------------------------------------------------------
    def run(self) -> dict:
        """Run (or resume) the pipeline. Every phase is keyed off persisted state, so the loop can
        be paused at any point (SIGINT/SIGTERM or --stop-after) and re-entered: a checkpoint is
        written after each evaluation and each phase boundary."""
        self._install_signal_handlers()
        self._session_t0 = time.time()  # only active sessions count toward wall-clock criteria
        if self.wall_mode:
            msg = f"  [wall] wall-clock mode: stop on {self.plateau_hours}h objective plateau"
            if self.min_hours:
                msg += f" after >= {self.min_hours}h run time"
            if self.success_stop is not None:
                msg += f", or on trained success_rate >= {self.success_stop}"
            if self.wall_elapsed:
                msg += f" (resuming at {self.wall_elapsed / 3600.0:.2f}h)"
            print(msg)
        if self.context_block is None:
            self.context_block = self.gather_context()
            self.save_checkpoint()
        if self.seeds is None:
            self.seeds = self.generate_seeds(self.context_block)
            self.save_checkpoint()
        context_block = self.context_block

        # EXPLORE: up to min(n_seeds, budget) diverse seeds, one iteration each.
        explore_cap = min(self.n_seeds, self.budget)
        seed_errors: list[str] = []

        def _explore_pass() -> str | None:
            while self.next_seed < len(self.seeds) and self.iters < explore_cap:
                if self._should_pause():
                    return "pause"
                cand = self.seeds[self.next_seed]
                self.next_seed += 1
                rec = self._eval(cand, "explore", errors=seed_errors)
                if rec is not None:
                    self.chains.append(
                        _Chain(name=str(cand.get("name") or f"seed{len(self.chains)}"),
                               raw_cand=cand, program=rec["program"],
                               diagnostics=rec["diagnostics"], best_obj=rec["objective"],
                               history=[rec["objective"]], frontier=_frontier(rec["diagnostics"])))
                self.save_checkpoint()
                reason = self._wall_stop()
                if reason and self.success_stop is not None and "success criterion" in reason:
                    print(f"  [stop] {reason}")
                    return "success"
            return None

        stop = _explore_pass()
        # SEED RETRY: every seed failed validation -- fatal for small --n-seeds if not handled.
        # A validation reject costs no budget iteration, so regenerating with the compile errors
        # fed back is cheap; give up after 2 retries.
        retries = 0
        while stop is None and not self.chains and seed_errors and retries < 2:
            retries += 1
            note = ("NOTE: your previous candidates FAILED VALIDATION and were rejected -- "
                    "regenerate them with these compile errors fixed:\n- "
                    + "\n- ".join(seed_errors[-6:]))
            print(f"  [seeds] all seeds rejected -> regenerating with error feedback "
                  f"(retry {retries}/2)")
            self.seeds = self.generate_seeds(context_block, retry_note=note)
            self.next_seed = 0
            self.save_checkpoint()
            stop = _explore_pass()
        if stop == "pause":
            return self._paused()
        if stop == "success":
            return self.finish()
        if not self.chains:
            raise RuntimeError("no seed candidate compiled; cannot refine")

        # REFINE: pick the best seed(s) -- one by default, a second only if competitive -- and
        # spend the remaining budget on revise->eval, with plateau/degradation early-stop.
        if self.chosen_idx is None:
            order = sorted(range(len(self.chains)),
                           key=lambda i: self.chains[i].best_obj, reverse=True)
            best = self.chains[order[0]]
            self.chosen_idx = [order[0]]
            # refine_top >= number of chains is an explicit request to refine EVERY chain
            # (breadth experiment); the competitiveness filter only applies to a partial top-k.
            refine_all = int(self.refine_top) >= len(self.chains)
            for j in order[1:max(1, int(self.refine_top))]:
                if refine_all or self.chains[j].best_obj >= best.best_obj - 0.2 * abs(best.best_obj):
                    self.chosen_idx.append(j)
            for i in self.chosen_idx:
                self.chains[i].active = True
            self.log["explore"] = [{"name": self.chains[i].name, "obj": self.chains[i].best_obj}
                                   for i in order]
            self.log["refining"] = [self.chains[i].name for i in self.chosen_idx]
            print(f"  [refine] seeds chosen: {self.log['refining']} "
                  f"(best explore obj={best.best_obj:+.3f})")
            # Explore is diversity probing, not hill-climbing: weaker seeds must not pre-charge the
            # global plateau counter (observed: best seed FIRST -> two weaker seeds + one rejected
            # refine tripped the global patience after a single refinement iteration).
            self.since_improve_global = 0

            if self.wall_mode and self.per_stage_iters:
                print("  [wall] per_stage_iters ignored in wall-clock mode (time criteria govern)")
            if self.per_stage_iters and not self.wall_mode and not self.budget_resized:
                # Guarantee every authored stage `per_stage_iters` improvement attempts should it
                # become the (failing) frontier: patience = per_stage_iters, and the refine budget
                # is sized from the ACTUAL stage count of the chosen seed(s) (LLM-authored, so only
                # known now). Falls back to the configured budget when there is no staged report.
                n_st = max((len(((self.chains[i].diagnostics or {}).get("stage_report") or {})
                                .get("stage_names") or []) for i in self.chosen_idx), default=0)
                if n_st > 0:
                    self.patience = self.per_stage_iters
                    self.budget = self.iters + self.per_stage_iters * n_st * len(self.chosen_idx)
                    self.budget_resized = True
                    print(f"  [budget] per_stage_iters={self.per_stage_iters} x {n_st} stages x "
                          f"{len(self.chosen_idx)} chain(s) -> budget={self.budget} "
                          f"(patience={self.patience})")
            self.save_checkpoint()
        chosen = [self.chains[i] for i in self.chosen_idx]

        revise_fail_streak = 0
        while self.iters < self.budget and any(c.active for c in chosen):
            if self._should_pause():
                return self._paused()
            reason = self._wall_stop()
            if reason:
                print(f"  [stop] {reason}")
                break
            if not self.wall_mode and self.since_improve_global >= self.patience:
                print(f"  [stop] global plateau: no gain > eps in {self.patience} iters")
                break
            active = [c for c in chosen if c.active]
            chain = active[self.rr % len(active)]
            self.rr += 1
            cand = self.revise(chain, context_block)
            if cand is None:  # reasoning call failed to yield a candidate; not a budget iteration
                revise_fail_streak += 1
                chain.since_improve += 1
                if not self.wall_mode and chain.since_improve >= self.patience:
                    chain.active = False
                if revise_fail_streak >= 10:
                    # In wall-clock mode chains never deactivate, so a dead LLM backend must not
                    # spin the loop forever; a checkpoint exists, resume when the backend is back.
                    print("  [stop] 10 consecutive revise calls yielded no candidate -- "
                          "LLM backend looks down; pausing")
                    return self._paused()
                self.save_checkpoint()
                continue
            # Probes and authored evals persist across revisions until the model replaces them: a
            # revision that omits them keeps measuring/testing what the previous iteration asked
            # for (evals MUST persist for the battery comparison to mean anything).
            if "probes" not in cand and isinstance(chain.raw_cand, dict) and chain.raw_cand.get("probes"):
                cand["probes"] = chain.raw_cand["probes"]
            if "evals" not in cand and isinstance(chain.raw_cand, dict) and chain.raw_cand.get("evals"):
                cand["evals"] = chain.raw_cand["evals"]
            rev_errors: list[str] = []
            rec = self._eval(cand, f"refine:{chain.name}", errors=rev_errors)
            if rec is None:  # uncompilable revision; no rollout consumed
                # Surface the compile error to the next revise call (prior_failed_revisions)
                # instead of silently dropping the attempt.
                if rev_errors:
                    chain.failed.append({"name": cand.get("name"), "objective": None,
                                         "rejected": rev_errors[-1]})
                self.save_checkpoint()
                continue
            revise_fail_streak = 0
            chain.history.append(rec["objective"])
            improved = rec["objective"] > chain.best_obj + self.eps
            new_frontier = _frontier(rec["diagnostics"])
            unlocked = new_frontier > chain.frontier
            # Authored-eval tie-break (partial selection): a revision whose objective stays within
            # the measured noise band may be adopted on a strict improvement of the SHARED authored
            # eval battery. Never overrides a real objective regression.
            eval_gain, eval_solved = _eval_battery_delta(chain.diagnostics, rec["diagnostics"])
            std = float((rec["diagnostics"] or {}).get("objective_batch_std") or 0.0)
            noise_band = max(self.eps, 2.0 * std)
            eval_accept = (not improved and not unlocked and eval_gain is not None
                           and rec["objective"] >= chain.best_obj - noise_band)
            if improved or unlocked or eval_accept:
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
                if eval_accept:
                    print(f"  [evals] {chain.name}: adopted on authored-eval improvement "
                          f"({eval_gain}) with objective within noise "
                          f"({rec['objective']:+.4f} vs {chain.best_obj:+.4f})")
                # A refinement that SOLVED a problem -- unlocked a stage or flipped a failing
                # authored eval to passing -- becomes a persistent BRANCH of the prior, indexed
                # with its performance breakdown (ultra-long refinement can fork from it later).
                if unlocked or eval_solved:
                    self._save_branch(chain, rec,
                                      reason=("frontier_unlock" if unlocked else "eval_solved"),
                                      solved=eval_solved)
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
            if not self.wall_mode and chain.since_improve >= self.patience:
                chain.active = False
                print(f"  [stop] chain {chain.name} plateaued (best={chain.best_obj:+.3f})")
            self.save_checkpoint()

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
            "n_seeds": self.n_seeds, "wall_hours": round(self._elapsed() / 3600.0, 3),
            "best": {"name": best["name"], "source": best["source"],
                     "objective": best["objective"], "accounting": best["accounting"],
                     "diagnostics": best["diagnostics"], "full": best["full"]},
            "trajectory": [{"iter": r["iter"], "source": r["source"], "name": r["name"],
                            "objective": r["objective"], "wrist_driven": r["accounting"].get("wrist_driven"),
                            "n_driven": r["accounting"].get("n_driven")}
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
        # UNSTABLE ENTRY (flicker): the stall stage is entered but almost immediately loses
        # dominance BACK to its predecessor. The broken boundary is then k-1 <-> k, not k -> k+1;
        # entry-side edits to stage k+1 cannot fix it, so the directive overrides the focus side.
        rev_in = reverse[k - 1] if (k >= 1 and k - 1 < len(reverse)
                                    and reverse[k - 1] is not None) else None
        occ_k = occ[k] if k < len(occ) else None
        ent_k = entered[k] if k < len(entered) else None
        flicker = (rev_in is not None and ent_k and rev_in >= 0.5 * float(ent_k)
                   and occ_k is not None and float(occ_k) < 0.2 * float(ent_k))
        if flicker:
            lines = [
                f"Stage residence [{table}]. UNSTABLE ENTRY into stage {k} "
                f"('{rep.get('stall_name')}'): it is entered in {_pct(ent_k)} of episodes but holds "
                f"dominance for only {_pct(occ_k)} of steps, and in {_pct(rev_in)} of episodes the "
                f"policy FALLS BACK to stage {k - 1} after entering. The stage {k - 1} <-> {k} "
                f"boundary oscillates: stage {k - 1}'s gate retakes control in the states stage {k} "
                f"produces (or stage {k}'s own channels immediately break its gate condition).",
                f"FOCUS = this unstable boundary: revise stages {k - 1} and {k} COHERENTLY (edit "
                f"menu option d) -- separate their gate conditions so each clearly dominates its own "
                f"region, and make stage {k}'s channels KEEP its own gate condition true while the "
                f"stage acts. Return every OTHER stage byte-identical to the input."]
        else:
            hand_line = ""
            if k < len(entered) and k < len(handoff) and handoff[k] is not None:
                hand_line = (f" Dwell-qualified: stage {k} is ENTERED in {_pct(entered[k])} of "
                             f"episodes but HANDS OFF to stage {k + 1} in only {_pct(handoff[k])} "
                             f"(conversion {_pct(conv[k] if k < len(conv) else None)}).")
            lines = [
                f"Stage residence [{table}]. The policy STALLS in stage {k} "
                f"('{rep.get('stall_name')}'): it reliably reaches this stage but rarely advances "
                f"to stage {k + 1}.{hand_line}",
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

    fa = rep.get("failure_attribution")
    if isinstance(fa, dict) and (fa.get("failure_rate") or 0) > 0.05:
        by = fa.get("by_stage_at_first_failure") or {}
        top = ", ".join(f"{nm} {100*v:.0f}%" for nm, v in
                        sorted(by.items(), key=lambda kv: -kv[1])[:3])
        lines.append(
            f"TASK FAILURE SIGNAL: the task's mistake indicator fires in "
            f"{_pct(fa.get('failure_rate'))} of episodes; the stage IN CONTROL at the first "
            f"failing step: {top}. The mistake belongs to that stage's channels -- fixing it "
            f"outranks advancing the chain.")

    skipped = rep.get("skipped_entry_stages") or []
    if skipped:
        lines.append(
            f"Note: stage(s) {skipped} ({', '.join(repr(names[i]) for i in skipped)}) are SKIPPED "
            f"-- never dwell-entered while deeper stages run. That is gate arithmetic (another "
            f"gate wins in those states), NOT necessarily a failure: if the deeper chain performs, "
            f"leave the skipped stage alone rather than forcing it to dominate.")
    pre = diagnostics.get("pretrained_prior")
    if isinstance(pre, dict) and pre:
        keys = ("success_rate", "grasp_rate", "reach_rate", "lift_reached_rate", "lift_max")
        row = "; ".join(f"{m} {pre.get(m)} -> {diagnostics.get(m)}" for m in keys
                        if pre.get(m) is not None or diagnostics.get(m) is not None)
        lines.append(
            f"PRE-TRAIN vs POST-TRAIN (prior alone at PPO iteration 0 -> after training): {row}. "
            f"A weakness already present pre-train is a PRIOR defect; one that appears (or a "
            f"strength that vanishes) only after training implicates the reward/training side, "
            f"not the prior -- do not 'fix' the prior for it.")
    trend = rep.get("stall_signal_trend")
    if isinstance(trend, dict) and trend:
        rows = "; ".join(f"{s} {_fmt(p)}" for s, p in trend.items())
        lines.append(f"While stage {k} is active, mean signal values over the FIRST -> SECOND half "
                     f"of the episode: {rows}.")
    ie = [r for r in (rep.get("intent_execution") or []) if r.get("gap", 0) >= 0.05]
    if ie:
        rows = "; ".join(f"{r['actuator']}: stage commands {r['stage_cmd']:+.2f} but "
                         f"{r['executed']:+.2f} executes (gap {r['gap']:.2f})" for r in ie[:4])
        lines.append(f"EXECUTED vs INTENDED while stage {k} is active -- the blended action "
                     f"differs from this stage's own channel commands on: {rows}. A large gap "
                     f"means other stages' channels dilute or cancel this stage's intent there.")
    tr_rows = [r for r in (rep.get("tracking") or [])
               if r.get("cmd_vs_measured_frac_of_range", 0) >= 0.15]
    if tr_rows:
        rows = "; ".join(f"{r['actuator']} {r['cmd_vs_measured_frac_of_range']:.2f}"
                         for r in tr_rows[:4])
        lines.append(f"COMMAND vs MEASURED position while stage {k} is active (mean |target - "
                     f"actual| as a fraction of the actuator's range): {rows}. A persistent gap "
                     f"means the joint is saturated, blocked by contact, or physically stopped -- "
                     f"commanding it harder cannot close that gap.")
    pr = rep.get("probe_report")
    if isinstance(pr, dict) and pr:
        prows = []
        for pname, e in pr.items():
            if e.get("error"):
                prows.append(f"{pname}: FAILED TO COMPILE ({e['error']})")
            else:
                el = e.get("early_late_mean") or [None, None]
                scope = f" [stage {e['stage']}]" if e.get("stage") is not None else ""
                prows.append(f"{pname}{scope} = {_fmt(el)} (min {e.get('min')}, max {e.get('max')})")
        lines.append("YOUR PROBES (the measurements you requested, early -> late episode means): "
                     + "; ".join(prows))
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

# TODO: Closing the gate-dominance / stage-success gap — feature plan

Status (2026-07-02): **Increment 1 (features 1, 3, 6) IMPLEMENTED** — transition-based success in
`stage_transition_stats` + `stage_occupancy` (entered/handoff/conversion/reverse, first-broken-hand-off
stall rule, `weakest_stage` endgame), edit menu in `revise_candidate.md` + suggestion heuristic in
`_stage_focus`, weakest-hand-off rendering with reverse-transition (fall-back) warning. Unit-tested
(synthetic episodes: clean chain / transient blip / reverse / out-of-order "air chain") and
offline-replayed on archived programs (dwell filter correctly discards one-step dominance:
reached_frac 0.06 vs entered_frac 0.0).

**Increment 2 (feature 2) IMPLEMENTED** — `_Chain.focus_side` ("entry" → successor's gate/channels
first; after `handoff_attempts` (default 2) rejections, rollback to "exit" = the stage itself, with
the rejected entry attempts carried as `prior_failed_revisions`); `_stage_focus(…, focus_side=…)`
renders FOCUS = ENTRY/EXIT directives (entry skipped when the stall stage itself is barely entered);
reset to entry on every accepted base. Applies to the terminal/weakest branch too.

**Increment 3 (feature 4) IMPLEMENTED** — optional per-stage `success:'<expr>'` (declared in
`framework_freeform_staged.md` + `_output_item`); compiled leniently in `stage_occupancy` (failures
→ `success_compile_errors`, never fatal); `authored_success_frac` = expr>0 for ≥dwell steps at/after
the stage's first dominance; `success_discrepancy` per stage (`handoff_without_success` /
`success_without_handoff` / `entered_without_success` for the terminal stage) rendered as a
DISCREPANCY line in `_stage_focus`. Features 5 and 7 remain planned.

**LIVE VALIDATION (2026-07-02, two full loop runs):**
- `runs/staged_handoff_val_20260702-191014`: seeds authored `success` exprs unprompted (0 compile
  errors); transition stats exposed an out-of-order air-chain (stage 2 entered 16.8% vs stage 1
  2.3% -- raw dominance would have called it "reached"); entry-side focus obeyed
  (`stage1_entry_nudge_close_range`); record explore obj +0.279. Also exposed a loop bug: explore
  seeds pre-charged the global plateau counter (run stopped after ONE refine) -- fixed by resetting
  `since_improve_global` at refine start.
- `runs/staged_handoff_val2_20260702-194333`: full budget used; frontier unlocks 0->1, 0->1
  (adopted DESPITE an objective dip -- the unlock-adoption path fired live), 1->2, 2->terminal;
  objective climbed 0.030 -> 0.084 -> 0.099 -> 0.185 along one chain and a REFINE won the run for
  the first time (`earlier_extract_when_partially_gripped`, +0.185). Final report: chain completes,
  all hand-offs >= 0.17, weakest_stage=0 armed for endgame pressure, and the discrepancy
  cross-check FIRED on real behavior: stage 2 `handoff_without_success` (hands off without its
  squeeze post-condition) and terminal `entered_without_success` (runs without `lift-0.05` holding
  -- honest: trained_success=0.0). Entry->exit ROLLBACK never triggered live (unlocks kept
  resetting focus before 2 consecutive rejections accrued) -- covered by unit tests; watch for it
  in longer runs. Companion to AGENTIC_PRIOR_SELECTION.md ("Stage
localization" and follow-on sections), which documents what is already built: stage occupancy +
stall localization, stall-direction evidence (signal/gate trends, self-lock), training-convergence
verdicts, frontier-aware refinement, and `--per-stage-iters`.

## Context

The `freeform_staged` feedback loop measures "stage reached" as *gate dominance* (the stage's gate
wins the blend for ≥1 step in an episode). That certifies gate arithmetic, not accomplishment, and
produces three failure classes:

1. **False positives** — a gate fires in a state that satisfies its expression while the stage's
   purpose is unmet (e.g. `gripped` is sigmoid of COMMANDED closure, so air-closure lights up the
   clamp/lift stages with no grip; observed: best staged candidate reached `caged_lift` in 14% of
   episodes with success 0).
2. **Misattribution** — an over-permissive stage-k gate hands off garbage; the observable symptom
   (stage k+1 never fires) blames the successor, so revision targets one stage too late.
3. **Transient dominance** — one noisy winning timestep counts as "reached"; no dwell or ordering
   requirement.

Everything below must stay **task-agnostic**: the framework supplies mechanisms only; task knowledge
lives in the LLM-authored candidate (its stages, gates, and — new — success expressions and probes).

Key code to build on:
- `freeform_priors.py` — `stage_occupancy()` (has `active[T,E]`, `sig_te`, `gates_te`, trend
  machinery), `compile_expr` (restricted AST), `freeform_signal_fn`.
- `agentic_orchestrator.py` — `_Chain` (frontier/failed/patience), `_stage_focus()` directive
  rendering, `revise()`, frontier-unlock acceptance, `per_stage_iters` budget resize.
- `ppo_arbiter.py` — `evaluate_candidate_ppo` diagnostics attachment, `training_convergence`.
- `prior_eval.py::validate_program` — passes `stages` through untouched (extra per-stage keys like
  `success` survive validation; `make_freeform_staged_prior_fn` reads only `gate`/`channels`).
- `prompts/` — `revise_candidate.md`, `framework_freeform_staged.md`, `seed_candidates.md`.

---

## Feature 1 — Transition-based stage success (next gate as the success predicate)

**What.** Stage k succeeded in an episode iff stage k+1 subsequently becomes dominant for ≥ `dwell`
consecutive steps (dwell ≈ 3) after k was dominant. Per stage report: `entered_frac` (current
`reached_frac`), `handoff_frac` (episodes with a dwell-qualified k→k+1 transition), and
`conversion = handoff_frac / entered_frac`. Stall rule becomes: deepest stage with entered_frac ≥
thresh but low conversion.

**Implementation.** Extend `stage_occupancy()`: `active[T,E]` already exists; add a numpy
run-length pass for dwell-filtered dominance runs, then ordered per-episode transition detection.
Keep old fields for compatibility. `_stage_focus` adds: "stage k is ENTERED in X% of episodes but
CONVERTS to k+1 in only Y%", and distinguishes "k entered but never converts" (k's fault) from
"k converts but k+1 never does" (k+1's fault) — the misattribution fix.

**Feasibility: high** (~40 lines in one function + rendering; no schema/prompt-format/PPO changes;
verifiable offline against archived programs). **Usefulness: high** — kills false-positive reach and
transient dominance; foundation for features 2, 6, 7.

## Feature 2 — Hand-off-focused revision with rollback (k+1 first, then k with k+1's evidence)

**What.** On a k→k+1 hand-off failure, spend the first 1–2 revision attempts on stage k+1 (its
gate = entry condition, and channels). If rejected, roll back to revising stage k, and the
k-directive carries the failed k+1 attempts (names/objectives, already in `chain.failed`) plus the
signal state k hands over — "the entry side was tried; now fix the exit side."

**Implementation.** `_Chain` gains `focus_stage: int | None`, `focus_attempts: int`. Refine loop: on
a new stall set focus = k+1; after `handoff_attempts` (default 2) rejections shift focus to k and
fold the evidence into the directive. `_stage_focus` gains a `focus` parameter ("Revise ONLY stage
{focus}" + rationale line). Per-stage window (`per_stage_iters`) covers the sum of both phases
(e.g. 4 = 2 on k+1 then 2 on k). Oscillation is prevented structurally: two phases per stall, then
normal plateau rules.

**Feasibility: high** (orchestrator + rendering only). **Usefulness: high** — the control policy
that makes feature 1's evidence actionable.

## Feature 3 — Gate rewriting as an explicit, first-class revision option

**What.** Gates are already editable (stages pass through whole), but the directive emphasizes
channels. Add an explicit EDIT MENU with guidance: self-lock ⇒ gates only; entry-side focus
(feature 2) ⇒ successor's gate is the primary lever; converged training + wrong-way trends ⇒ gate
boundary misplaced (REWRITE the condition, don't rescale); plus "add a guard term" as a named
micro-edit.

**Implementation.** Prompt-only: extend `revise_candidate.md` (menu) and have `_stage_focus` name
which menu entry the evidence supports. `compile_expr` already accepts arbitrary gate expressions.

**Feasibility: trivial.** **Usefulness: medium-high** — the replay showed the model reaches for gate
edits when told the lock is in the gates; this generalizes that into standing guidance.

## Feature 4 — LLM-authored per-stage `success` expressions (backup predicate)

**What.** Candidates may include `success: "<expr>"` per stage — the author's explicit
post-condition over the same signals, compiled by the same restricted AST. Used as a
*backup/cross-check*, not the primary predicate (a wrong success expr is its own failure mode):
when transition conversion (feature 1) and authored success disagree — conversion high, authored
success ≈ 0 — flag a **dominance-success discrepancy**: "your own success test for stage k rarely
passes even though the hand-off fires — either the gate hands off too early or the success expr is
wrong; fix one."

**Implementation.** Schema survives `validate_program` unchanged; verify the staged compiler ignores
the extra key (it does). In `stage_occupancy`, compile `success` when present; report per-stage
`authored_success_frac` — recommend: expr > 0 for ≥dwell steps anywhere in the episode suffix from
first k-dominance (simple, monotone) — plus the discrepancy flag. Prompts: one paragraph each in
`framework_freeform_staged.md` and `seed_candidates.md`.

**Feasibility: medium-high** (eval-window definition is the only subtlety). **Usefulness:
medium-high** — targets the residual gap feature 1 can't see (a hand-off chain whose gates are all
too permissive in a correlated way); disagreement is informative in both directions.

## Feature 5 — LLM-authored diagnostic evals (probe expressions)

**What.** Alongside a revision, the model may request up to ~4 named probe expressions over the
generic signals (optionally stage-masked), e.g. "object xy-speed while stage 2 active". The
framework evaluates them on the next eval's visited states (masked early/late means + min/max, same
machinery as `stall_signal_trend`) and returns numbers in the following revision's diagnostics —
a hypothesis-test loop: reason over failure modes → request the discriminating measurement → get it
next iteration. This is how the LLM gets "more and better context-dependent information" without the
framework learning task nouns.

**Implementation.** Candidate JSON gains optional `probes: [{name, expr, stage?}]`; compile via
`compile_expr` (non-compiling probes reported, not fatal); evaluate in `stage_occupancy` (it already
has `sig_te`, `active`) → `probe_report` in diagnostics; document + echo results in
`revise_candidate.md`; persist probes on the chain until replaced. Cap N; round aggressively
(prompt-size control). Be explicit in the prompt that the vocabulary is the closed generic signal
set.

**Feasibility: medium.** **Usefulness: high with variance** — the stall-direction problem was
exactly "the decisive measurement existed but wasn't surfaced"; probes let the model request such
measurements itself. Build after 1–2 prove the consumption side.

## Feature 6 — Chain-completion gap analysis (all gates pass, task still fails)

**What.** When `reaches_terminal` is true (stall machinery goes quiet) but the objective is far
below what the chain implies, localize the *leak*: worst conversion (feature 1), lowest authored
success (feature 4), or highest **reverse-transition** rate (k+1 falling back to k — grip lost).
Emit `weakest_stage` with the full trend/direction evidence block, so `_stage_focus` keeps giving
stage-localized directives in the endgame instead of "refine whichever stage the diagnostics
implicate".

**Implementation.** In `stage_occupancy`: reverse-transition counts from the same `active` array;
`weakest_stage = argmin(conversion)` on terminal; compute the trend block for it. `_stage_focus`
terminal branch renders it. Refine loop: `focus_stage = weakest_stage` (plugs into feature 2's
plumbing).

**Feasibility: high** once feature 1 exists. **Usefulness: high** — long training runs spend most of
their budget in exactly this endgame state; without this, the stage machinery switches off right
when the budget concentrates.

## Feature 7 — Recursive stage decomposition (split a hard stage into its own sub-chain)

**What.** When a stage exhausts its improvement window (both phases of feature 2 failed) AND
training is `converged`, offer a decomposition move: replace stage k with 2–3 finer stages with
clean hand-offs, all other stages byte-identical. The stage list is flat, so this is a splice —
"recursion" happens in candidate space, not the framework.

**Implementation.** A `decompose` escalation in the refine loop: when a chain would plateau-stop at
stage k, one extra LLM call (new `prompts/decompose_stage.md`: current stage JSON + accumulated
evidence + "split ONLY this stage") and one extra evaluation before deactivating the chain. Counts
as a normal iteration; the new sub-stages are ordinary stages, so features 1/2/6 apply to them
automatically. Guards: cap total stages (~8); offer once per chain; only on plateau + converged.

**Feasibility: medium** (mechanics simple; risks: finer hard-blend gates = more self-lock surface;
comparability resets). **Usefulness: medium, high ceiling** — the only feature that grows the
*structure* when structure is the bottleneck (e.g. `five_finger_clamp` plausibly needs
pre-shape/close/verify); must come last since its value depends on 1/2/6 correctly ruling out
budget and gate-boundary causes first.

---

## Build order

1. **Increment 1 (diagnostic):** Feature 1 + Feature 3 + Feature 6 — all in `stage_occupancy` +
   `_stage_focus` + `revise_candidate.md`.
2. **Increment 2 (control):** Feature 2 — successor-first focus with rollback.
3. **Increment 3 (schema):** Feature 4 — authored success exprs + discrepancy flag.
4. **Increment 4 (extensibility):** Feature 5 — probes.
5. **Increment 5 (structure):** Feature 7 — decomposition, gated behind plateau + converged.

Files touched: `freeform_priors.py`, `agentic_orchestrator.py`, `prompts/revise_candidate.md`,
`prompts/framework_freeform_staged.md`, `prompts/seed_candidates.md`, new
`prompts/decompose_stage.md` (feature 7). `ppo_arbiter.py` unchanged (everything flows through
diagnostics).

## Verification per increment

- **Unit (no env):** synthetic `active`/signal arrays through transition/dwell/conversion code
  (episodes: clean hand-off, transient blip, reverse transition, correlated all-gates-fire);
  `_stage_focus` rendering for every new branch (hand-off focus, rollback-with-context,
  weakest-stage, discrepancy).
- **Offline replay (env, no LLM, minutes):** re-run `stage_occupancy` on archived best programs
  (`runs/staged_stagefocus_20260702-000302`, `runs/staged_stalldir_20260702-125823`) — the
  air-closure candidate should show conversion ≈ 0 where `reached_frac` looked healthy.
- **Live replay (one revise + one PPO eval, ~20 min):** repeat the `runs/stalldir_replay` pattern at
  the stage-2 plateau: successor-first directive, then rollback directive; check the revision
  targets follow the focus.
- **Full loop (hours):** `--per-stage-iters 4 --ppo-train-seconds 420+`; success = frontier advances
  past stage 2, or a clean converged plateau with localized evidence at every step.

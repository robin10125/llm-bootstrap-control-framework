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

## Feature 8 — LLM-authored parallel stage tracks (overlapping stage execution)

**What.** Allow a candidate to author independent stage tracks whose active intervals can overlap,
instead of forcing every useful behavior through one global first-unfinished ladder. A track would
still be stateless and gate-driven, but it could run concurrently with another track when both of
their observable conditions hold. This is for mechanisms whose progress conditions are independent
enough to be maintained or prepared in parallel while the main chain advances.

**Implementation sketch.** Extend the staged representation with optional `tracks`, each containing
its own ordered progress ladder and per-track stages. The executor combines active stages across
tracks by summing channels, with explicit conflict rules when two active stages command the same
actuator/group (for example: reject at validation unless the candidate declares a priority,
exclusive group, or blend rule). Diagnostics need per-track occupancy, cross-track overlap
fractions, actuator-conflict counts, and hand-off stats within each track. Revision prompts should
ask the author to keep track dependencies observable: a parallel stage must have a clear entry
condition, exit activation, and suppression condition if it should stop when another track changes
state.

**Feasibility: medium.** Requires representation/schema changes, staged executor changes, validation
for actuator conflicts, and new diagnostics. **Usefulness: potentially high** when the one-stage
ladder serializes independent preparation/maintenance work and consumes the rollout budget, but it
adds a new failure surface: hidden cross-track competition, persistent stale commands, and harder
attribution when a shared signal moves the wrong way.

---

## Future experiment: soft vs hard blend (controlled)

Observational evidence strongly favors `blend='soft'` — the winners of 4/4 staged runs were soft
except the self-locking `hard_guarded_funnel_grip` (0.041, the project's canonical failure case),
and the self-lock failure mode is *structural* to hard blend (argmax can trap the policy in a stage
whose own actions keep its gate winning; soft always gives weight to a rising successor gate).
Other hard-blend candidates placed at the bottom of their batches (`hard_guarded_handoff` +0.018).
But blend is confounded with everything else (each candidate differs in stages/gates/channels; only
~4 hard candidates sampled), so this is NOT rigorously established.

**Controlled test (cheap, existing machinery):** take the 2–3 best staged programs, flip ONLY the
`blend` field, and run each variant through `evaluate_candidate_ppo` at matched budget
(~8 min/eval, ~6 evals ≈ 1 h). This isolates the blend variable the same way PRELIM_dsl_vs_freeform
isolated the sub-prior representation. Secondary readout: whether hard blend's crisper stage
separation makes the occupancy/hand-off diagnostics sharper enough to matter for the feedback loop.
Until run, treat **soft as the default recommendation**.

## Future feature — Pacing curriculum: learn the slow rollout, then learn it sped up

**What.** Two-phase training over rollout length / pace. **Phase 1:** learn the task at a GENEROUS
rollout (long episode, the prior's slow asymptotic pacing acceptable) so the policy actually reaches
the terminal stage and gets the grasp/lift at all. **Phase 2:** progressively SHORTEN the episode
budget (and/or tighten a pace/speed requirement) across a curriculum so the *same* accomplished
behavior is compressed into the target 20 s budget. Separates "can it do the task at all" from "can
it do it fast" — a slow-but-correct prior/policy is accelerated instead of discarded for missing the
budget.

**Why.** The timing/fit work (`est_seconds`, `time_report.measured_vs_est_ratio`, the `PACE`
directive) makes a slow prior FAIL the 20 s rollout (later stages never run) even when the early
behavior is correct. Those features *diagnose* slowness and push the author toward saturated
(non-asymptotic) channels; this feature is the complementary *recovery* path — when the correct
motion is inherently multi-stage and simply needs the budget first, learn it long, then compress.

**Implementation sketch.** In the PPO arbiter / training loop (`evaluate_candidate_ppo`): a schedule
that starts `episode_seconds` high (e.g. 40–60 s) and decays toward the target (20 s) as training
progresses, advancing the compression only once the chain reaches terminal at the current horizon
(use the existing `time_report` / conversion stats as the curriculum gate). Alternative lever: a
reward term that increasingly rewards reaching each stage EARLIER. Stays task-agnostic — the
curriculum is over the generic rollout-length/pace hyperparameter, never task nouns.

**Feasibility: medium** (training-loop change; horizon is already a config hyperparam).
**Usefulness: high for slow-but-correct priors** — the natural partner to the est_seconds/pace
diagnostics: they DIAGNOSE slowness, this one RECOVERS from it by learning the fast version rather
than requiring it be authored up front.

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

##Future Feature - Vision feedback
(expand on this)

##Future Feature - Prevent action prior from fighting neural policy

Maybe focus on softer biases that work within neural policy system.  Use neural policy interface as the same interface as the prior/align prior interface with neural policy interface.

Potential avenues of expoloration:
  1. Prior As Exploration Proposal
  Use the prior to bias sampling, not final policy semantics.

  The neural policy still defines the learned action distribution, but rollout collection sometimes samples candidate
  actions near the prior suggestion. The stored PPO action/log-prob must correspond to the actual neural distribution or be
  handled as off-policy data.

  Useful variants:

  - mixture sampling: sometimes sample from policy, sometimes from prior-neighborhood proposals,
  - prior-centered exploration noise early in training,
  - anneal prior proposal probability to zero,
  - use prior only in low-confidence states.

  This helps discover useful regions without permanently constraining the final policy.

  2. Prior As Curriculum / State-Visitation Shaper
  Use the prior to bring the agent into informative states, then train the neural policy there.

  Examples:

  - rollout reset/state collection via prior-driven episodes,
  - prior-assisted warmup episodes whose visited states seed PPO batches,
  - fragment training where the prior helps reach later-stage states, but the policy learns from those states independently,
  - staged curriculum based on authored stage.success expressions.

  This avoids “copy this action” and instead says “make sure training sees the relevant parts of the task.”

  3. Prior As Value Shaping
  Convert prior structure into value/reward information, not action commands.

  Examples:

  - reward progress on authored stage success expressions,
  - reward reduction in authored constraint violations,
  - reward reaching states where future task success is more likely,
  - add a critic auxiliary target estimating prior-defined progress.

  This is probably the cleanest fit for your stated goal: the policy can choose any action, including holding still, but it
  receives denser information about which states are promising.

  4. Prior As Advantage Baseline Or Critic Feature
  Feed prior diagnostics into the critic, not the actor action path.

  Examples:

  - active stage id/weights,
  - prior suggested action norm,
  - prior-policy disagreement magnitude,
  - stage success margin,
  - authored constraint margins.

  The critic can learn that some states are valuable because the prior sees structure there, while the actor remains
  expressive.

  Important constraint: if these become observation features for the neural policy, they change the policy interface. That
  can be good, but it should be explicit.

  5. Prior As Auxiliary Prediction Task
  Train the network to understand the prior without forcing it to obey.

  Auxiliary losses:

  - predict active stage,
  - predict prior-suggested action,
  - predict next-stage success margin,
  - predict whether a prior channel would be saturated/blocked,
  - predict authored probe outcomes.

  This shapes representation learning. The actor head can still ignore the prior when task reward says otherwise.

  6. Prior As Regularizer, Not Target
  Instead of supervised imitation, use a weak, conditional regularizer.

  Examples:

  - penalize disagreement only when the prior has high confidence,
  - penalize disagreement only early in training,
  - penalize disagreement only for selected actuator groups,
  - penalize large disagreement in value-equivalent states, not all states,
  - use KL-to-prior as an annealed exploration prior, not a permanent constraint.

  This is safer than hard action bias, but still risks teaching cancellation if too strong.

  7. Prior As Action-Ranking Hint
  Rather than adding a vector to the action, use the prior to rank sampled candidates.

  Flow:

  1. Neural policy proposes several candidate actions.
  2. Prior scores them structurally.
  3. Environment executes the selected neural-proposed action.
  4. PPO trains on the chosen action with correct sampling accounting.

  This keeps executed actions inside the policy’s support and lets the prior guide selection rather than override.

  8. Prior As Safety / Constraint Critic
  Use authored constraints as penalties or diagnostics, not hard action projection.

  Examples:

  - contact-force budget violations,
  - commanded-vs-measured gap budgets,
  - speed/oscillation budgets,
  - stage-local constraint margins.

  The policy learns that some action/state transitions are costly. It can still violate them if the real task objective
  justifies it, unless you intentionally make them hard constraints.

  9. Prior As Teacher For “What To Attend To”
  Use prior signals, probes, and stages to decide what diagnostics to collect and what failures to explain.

  Examples:

  - stage occupancy,
  - prior-policy disagreement by actuator group,
  - success-expression trend,
  - constraint violation trend,
  - which authored channels would have fired in failed states.

  This guides revision and analysis without touching action selection.

  10. Prior As Search Space For Learned Options
  Instead of executing prior channels directly, use them to define latent option labels or skill regions.

  Examples:

  - stage-conditioned policy heads,
  - option-conditioned value functions,
  - learned latent conditioned on prior stage,
  - policy learns continuous skills corresponding to prior-authored phases.

  This is more invasive, but can make long-horizon learning easier without requiring the prior vector to be canceled joint-
  by-joint.

  My recommendation: make the default advisor_only stack use three mechanisms first: stage-success reward shaping, critic/
  diagnostic features, and prior-guided state visitation. Keep action bias and residual-scaled prior only as ablations.



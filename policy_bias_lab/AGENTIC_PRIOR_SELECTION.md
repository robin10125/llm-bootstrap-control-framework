# Agentic, robot/task-agnostic action-prior selection

A refactor of the action-prior prompting system. Goal: make prior authoring **completely robot-,
task-, and environment-agnostic** by injecting those details *modularly*, giving the model free
rein to design an optimal prior from the actual robot DOF, under an **agentic, budget-bounded**
selection loop that ends in a compilable prior.

This replaces the hardcoded `ACTION_GROUPS` / `PRIOR_DIRECTIONS` vocabulary and the single
fixed candidate prompt in `action_priors.py`.

## Why (motivating defect)

The old vocabulary is hardcoded and omits DOF. Concrete symptom: the Shadow hand's **wrist**
(`rh_A_WRJ1/2`, actuated, in the action space) is never biased because (a) no `wrist` group
exists, (b) no tilt direction exists, and (c) `close_hand` zeroes on the wrist (open pose ==
close pose). So no prior — LLM or hand-authored — can articulate it, and PPO suppresses the
unrewarded dimension. **Root cause: the action vocabulary is fixed instead of derived from the
robot.** The load-bearing principle of this refactor: *the model is shown the robot's real
controllable DOF and composes priors over them.*

## Principles
- **Vocabulary is derived from the robot spec, not hardcoded.** Agnosticism and "free rein" both
  reduce to this. A new robot/task is supported by swapping injected specs; no code changes.
- **The vocabulary must form an approximate complete motion basis.** Every joint must be *actively
  movable* from the priors (both directions) — holding/stabilize is not sufficient. The motion
  primitives (`freeform_priors.derive_motion_basis`) span the robot's legal configuration space, so
  any reachable orientation is a combination of primitives; `check_basis_complete` rejects any DOF
  that is un-drivable or single-sign-only. This is the structural fix for the wrist-omission class
  of defect.
- **Compilable & system-fitting.** The output must compile into `composed_priors` (modes:
  `monolithic | gated(soft|subgoal|options|stacked) | reactive_law | dmp`) and is validated by
  rollout. The prior-system spec is injected so the model knows the target representation.
- **Empirically grounded.** An LLM proposes; the contact-gated rollout selector (real-world
  observable signals only) decides. Cheap to reproduce on hardware.
- **Staged, single-job prompts.** Each LLM call has exactly one task — no mega-prompt.
- **Understand before structuring.** The model forms understanding first; action groupings emerge
  from that understanding, not from a rigid template fixed up front.

## Injected specs (modular, schema-described in the base template)

1. **Robot spec** — morphology & control. Actuator table `[{name, joint_type, range, drives
   (body/dof), semantic_tag}]`; kinematic groupings (sub-chains); proprioceptive observables;
   control law (incremental-position / torque, `action_scale`); embodiment quirks (e.g. the
   base→world sign flip `M=diag(1,-1,-1)`).
2. **Task spec** — goal (natural language); success predicate (observable); available reward
   signals; constraints / anti-exploits (anti-fling, sim-to-real, "no non-transferable tricks").
3. **Environment spec** — scene; object(s) `{size, mass, pose_distribution}`; surfaces; observable
   state-vector layout; runtime perception signals (so gates reference only real signals).
4. **Prior-system spec** — the *target representation*: `composed_priors` modes, gating
   disciplines, operator semantics, weight/stateless constraints, what is runtime-tunable. Tells
   the model what KIND of artifact to emit and that it must compile (e.g. "a stacked-gate prior").

## Base template scaffolding (robot/task-invariant)
- **Framing** — what an action prior is mechanically (weak pre-tanh mean shift biasing PPO
  exploration; reactive; *stateless* fn(obs, weights); no fixed coordinates; gated on observable
  signals).
- **Injection map** — labeled, delimited sections that identify each spec block and how its fields
  relate. (The "contextualize and identify the injected information" requirement.)
- **Reasoning approach** — references the Stage-0 context and the strategy spine (below).
- **Output contract** — references robot-derived primitives + gates + rationale, in the
  prior-system schema.

## Stage 0 — Context generation (parallel, single-job calls; outputs injected as CONTEXT)

Each returns structured context for later stages; **not** secondary instructions inside the
generator prompt (keeps the generator's intent clean).

- **C1 — itemized execution description.** What each joint/actuator does over a *successful* run,
  in excruciating detail (per-actuator, phase-by-phase). Given robot+task+env. *This is context,
  so the generator sees a rich per-joint narrative instead of just "grab the object."*
- **C2 — failure-mode & reward-hack enumeration.** How the task is commonly failed/exploited
  (fling, air-closure, fisting, etc.). Given task+env+constraints.
- **C3 — kinematic / affordance analysis.** Reachable workspace + object geometry → required
  contact configuration (caging, opposition, surface normals) → implicated DOF. Given robot+env.
  **EXPERIMENT FLAG:** inject before generation vs. after a first draft — test both orderings.
- **C4 — human/biological analogy** *(optional, behind a flag).* How a human effector does it,
  mapped to the robot's DOF. Likely redundant with general reasoning; included only to A/B whether
  analogy helps recruit unused DOF (e.g. wrist tilt).

## Stage 1 — Understand + flexible vocabulary
The agent reads the four specs + Stage-0 context, forms its understanding, and enumerates the
relevant controllable DOF and *ad-hoc* groupings **as the problem requires** — explicitly not a
rigid pre-committed template. This is where unused DOF (the wrist) surface.

## Stage 2 — Diverse seed candidates
Generate a small set of genuinely different compilable candidates (different modes / structures /
strategies), each grounded in the context and the prior-system schema.

## Stage 3 — Empirical evaluation (consumes budget)
Compile each candidate and evaluate it. **The arbiter is a SHORT PPO run, ranked on the TRAINED
contact-gated success** (`ppo_arbiter.evaluate_candidate_ppo` → `graded_objective`), NOT the
open-loop rollout score. Rationale: the open-loop scorer is demonstrably **blind and can invert the
PPO ranking** (prelim: DSL open-loop −0.10 "beat" free-form −0.57, yet free-form trained better), so
optimizing it — especially with a refinement loop — can drive *away* from the PPO-best prior. The
open-loop score (`prior_eval.score_program`) is retained only as a cheap **prefilter** (compile + DOF
accounting reject non-compiling/blind candidates before a PPO run is spent). One short-PPO evaluation
= **one iteration**; the budget governs the scarce real-rollout-equivalent resource.

## Stage 4 — Iterative revision (consumes budget)
The selector agent critiques a promising seed against the failure modes (C2), the *actual* DOF, and
the **trained policy's behavioral diagnostics** (`ppo_arbiter.behavioral_diagnostics`: reach/grasp/
lift/closure/saturation + an explicit failure-mode read — air-closure, grasp-no-lift, flinging,
transient-only, saturating), emits a revised candidate, and re-evaluates. So the model fixes what
the policy actually *did*, not an open-loop proxy. Repeat under budget.

### Stage localization for `freeform_staged` (built) — and its open problem
For `freeform_staged`, the arbiter also attaches a **task-agnostic stage-occupancy report**
(`freeform_priors.stage_occupancy`, fed obs from `evaluate_ppo_policy(return_obs=True)`): it recomputes
the *model's own* stage gates over the states the trained policy actually visited, and reports, per
authored stage, the time-share (`occupancy`), the fraction of episodes that ever reach it
(`reached_frac`), and the deepest reliably-reached non-terminal stage as `stall_stage`. `revise()` turns
this into a directive (`_stage_focus`) that tells the LLM to revise **only** the stalling stage and return
every other stage byte-identical — so each refine is a controlled single-stage ablation. This reads only
the stages/gates the model wrote (no task nouns), so it generalizes to any staged prior on any task.

**KNOWN PROBLEM (surface the failing signals AT the stage).** Localization tells the model *where* it is
stuck but not *which direction* to fix it, and the LLM's default reading of "you stall in stage k" is
"make stage k push harder" — which is often exactly wrong. Observed in
`runs/staged_stagefocus_20260702-000302`: the best candidate (`hard_guarded_funnel_grip`, hard blend) had
`occupancy[stage0]=1.000` and `reached_frac[stage1]=0.012` — a **self-reinforcing lock**. Its stage-0
"recover_far_or_lost" gate stays true because stage 0's own aggressive centering **displaces the target**
(`obj_xy_drift=0.73 m`), which keeps "far/lost" true, which (under the hard/argmax blend) keeps stage 0
winning forever. The two stage-0-focused refines the loop produced were named
`stronger_far_approach_centering` / `stage0_bidirectional_recenter_and_height_recover` — i.e. the model
pushed *harder*, tightening the lock, and the objective **dropped** 0.041 → 0.036 → 0.022 before plateau
early-stop.

The decisive evidence (`obj_xy_drift` large and *growing* — the stall stage is diverging the very signal
it acts on) is present in the diagnostics but was **not surfaced at the stalling stage**, so the model
defaulted to "try harder."

**FIX (built 2026-07-02, task-agnostic).** `stage_occupancy` now attaches *direction* evidence for the
stall stage, computed only over timesteps where that stage is dominant:
- `stall_signal_trend`: each generic signal's mean over the first vs second half of the episode — the
  model can see which signal is moving the wrong way *under its own stage's channels*.
- `stall_gate` / `next_gate`: the raw gate values (same episode-half split) of the stalling stage and its
  successor, plus which signals the successor's gate expression reads (word-boundary match over the
  generic signal names). A flat/falling next-gate value ⇒ the channels are not moving the state toward
  the hand-off.
- `self_lock` (structural): `blend=="hard"` ∧ `occupancy[stall]≥0.95` ∧ successor `reached_frac≤0.05` ⇒
  the argmax gate never yields; no channel change alone can exit the stage.

`_stage_focus` renders these into the revise directive: the per-signal trend table, the next gate's
expression + trajectory, an explicit "NOT rising ⇒ don't push harder, use a gentler/decelerating or
sign-corrected response, or reshape the hand-off" instruction, and — when `self_lock` — "fix the GATES,
not the channel strength." `revise_candidate.md` mirrors this ("use the signal trends to pick the
DIRECTION of your change"). Everything is derived from the model's own gates and the generic signal set;
no task nouns.

**VALIDATED (2026-07-02, replay in `runs/stalldir_replay/`).** A single revise was replayed against the
archived self-locked `hard_guarded_funnel_grip`. The fresh eval reproduced the lock (stall 0, occupancy
0.9994, `self_lock=true`; the prompt showed `near` falling 0.027→0.007 and `palm_obj_dist` diverging
0.182→0.416 under stage 0, next-gate value decaying). Where the old loop's refines were named
`stronger_*` and DROPPED the objective (0.041→0.036→0.022), the direction-informed refine produced
`stage0_gate_handoff_and_decelerated_recover` — gate narrowing + decelerated recover, rationale citing
the self-lock — and one refine broke the lock: objective +0.0367→+0.0406, `self_lock` cleared, stall
moved downstream 0→2 (`five_finger_clamp`), reached_frac stage1 0.023→0.27 and stage2 0.004→0.41. The
new stall's report shows a slowly *converging* grasp (next gate rising 0.054→0.068), i.e. normal
iterative-refinement territory rather than a divergence.

### Under-training vs prior-defect discrimination (built 2026-07-02)
A candidate can score low for two different reasons: the prior is wrong, or the prior is fine and the
short-PPO budget ran out — and the revision loop must not "fix" the latter (observed in
`runs/stalldir_frontier`: three stage-2 revisions of a slowly-converging grasp all scored *worse* than
the base, plateau-stopping the loop; nothing told the model the base might simply be under-trained).
`ppo_arbiter.training_convergence(rows)` compares each training metric (env `base_return` + the generic
progress rates) over the middle vs last third of training iterations (first third = warmup noise):
- any metric still clearly rising at budget end ⇒ **`undertrained`** — `_stage_focus` appends a
  TRAINING BUDGET CAVEAT ("the low score may reflect UNDER-TRAINING, not a prior defect; prefer a
  MINIMAL change over restructuring").
- all flat/falling ⇒ **`converged`** — "the prior, not the budget, is the limiter; a real change is
  warranted."
The verdict is attached as `diagnostics["training_report"]` (so it also appears verbatim in the
diagnostics JSON of the revise prompt) and the caveat renders on every `_stage_focus` branch. Purely
budget/telemetry-based; no task nouns.

### Frontier-aware refinement (built 2026-07-02)
Objective-only acceptance rejects exactly the revisions that matter for long runs: a revision that
UNLOCKS a deeper stage usually starts out with a *worse* scalar objective (the newly reached stage
doesn't work yet), so the old loop would discard it and plateau-stop (observed in
`runs/stalldir_frontier`: a revision faintly reaching stages 3–4 was rejected on objective). The refine
loop now:
- tracks each chain's **stage frontier** (`_frontier`: stall index, or n_stages when terminal is
  reached) alongside its best objective;
- **adopts** a revision as the new base when the objective improves OR the frontier advances. An
  unlock resets the chain's patience window (and the global one), giving the newly unlocked stage its
  own round of iterative improvement; `chain.best_obj` keeps the high-water mark so the objective
  ratchet stays honest, and `finish()` still returns the best candidate by objective — a failed unlock
  line cannot win the run;
- feeds **rejected revisions** (`prior_failed_revisions`: name + objective, cleared on each accepted
  base) back into the revise diagnostics, with a prompt instruction not to re-propose them.
For LONG training runs, raise `--ppo-train-seconds`: `training_convergence` then returns `converged`
verdicts (making real changes warranted) instead of under-training caveats, and the frontier machinery
gives each unlocked stage its improvement window instead of dying on the first objective dip.

**Per-stage budget guarantee:** `--per-stage-iters N` sets `patience = N` and, once the refine seed(s)
are chosen (the stage count is LLM-authored, so only known then), resizes the budget to
`explore_iters + N × n_stages × n_chains` — every authored stage is guaranteed N improvement attempts
if it becomes the failing frontier. E.g. a 5-stage seed with N=4: budget = 3 + 20 = 23 iterations.

### Gate dominance ≠ stage success: transition-based success (increment 1 built 2026-07-02)
"Reached" via raw gate dominance (one winning step) certifies gate arithmetic, not accomplishment.
`stage_transition_stats` (freeform_priors.py) now defines stage success by the model's OWN hand-off:
stage k succeeded in an episode iff stage k+1 subsequently stays dominant ≥ `dwell` (3) consecutive
steps after k was dominant — ordered, so a correlated "air chain" firing out of sequence doesn't count.
Per stage: `entered_frac` (dwell-qualified entry), `handoff_frac` (ordered k→k+1 transitions),
`conversion` (= handoff/reached), `reverse_frac` (fall-backs k+1→k, e.g. grip lost). The stall rule is
now the FIRST BROKEN HAND-OFF (shallowest reliably-entered stage whose successor rarely dwells); when
no hand-off is broken, `weakest_stage` (lowest hand-off rate) keeps the endgame stage-localized —
`_stage_focus` renders "the chain COMPLETES, but its WEAKEST hand-off is stage k" with the full trend
evidence block and a fall-back warning when `reverse_frac ≥ 0.2`. The revise prompt gained an EDIT
MENU ((a) reshape channel response, (b) nudge gate threshold, (c) REWRITE gate condition, (d)
restructure the hand-off), and `_stage_focus` suggests the entry the evidence supports (self-lock ⇒ d;
next-gate rising ⇒ b; converged + not approaching ⇒ c; else a). Validated: unit tests over synthetic
episodes (clean chain / transient blip / reverse / out-of-order), and offline replay on archived
programs showed the dwell filter discarding transient dominance (reached 0.06 vs entered 0.0).
**Hand-off-focused revision with rollback (built 2026-07-02):** each chain revises the ENTRY side of
the broken hand-off first (stage k+1's gate/channels — the cheap fix); after `handoff_attempts`
(default 2) rejected attempts it rolls back to the EXIT side (stage k itself), carrying the rejected
entry attempts as `prior_failed_revisions` in the prompt. Focus resets to entry on every accepted
base; the entry phase is skipped when the stall stage itself is barely entered (its own activation is
the problem). Applies equally to the endgame weakest-hand-off focus.

**Authored success expressions (built 2026-07-02):** stages may declare `success:'<expr>'` — the
author's post-condition over the same signals (declared in `framework_freeform_staged.md` and the
output schema). `stage_occupancy` compiles them leniently (`success_compile_errors`, never fatal) and
reports `authored_success_frac` (expr > 0 for ≥ dwell steps at/after the stage's first dominance) plus
a per-stage `success_discrepancy` cross-check against the hand-off predicate:
`handoff_without_success` (gate hands off before the job is done, or the success expr is wrong),
`success_without_handoff` (successor's entry gate too strict), `entered_without_success` (terminal
stage runs without accomplishing its post-condition). `_stage_focus` renders these as DISCREPANCY
lines with both percentages. Remaining planned features (probes, decomposition):
**TODO_STAGE_SUCCESS_FEATURES.md**.

## Orchestrator — agentic within a fixed structure

One **iteration = one short-PPO evaluation**. Budget = a single hyperparameter `B` (default **10**).
The agent is agentic *within* this structure (it decides early-stop, which seeds to refine, and
budget split), not free-form:

```
explore:  evaluate up to 3 DIVERSE seed strategies (1 iter each).     # 1..3 iters
refine:   pick the best seed(s) — one by default, two if multiple look
          good — and spend the REMAINING budget on revise→eval cycles,
          agent allocating across the chosen seed(s).                  # B - (explore) iters
finish:   return the best seed overall by the trained objective.
```

**Default `n_seeds=3`, `B=10` (≈3 explore + 7 refine) are empirically set** by the marginal-value
run (`run_marginal_value`, mv_20260630-183954): breadth **saturates by ~3 seeds** (seed 2 added
~0.001, seed 4 added 0), while depth is a **fat-tailed** lever (a single revision tripled the best
program, 0.042→0.130) — so budget is deliberately skewed toward refinement, with plateau early-stop
trimming it. Both remain agent-/CLI-adjustable. (Caveat: n=2 reps; provisional.)

**Early stopping is performance-dynamics-based, NOT threshold-based.** The agent does not stop a
line of work because a score crossed some "promising" bar; it stops when the contact-gated
objective **plateaus or consistently degrades**:
- maintain the best-so-far score for the line (a refinement chain, or the whole run);
- with patience window `w` (default 3) and tolerance `eps`, stop a refinement chain when the last
  `w` revisions show no improvement `> eps` over that chain's best (**plateau**) or a monotone/
  near-monotone decrease (**degradation**) — then move to the next promising seed or finish;
- at the loop level, stop spending budget early when best-so-far has not improved `> eps` over the
  last `w` iterations, even if budget remains.

This avoids a brittle "promising threshold" hyperparameter; the agent keeps improving a seed until
it stops paying off, rather than quitting as soon as something is merely good enough.

- Stage-0 context calls and revision *reasoning* calls are LLM calls, **not** counted against `B`
  (the budget governs expensive rollout-evaluations — the real-robot-relevant resource); the agent
  is instructed to be frugal with reasoning calls regardless.
- Default split at `B=10`: up to 3 explore, ≥7 refine (empirically set; see above). Agent-adjustable.
- Justification for real-world embodied training: each iteration maps to a small batch of real
  rollouts; "diverse-then-refine" spends a fixed real-trial budget the way a human experimenter
  would, and subsumes a static portfolio (diversity at seed, depth via revision).

### Checkpoint / pause / resume (built 2026-07-02)

Long runs (`--per-stage-iters 4 --ppo-train-seconds 420+` is hours) will be paused and resumed
repeatedly, so the orchestrator persists its full progress to `<out>/checkpoint.pkl` (atomic
tmp+rename pickle) **after every evaluation** and at every phase boundary (context done, seeds done,
seed selection done). Persisted state: iteration count, all evaluated records (trained `best_params`
device-fetched to numpy), the context block, the seed list + explore cursor, every refinement chain
(`_Chain` incl. frontier / focus_side / failed-revision history), the chosen-chain indices,
round-robin cursor, plateau counters, and any `per_stage_iters` budget resize.

- **Pause**: `SIGINT`/`SIGTERM` (Ctrl+C, or `kill <pid>` on a detached run) sets a flag; the
  in-flight evaluation finishes, a final checkpoint is written, and the process exits cleanly with a
  `[paused]` line showing the resume command. A second signal aborts immediately — safe, because the
  checkpoint after the last finished eval is already on disk (a hard `SIGKILL`/power loss likewise
  loses at most the in-flight eval). `--stop-after N` pauses after N evaluations this session
  (planned pauses, e.g. "run 3 more iterations tonight").
- **Resume**: `run_agentic_selection --out <same dir> --resume` reloads config + state and re-enters
  the loop exactly where it left off (mid-explore, pre-refine, or mid-refine). Config comes from the
  checkpoint; any flag passed explicitly on the resume command line overrides it — in particular
  `--budget <bigger>` extends a run whose budget ran out or whose chains plateaued
  (`extend_budget` reactivates the chosen chains with a fresh patience window).
- Validated end-to-end (offline smoke, 3 simulated sessions through disk): pause mid-explore →
  resume → pause pre-refine → resume → completion, with contiguous iteration numbering and all
  explore/refine records preserved in the final `report.json`.

### Live metrics dashboard (built 2026-07-02)

`<out>/dashboard.html` — a self-contained HTML page (inline SVG, no JS/CDN/server/deps) written by
`run_dashboard.write_dashboard` from the same payload as the checkpoint, on every evaluation. Open
it once in a browser (`file://` works); a 20 s meta-refresh keeps it live. Built to answer "how much
longer should I run this?":

- **stat cards**: iters/budget, best objective + candidate, global plateau counter, wall-clock
  **ETA to budget** (mean of the last 5 eval durations × remaining evals), latest training verdict;
- **run guidance** box (heuristics): plateau imminent / all chains dead / budget exhausted (with the
  `--resume --budget N` remedy), latest-candidate `undertrained` → raise `--ppo-train-seconds`,
  recent frontier unlocks → keep running, best-objective still improving within the patience window;
- **charts**: objective per evaluation (per-chain series + best-so-far step envelope);
  trained-policy metrics (success/grasp/reach/lift per eval); **stage frontier progression** per
  chain vs the terminal line; **per-stage gate performance** bars for each refining chain's current
  base (entered / hand-off / conversion / authored-success — the transition-stats data);
  the latest eval's **PPO learning curve** (from telemetry rows now persisted through
  `evaluate_candidate_ppo` → `full.telemetry`, ≤150 downsampled rows);
- **tables**: chain status (role, best obj, frontier/n_stages, focus side, plateau, active) and the
  last 10 evaluations with failure-mode flags.

Regenerate offline for any paused/finished run: `python -m policy_bias_lab.run_dashboard runs/<dir>`.
Dashboard writing is wrapped in try/except inside `save_checkpoint` — a rendering bug can never
take the run down. **Off by default** (paused per user request 2026-07-02): enable with
`--dashboard`, or regenerate manually from the checkpoint at any time.

### Long-duration training with the selected prior: `run_long_ppo` (built 2026-07-02)

**Division of labor (user-set):** the selection loop keeps its FIXED budget of LLM revision calls
-- its job is only to pick a good prior. The long-duration question ("what happens when a policy
trains under that prior for many hours") is answered by `run_long_ppo`, which makes ZERO LLM calls:
one continuous PPO run (single XLA compile; Adam state, params, iteration count, run-time clocks
all in `resume.pkl`) on a fixed `--program best_program.json`, optionally warm-started from the
selection arbiter's weights (`--init-params best_params.pkl`, same 256x256 net). Termination only
on:

- **training plateau**: the best-checkpoint metric (sustained contact-gated success + gated-lift /
  grasp tie-breaks, same metric `train_ppo_arm` uses to pick checkpoints) unimproved by
  `--plateau-eps` for `--plateau-hours` (default 2 h), gated by `--min-hours` (default 8 h). At the
  default eps (1e-5), new sustained success or gated-lift progress resets the clock; bare
  grasp-rate wiggle does not.
- **sustained success**: rolling mean (`--success-window`, default 10 iters) of the training
  sustained contact-gated hold rate reaches `--success-stop` (default 0.8).

Pause with SIGINT/SIGTERM (finishes the current PPO iteration, ~seconds) or `--stop-after-iters`;
resume with `--resume`; `resume.pkl` also refreshes every `--save-every-iters` (crash insurance);
paused time never counts toward the criteria; per-iteration metrics append to `metrics.jsonl`
across sessions; on termination the global-best params get a full held-out eval into
`final_report.json`. Implemented via three hooks added to `train_ppo_arm`
(`initial_opt_state`, per-iteration `control_fn`, `state_out`) -- existing callers unaffected.

### Reward-mode experiment arms for long training (built 2026-07-03)

The first long run (`runs/longppo_20260702-234528`, 2.1 h) reward-hacked: the builtin base
reward's dense state-based closure term paid ~12/episode for AIR-CLOSURE hovering (fingers curled
at 3.3 cm, no contact, no lift; arithmetic match: estimated 11.3 vs observed 11.8), base_return
rose 5x while reach/grasp/lift collapsed, and the shaping template that penalizes exactly this
(`closure_contact_consistency`) fired but at 1/6 the exploit's income. Design principles adopted
(user): shaping must target CURRENT weaknesses, not blindly pay mastered behavior; lift +
sustained lift must dominate intermediate income; intermediate rewards should be stage-gated like
the action prior so premature behavior (grasp before reach) is never paid.

Three arms in `reward_modes.py`, selected by `run_long_ppo --reward-mode ...` (no changes to the
shared env; intermediate base terms are zeroed via EnvConfig overrides and replacements live in a
`shaping_fn(prev_eval, eval_vec, obs)` hook in the collect loop):

- **`lift_only`** -- env pays ONLY the contact-gated lift terms (w_lift/w_lift_pot/w_lift_hold/
  w_success); all intermediate income (approach, closure, contact, hold) and all template shaping
  removed. Tests whether the action prior alone can carry the policy to contact.
- **`adjusted`** -- the diagnosis fixes: closure paid on PROGRESS (telescoping Δclosure, bounded
  ~0.25 total vs the old 12/episode) behind a tight 0.02 m fingertip gate; contact paid on
  progress (Δcontact_gate, symmetric so losing a grip costs); empty-hand squeezing penalized at
  the magnitude the old exploit paid (-0.25/step); env keeps its safe potential-based approach
  terms.
- **`stage_gated`** -- same adjusted terms, but each multiplied by the prior program's OWN stage
  weights (`make_stage_weight_fn`, identical soft/hard blend the prior acts with): approach terms
  pay only while a base-driving stage is active, closure/contact only while a hand-driving stage
  is active (stage->term mapping read mechanically from which actuators each stage's channels
  drive -- task-agnostic). The squeeze penalty stays ungated. Contrib slots 0-4 in metrics.jsonl:
  approach_potential, closure_progress, contact_progress, empty_squeeze_penalty, hand_gate_mean.

All three keep the lift terms dominant by construction (held 5 cm lift earns ~0.6/step + height
potential + 3/step sustained-aloft vs bounded intermediate totals of ~1-2 per episode).

### Wall-clock termination for the selection loop itself (built 2026-07-02; superseded for the
long-training use case by `run_long_ppo` above -- kept as an opt-in)

`--plateau-hours H` switches the loop into **wall-clock mode**: all iteration-count stopping is
disabled (no global-patience stop, chains are never deactivated, `per_stage_iters` resize ignored,
`--budget` defaults to unlimited), and the run terminates only on:

- **prolonged plateau**: the best objective has not improved (`> eps`) for `--plateau-hours` hours
  of run time, gated by `--min-hours` (no plateau stop before that much total run time); or
- **success**: any evaluation's `trained_success` — the *sustained* contact-gated hold rate over
  the eval horizon — reaches `--success-stop`.

Run time = accumulated ACTIVE seconds across sessions (`wall_elapsed`, persisted in the
checkpoint): pausing and resuming does not count paused time toward `--min-hours` or the plateau
window, and the improvement clock (`t_improve_elapsed`) survives resume. A dead LLM backend cannot
spin the loop forever: 10 consecutive revise calls yielding no candidate auto-pauses (checkpointed,
resume when the backend is back). `report.json` records `wall_hours`. Validated: no-env unit tests
of the criteria + env-backed loop test (thresholds scaled to seconds) confirming chains outlive
`patience`, the plateau stop fires, and the clock persists across pause/resume.

## Integration
- Replaces `action_priors.build_candidate_prompt` / `build_action_prior_prompt` and the fixed
  `ACTION_GROUPS`/`PRIOR_DIRECTIONS` with: robot-spec-derived vocabulary + the staged prompt
  builders + the agentic orchestrator.
- Output compiles via `composed_priors.make_composed_prior_fn` (already supports
  gated/stacked/reactive_law/dmp); the orchestrator drives the existing rollout scorer.
- Specs are built once from the env (robot/task/env) + a static prior-system description.

## Implementation (built)
- **`agentic_orchestrator.py`** — `AgenticOrchestrator` dataclass implementing the full pipeline:
  `gather_context` (Stage 0, `ThreadPoolExecutor` parallel C1/C2/C3 + optional C4),
  `generate_seeds` (Stage 2, one call → ≤`n_seeds` diverse candidates, hand-fallback if empty),
  `_eval` (Stage 3, one `prior_eval.score_program` rollout == one budget iteration, with global
  plateau tracking), `revise` + `run` refine loop (Stage 4, round-robin over the chosen seed
  chain(s), per-chain `since_improve`/`patience` plateau-or-degradation early-stop, global
  early-stop), `finish` (writes `best_program.json` + `report.json` with the full eval trajectory).
  Seed-refine choice: best seed always; 2nd only if within 20% of best. Budget counts only
  rollout-evals; context/reasoning calls are free. `llm_backend in {none,fixture}` runs offline.
- **`run_agentic_selection.py`** — CLI: `--out --rep {dsl,freeform} --dof-mode --budget --n-seeds
  --patience --eps --human-analogy --llm-backend`. Emits `best_program.json` to hand to
  `run_prior_ppo --prior-program-arm ARM=best_program.json`.
- **`prior_eval.py`** — shared open-loop scorer/validator/accounting (`score_program`,
  `validate_program`, `accounting`), extracted from `run_dsl_vs_freeform` so both it and the
  orchestrator use one contact-gated rollout definition.
- **Staged prompt templates** in `prompts/`: `context_execution.md` (C1), `context_failure_modes.md`
  (C2), `context_kinematics.md` (C3), `context_human_analogy.md` (C4), `seed_candidates.md`
  (Stage 2), `revise_candidate.md` (Stage 4); each a single-job prompt over a shared spec block.

## Open / experiment flags
- C3 ordering (before generation vs. after first draft).
- C4 (human analogy) on/off.
- Seed diversity count (default ≤8) and explore/refine split.
- Early-stop patience `w` (default 3) and tolerance `eps` (plateau/degradation detection).
- Choosing 1 vs 2 seeds to refine (refine the second only if it is competitive with the best).
- **Output representation: constrained robot-derived DSL vs. free-form symbolic.** To be settled by
  a preliminary experiment (see `PRELIM_dsl_vs_freeform.md`), not assumed. Constrained DSL = groups
  (subsets of real actuators) + composable directions + gates → `composed_priors`. Free-form =
  per-group symbolic expressions over observable signals, compiled by a restricted evaluator. The
  experiment compares best objective, compile-success rate, DOF coverage (does it recruit the
  wrist?), and candidate diversity.

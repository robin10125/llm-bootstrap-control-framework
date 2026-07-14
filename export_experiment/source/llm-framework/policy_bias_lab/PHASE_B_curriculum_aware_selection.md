# Phase B — LLM-authored phase programs + curriculum-aware selection

Phase A (done) gives a **closed-loop curriculum phase controller** (`phase_controller.py`):
a state machine over the symbolic action vocabulary whose phases advance on observable-signal
gates (e.g. `n_contacts >= 1` → close → `closure >= 0.5` → lift). It ships with a hand-written
`default_phase_program("lift")`, a staged contact-gated validator (`validate_phase_program`),
a real-robot per-step loop (`PhaseController.step_single`), and warm-start BC dataset
generation (`PhaseController.bc_dataset`), wired into the `*_supervised_init` arms via
`--phase-teacher`.

Phase B replaces the *hand-written* program with a **reasoning-LLM-authored, candidate-selected**
program, closing the loop on the thesis: the LLM reasons out the curriculum, the runner
validates it objectively with few rollouts, no simulation-only state required.

## B1 — LLM authors the phase program

Add `load_phase_program_candidates(cfg, env)` to `action_priors.py` (or a new
`phase_programs.py`), mirroring the existing candidate flow:

- **Prompt** the model to emit a *phase program* JSON (not a flat rule list): an ordered list
  of phases, each `{name, rules:[{group,direction,weight}], advance_when:[{signal,op,value}],
  min_steps}`. Reuse `sanitize_phase_program()` for validation/clamping.
- Give the model the allowed `ACTION_GROUPS`, `PRIOR_DIRECTIONS`, and the **observable signal
  names** (`schema.EVAL_FIELDS`) it may gate on. Emphasize: gates must reference only signals a
  real robot can sense; phases advance on *achieved subgoals*, never on a timestep.
- Frame the task as "design the natural curriculum (e.g. approach → contact → close → lift)".
  The LLM is good at this; the structure is what a single static prior cannot express.

## B2 — Curriculum-aware selection among candidate programs

For each candidate program, build a `PhaseController` and call `validate_phase_program()`.
Select **deterministically** by `staged_score` (already implemented):

```
staged_score = Σ_k phase_reach[k]            # monotone curriculum depth → graded partial credit
             + 4 · contact_gated_success      # full contact-grasp-lift
             + 1 · contact_conditioned_lift
             − 3 · fling_fraction             # anti-fling
```

`phase_reach` being monotone is the key: it discriminates candidates by *how deep into the
curriculum* they get even before any of them completes a full lift — fixing the null-signal
failure that flat end-to-end scoring had. No qualitative LLM Pareto pick.

Reject-and-fallback gate already exists: `--phase-teacher-min-progress` drops a teacher whose
`final_phase_frac` is too low, falling back to no warm-start so a bad teacher can't poison BC.

## B3 — Distill teacher → static prior (optional, for the ablation)

The `reward_action_prior` arm uses a *static* prior, the teacher is *closed-loop*. To keep the
static-vs-teacher comparison meaningful, fit a single static prior that best reproduces the
teacher's early-phase action distribution (least-squares over the approach/contact phase rules)
and feed it through the existing action-prior path. This gives: baseline vs static-prior vs
closed-loop-teacher-warm-start, all from the same LLM reasoning.

## B4 — Mid-run, feedback-driven program edits (stretch)

The 50% checkpoint can re-author *gates/weights* (not structure) from training diagnostics,
the same way `DynamicRewardCoach` rewrites reward terms — e.g. loosen a `closure` gate that no
env is passing. Structure changes require recompiling the padded arrays; weight/threshold
changes do not.

## Real-world / portability checklist (must stay true in B)

- Gates reference observable signals by **name**, resolved against a task-specific
  `field_index` — a new task/robot supplies its own `EVAL_FIELDS` and works unchanged.
- Validation uses only sensor-observable quantities and a handful of rollouts → reproducible on
  hardware (`PhaseController.step_single` is the on-robot execution path).
- Action semantics live once in `symbolic_control.make_rule_action_fn` (shared by priors and the
  controller) so authored programs mean the same thing everywhere.
- Keep the reject-and-fallback validation gate: never commit an unvalidated teacher to BC.

## Files

- `phase_controller.py` — controller, staged validator, BC dataset, real-robot loop (Phase A).
- `symbolic_control.py` — shared symbolic action primitives (single source of truth).
- `action_priors.py` — candidate generation/selection to extend for phase programs (B1/B2).
- `run_dynamic_reward_experiment.py` — `--phase-teacher`, `--phase-program-json`,
  `--phase-teacher-min-progress`; pass the selected program in via `--phase-program-json` or
  wire `load_phase_program_candidates` directly.

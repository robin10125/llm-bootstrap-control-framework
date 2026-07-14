# Policy-clocked action paths

Implementation of option 4 from the sequencing discussion (full rationale in `DESIGN.md`): the
LLM authors WHAT (an ordered list of closed-loop behavior segments, recovery edges, measurement
hints); the learned policy owns WHEN (progression). There are no gates, no argmax, and no authored
thresholds anywhere in the control loop — switch timing is an ordinary PPO action with an exact
log-probability, so it is trained by reward every iteration instead of needing an LLM round-trip.

## Files

- `clocked_paths_ppo.py` — IR (validation + mechanical `freeform_staged` conversion), compiled
  segment runtime, progression dynamics, actor/critic with hint features, PPO with annealed
  imitation, arbiter-parity evaluation.
- `run_clocked_ppo.py` — CLI runner.
- `DESIGN.md` — first-principles design document.

## Mechanism

- **Progression head**: the actor outputs the actuator residual plus `rate ∈ [0, rate_max]`;
  within-segment phase integrates `u += rate·dt/est_seconds[seg]`, advancing at `u ≥ 1`
  (monotone by construction). Segments with a declared `recovery` edge add a recover head
  (bounded per episode, cooldown).
- **Structural guarantees**: min dwell (no chatter), forced advance after
  `dwell_slack × est_seconds` (a stall becomes the measured `forced_advance_frac`, never a dead
  run), bounded recoveries.
- **Hints, not switches**: `done_hint`/`abort_hint` values feed actor+critic as continuous
  features; an annealed imitation loss (`--imitation-coef`) initializes the progression head at
  the authored decision; a potential-based term on `phi = (seg + u)/n` shapes reward.
- **Composition**: `env_action = clip(segment_action + residual_scale · residual)`. The deployed
  controller is the complete hybrid (minimality is explicitly not a goal here); evaluation is
  deterministic, full-episode, scored with the same `eval_summary` columns and
  `task_graded_objective` as every other arm.
- **`--progression autopilot`**: advancement from dwell-filtered hints + timeouts only — the
  "hardened gates" conservative option, serving as the ablation arm and as the semantics the
  imitation target encodes.

## Program format

Native `clocked_paths` programs (see `DESIGN.md` for the JSON shape) or any `freeform_staged`
program, converted mechanically: stage channels → segment paths, the NEXT stage's gate →
`done_hint`, `success` carried, no recovery edges synthesized. Caveat on converted programs:
staged gates were authored for argmax comparison, so `gate > 0` is a weaker "done" signal than a
natively authored signed margin — converted `done_hint`s are usable initialization but native
candidates should author `done_hint` as `>0 ⇔ segment did its job`. Channel and hint expressions
may reference `u` (within-segment phase), enabling phase-parameterized paths.

## Running

```bash
PY=.venv/bin/python
RUN=policy_bias_lab/experimental/policy-clocked-paths/run_clocked_ppo.py
PROG=runs/agentic_v4_20260704/best_program.json   # freeform_staged, auto-converted

$PY $RUN --out runs/clocked_learned_$(date +%m%d)   --program $PROG --progression learned
$PY $RUN --out runs/clocked_autopilot_$(date +%m%d) --program $PROG --progression autopilot
```

Key telemetry in `metrics.jsonl`: `seg_occupancy`, `rate_mean`, `hint_agreement` (does the
learned head advance when the authored hint says done?), `forced_advance_frac` (authored path
can't reach its own hand-off → revise channels), `recover_frac`, `imit_coef`, `imit_loss`.
Headline comparison number: `final_report.json:eval_graded_objective`.

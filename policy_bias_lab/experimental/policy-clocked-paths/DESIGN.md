# Policy-Clocked Action Paths: Sequencing Overhaul

## Problem

The staged-prior system sequences LLM-authored behaviors with authored gate expressions —
instantaneous threshold predicates over position/contact terms, resolved by hard argmax with a
monotone cursor. This consistently fails, and the failures are structural, not tunable:

1. **Thresholds must be guessed blind.** The LLM writes `palm_dist < 0.05` for a physics it cannot
   observe or calibrate. A few millimeters of error means the gate never fires or fires constantly.
2. **Point predicates, persistent phenomena.** A gate tests one step; real hand-offs are regions of
   state space held over time. Signal noise produces blips that hard argmax converts into switches.
3. **Every failure is total.** The monotone cursor makes a single false positive an unrecoverable
   premature advance, and a false negative a permanent stall. There is no graceful degradation.
4. **Switching has no gradient.** A mis-timed hand-off can only be fixed by a full LLM revision
   round-trip (generate → PPO eval → diagnose → revise). The one quantity that most needs local
   adaptation — *when* — is the one quantity nothing in the system can adapt.
5. **Distribution shift.** Gates are authored and validated on prior-driven rollouts, then consumed
   on states the trained policy visits.
6. **No recovery semantics.** "Grip lost → re-grasp" is inexpressible; the cursor cannot regress.

## Principle

**The LLM authors WHAT; the learned policy owns WHEN.**

LLMs are good at task decomposition and at writing closed-loop motion structure ("move the palm
toward the object, then wrap the fingers, then extend the base"). They are bad at calibrating scalar
thresholds in an unseen simulator. So put the behavioral content in authored artifacts and the
switching authority in the trainable component, where reward provides a gradient every iteration.

Authored conditions do not disappear — they are demoted from *control* to *hints*: features,
shaping, and initialization for the learned switch. A wrong authored threshold then costs a biased
starting point instead of a broken controller.

Corollary: every remaining structural mechanism must guarantee progress and bound damage
(min/max dwell, bounded recoveries), so that no single authored error can be total.

The prior is NOT required to be minimal here (contrast `experimental/quiet-prior/`): the deployed
controller may permanently include the segment machinery plus the learned policy as one unit.

## Architecture

### Authored representation (IR)

The candidate is an ordered list of **segments** — closed-loop action paths — plus optional
recovery edges. All expressions use the existing restricted-AST machinery over injected observables
and authored signals; the framework contributes no task nouns.

```json
{
  "mode": "clocked_paths",
  "signals": { "authored_quantity": "<expr over raw observables>" },
  "segments": [
    {
      "name": "segment_0",
      "channels": [
        { "actuators": ["<injected name or group>"],
          "expr": "<closed-loop expr; may reference u>" }
      ],
      "est_seconds": 1.5,
      "done_hint": "<expr; HINT for advancing, not a switch>",
      "abort_hint": "<expr; optional>",
      "recovery": "<segment name to fall back to; optional>",
      "success": "<measurement expr for diagnostics/shaping>"
    }
  ],
  "unused_dofs": []
}
```

- `channels` are functions of current observables (targets track the object — feedback, not
  playback) and may additionally reference `u`, the within-segment phase in [0, 1], so a segment
  can be a parameterized path (pre-shape trajectories, timed squeezes) rather than a fixed posture.
- `est_seconds` is a pacing budget, not a schedule: it scales the phase-rate action and sets the
  forced-advance timeout. Duration estimates are far more robustly authorable than metric
  thresholds — being wrong by 2x degrades pacing, not correctness.
- `done_hint` / `abort_hint` never switch anything directly (see Hints below).
- `recovery` declares the only permitted regressions. Default: none (linear chain).

### Progression state and the learned switch

Each environment carries `(seg, u)`: segment index and within-segment phase. The policy's action
space is extended with a small **progression head** alongside the actuator dimensions:

- **Continuous clock (default).** A scalar `rate ∈ [0, rate_max]` (squashed). Phase integrates
  `u += rate * dt / est_seconds[seg]`; at `u >= 1` the cursor advances and `u` resets. Monotone by
  construction, no discrete chatter, always differentiable; "hold" is `rate ≈ 0`, "advance now" is
  `rate = rate_max`.
- **Recovery head (only when any segment declares `recovery`).** A binary head, masked invalid
  where no edge exists; firing it jumps to the declared fallback segment. Bounded per episode.

The environment action within a segment composes as today: `env_action = clip(segment_action +
residual)` with the existing learned prior-scale machinery (composition is orthogonal to this
overhaul; exclusive per-actuator ownership from quiet-prior DESIGN.md can be swapped in later
without touching sequencing).

### Structural guarantees (framework, task-agnostic)

- **Min dwell / switch cooldown:** a segment holds for at least `min_dwell` steps; recoveries have
  a cooldown. Kills chatter regardless of what the policy or hints do.
- **Max dwell (progress guarantee):** if a segment exceeds `slack * est_seconds` (slack ~3), the
  cursor force-advances. No authored or learned failure can produce an infinite stall; a stall
  becomes a measured event (`forced_advance_rate`) instead of a dead run.
- **Bounded recoveries:** at most `k` recovery jumps per episode; further recover actions are
  masked. No oscillation loops.
- **Monotone otherwise:** only declared edges regress.

Every failure mode is thereby converted from *total* to *graded and measured*.

### Where the authored hints go

1. **Observation features (explicit interface change).** Actor and critic both receive: segment
   one-hot, `u`, normalized dwell, `done_hint`/`abort_hint` values, authored success margins. The
   policy interface changes — deliberately and visibly, per the standing constraint that such
   changes be explicit. The switch decision is state-conditional because the policy sees the same
   evidence the authored gate would have tested, as *continuous features* rather than a binary
   already-thresholded verdict.
2. **Shaping.** Potential-based term on ladder progress `phi = (seg + u) / n_segments` (cannot
   change the optimal policy) plus the existing authored-`success` shaping. Progress pays; the
   forced-advance path pays less than an earned advance only through the task reward, never through
   a framework-authored bonus.
3. **Initialization by imitation (annealed).** An auxiliary cross-entropy on the progression head
   toward the authored hint decision (`done_hint > 0` → advance, `abort_hint > 0` → recover),
   weight annealed to zero over early training. Training starts near the authored controller;
   reward then tunes the timing. A wrong hint biases the first iterations and nothing after.

### PPO integration

- **Factored log-probability:** squashed-Gaussian over actuator dims + Beta/squashed-Gaussian over
  `rate` + Bernoulli over recover, summed. This is the same per-group factoring quiet-prior
  Increment 3 requires — shared infrastructure.
- **Fully on-policy.** The progression head is an ordinary action: no off-policy correction, no
  masking, no importance weighting. The stored logp is exact. GAE, minibatching, and the fragment
  skeleton are unchanged; `(seg, u)` rides in the carried env state exactly as `cursor` does today,
  reset at episode boundaries.
- **Evaluation parity:** deterministic full-episode eval of the complete hybrid (segments + policy,
  since minimality is not required), scored with the same `eval_summary` columns and
  `task_graded_objective` as every other arm.

### Diagnostics → revision loop

Per segment, per training window:

- dwell distribution and `forced_advance_rate` (high = the path cannot achieve its own hand-off —
  revise the *channels*, not a threshold);
- learned-switch vs hint agreement rate and lag (the learned switcher is a **measurement
  instrument**: report signal snapshots at learned advance times back to the LLM — "you said
  advance at palm_dist < 0.05; the trained policy advances at 0.021 ± 0.004" — so revision produces
  calibrated hints instead of guesses);
- recovery usage and post-recovery success (high churn = fragile grip → revise the gripping
  segment);
- authored `success` rates, assisted vs residual-off;
- occupancy/conversion per segment (the existing stall-localization machinery consumes `seg`
  exactly as it consumes the cursor today).

Revision pressure stays statistical and mechanical; interpretation stays with the LLM.

## Why each old failure mode is closed

| Failure | Mechanism that absorbs it |
|---|---|
| Guessed thresholds | Not in the control loop; timing is learned by reward gradient |
| Noise blips → spurious switch | Continuous clock + min dwell; no argmax anywhere |
| Permanent stall (gate never fires) | Max-dwell forced advance; stall becomes a metric |
| Premature advance (gate too eager) | Policy learns `rate ≈ 0` where advancing loses reward; declared recovery edges |
| No gradient on timing | Progression is an action with a log-probability |
| Distribution shift | The switcher trains on exactly the state distribution it runs on |
| No recovery semantics | Declared, bounded recovery edges |
| LLM round-trip to fix timing | Timing fixed by PPO within a run; revision loop reserved for structure |

## Build increments

1. **Hardened autopilot (no new action dims).** IR + runtime where progression is driven by
   hints alone but wrapped in the guarantees: dwell-filtered hysteresis on `done_hint`, max-dwell
   forced advance, bounded declared recoveries. This is the conservative fallback option in its own
   right ("behavior-tree hardening") and immediately testable against the current staged prior.
   It also exercises the runtime `(seg, u)` state, diagnostics, and IR before any PPO change.
2. **Progression head.** Factored logp, continuous clock, imitation-initialized from the hints,
   hint features into actor/critic. The autopilot from (1) becomes the imitation target and an
   ablation arm.
3. **Shaping + recovery head.** Potential on `(seg + u)/n`, authored-success shaping, recovery
   edges with bounds.
4. **Revision-loop integration.** Segment diagnostics into the orchestrator's directive rendering;
   switch-time signal snapshots as calibration feedback.
5. **Seed-matched comparison.** Same budget: current staged prior · hardened autopilot (1) ·
   policy-clocked (2–3) · no-prior control. Headline: `eval_graded_objective`; secondaries:
   forced-advance rate, learned-vs-hint lag, variance across seeds.

## Relation to quiet-prior

Orthogonal axes. Quiet-prior governs *how much* authority the prior has and how it retires;
this governs *how behaviors are sequenced*. Shared: factored per-group log-probabilities,
measurement-not-interpretation discipline, task-agnostic validation. Divergent: minimality is
explicitly relaxed here — the segment machinery may ship in the final controller, so no retirement
scheduler is needed; residual-off performance is a diagnostic, not the score.

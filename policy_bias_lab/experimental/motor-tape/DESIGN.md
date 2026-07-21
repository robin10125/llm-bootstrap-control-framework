# Motor Tape: Reset-Anchored Command Tapes with Cerebellar Correction Heads

## Problem

In the staged/gated prior systems, future behaviour is invisible until the robot is in the state
that elicits it: a stage's channels exist only behind its gate, so nothing — not the policy, not
the critic, not a reviewer of the artifact — can see what the prior will do next, and general
coordination of movements across time cannot be planned, only chained reactively. The clocked
framework fixed WHO owns switching but kept the plan opaque (the policy sees only the current
segment one-hot + phase) and immutable (networks ride on it; they cannot reshape it).

## Principle (the brain isomorphism)

**Motor cortex authors the full command sequence; the cerebellum corrects it online.**

- The LLM (motor cortex) authors a FULL TRAJECTORY: timed keyframes compiled ONCE per episode —
  on the reset observation, i.e. conditioned on the state at planning time, like a reach plan
  generated for a seen target — into a concrete numeric command tape `q_des[T+1, nu]` of absolute
  ctrl targets. Playback is feedforward; no expression reads sensors after reset.
- The learned networks (cerebellum) receive an **efference copy** — the tape's current command
  and a lookahead window of upcoming commands, as features to actor AND critic — plus all
  sensors, and correct the plan three ways:
    1. **residual** (per-step, added to the tracking action): execution errors the plan cannot
       anticipate;
    2. **rate** (time-warp, `[rate_lo, rate_hi]`, imitation-annealed toward the authored pace):
       the cerebellar timing function — pause the tape under contact, hurry free space;
    3. **modulation** (optional, flag-gated): a bounded, smoothness-regularized offset applied to
       the tape COMMAND before the tracking law — bends the plan itself rather than perturbing
       execution. Structurally absent when off (`--modulation off` ⇒ identical network/logp/loss
       to the no-modulation ablation), so it is cleanly A/B-testable.

What this buys over the gated/clocked priors: the whole plan is visible (to the networks as
lookahead features, to humans as a plottable tape), coordination across actuators and time is
authored explicitly on one timeline, and the plan is mutable by learning instead of only
switchable.

## Files

- `motor_tape.py` — IR, validation (`validate_motor_tape`, hard compile gate on a real reset),
  reset-time compilation (`compile_tape` → vmap/jit-safe `tape_from_obs`), numeric accounting
  (`tape_report`: coverage / out-of-range knots / over-slew segments), autopilot scoring
  (`score_tape`: contact-gated open-loop columns comparable to `prior_eval.score_program`, plus
  task eval fields + `task_graded_objective` comparable to trained arms).
- `motor_tape_ppo.py` — PPO trainer: single squashed-Gaussian action vector
  `[residual? | rate? | modulation?]` (exact logp via `squashed_gaussian_logp`), efference-copy
  features (dim `nu·(1+len(lookaheads))+2`), potential shaping on `phi = s/T`, rate imitation
  anneal, modulation smoothness + L2 aux losses, fragment skeleton and eval parity mirrored from
  `alternative-methods/alt_methods_ppo.py` / `policy-clocked-paths/clocked_paths_ppo.py`.
- `run_motor_tape.py` — runner; arms: `--tape-only` (autopilot floor), `--no-rate`
  (tape+residual), default (tape+residual+rate), `--modulation on` (+bender). Every trained
  final_report carries the autopilot floor alongside.
- `generate_motor_tape.py` — single-shot LLM generation (codex backend) with the standard
  validation-retry loop; prompts `prompts/framework_motor_tape.md` +
  `prompts/representation_motor_tape.md` (task-agnostic; task/robot content injected via
  `$spec_block`/context block, slew ceiling and episode budget substituted from the env).
- `test_motor_tape.py` — interpolation math (no env) + compile/validation against the real env.
- `example_tape.json` — hand-written smoke plan.

## Key mechanics (detail)

- **Keyframe semantics**: `t` = ARRIVE-BY deadline. Implicit knot at t=0 = the actuator's reset
  ctrl (so the first segment interpolates from the spawn pose; no step discontinuity). Actuators
  in no keyframe hold spawn pose; hold-last after an actuator's final knot. Group targets expand
  per member with RESET-SELF bindings (`ctrl_init_self`, `ctrl_lo_self`, `ctrl_hi_self`);
  exact-name entries override group entries within a keyframe.
- **Interpolation**: per actuator between its own knots; `linear` or `minjerk`
  (quintic `10u³−15u⁴+6u⁵`, zero velocity/acceleration at knots, never leaves the knot interval).
  Both affine in knot values ⇒ the tape is a static weight matrix times reset-evaluated knot
  values (one matmul per actuator inside `tape_from_obs`).
- **Units**: knot exprs evaluate to ABSOLUTE ctrl targets (clipped to ctrlrange at build).
  Playback tracking law: `a_ff = clip((q_eff(s') − ctrl)/action_scale, −1, 1)` — one-step-exact
  when unsaturated (env applies `ctrl' = clip(ctrl + a·action_scale)`); above the slew ceiling
  (`action_scale/control_dt` = 2 ctrl-units/s) it degrades to a max-rate ramp, visible to the
  networks as tracking error and to authors via the `tape_report` over-slew warning.
- **Time-warp lookup**: phase `s` in tape steps; `s' = min(s + rate, T)`; linear lookup between
  tape samples; lookahead features indexed in PLAN steps (what the plan will command), clamped
  hold-last.
- **Reset handling**: `env.step` never auto-resets (done only at horizon); episodes reset at the
  Python level between fragments, where the trainer rebuilds `tape = vmap(tape_from_obs)(reset
  obs)` — per-env tapes for per-env spawns. `tape_from_obs` is itself pure JAX, so an
  auto-resetting env could recompute it under `lax.cond` without redesign.
- **Memory**: tape `[256, 801, 26]` f32 ≈ 21 MB. `eval_envs` defaults to 128 (256-env eval OOMed
  for clocked on this 15 GB machine).

## v2 notes (not built)

- Orchestrator integration: revision loop fed `score_tape` + tape diagnostics (tracking-error
  hotspots, rate-head dwell profile = where the policy stretches/compresses the plan — a
  measurement instrument for recalibrating keyframe times), decomposition of the tape's phases.
- Per-episode plan deltas (motor adaptation): a head conditioned on reset obs offsetting the knot
  values once per episode, within LLM-declared parameter ranges. Deliberately excluded from v1.
- Task-space keyframes via IK primitives (see the action-representation memory) — orthogonal.

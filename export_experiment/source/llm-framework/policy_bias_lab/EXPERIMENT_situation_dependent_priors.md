# Situation-dependent action priors

Goal: move the action prior from a single, always-on, time-invariant mean-shift to a
**situation-dependent** bias that effectively models a *sequence of behaviours* (approach →
grasp → lift) while staying a reactive function of the observation (no scripts, no clocks,
no fixed trajectories baked in). Everything here is portable: behaviour is expressed over
symbolic action groups/directions and **observable signals**, so the same prior runs on a
real robot from perception + proprioception.

This note covers the **action-prior bucket** only. The supervised-init bucket (codex
receding-horizon trajectory generation; DMP-as-demo-generator) is **deferred** — see
"Deferred" at the end.

## Background: what a prior is today

A prior is a set of legacy rules `{group, direction, weight}` compiled to a per-step mean
shift added to the policy's pre-tanh action (`bias.weighted_action_prior(obs, weights)`,
applied in `ppo_bias.policy_dist`). Directions split into:
- **endpoint-seeking servo** (`toward_object_xy`): drives a runtime target, settles to 0;
- **constant-rate bias** (`close_hand/open_hand/lower_base/raise_base/stabilize`): a fixed
  per-step push that integrates with the incremental controller.

All rules are summed every step, always-on, no ordering. The curriculum (approach→grasp→
lift) is *not* in the prior today — it emerges from physics + the contact-gated reward, or
lives in the separate `phase_controller.py` teacher. The base-direction operators are
calibrated to the flipped base (`M = diag(+1,-1,-1)`) and the training path now shares the
scorer's calibrated operators (`bias._rule_vector` == `symbolic_control.make_rule_action_fn`).

## Observable-signal constraint (important)

The live policy observation is
`[base_q, base_v, hand_q, hand_v, obj_pos, obj_vel, palm_pos, obj_rel, ctrl]`.
It contains `obj_rel` and `ctrl` but **not** `n_contacts` or `min_finger_dist` (those need
fingertip↔object distances, computed only in the eval vector). So a *live* prior can gate
only on signals derivable from `obs`:

| signal | source in obs | hardware analog |
|---|---|---|
| `palm_obj_dist` | `‖obj_rel‖` (exact — obj_rel is obj−grasp_site) | object tracking |
| `obj_rel` (xyz) | trailing-3-before-ctrl | object tracking |
| `closure` | normalized `ctrl[hand_ids]` | proprioception |
| `lift` | `obj_pos[z] − z0` (z0 = rest height const) | object tracking |
| `obj_vel`, `palm_pos` | in obs | tracking / FK |

Consequence: gating uses **proximity + closure + lift** as the situation signals, *not*
contact count. This is hardware-realizable and keeps the prior a pure function of `obs`
(minimal wiring). The reward stays contact-gated for anti-fling correctness; the prior's
gating only decides *which sub-behaviour is active*, so the proxy is sufficient.
(Extension, not done here: append contact/tactile signals to `obs` for crisper gates — a
network-input change requiring re-validation.)

## Shared substrate

**Legacy-prior library `L`** (codex-authored, each a normal `{group,direction,weight}`
rule-set; "legacy" = same form as past experiments, *not* a fixed catalog):
- `P_approach`: `toward_object_xy` (base_xy) + `lower_base` (base_z)
- `P_grasp`: `close_hand` on thumb/index/middle
- `P_lift`: `raise_base` (base_z) + light `close_hand` (hold)

**Composition:** `prior(obs) = clip( Σ_i g_i(s(obs)) · P_i(obs), −1, 1 )` added to the policy
mean. `P_i` uses the calibrated operators. Only `g_i` varies across the gating disciplines.

## Prior-bucket strategies

### #2 + gating disciplines

- **A.2 — declarative subgoal (hard switch).** Ordered subgoals with observable predicates:
  `approach until palm_obj_dist<ε` → `grasp until closure>κ` → `lift until lift>h`. A pointer
  selects exactly one active sub-prior, advancing when its predicate holds. (= the phase
  controller's logic promoted to a live prior.)
- **A.3 — options with commitment.** Same predicates + per-option termination and
  hysteresis/min-dwell, so a noisy signal can't flip the active option back. A latch over A.2.
- **A.1 — soft gating (continuous blend).** `g_i` are smooth sigmoids over signal distance to
  each prior's home region (e.g. `g_grasp = σ((ε−palm_dist)/τ)·σ((κ−closure)/τ)`),
  normalized. Differentiable, no boundary discontinuity. The genuinely new mechanism.

### B.5 — fused reactive law (`prior_reactive_law`)
One compiled function `desired = f(s)` evaluated every step: descend ∝ palm-gap; seek object
xy; close ∝ proximity·(1−grip); raise ∝ grip·secureness. Encodes A.1/A.2/A.3's logic
implicitly in one continuous controller. Reuses the operator/compile path.

### B.6 — DMP prior (`prior_dmp`)
Per group, `τ²ẍ = α(β(g−x) − τẋ) + f(s)` with **goal `g` bound to the observed object pose**
and **phase `s` advanced by progress** (palm-gap shrink / closure / lift), not a clock. Start
with `f≈0` (critically-damped goal servo) and optionally a codex-authored shape. Output →
desired ctrl delta → mean-shift. Qualifies as a *reactive* prior precisely because of the
goal-binding + progress-phase (without them it degrades to a fixed demo, i.e. the deferred
bucket).

## Runs

**Run 1 — bake-off** (one shared library `L`, identical reward/resets/seeds/budget):
`prior_monolithic` (control = today's single always-on prior) · `prior_gate_soft` (A.1) ·
`prior_gate_subgoal` (A.2) · `prior_gate_options` (A.3).

**Run 2 — stacked**: `prior_gate_stacked` = A.2 regions + A.3 commitment latch + A.1 soft
transitions, vs. the best bake-off arm + `prior_monolithic`.

**Plus** `prior_reactive_law` (B.5) and `prior_dmp` (B.6) as their own arms.

## Experimental controls
- Constant across arms: env config (post lift-reward rebalance: `contact_lift_floor=0.3`,
  `w_lift_hold=3.0`, `w_lift_pot=20`), reset distribution, seeds, env/eval budget, `--no-coach`
  (fixed shaping ⇒ identical reward), calibrated operators.
- **Prerequisite (done):** best-checkpoint selection falls back to a graded grasp/lift score
  when sustained success == 0 (was pinning to iter-0). `ppo_bias.train_ppo_arm`.
- **Train long enough:** 63 iters (the prior-fix quick test) was ~10× too short to see the
  grasp climb; budget several hundred iters/arm ⇒ stage the arms, don't cram into one short run.
- Metrics per arm: `grasp_rate` climb curve, sustained `contact_gated_success`,
  `lift_reached_rate`, `contact_engagement`, `fling_fraction` (must stay ~0).
- **Likely side benefit:** a gated sub-prior gets *full authority in its region*, so the
  composed prior may reach contact open-loop where the single weak prior scored `engage=0` —
  potentially also mitigating the selection-blindness problem.

## Implementation map
- `policy_bias_lab/composed_priors.py` (new): compiles a `prior_program` spec into a JAX
  `prior_fn(obs, weights) -> mean_shift`; modes `monolithic | gated(soft|subgoal|options) |
  reactive_law | dmp`; shared calibrated operators + `signals_from_obs`.
- `bias.py`: `compile_bias` builds the `prior_fn` when the spec carries a `prior_program`;
  `weighted_action_prior` dispatches to it (else legacy rule sum). Back-compatible.
- `ppo_bias.py`: unchanged call site (`weighted_action_prior(obs, weights)`); eval-bug fix.
- `run_dynamic_reward_experiment.py`: new arms + CLI to choose the prior program; codex
  authors `L` / gates / law / DMP params (fixture fallback for offline runs).

## Deferred (supervised-init bucket)
- Your #1: single-env receding-horizon codex (codex emits a macro-action every K steps,
  rolled open-loop until goal) → BC warm-start.
- B.6 as demo generator: roll the goal-bound DMP across resets → BC dataset.
Not implemented in this pass.

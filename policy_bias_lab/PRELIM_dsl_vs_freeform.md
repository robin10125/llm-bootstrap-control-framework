# Preliminary experiment: constrained DSL vs. free-form symbolic priors

Settles the output-representation decision for the agentic prior-selection refactor
(`AGENTIC_PRIOR_SELECTION.md`). Both representations are LLM-generated; the winner is decided by a
short PPO train, not just the open-loop scorer.

## The two representations (both stateless `fn(obs) -> mean_shift`)

**A — Constrained robot-derived DSL.** Rules `{group = subset of the real actuators, direction =
composable calibrated operator, weight}` + gate thresholds → `composed_priors` gated/stacked.
Bounded; **always compiles**; the base→world sign calibration is hidden inside the operators.

**B — Free-form symbolic.** Per channel (a set of actuators it names), an **expression for the
mean-shift over observable signals**, e.g.
```
base_z          : -0.6 * clip(palm_obj_dist - 0.04, 0, 1) * (1 - gripped)
wrist_flex      :  0.3 * gripped
thumb,index,mid :  0.5 * near * (1 - gripped)
```
Signals (`palm_obj_dist, closure, gripped, near, lift, obj_rel_x/y/z, ...`) + a whitelisted op set
(`clip, sigmoid, min, max, abs, + - * /`, comparisons) compiled by a restricted AST evaluator into
a JAX fn. Maximally expressive; the model writes the controller and the raw-actuator sign
conventions are documented in the robot spec (so this also tests whether the LLM can handle
calibration the DSL hides).

## DOF-completeness requirement (both representations)
Stronger than "mention every DOF": **the representation's vocabulary must be able to ACTIVELY
MOVE every joint** — holding/stabilize is NOT sufficient. The motion primitives must form an
**approximate complete basis (loose sense) over the robot's legal configuration space**: every
actuator drivable in **both** directions, so any reachable orientation is a combination of
primitives.

- **DSL:** the direction vocabulary is robot-derived (`freeform_priors.derive_motion_basis`) — a
  signed articulation primitive per semantic group (`group` granularity = approximate basis) or per
  actuator (`actuator` granularity = full basis), PLUS the calibrated task operators
  (`toward_object_xy`/`lower_base`/`raise_base`/`close_hand`/`open_hand`). Validated by
  `check_basis_complete`, which fails on any joint that is `not_drivable` or `single_sign_only`.
  This is what makes the wrist (and any DOF) actually usable, not just named.
- **Free-form:** intrinsically able to drive any actuator via its expression; the per-candidate
  `check_dof_complete` still confirms each candidate engages the full DOF set (repair only as a
  last-resort hold).

The generator prompt lists every actuator from the robot spec and the basis primitives, and
states that the prior must be able to move each one.

## Protocol
Hold constant: robot/task/env specs, Stage-0 context fed to the generator, candidate count
(~8 diverse), the rollout scorer, seeds. Two conditions: **DSL** and **FREE**.
1. LLM generates ~8 candidates per condition (DOF-complete, validated/repaired).
2. Compile each; record compile-success.
3. Score each via the contact-gated open-loop rollout scorer.
4. **Short PPO train (~30 min) of the best candidate per condition** → grasp/lift (the decisive
   signal, since the open-loop scorer is blind to what PPO can learn from a prior).

## Metrics
1. **Compile-success rate** — FREE's key risk; DSL ≈ 100% by construction.
2. **DOF coverage / wrist recruitment** — # actuators with nonzero bias; is the wrist used?
   Directly tests the motivating defect.
3. **Open-loop reachability** — contact-gated rollout score (caveat: weak priors score ~0; a strong
   FREE controller may reach contact where DSL doesn't).
4. **Candidate diversity** — spread of strategies/structures.
5. **Downstream PPO** — grasp_rate / lift_max / sustained success of the best per condition.

## Build pieces
- Robot-spec extractor: enumerate all actuators (name, joint type, range, driven body, world-sign
  convention) → injectable spec. (`composed_priors` / new helper.)
- DOF-completeness validator (+ repair).
- Free-form expression compiler (restricted AST safe-eval → stateless JAX fn).
- LLM generation prompts (DSL, FREE) with injected specs + Stage-0 context + DOF-completeness.
- Comparison harness: generate→validate→compile→score for both, then short PPO of the best each.

## Notes
- LLM-generated (not hand-authored): the experiment is about how well a model *uses* each
  representation, so the generator is the LLM under both.
- This is decoupled from the full orchestrator; it only decides the representation.

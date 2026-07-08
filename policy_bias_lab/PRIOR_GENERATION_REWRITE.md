# Prior Generation Rewrite

This document describes the active rewrite direction for action-prior generation.

## Generation Flow

1. **Comprehensive execution account**
   The first LLM call produces a task-specific but framework-external account of completing the
   task successfully with the injected robot. It must cover every actuator or semantic body group:
   what moves, what holds still, what raw observables show correct behavior, and what raw
   observables show interference.

2. **Observable procedure**
   The second LLM call turns that account into chronological phases with observable exit
   predicates. These exits are validated by compiling them over raw observables.

3. **Typed prior candidate**
   Seed/revision prompts now ask for a versioned candidate shape:
   `signals`, optional tunable `parameters`, stages with `success` exit measurements, progress
   ladder gates, channels, probes, evals, and unused-DOF accounting.

4. **IR normalization and static audit**
   `prior_ir.py` normalizes candidate JSON into a typed intermediate representation and checks
   structural issues before rollout budget is spent.

5. **Compilation and arbitration**
   `prior_eval.validate_program` compiles the normalized candidate. The short-PPO arbiter remains
   the expensive selector; cheap simulation-only scoring is retained for prefilters or calibration.

## Parameter Calibration

The LLM should author structure and declare uncertain numeric constants as:

```json
"parameters": {
  "threshold_name": {"init": 0.1, "range": [0.02, 0.2]},
  "gain_name": {"init": 0.3, "range": [0.05, 0.6]}
}
```

Parameter names are scalar constants available inside signal, gate, success, channel, probe, and
eval expressions. `prior_calibration.py` provides a generic bounded search over these values with a
caller-provided simulator/scorer. This supports a future exploration phase:

1. Generate candidate structure.
2. Run cheap parameter search in sim.
3. Compile the best parameter setting.
4. Spend short-PPO budget only on calibrated structures.

The calibration objective must come from task data or a caller-supplied scorer, not framework prose.

## Robotics Tool Layer

A robotics-tool layer should sit between the LLM-authored procedure and prior compilation. The LLM
chooses and configures tools; the framework exposes generic tool outputs as injected data.

Useful non-neural tools:

- **Forward kinematics:** compute poses, Jacobians, reachable sets, and actuator-to-body influence.
- **Inverse kinematics:** solve target body poses or relative pose constraints, returning joint
  targets, residuals, and feasibility margins.
- **Trajectory interpolation:** produce smooth joint/actuator target paths subject to velocity and
  acceleration limits.
- **Motion planning:** sample or optimize collision-aware paths over the robot configuration space.
- **Operational-space control:** convert task-space errors into joint-space commands via Jacobians.
- **Contact/force control:** enforce force ceilings and compliant behavior from measured contact
  forces and commanded-vs-measured gaps.
- **System identification/calibration:** estimate effective gains, latency, friction, and tracking
  error from rollouts.
- **Classical optimization:** tune thresholds/gains or solve constrained stage-local objectives.

Later neural tools can be added behind the same boundary:

- learned IK warm starts;
- learned residual controllers;
- learned contact classifiers from raw observations;
- diffusion or sampling policies used as proposal generators;
- value/model predictors used only as calibrated simulators or advisors.

The key boundary is that tools may return measurements, feasible targets, paths, residuals, and
controller candidates. They should not inject task-specific interpretation into framework prompts.
The LLM decides how tool outputs relate to the task using the task spec and its own authored
execution account.

## Proposed Tool-Oriented Pipeline

1. LLM authors comprehensive execution account.
2. LLM requests generic tool analyses for phases that need geometry, reachability, force, or timing.
3. Tools return structured data: feasible target sets, IK residuals, path constraints, controller
   coefficients, and failure certificates.
4. LLM authors typed prior IR using both the execution account and tool outputs.
5. Static audit checks observability, actuator accounting, gate sequencing, units, and compilability.
6. Parameter calibration explores declared scalar ranges.
7. Short-PPO arbiter selects among calibrated candidates.
8. Diagnostics and authored probes/evals drive revisions.

This keeps the framework task-agnostic while allowing the LLM to control effective robotics methods
instead of hand-authoring every controller from scratch.

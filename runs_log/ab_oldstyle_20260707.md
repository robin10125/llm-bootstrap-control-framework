# Ab Oldstyle

- **Run directory:** [`runs/ab_oldstyle_20260707`](../runs/ab_oldstyle_20260707)
- **Date:** 2026-07-07 (directory timestamp)
- **Status:** completed or completed-with-caveats
- **Experiment class:** agentic prior generation/selection study
- **Recorded task:** grasp and lift a 5cm cube off the table
- **Training metric rows recovered:** 0

## Abstract

This entry reconstructs `ab_oldstyle_20260707`, an agentic prior generation/selection study. The account is based only on artifacts that survive in the repository: machine-readable configuration, candidate programs, selection diagnostics, training metrics, evaluations, reports, and logs. The run targeted grasp and lift a 5cm cube off the table. Its purpose and comparisons are inferred from the recorded arm names and configuration where no authored abstract survives; those inferences are identified by this provenance-first framing rather than presented as new experimental facts.

## Experimental structure

No machine-readable configuration file survives at the run root. Structure below is reconstructed from reports, result files, subdirectory names, and logs.
### Selection-loop structure

| Field | Value |
|---|---|
| representation | freeform_staged |
| dof_mode | encourage |
| arbiter | prior_only |
| budget | 3 |
| iters_used | 3 |
| n_seeds | 3 |
| wall_hours | 0.254 |

### Arms and sub-runs represented by directories

`llm`

## Results

### Recorded results: `report.json`

| Best-candidate field | Value |
|---|---|
| name | stabilized_descent_handoff |
| source | refine:wrap_cage_then_thumb_lock |
| objective | 0.03856 |
| accounting.n_driven | 23 |
| accounting.n_unused_listed | 0 |
| accounting.wrist_driven | yes |

| Diagnostic | Value |
|---|---|
| success_rate | 0 |
| reach_rate | 0 |
| grasp_rate | 0 |
| grasp_lift_rate | 0 |
| lift_reached_rate | 0 |
| lift_max | 0.0062 |
| base_return | 3.0729 |
| train_return | 3.0902 |
| shaped_return | 0.0173 |
| action_abs_mean | 0.0112 |
| saturation_frac | 0 |
| note | behavior at PPO iteration 0: the prior acting through an UNTRAINED policy (network output ~0, so this is essentially the prior alone). Compare with the trained metrics -- a defect present here is a PRIOR problem; one that only appears af… |
| base_reward_weight | 1 |
| instant_success_rate | 0 |
| hard_clip_frac | 0 |
| failure_rate | 0.0029 |
| calm_frac | 0.9971 |
| reach_rate_calm | 0 |
| objective_batch_std | 0.0001 |

Stage-flow measurements for the selected candidate:

| # | Stage | Occupancy | Reached | Entered | Handoff | Conversion | Authored success |
|---|---|---|---|---|---|---|---|
| 0 | spread_open_cage | 0.7254 | 1 | 1 | 0.9707 | 0.9707 | 1 |
| 1 | low_descent_and_center | 0.1309 | 1 | 0.9775 | 0 | 0 | 1 |
| 2 | four_finger_cage_contact | 0 | 0 | 0 | 0 | 0 | 0 |
| 3 | thumb_lock_in | 0.0004 | 0.0254 | 0.0146 | 0.0244 | 0.9615 | 0.0254 |
| 4 | distributed_preload | 0.1167 | 0.998 | 0.96 | 0.3584 | 0.3591 | 0.998 |
| 5 | vertical_breakaway | 0.0266 | 0.8311 | 0.3584 | 0 | 0 | 0.8311 |
| 6 | clearance_lift_hold | 0 | 0 | 0 | — | — | 0 |

The recorded structural stall was `low_descent_and_center` (index 1); terminal stage reached: no.

Candidate trajectory:

| iter | source | name | objective | frontier | wrist_driven | n_driven |
|---|---|---|---|---|---|---|
| 1 | explore | wrap_cage_then_thumb_lock | 0.038445 | — | yes | 23 |
| 2 | explore | reactive_force_servo_minimal_pinch | 0.024386 | — | yes | 23 |
| 3 | refine:wrap_cage_then_thumb_lock | stabilized_descent_handoff | 0.03856 | — | yes | 23 |

## What the results mean and major discoveries

- The selection loop chose `stabilized_descent_handoff` with objective 0.03856. Selection therefore established a best candidate relative to the tested pool; it did not by itself establish task mastery.
- Recorded success was zero while the associated lift/progress measurement was 0. The candidate generated partial behavior under the run's diagnostics but did not satisfy the full success predicate in that evaluation.
- The stage-flow diagnostic localized the dominant structural bottleneck to `low_descent_and_center`. This is a measured control-flow finding from the authored program, useful for revision without treating the stage label as a framework-authored diagnosis.

## Limitations and reading guidance

- Metrics are reported under the names used by the historical environment and runner. Similar-looking metrics from different generations of the code are not assumed to be numerically interchangeable.
- A maximum observed in training is not treated as held-out performance. Terminal evaluation values are preferred whenever present.
- Missing summaries, empty metric streams, failed retries, and incomplete arms are preserved as negative evidence rather than silently dropped.
- Candidate names and stage names are LLM-authored run artifacts. Interpretations in this historical log do not alter framework prompts, defaults, diagnostics, or task logic.

## Artifact provenance

| Artifact type | Count |
|---|---|
| .json | 14 |
| .md | 8 |
| .txt | 8 |
| .pkl | 1 |

Primary evidence files:

- [`report.json`](../runs/ab_oldstyle_20260707/report.json)

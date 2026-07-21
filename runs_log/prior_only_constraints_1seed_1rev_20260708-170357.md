# Prior Only Constraints 1seed 1rev

- **Run directory:** [`runs/prior_only_constraints_1seed_1rev_20260708-170357`](../runs/prior_only_constraints_1seed_1rev_20260708-170357)
- **Date:** 2026-07-08 17:03:57 (directory timestamp)
- **Status:** completed or completed-with-caveats
- **Experiment class:** agentic prior generation/selection study
- **Recorded task:** grasp and lift a 5cm cube off the table
- **Training metric rows recovered:** 0

## Abstract

This entry reconstructs `prior_only_constraints_1seed_1rev_20260708-170357`, an agentic prior generation/selection study. The account is based only on artifacts that survive in the repository: machine-readable configuration, candidate programs, selection diagnostics, training metrics, evaluations, reports, and logs. The run targeted grasp and lift a 5cm cube off the table. Its purpose and comparisons are inferred from the recorded arm names and configuration where no authored abstract survives; those inferences are identified by this provenance-first framing rather than presented as new experimental facts.

## Experimental structure

No machine-readable configuration file survives at the run root. Structure below is reconstructed from reports, result files, subdirectory names, and logs.
### Selection-loop structure

| Field | Value |
|---|---|
| representation | freeform_staged |
| dof_mode | consider |
| arbiter | prior_only |
| budget | 2 |
| iters_used | 2 |
| n_seeds | 1 |
| wall_hours | 0.162 |

### Arms and sub-runs represented by directories

`llm`

## Results

### Recorded results: `report.json`

| Best-candidate field | Value |
|---|---|
| name | reachable_wrist_then_three_point_gentle_lift |
| source | explore |
| objective | 0.022487 |
| accounting.n_driven | 23 |
| accounting.n_unused_listed | 7 |
| accounting.wrist_driven | yes |

| Diagnostic | Value |
|---|---|
| success_rate | 0 |
| reach_rate | 0 |
| grasp_rate | 0 |
| grasp_lift_rate | 0 |
| lift_reached_rate | 0 |
| lift_max | 0 |
| base_return | 2.9706 |
| train_return | 2.9706 |
| shaped_return | 0 |
| action_abs_mean | 0.0124 |
| saturation_frac | 0 |
| note | behavior at PPO iteration 0: the prior acting through an UNTRAINED policy (network output ~0, so this is essentially the prior alone). Compare with the trained metrics -- a defect present here is a PRIOR problem; one that only appears af… |
| base_reward_weight | 1 |
| instant_success_rate | 0 |
| hard_clip_frac | 0 |
| failure_rate | 0 |
| calm_frac | 1 |
| reach_rate_calm | 0 |

Stage-flow measurements for the selected candidate:

| # | Stage | Occupancy | Reached | Entered | Handoff | Conversion | Authored success |
|---|---|---|---|---|---|---|---|
| 0 | wrist_attitude | 0.1899 | 1 | 1 | 0 | 0 | 1 |
| 1 | open_long_fingers | 0 | 0 | 0 | 0 | 0 | 0 |
| 2 | thumb_opposition_shape | 0 | 0 | 0 | 0 | 0 | 0 |
| 3 | center_xy | 0 | 0 | 0 | 0 | 0 | 0 |
| 4 | descend_to_side_height | 0 | 0 | 0 | 0 | 0 | 0 |
| 5 | first_opposing_touch | 0.8101 | 1 | 1 | 0 | 0 | 0 |
| 6 | secure_load | 0 | 0 | 0 | 0 | 0 | 0 |
| 7 | lift_with_grasp | 0 | 0 | 0 | 0 | 0 | 0 |
| 8 | stabilize_hold | 0 | 0 | 0 | — | — | 0 |

The recorded structural stall was `wrist_attitude` (index 0); terminal stage reached: no.

Candidate trajectory:

| iter | source | name | objective | frontier | wrist_driven | n_driven |
|---|---|---|---|---|---|---|
| 1 | explore | reachable_wrist_then_three_point_gentle_lift | 0.022487 | — | yes | 23 |
| 2 | refine:reachable_wrist_then_three_point_gentle_lift | wrist_handoff_to_opening_despite_clearance_fix | 0.022487 | — | yes | 23 |

## What the results mean and major discoveries

- The selection loop chose `reachable_wrist_then_three_point_gentle_lift` with objective 0.022487. Selection therefore established a best candidate relative to the tested pool; it did not by itself establish task mastery.
- Recorded success was zero while the associated lift/progress measurement was 0. The candidate generated partial behavior under the run's diagnostics but did not satisfy the full success predicate in that evaluation.
- The stage-flow diagnostic localized the dominant structural bottleneck to `wrist_attitude`. This is a measured control-flow finding from the authored program, useful for revision without treating the stage label as a framework-authored diagnosis.

## Limitations and reading guidance

- Metrics are reported under the names used by the historical environment and runner. Similar-looking metrics from different generations of the code are not assumed to be numerically interchangeable.
- A maximum observed in training is not treated as held-out performance. Terminal evaluation values are preferred whenever present.
- Missing summaries, empty metric streams, failed retries, and incomplete arms are preserved as negative evidence rather than silently dropped.
- Candidate names and stage names are LLM-authored run artifacts. Interpretations in this historical log do not alter framework prompts, defaults, diagnostics, or task logic.

## Artifact provenance

| Artifact type | Count |
|---|---|
| .json | 8 |
| .md | 5 |
| .txt | 5 |
| .pkl | 1 |

Primary evidence files:

- [`report.json`](../runs/prior_only_constraints_1seed_1rev_20260708-170357/report.json)

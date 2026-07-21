# Prior Only Gatefix 1seed 4rev

- **Run directory:** [`runs/prior_only_gatefix_1seed_4rev_20260708-134057`](../runs/prior_only_gatefix_1seed_4rev_20260708-134057)
- **Date:** 2026-07-08 13:40:57 (directory timestamp)
- **Status:** completed or completed-with-caveats
- **Experiment class:** agentic prior generation/selection study
- **Recorded task:** grasp and lift a 5cm cube off the table
- **Training metric rows recovered:** 0

## Abstract

This entry reconstructs `prior_only_gatefix_1seed_4rev_20260708-134057`, an agentic prior generation/selection study. The account is based only on artifacts that survive in the repository: machine-readable configuration, candidate programs, selection diagnostics, training metrics, evaluations, reports, and logs. The run targeted grasp and lift a 5cm cube off the table. Its purpose and comparisons are inferred from the recorded arm names and configuration where no authored abstract survives; those inferences are identified by this provenance-first framing rather than presented as new experimental facts.

## Experimental structure

No machine-readable configuration file survives at the run root. Structure below is reconstructed from reports, result files, subdirectory names, and logs.
### Selection-loop structure

| Field | Value |
|---|---|
| representation | freeform_staged |
| dof_mode | consider |
| arbiter | prior_only |
| budget | 5 |
| iters_used | 5 |
| n_seeds | 1 |
| wall_hours | 0.34 |

### Arms and sub-runs represented by directories

`branches`, `llm`, `videos`

## Results

### Recorded results: `report.json`

| Best-candidate field | Value |
|---|---|
| name | center_descent_handoff_activation_fix |
| source | refine:tilted_open_cage_thumb_index_middle_lift |
| objective | 0.115163 |
| accounting.n_driven | 23 |
| accounting.n_unused_listed | 0 |
| accounting.wrist_driven | yes |

| Diagnostic | Value |
|---|---|
| success_rate | 0 |
| reach_rate | 0.2578 |
| grasp_rate | 0.1055 |
| grasp_lift_rate | 0.0078 |
| lift_reached_rate | 0.0273 |
| lift_max | 0.0062 |
| base_return | 5.924 |
| train_return | 5.8846 |
| shaped_return | -0.0394 |
| action_abs_mean | 0.0332 |
| saturation_frac | 0 |
| note | behavior at PPO iteration 0: the prior acting through an UNTRAINED policy (network output ~0, so this is essentially the prior alone). Compare with the trained metrics -- a defect present here is a PRIOR problem; one that only appears af… |
| base_reward_weight | 1 |
| instant_success_rate | 0 |
| hard_clip_frac | 0 |
| failure_rate | 0.1523 |
| calm_frac | 0.8477 |
| reach_rate_calm | 0.1055 |

Stage-flow measurements for the selected candidate:

| # | Stage | Occupancy | Reached | Entered | Handoff | Conversion | Authored success |
|---|---|---|---|---|---|---|---|
| 0 | establish_wrist_attitude | 0.5038 | 1 | 1 | 1 | 1 | 1 |
| 1 | pre_shape_open_cage | 0.088 | 1 | 1 | 1 | 1 | 1 |
| 2 | lateral_center | 0.375 | 1 | 1 | 0.5469 | 0.5469 | 0.5234 |
| 3 | controlled_descent | 0.0332 | 1 | 0.5469 | 0 | 0 | 0.1172 |
| 4 | light_opposing_contact | 0 | 0 | 0 | 0 | 0 | 0 |
| 5 | build_support | 0 | 0 | 0 | 0 | 0 | 0 |
| 6 | initial_lift_breakaway | 0 | 0 | 0 | 0 | 0 | 0 |
| 7 | stable_lifted_hold | 0 | 0 | 0 | — | — | 0 |

The recorded structural stall was `controlled_descent` (index 3); terminal stage reached: no.

Candidate trajectory:

| iter | source | name | objective | frontier | wrist_driven | n_driven |
|---|---|---|---|---|---|---|
| 1 | explore | tilted_open_cage_thumb_index_middle_lift | 0.019343 | — | yes | 23 |
| 2 | refine:tilted_open_cage_thumb_index_middle_lift | wrist_handoff_activation_fix | 0.019344 | — | yes | 23 |
| 3 | refine:tilted_open_cage_thumb_index_middle_lift | handoff_activation_rescale_for_wrist_and_shape | 0.019381 | — | yes | 23 |
| 4 | refine:tilted_open_cage_thumb_index_middle_lift | descent_takes_over_on_centered_activation | 0.037427 | — | yes | 23 |
| 5 | refine:tilted_open_cage_thumb_index_middle_lift | center_descent_handoff_activation_fix | 0.115163 | — | yes | 23 |

## What the results mean and major discoveries

- The selection loop chose `center_descent_handoff_activation_fix` with objective 0.115163. Selection therefore established a best candidate relative to the tested pool; it did not by itself establish task mastery.
- Recorded success was zero while the associated lift/progress measurement was 0.0273. The candidate generated partial behavior under the run's diagnostics but did not satisfy the full success predicate in that evaluation.
- The stage-flow diagnostic localized the dominant structural bottleneck to `controlled_descent`. This is a measured control-flow finding from the authored program, useful for revision without treating the stage label as a framework-authored diagnosis.

## Limitations and reading guidance

- Metrics are reported under the names used by the historical environment and runner. Similar-looking metrics from different generations of the code are not assumed to be numerically interchangeable.
- A maximum observed in training is not treated as held-out performance. Terminal evaluation values are preferred whenever present.
- Missing summaries, empty metric streams, failed retries, and incomplete arms are preserved as negative evidence rather than silently dropped.
- Candidate names and stage names are LLM-authored run artifacts. Interpretations in this historical log do not alter framework prompts, defaults, diagnostics, or task logic.

## Artifact provenance

| Artifact type | Count |
|---|---|
| .json | 16 |
| .md | 8 |
| .txt | 8 |
| .html | 1 |
| .mp4 | 1 |
| .pkl | 1 |

Primary evidence files:

- [`report.json`](../runs/prior_only_gatefix_1seed_4rev_20260708-134057/report.json)

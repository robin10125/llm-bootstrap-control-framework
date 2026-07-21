# Prior Only Reset 1seed 4rev

- **Run directory:** [`runs/prior_only_reset_1seed_4rev_20260708-175339`](../runs/prior_only_reset_1seed_4rev_20260708-175339)
- **Date:** 2026-07-08 17:53:39 (directory timestamp)
- **Status:** completed or completed-with-caveats
- **Experiment class:** agentic prior generation/selection study
- **Recorded task:** grasp and lift a 5cm cube off the table
- **Training metric rows recovered:** 0

## Abstract

This entry reconstructs `prior_only_reset_1seed_4rev_20260708-175339`, an agentic prior generation/selection study. The account is based only on artifacts that survive in the repository: machine-readable configuration, candidate programs, selection diagnostics, training metrics, evaluations, reports, and logs. The run targeted grasp and lift a 5cm cube off the table. Its purpose and comparisons are inferred from the recorded arm names and configuration where no authored abstract survives; those inferences are identified by this provenance-first framing rather than presented as new experimental facts.

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
| wall_hours | 0.418 |

### Arms and sub-runs represented by directories

`llm`

## Results

### Recorded results: `report.json`

| Best-candidate field | Value |
|---|---|
| name | contact_force_gated_descent_and_centering |
| source | refine:reachable_wrist_opposed_finger_thumb_lift |
| objective | 0.158495 |
| accounting.n_driven | 23 |
| accounting.n_unused_listed | 0 |
| accounting.wrist_driven | yes |

| Diagnostic | Value |
|---|---|
| graded_objective | 0.1585 |
| success_rate | 0.043 |
| reach_rate | 0.4219 |
| grasp_rate | 0 |
| grasp_lift_rate | 0 |
| lift_reached_rate | 0.043 |
| lift_max | 0.0123 |
| base_return | 4.7782 |
| train_return | 5.3266 |
| action_abs_mean | 0.0144 |
| note | behavior at PPO iteration 0: the prior acting through an UNTRAINED policy (network output ~0, so this is essentially the prior alone). Compare with the trained metrics -- a defect present here is a PRIOR problem; one that only appears af… |
| task_fitness | 0.3344 |
| stage_return | 0.5484 |
| prior_scale_mean | 0.9998 |

Candidate trajectory:

| iter | source | name | objective | frontier | wrist_driven | n_driven |
|---|---|---|---|---|---|---|
| 1 | explore | reachable_wrist_opposed_finger_thumb_lift | 0.156837 | — | yes | 23 |
| 2 | refine:reachable_wrist_opposed_finger_thumb_lift | focused_wrist_completion_sign_fix | 0.030648 | — | yes | 23 |
| 3 | refine:reachable_wrist_opposed_finger_thumb_lift | contact_force_gated_descent_and_centering | 0.158495 | — | yes | 23 |
| 4 | refine:reachable_wrist_opposed_finger_thumb_lift | targeted_wrist_completion_sign_fix | 0.14599 | — | yes | 23 |
| 5 | refine:reachable_wrist_opposed_finger_thumb_lift | targeted_precontact_gentler_alignment_and_real_wrist_completion | 0.040039 | — | yes | 23 |

## What the results mean and major discoveries

- The selection loop chose `contact_force_gated_descent_and_centering` with objective 0.158495. Selection therefore established a best candidate relative to the tested pool; it did not by itself establish task mastery.

## Limitations and reading guidance

- Metrics are reported under the names used by the historical environment and runner. Similar-looking metrics from different generations of the code are not assumed to be numerically interchangeable.
- A maximum observed in training is not treated as held-out performance. Terminal evaluation values are preferred whenever present.
- Missing summaries, empty metric streams, failed retries, and incomplete arms are preserved as negative evidence rather than silently dropped.
- Candidate names and stage names are LLM-authored run artifacts. Interpretations in this historical log do not alter framework prompts, defaults, diagnostics, or task logic.

## Artifact provenance

| Artifact type | Count |
|---|---|
| .json | 11 |
| .md | 8 |
| .txt | 8 |
| .pkl | 1 |

Primary evidence files:

- [`report.json`](../runs/prior_only_reset_1seed_4rev_20260708-175339/report.json)

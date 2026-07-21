# Monotone Prioronly

- **Run directory:** [`runs/monotone_prioronly_20260709-144612`](../runs/monotone_prioronly_20260709-144612)
- **Date:** 2026-07-09 14:46:12 (directory timestamp)
- **Status:** completed or completed-with-caveats
- **Experiment class:** agentic prior generation/selection study
- **Recorded task:** grasp and lift a 5cm cube off the table
- **Training metric rows recovered:** 0

## Abstract

This entry reconstructs `monotone_prioronly_20260709-144612`, an agentic prior generation/selection study. The account is based only on artifacts that survive in the repository: machine-readable configuration, candidate programs, selection diagnostics, training metrics, evaluations, reports, and logs. The run targeted grasp and lift a 5cm cube off the table. Its purpose and comparisons are inferred from the recorded arm names and configuration where no authored abstract survives; those inferences are identified by this provenance-first framing rather than presented as new experimental facts.

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
| wall_hours | 0.444 |

### Arms and sub-runs represented by directories

`branches`, `llm`, `videos`

## Results

### Recorded results: `report.json`

| Best-candidate field | Value |
|---|---|
| name | stage7_early_takeover_with_contact_guarded_raise |
| source | refine:reachable_pitch_opposed_digit_lift |
| objective | 0.678336 |
| accounting.n_driven | 22 |
| accounting.n_unused_listed | 2 |
| accounting.wrist_driven | yes |

| Diagnostic | Value |
|---|---|
| graded_objective | 0.6783 |
| success_rate | 0 |
| reach_rate | 1 |
| grasp_rate | 0.8516 |
| grasp_lift_rate | 0 |
| lift_reached_rate | 0 |
| lift_max | 0.0063 |
| base_return | 9.7429 |
| train_return | 12.4637 |
| action_abs_mean | 0.0128 |
| note | behavior at PPO iteration 0: the prior acting through an UNTRAINED policy (network output ~0, so this is essentially the prior alone). Compare with the trained metrics -- a defect present here is a PRIOR problem; one that only appears af… |
| task_fitness | 0.6314 |
| stage_return | 2.7208 |
| prior_scale_mean | 0.9976 |

Stage-flow measurements for the selected candidate:

| # | Stage | Occupancy | Reached | Entered | Handoff | Conversion | Authored success |
|---|---|---|---|---|---|---|---|
| 0 | wrist_reachable_working_pitch | 0.117 | 1 | 1 | 0.6328 | 0.6328 | 1 |
| 1 | digit_and_thumb_preshape_clear | 0.0408 | 1 | 0.6328 | 0.6289 | 0.6289 | 0.7188 |
| 2 | horizontal_alignment_above_cube | 0.0119 | 1 | 0.6289 | 1 | 1 | 1 |
| 3 | guarded_descent_to_side_contact_height | 0.2299 | 1 | 1 | 0.8867 | 0.8867 | 0.2461 |
| 4 | first_light_opposing_touch | 0.0606 | 0.9805 | 0.8867 | 0.8047 | 0.8207 | 0 |
| 5 | balanced_capture_without_sliding | 0.0605 | 0.9688 | 0.8047 | 0.8555 | 0.8831 | 0 |
| 6 | initial_lift_with_grasp_locked | 0.2122 | 0.957 | 0.8555 | 0.6836 | 0.7143 | 0 |
| 7 | raise_and_stabilize_clear_of_table | 0.2672 | 0.6836 | 0.6836 | — | — | 0 |

Candidate trajectory:

| iter | source | name | objective | frontier | wrist_driven | n_driven |
|---|---|---|---|---|---|---|
| 1 | explore | reachable_pitch_opposed_digit_lift | 0.268369 | 3 | yes | 22 |
| 2 | refine:reachable_pitch_opposed_digit_lift | earlier_touch_handoff_from_descent | 0.720975 | 4 | yes | 22 |
| 3 | refine:reachable_pitch_opposed_digit_lift | targeted_touch_to_capture_entry_v1 | 0.632734 | 5 | yes | 22 |
| 4 | refine:reachable_pitch_opposed_digit_lift | stage6_early_lift_entry_from_light_capture | 0.70303 | 6 | yes | 22 |
| 5 | refine:reachable_pitch_opposed_digit_lift | stage7_early_takeover_with_contact_guarded_raise | 0.678336 | 8 | yes | 22 |

## What the results mean and major discoveries

- The selection loop chose `stage7_early_takeover_with_contact_guarded_raise` with objective 0.678336. Selection therefore established a best candidate relative to the tested pool; it did not by itself establish task mastery.
- Recorded success was zero while the associated lift/progress measurement was 0. The candidate generated partial behavior under the run's diagnostics but did not satisfy the full success predicate in that evaluation.

## Limitations and reading guidance

- Metrics are reported under the names used by the historical environment and runner. Similar-looking metrics from different generations of the code are not assumed to be numerically interchangeable.
- A maximum observed in training is not treated as held-out performance. Terminal evaluation values are preferred whenever present.
- Missing summaries, empty metric streams, failed retries, and incomplete arms are preserved as negative evidence rather than silently dropped.
- Candidate names and stage names are LLM-authored run artifacts. Interpretations in this historical log do not alter framework prompts, defaults, diagnostics, or task logic.

## Artifact provenance

| Artifact type | Count |
|---|---|
| .json | 18 |
| .md | 8 |
| .txt | 8 |
| .html | 1 |
| .mp4 | 1 |
| .pkl | 1 |

Primary evidence files:

- [`report.json`](../runs/monotone_prioronly_20260709-144612/report.json)

# Prior Refine Mb

- **Run directory:** [`runs/prior_refine_mb_20260704`](../runs/prior_refine_mb_20260704)
- **Date:** 2026-07-04 (directory timestamp)
- **Status:** completed or completed-with-caveats
- **Experiment class:** agentic prior generation/selection study
- **Recorded task:** grasp and lift a 5cm cube off the table
- **Training metric rows recovered:** 0

## Abstract

This entry reconstructs `prior_refine_mb_20260704`, an agentic prior generation/selection study. The account is based only on artifacts that survive in the repository: machine-readable configuration, candidate programs, selection diagnostics, training metrics, evaluations, reports, and logs. The run targeted grasp and lift a 5cm cube off the table. Its purpose and comparisons are inferred from the recorded arm names and configuration where no authored abstract survives; those inferences are identified by this provenance-first framing rather than presented as new experimental facts.

## Experimental structure

No machine-readable configuration file survives at the run root. Structure below is reconstructed from reports, result files, subdirectory names, and logs.
### Selection-loop structure

| Field | Value |
|---|---|
| representation | freeform_staged |
| dof_mode | encourage |
| arbiter | prior_only |
| budget | 13 |
| iters_used | 13 |
| n_seeds | 3 |
| wall_hours | 0.733 |

### Arms and sub-runs represented by directories

`llm`

## Results

### Recorded results: `report.json`

| Best-candidate field | Value |
|---|---|
| name | thumb_index_pinch_then_reinforce |
| source | explore |
| objective | 0.07129 |
| accounting.n_driven | 23 |
| accounting.n_unused_listed | 0 |
| accounting.wrist_driven | yes |

| Diagnostic | Value |
|---|---|
| success_rate | 0 |
| reach_rate | 0.1191 |
| grasp_rate | 0.0527 |
| grasp_lift_rate | 0.0156 |
| lift_reached_rate | 0.376 |
| lift_max | 0.0517 |
| base_return | 1.645 |
| train_return | 1.485 |
| shaped_return | -0.16 |
| action_abs_mean | 0.2494 |
| saturation_frac | 0 |
| note | behavior at PPO iteration 0: the prior acting through an UNTRAINED policy (network output ~0, so this is essentially the prior alone). Compare with the trained metrics -- a defect present here is a PRIOR problem; one that only appears af… |
| base_reward_weight | 1 |
| instant_success_rate | 0 |
| hard_clip_frac | 0 |
| objective_batch_std | 0.0036 |

Stage-flow measurements for the selected candidate:

| # | Stage | Occupancy | Reached | Entered | Handoff | Conversion | Authored success |
|---|---|---|---|---|---|---|---|
| 0 | precision_approach | 0.4851 | 1 | 0.8184 | 0.957 | 0.957 | 0.0107 |
| 1 | thumb_index_pinch | 0.0943 | 0.9863 | 0.957 | 0.4824 | 0.4891 | 0.001 |
| 2 | reinforce_wrap | 0.0154 | 0.5312 | 0.4824 | 0.5117 | 0.9632 | 0 |
| 3 | pinch_lift_recover | 0.4052 | 0.5293 | 0.5166 | — | — | 0.0381 |

Candidate trajectory:

| iter | source | name | objective | frontier | wrist_driven | n_driven |
|---|---|---|---|---|---|---|
| 1 | explore | center_then_opposed_lift | 0.043555 | — | yes | 23 |
| 2 | explore | vertical_settle_cage | 0.045312 | — | yes | 23 |
| 3 | explore | thumb_index_pinch_then_reinforce | 0.07129 | — | yes | 23 |
| 4 | refine:thumb_index_pinch_then_reinforce | reinforce_wrap_entry_recenter | 0.032911 | — | yes | 23 |
| 5 | refine:thumb_index_pinch_then_reinforce | reinforce_wrap_earlier_decelerated_entry | 0.037305 | — | yes | 23 |
| 6 | refine:thumb_index_pinch_then_reinforce | stage1_near_gated_gentle_pinch | 0.035254 | — | yes | 23 |
| 7 | refine:thumb_index_pinch_then_reinforce | stage1_near_preserving_decelerated_pinch | 0.031739 | — | yes | 23 |
| 8 | refine:thumb_index_pinch_then_reinforce | stage1_distance_yield_and_decelerated_pinch | 0.030273 | — | yes | 23 |
| 9 | refine:thumb_index_pinch_then_reinforce | stage1_decelerated_contact_preserving_pinch | 0.06338 | — | yes | 23 |
| 10 | refine:thumb_index_pinch_then_reinforce | stage1_near_preserving_sign_corrected_pinch | 0.03418 | — | yes | 23 |
| 11 | refine:thumb_index_pinch_then_reinforce | stage1_contact_gated_pinch | 0.032812 | — | yes | 23 |
| 12 | refine:thumb_index_pinch_then_reinforce | stage1_near_guarded_gentle_pinch | 0.029688 | — | yes | 23 |
| 13 | refine:thumb_index_pinch_then_reinforce | stage1_x_decelerated_contact_preserving_pinch | 0.057617 | — | yes | 23 |

## What the results mean and major discoveries

- The selection loop chose `thumb_index_pinch_then_reinforce` with objective 0.07129. Selection therefore established a best candidate relative to the tested pool; it did not by itself establish task mastery.
- Recorded success was zero while the associated lift/progress measurement was 0.376. The candidate generated partial behavior under the run's diagnostics but did not satisfy the full success predicate in that evaluation.

## Limitations and reading guidance

- Metrics are reported under the names used by the historical environment and runner. Similar-looking metrics from different generations of the code are not assumed to be numerically interchangeable.
- A maximum observed in training is not treated as held-out performance. Terminal evaluation values are preferred whenever present.
- Missing summaries, empty metric streams, failed retries, and incomplete arms are preserved as negative evidence rather than silently dropped.
- Candidate names and stage names are LLM-authored run artifacts. Interpretations in this historical log do not alter framework prompts, defaults, diagnostics, or task logic.

## Artifact provenance

| Artifact type | Count |
|---|---|
| .json | 19 |
| .md | 15 |
| .txt | 15 |
| .pkl | 1 |

Primary evidence files:

- [`report.json`](../runs/prior_refine_mb_20260704/report.json)

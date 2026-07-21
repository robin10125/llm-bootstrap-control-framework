# Lift Sustain 4seed 4refine Shortppo

- **Run directory:** [`runs/lift_sustain_4seed_4refine_shortppo_20260704`](../runs/lift_sustain_4seed_4refine_shortppo_20260704)
- **Date:** 2026-07-04 (directory timestamp)
- **Status:** completed or completed-with-caveats
- **Experiment class:** PPO bias/reward/prior experiment
- **Recorded task:** grasp and lift a 5cm cube off the table
- **Training metric rows recovered:** 0

## Abstract

This entry reconstructs `lift_sustain_4seed_4refine_shortppo_20260704`, a PPO bias/reward/prior experiment. The account is based only on artifacts that survive in the repository: machine-readable configuration, candidate programs, selection diagnostics, training metrics, evaluations, reports, and logs. The run targeted grasp and lift a 5cm cube off the table. Its purpose and comparisons are inferred from the recorded arm names and configuration where no authored abstract survives; those inferences are identified by this provenance-first framing rather than presented as new experimental facts.

## Experimental structure

No machine-readable configuration file survives at the run root. Structure below is reconstructed from reports, result files, subdirectory names, and logs.
### Selection-loop structure

| Field | Value |
|---|---|
| representation | freeform_staged |
| dof_mode | encourage |
| arbiter | short_ppo |
| budget | 8 |
| iters_used | 8 |
| n_seeds | 4 |
| wall_hours | 1.121 |

### Arms and sub-runs represented by directories

`llm`, `ppo`, `selection_videos`

## Results

### Recorded results: `report.json`

| Best-candidate field | Value |
|---|---|
| name | thumb_first_opposition_then_three_finger_wrap |
| source | explore |
| objective | 0.033203 |
| accounting.n_driven | 23 |
| accounting.n_unused_listed | 0 |
| accounting.wrist_driven | yes |

| Diagnostic | Value |
|---|---|
| success_rate | 0 |
| reach_rate | 0.1484 |
| grasp_rate | 0.0039 |
| grasp_lift_rate | 0 |
| lift_reached_rate | 0.3438 |
| lift_max | 0.0457 |
| base_return | 0.1705 |
| train_return | 0.0478 |
| shaped_return | -0.1227 |
| action_abs_mean | 0.249 |
| saturation_frac | 0 |
| base_reward_weight | 1 |
| instant_success_rate | 0 |
| hard_clip_frac | 0 |

Stage-flow measurements for the selected candidate:

| # | Stage | Occupancy | Reached | Entered | Handoff | Conversion | Authored success |
|---|---|---|---|---|---|---|---|
| 0 | wide_xy_approach | 0.9611 | 1 | 1 | 0.25 | 0.25 | 0.3633 |
| 1 | thumb_opposition_set | 0.0348 | 0.9961 | 0.2812 | 0.0273 | 0.0275 | 0.0273 |
| 2 | three_finger_wrap | 0.0041 | 0.0547 | 0.0273 | 0 | 0 | 0 |
| 3 | little_guard_lift | 0 | 0 | 0 | — | — | 0 |

The recorded structural stall was `thumb_opposition_set` (index 1); terminal stage reached: no.

Candidate trajectory:

| iter | source | name | objective | frontier | wrist_driven | n_driven |
|---|---|---|---|---|---|---|
| 1 | explore | centered_cage_then_vertical_carry | 0.013281 | — | yes | 23 |
| 2 | explore | thumb_first_opposition_then_three_finger_wrap | 0.033203 | — | yes | 23 |
| 3 | explore | low_cup_scoop_without_lateral_sweep | 0.021875 | — | yes | 23 |
| 4 | explore | progressive_precision_pinch_with_base_microservo | 0.016406 | — | yes | 23 |
| 5 | refine:thumb_first_opposition_then_three_finger_wrap | gentler_wrap_entry_from_centered_upper | 0.030469 | — | yes | 23 |
| 6 | refine:thumb_first_opposition_then_three_finger_wrap | earlier_wrap_takeover_from_centered_high_gap | 0.015625 | — | yes | 23 |
| 7 | refine:thumb_first_opposition_then_three_finger_wrap | gentler_thumb_set_preserve_speed | 0.008594 | — | yes | 23 |
| 8 | refine:thumb_first_opposition_then_three_finger_wrap | gentler_thumb_preserve_centering_handoff | 0.025 | — | yes | 23 |

## What the results mean and major discoveries

- The selection loop chose `thumb_first_opposition_then_three_finger_wrap` with objective 0.033203. Selection therefore established a best candidate relative to the tested pool; it did not by itself establish task mastery.
- Recorded success was zero while the associated lift/progress measurement was 0.3438. The candidate generated partial behavior under the run's diagnostics but did not satisfy the full success predicate in that evaluation.
- The stage-flow diagnostic localized the dominant structural bottleneck to `thumb_opposition_set`. This is a measured control-flow finding from the authored program, useful for revision without treating the stage label as a framework-authored diagnosis.

## Limitations and reading guidance

- Metrics are reported under the names used by the historical environment and runner. Similar-looking metrics from different generations of the code are not assumed to be numerically interchangeable.
- A maximum observed in training is not treated as held-out performance. Terminal evaluation values are preferred whenever present.
- Missing summaries, empty metric streams, failed retries, and incomplete arms are preserved as negative evidence rather than silently dropped.
- Candidate names and stage names are LLM-authored run artifacts. Interpretations in this historical log do not alter framework prompts, defaults, diagnostics, or task logic.

## Artifact provenance

| Artifact type | Count |
|---|---|
| .pkl | 58 |
| .json | 18 |
| .md | 9 |
| .txt | 9 |
| .mp4 | 6 |
| .csv | 1 |
| .html | 1 |
| .py | 1 |

Primary evidence files:

- [`report.json`](../runs/lift_sustain_4seed_4refine_shortppo_20260704/report.json)

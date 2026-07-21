# Staged Stagefocus

- **Run directory:** [`runs/staged_stagefocus_20260702-000302`](../runs/staged_stagefocus_20260702-000302)
- **Date:** 2026-07-02 00:03:02 (directory timestamp)
- **Status:** completed or completed-with-caveats
- **Experiment class:** PPO bias/reward/prior experiment
- **Recorded task:** grasp and lift a 5cm cube off the table
- **Training metric rows recovered:** 0

## Abstract

This entry reconstructs `staged_stagefocus_20260702-000302`, a PPO bias/reward/prior experiment. The account is based only on artifacts that survive in the repository: machine-readable configuration, candidate programs, selection diagnostics, training metrics, evaluations, reports, and logs. The run targeted grasp and lift a 5cm cube off the table. Its purpose and comparisons are inferred from the recorded arm names and configuration where no authored abstract survives; those inferences are identified by this provenance-first framing rather than presented as new experimental facts.

## Experimental structure

No machine-readable configuration file survives at the run root. Structure below is reconstructed from reports, result files, subdirectory names, and logs.
### Selection-loop structure

| Field | Value |
|---|---|
| representation | freeform_staged |
| dof_mode | encourage |
| arbiter | short_ppo |
| budget | 10 |
| iters_used | 5 |
| n_seeds | 3 |

### Arms and sub-runs represented by directories

`llm`, `ppo`

## Results

### Recorded results: `report.json`

| Best-candidate field | Value |
|---|---|
| name | hard_guarded_funnel_grip |
| source | explore |
| objective | 0.040625 |
| accounting.n_driven | 23 |
| accounting.n_unused_listed | 0 |
| accounting.wrist_driven | yes |

| Diagnostic | Value |
|---|---|
| reach_rate | 0.2031 |
| grasp_rate | 0 |
| lift_reached_rate | 0.4219 |
| lift_max | 0.0572 |
| saturation_frac | 0 |
| trained_success | 0 |
| instant_success | 0 |
| mean_contacts | 0.2188 |
| mean_closure | 0.231 |
| obj_xy_drift | 0.7312 |

Stage-flow measurements for the selected candidate:

| # | Stage | Occupancy | Reached | Entered | Handoff | Conversion | Authored success |
|---|---|---|---|---|---|---|---|
| 0 | recover_far_or_lost | 0.9997 | 1 | — | — | — | — |
| 1 | funnel_contact_geometry | 0.0002 | 0.0117 | — | — | — | — |
| 2 | five_finger_clamp | 0 | 0.0039 | — | — | — | — |
| 3 | lift_guarded | 0 | 0 | — | — | — | — |
| 4 | airborne_hold | 0 | 0 | — | — | — | — |

The recorded structural stall was `recover_far_or_lost` (index 0); terminal stage reached: no.

Candidate trajectory:

| iter | source | name | objective | frontier | wrist_driven | n_driven |
|---|---|---|---|---|---|---|
| 1 | explore | center_then_opposed_pinch_lift | 0.038281 | — | yes | 23 |
| 2 | explore | hard_guarded_funnel_grip | 0.040625 | — | yes | 23 |
| 3 | explore | low_scoop_avoidance_tripod_with_side_fence | 0.033594 | — | yes | 23 |
| 4 | refine:hard_guarded_funnel_grip | stage0_bidirectional_recenter_and_height_recover | 0.035938 | — | yes | 23 |
| 5 | refine:center_then_opposed_pinch_lift | stronger_far_approach_centering | 0.021875 | — | yes | 23 |

## What the results mean and major discoveries

- The selection loop chose `hard_guarded_funnel_grip` with objective 0.040625. Selection therefore established a best candidate relative to the tested pool; it did not by itself establish task mastery.
- The stage-flow diagnostic localized the dominant structural bottleneck to `recover_far_or_lost`. This is a measured control-flow finding from the authored program, useful for revision without treating the stage label as a framework-authored diagnosis.

## Limitations and reading guidance

- Metrics are reported under the names used by the historical environment and runner. Similar-looking metrics from different generations of the code are not assumed to be numerically interchangeable.
- A maximum observed in training is not treated as held-out performance. Terminal evaluation values are preferred whenever present.
- Missing summaries, empty metric streams, failed retries, and incomplete arms are preserved as negative evidence rather than silently dropped.
- Candidate names and stage names are LLM-authored run artifacts. Interpretations in this historical log do not alter framework prompts, defaults, diagnostics, or task logic.

## Artifact provenance

| Artifact type | Count |
|---|---|
| .pkl | 36 |
| .json | 11 |
| .md | 7 |
| .txt | 7 |

Primary evidence files:

- [`report.json`](../runs/staged_stagefocus_20260702-000302/report.json)

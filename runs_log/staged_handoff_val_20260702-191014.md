# Staged Handoff Val

- **Run directory:** [`runs/staged_handoff_val_20260702-191014`](../runs/staged_handoff_val_20260702-191014)
- **Date:** 2026-07-02 19:10:14 (directory timestamp)
- **Status:** completed or completed-with-caveats
- **Experiment class:** PPO bias/reward/prior experiment
- **Recorded task:** grasp and lift a 5cm cube off the table
- **Training metric rows recovered:** 0

## Abstract

This entry reconstructs `staged_handoff_val_20260702-191014`, a PPO bias/reward/prior experiment. The account is based only on artifacts that survive in the repository: machine-readable configuration, candidate programs, selection diagnostics, training metrics, evaluations, reports, and logs. The run targeted grasp and lift a 5cm cube off the table. Its purpose and comparisons are inferred from the recorded arm names and configuration where no authored abstract survives; those inferences are identified by this provenance-first framing rather than presented as new experimental facts.

## Experimental structure

No machine-readable configuration file survives at the run root. Structure below is reconstructed from reports, result files, subdirectory names, and logs.
### Selection-loop structure

| Field | Value |
|---|---|
| representation | freeform_staged |
| dof_mode | encourage |
| arbiter | short_ppo |
| budget | 10 |
| iters_used | 4 |
| n_seeds | 3 |

### Arms and sub-runs represented by directories

`llm`, `ppo`

## Results

### Recorded results: `report.json`

| Best-candidate field | Value |
|---|---|
| name | continuous_center_squeeze_lift |
| source | explore |
| objective | 0.279297 |
| accounting.n_driven | 23 |
| accounting.n_unused_listed | 0 |
| accounting.wrist_driven | yes |

| Diagnostic | Value |
|---|---|
| reach_rate | 0.2539 |
| grasp_rate | 0.2539 |
| lift_reached_rate | 0.4219 |
| lift_max | 0.0606 |
| saturation_frac | 0 |
| trained_success | 0 |
| instant_success | 0 |
| mean_contacts | 0.2812 |
| mean_closure | 0.602 |
| obj_xy_drift | 0.762 |

Stage-flow measurements for the selected candidate:

| # | Stage | Occupancy | Reached | Entered | Handoff | Conversion | Authored success |
|---|---|---|---|---|---|---|---|
| 0 | center_and_descend | 0.9708 | 1 | 1 | 0.0234 | 0.0234 | 0.0039 |
| 1 | opposed_power_grasp | 0.0017 | 0.0586 | 0.0234 | 0.0234 | 0.4 | 0 |
| 2 | secure_vertical_lift | 0.0275 | 0.2695 | 0.168 | — | — | 0.0508 |

The recorded structural stall was `center_and_descend` (index 0); terminal stage reached: no.

Candidate trajectory:

| iter | source | name | objective | frontier | wrist_driven | n_driven |
|---|---|---|---|---|---|---|
| 1 | explore | continuous_center_squeeze_lift | 0.279297 | — | yes | 23 |
| 2 | explore | hard_milestone_precision_then_hoist | 0.184766 | — | yes | 23 |
| 3 | explore | enveloping_cup_with_slip_guard | 0.111718 | — | yes | 23 |
| 4 | refine:continuous_center_squeeze_lift | stage1_entry_nudge_close_range | 0.193359 | — | yes | 23 |

## What the results mean and major discoveries

- The selection loop chose `continuous_center_squeeze_lift` with objective 0.279297. Selection therefore established a best candidate relative to the tested pool; it did not by itself establish task mastery.
- The stage-flow diagnostic localized the dominant structural bottleneck to `center_and_descend`. This is a measured control-flow finding from the authored program, useful for revision without treating the stage label as a framework-authored diagnosis.

## Limitations and reading guidance

- Metrics are reported under the names used by the historical environment and runner. Similar-looking metrics from different generations of the code are not assumed to be numerically interchangeable.
- A maximum observed in training is not treated as held-out performance. Terminal evaluation values are preferred whenever present.
- Missing summaries, empty metric streams, failed retries, and incomplete arms are preserved as negative evidence rather than silently dropped.
- Candidate names and stage names are LLM-authored run artifacts. Interpretations in this historical log do not alter framework prompts, defaults, diagnostics, or task logic.

## Artifact provenance

| Artifact type | Count |
|---|---|
| .pkl | 29 |
| .json | 10 |
| .md | 6 |
| .txt | 6 |

Primary evidence files:

- [`report.json`](../runs/staged_handoff_val_20260702-191014/report.json)

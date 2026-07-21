# Staged Handoff Val2

- **Run directory:** [`runs/staged_handoff_val2_20260702-194333`](../runs/staged_handoff_val2_20260702-194333)
- **Date:** 2026-07-02 19:43:33 (directory timestamp)
- **Status:** completed or completed-with-caveats
- **Experiment class:** PPO bias/reward/prior experiment
- **Recorded task:** grasp and lift a 5cm cube off the table
- **Training metric rows recovered:** 0

## Abstract

This entry reconstructs `staged_handoff_val2_20260702-194333`, a PPO bias/reward/prior experiment. The account is based only on artifacts that survive in the repository: machine-readable configuration, candidate programs, selection diagnostics, training metrics, evaluations, reports, and logs. The run targeted grasp and lift a 5cm cube off the table. Its purpose and comparisons are inferred from the recorded arm names and configuration where no authored abstract survives; those inferences are identified by this provenance-first framing rather than presented as new experimental facts.

## Experimental structure

No machine-readable configuration file survives at the run root. Structure below is reconstructed from reports, result files, subdirectory names, and logs.
### Selection-loop structure

| Field | Value |
|---|---|
| representation | freeform_staged |
| dof_mode | encourage |
| arbiter | short_ppo |
| budget | 10 |
| iters_used | 10 |
| n_seeds | 3 |

### Arms and sub-runs represented by directories

`llm`, `ppo`

## Results

### Recorded results: `report.json`

| Best-candidate field | Value |
|---|---|
| name | earlier_extract_when_partially_gripped |
| source | refine:soft_power_cup_from_above |
| objective | 0.184766 |
| accounting.n_driven | 23 |
| accounting.n_unused_listed | 0 |
| accounting.wrist_driven | yes |

| Diagnostic | Value |
|---|---|
| reach_rate | 0.168 |
| grasp_rate | 0.168 |
| lift_reached_rate | 0.2656 |
| lift_max | 0.046 |
| saturation_frac | 0 |
| trained_success | 0 |
| instant_success | 0 |
| mean_contacts | 0.1758 |
| mean_closure | 0.6545 |
| obj_xy_drift | 0.5699 |

Stage-flow measurements for the selected candidate:

| # | Stage | Occupancy | Reached | Entered | Handoff | Conversion | Authored success |
|---|---|---|---|---|---|---|---|
| 0 | wide_power_aperture | 0.8531 | 1 | 1 | 0.1719 | 0.1719 | 0.0781 |
| 1 | cup_and_capture | 0.0142 | 0.2422 | 0.1719 | 0.1953 | 0.8065 | 0.0938 |
| 2 | squeeze_equalize | 0.1057 | 0.6445 | 0.5547 | 0.2031 | 0.3152 | 0 |
| 3 | vertical_extract_power_grasp | 0.027 | 0.3086 | 0.2148 | — | — | 0 |

Candidate trajectory:

| iter | source | name | objective | frontier | wrist_driven | n_driven |
|---|---|---|---|---|---|---|
| 1 | explore | soft_center_opposed_pinch | 0.026562 | — | yes | 23 |
| 2 | explore | hard_guarded_handoff | 0.017969 | — | yes | 23 |
| 3 | explore | soft_power_cup_from_above | 0.029688 | — | yes | 23 |
| 4 | refine:soft_power_cup_from_above | earlier_capture_handoff | 0.084375 | — | yes | 23 |
| 5 | refine:soft_center_opposed_pinch | fine_pregrasp_entry_polish | 0.010156 | — | yes | 23 |
| 6 | refine:soft_power_cup_from_above | squeeze_handoff_gate_nudge | 0.06875 | — | yes | 23 |
| 7 | refine:soft_center_opposed_pinch | fine_pregrasp_entry_gate_relax | 0.014844 | — | yes | 23 |
| 8 | refine:soft_power_cup_from_above | squeeze_entry_contact_gate | 0.098828 | — | yes | 23 |
| 9 | refine:soft_center_opposed_pinch | opposed_close_entry_polish | 0.014844 | — | yes | 23 |
| 10 | refine:soft_power_cup_from_above | earlier_extract_when_partially_gripped | 0.184766 | — | yes | 23 |

## What the results mean and major discoveries

- The selection loop chose `earlier_extract_when_partially_gripped` with objective 0.184766. Selection therefore established a best candidate relative to the tested pool; it did not by itself establish task mastery.

## Limitations and reading guidance

- Metrics are reported under the names used by the historical environment and runner. Similar-looking metrics from different generations of the code are not assumed to be numerically interchangeable.
- A maximum observed in training is not treated as held-out performance. Terminal evaluation values are preferred whenever present.
- Missing summaries, empty metric streams, failed retries, and incomplete arms are preserved as negative evidence rather than silently dropped.
- Candidate names and stage names are LLM-authored run artifacts. Interpretations in this historical log do not alter framework prompts, defaults, diagnostics, or task logic.

## Artifact provenance

| Artifact type | Count |
|---|---|
| .pkl | 71 |
| .json | 11 |
| .md | 7 |
| .txt | 7 |

Primary evidence files:

- [`report.json`](../runs/staged_handoff_val2_20260702-194333/report.json)

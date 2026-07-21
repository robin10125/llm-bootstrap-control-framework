# Prior Refine

- **Run directory:** [`runs/prior_refine_20260704`](../runs/prior_refine_20260704)
- **Date:** 2026-07-04 (directory timestamp)
- **Status:** completed or completed-with-caveats
- **Experiment class:** agentic prior generation/selection study
- **Recorded task:** grasp and lift a 5cm cube off the table
- **Training metric rows recovered:** 0

## Abstract

This entry reconstructs `prior_refine_20260704`, an agentic prior generation/selection study. The account is based only on artifacts that survive in the repository: machine-readable configuration, candidate programs, selection diagnostics, training metrics, evaluations, reports, and logs. The run targeted grasp and lift a 5cm cube off the table. Its purpose and comparisons are inferred from the recorded arm names and configuration where no authored abstract survives; those inferences are identified by this provenance-first framing rather than presented as new experimental facts.

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
| wall_hours | 0.522 |

### Arms and sub-runs represented by directories

`llm`

## Results

### Recorded results: `report.json`

| Best-candidate field | Value |
|---|---|
| name | wrap_entry_on_moderate_contact |
| source | refine:pinch_first_then_wrap |
| objective | 0.034765 |
| accounting.n_driven | 23 |
| accounting.n_unused_listed | 0 |
| accounting.wrist_driven | yes |

| Diagnostic | Value |
|---|---|
| success_rate | 0 |
| reach_rate | 0.1562 |
| grasp_rate | 0.0039 |
| grasp_lift_rate | 0 |
| lift_reached_rate | 0.6016 |
| lift_max | 0.0778 |
| base_return | 1.1939 |
| train_return | 1.0559 |
| shaped_return | -0.138 |
| action_abs_mean | 0.2541 |
| saturation_frac | 0 |
| note | behavior at PPO iteration 0: the prior acting through an UNTRAINED policy (network output ~0, so this is essentially the prior alone). Compare with the trained metrics -- a defect present here is a PRIOR problem; one that only appears af… |
| base_reward_weight | 1 |
| instant_success_rate | 0 |
| hard_clip_frac | 0 |

Stage-flow measurements for the selected candidate:

| # | Stage | Occupancy | Reached | Entered | Handoff | Conversion | Authored success |
|---|---|---|---|---|---|---|---|
| 0 | wide_pinching_pregrasp | 0.9782 | 1 | 1 | 0.0859 | 0.0859 | 0.0273 |
| 1 | primary_pinch | 0.0123 | 0.4688 | 0.0859 | 0.1367 | 0.2917 | 0 |
| 2 | wrap_and_lock | 0.0079 | 0.332 | 0.1875 | 0.0078 | 0.0235 | 0.0039 |
| 3 | vertical_takeoff | 0.0016 | 0.0078 | 0.0078 | — | — | 0 |

The recorded structural stall was `wide_pinching_pregrasp` (index 0); terminal stage reached: no.

Candidate trajectory:

| iter | source | name | objective | frontier | wrist_driven | n_driven |
|---|---|---|---|---|---|---|
| 1 | explore | center_open_close_lift | 0.013281 | — | yes | 23 |
| 2 | explore | pinch_first_then_wrap | 0.03164 | — | yes | 23 |
| 3 | explore | low_preload_scoopless_cage | 0.029688 | — | yes | 23 |
| 4 | refine:pinch_first_then_wrap | lower_wrap_entry_on_real_primary_contact | 0.025781 | — | yes | 23 |
| 5 | refine:pinch_first_then_wrap | wrap_entry_on_moderate_contact | 0.034765 | — | yes | 23 |
| 6 | refine:pinch_first_then_wrap | primary_pinch_earlier_entry_on_pregrasp_stall | 0.030469 | — | yes | 23 |
| 7 | refine:pinch_first_then_wrap | vertical_takeoff_entry_nudge | 0.029297 | — | yes | 23 |
| 8 | refine:pinch_first_then_wrap | vertical_takeoff_close_contact_entry | 0.028906 | — | yes | 23 |
| 9 | refine:pinch_first_then_wrap | wrap_gate_yields_on_partial_grip | 0.029297 | — | yes | 23 |
| 10 | refine:pinch_first_then_wrap | wrap_gate_lower_exit_bias | 0.025781 | — | yes | 23 |
| 11 | refine:pinch_first_then_wrap | wrap_and_lock_grip_weight_yield | 0.028515 | — | yes | 23 |
| 12 | refine:pinch_first_then_wrap | wrap_gate_exits_on_grip_progress | 0.028906 | — | yes | 23 |
| 13 | refine:pinch_first_then_wrap | wrap_gate_small_exit_bias_cut | 0.032422 | — | yes | 23 |

## What the results mean and major discoveries

- The selection loop chose `wrap_entry_on_moderate_contact` with objective 0.034765. Selection therefore established a best candidate relative to the tested pool; it did not by itself establish task mastery.
- Recorded success was zero while the associated lift/progress measurement was 0.6016. The candidate generated partial behavior under the run's diagnostics but did not satisfy the full success predicate in that evaluation.
- The stage-flow diagnostic localized the dominant structural bottleneck to `wide_pinching_pregrasp`. This is a measured control-flow finding from the authored program, useful for revision without treating the stage label as a framework-authored diagnosis.

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

- [`report.json`](../runs/prior_refine_20260704/report.json)

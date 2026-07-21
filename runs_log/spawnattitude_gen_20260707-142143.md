# Spawnattitude Gen

- **Run directory:** [`runs/spawnattitude_gen_20260707-142143`](../runs/spawnattitude_gen_20260707-142143)
- **Date:** 2026-07-07 14:21:43 (directory timestamp)
- **Status:** completed or completed-with-caveats
- **Experiment class:** agentic prior generation/selection study
- **Recorded task:** grasp and lift a 5cm cube off the table
- **Training metric rows recovered:** 0

## Abstract

This entry reconstructs `spawnattitude_gen_20260707-142143`, an agentic prior generation/selection study. The account is based only on artifacts that survive in the repository: machine-readable configuration, candidate programs, selection diagnostics, training metrics, evaluations, reports, and logs. The run targeted grasp and lift a 5cm cube off the table. Its purpose and comparisons are inferred from the recorded arm names and configuration where no authored abstract survives; those inferences are identified by this provenance-first framing rather than presented as new experimental facts.

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
| wall_hours | 0.327 |

### Arms and sub-runs represented by directories

`branches`, `llm`

## Results

### Recorded results: `report.json`

| Best-candidate field | Value |
|---|---|
| name | settle_gate_offset_match |
| source | refine:gentle_side_pinch_thumb_middle_index |
| objective | 0.260656 |
| accounting.n_driven | 19 |
| accounting.n_unused_listed | 4 |
| accounting.wrist_driven | yes |

| Diagnostic | Value |
|---|---|
| success_rate | 0 |
| reach_rate | 1 |
| grasp_rate | 0.0391 |
| grasp_lift_rate | 0 |
| lift_reached_rate | 0 |
| lift_max | 0.0037 |
| base_return | 77.688 |
| train_return | 77.5839 |
| shaped_return | -0.104 |
| action_abs_mean | 0.0043 |
| saturation_frac | 0 |
| note | behavior at PPO iteration 0: the prior acting through an UNTRAINED policy (network output ~0, so this is essentially the prior alone). Compare with the trained metrics -- a defect present here is a PRIOR problem; one that only appears af… |
| base_reward_weight | 1 |
| instant_success_rate | 0 |
| hard_clip_frac | 0 |
| failure_rate | 0 |
| calm_frac | 1 |
| reach_rate_calm | 1 |

Stage-flow measurements for the selected candidate:

| # | Stage | Occupancy | Reached | Entered | Handoff | Conversion | Authored success |
|---|---|---|---|---|---|---|---|
| 0 | preshape_free_space | 0.7809 | 1 | 1 | 1 | 1 | 1 |
| 1 | transport_to_side_offset | 0.2191 | 1 | 1 | 0 | 0 | 0.0039 |
| 2 | settle_before_touch | 0 | 0 | 0 | 0 | 0 | 0 |
| 3 | establish_gentle_opposing_contact | 0 | 0 | 0 | 0 | 0 | 0 |
| 4 | load_stable_grip | 0 | 0 | 0 | 0 | 0 | 0 |
| 5 | lift_off_table | 0 | 0 | 0 | 0 | 0 | 0 |
| 6 | stabilize_lifted_hold | 0 | 0 | 0 | — | — | 0 |

The recorded structural stall was `transport_to_side_offset` (index 1); terminal stage reached: no.

Candidate trajectory:

| iter | source | name | objective | frontier | wrist_driven | n_driven |
|---|---|---|---|---|---|---|
| 1 | explore | gentle_side_pinch_thumb_middle_index | 0.01836 | — | yes | 19 |
| 2 | refine:gentle_side_pinch_thumb_middle_index | handoff_to_transport_when_preshape_margin_positive | 0.255964 | — | yes | 19 |
| 3 | refine:gentle_side_pinch_thumb_middle_index | handoff_to_settle_with_vertical_margin | 0.258394 | — | yes | 19 |
| 4 | refine:gentle_side_pinch_thumb_middle_index | settle_gate_entry_nudge | 0.246627 | — | yes | 19 |
| 5 | refine:gentle_side_pinch_thumb_middle_index | settle_gate_offset_match | 0.260656 | — | yes | 19 |

## What the results mean and major discoveries

- The selection loop chose `settle_gate_offset_match` with objective 0.260656. Selection therefore established a best candidate relative to the tested pool; it did not by itself establish task mastery.
- Recorded success was zero while the associated lift/progress measurement was 0. The candidate generated partial behavior under the run's diagnostics but did not satisfy the full success predicate in that evaluation.
- The stage-flow diagnostic localized the dominant structural bottleneck to `transport_to_side_offset`. This is a measured control-flow finding from the authored program, useful for revision without treating the stage label as a framework-authored diagnosis.

## Limitations and reading guidance

- Metrics are reported under the names used by the historical environment and runner. Similar-looking metrics from different generations of the code are not assumed to be numerically interchangeable.
- A maximum observed in training is not treated as held-out performance. Terminal evaluation values are preferred whenever present.
- Missing summaries, empty metric streams, failed retries, and incomplete arms are preserved as negative evidence rather than silently dropped.
- Candidate names and stage names are LLM-authored run artifacts. Interpretations in this historical log do not alter framework prompts, defaults, diagnostics, or task logic.

## Artifact provenance

| Artifact type | Count |
|---|---|
| .json | 13 |
| .md | 8 |
| .txt | 8 |
| .mp4 | 1 |
| .pkl | 1 |

Primary evidence files:

- [`report.json`](../runs/spawnattitude_gen_20260707-142143/report.json)

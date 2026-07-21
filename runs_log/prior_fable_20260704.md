# Prior Fable

- **Run directory:** [`runs/prior_fable_20260704`](../runs/prior_fable_20260704)
- **Date:** 2026-07-04 (directory timestamp)
- **Status:** completed or completed-with-caveats
- **Experiment class:** agentic prior generation/selection study
- **Recorded task:** grasp and lift a 5cm cube off the table
- **Training metric rows recovered:** 0

## Abstract

This entry reconstructs `prior_fable_20260704`, an agentic prior generation/selection study. The account is based only on artifacts that survive in the repository: machine-readable configuration, candidate programs, selection diagnostics, training metrics, evaluations, reports, and logs. The run targeted grasp and lift a 5cm cube off the table. Its purpose and comparisons are inferred from the recorded arm names and configuration where no authored abstract survives; those inferences are identified by this provenance-first framing rather than presented as new experimental facts.

## Experimental structure

No machine-readable configuration file survives at the run root. Structure below is reconstructed from reports, result files, subdirectory names, and logs.
### Selection-loop structure

| Field | Value |
|---|---|
| representation | freeform_staged |
| dof_mode | encourage |
| arbiter | prior_only |
| budget | 1 |
| iters_used | 1 |
| n_seeds | 1 |
| wall_hours | 0.091 |

### Arms and sub-runs represented by directories

`llm`

## Results

### Recorded results: `report.json`

| Best-candidate field | Value |
|---|---|
| name | top_grasp_align_then_cage |
| source | explore |
| objective | 0.042383 |
| accounting.n_driven | 21 |
| accounting.n_unused_listed | 2 |
| accounting.wrist_driven | no |

| Diagnostic | Value |
|---|---|
| success_rate | 0 |
| reach_rate | 0.2119 |
| grasp_rate | 0 |
| grasp_lift_rate | 0 |
| lift_reached_rate | 0.5752 |
| lift_max | 0.0783 |
| base_return | 1.2652 |
| train_return | 1.0881 |
| shaped_return | -0.1771 |
| action_abs_mean | 0.1135 |
| saturation_frac | 0 |
| note | behavior at PPO iteration 0: the prior acting through an UNTRAINED policy (network output ~0, so this is essentially the prior alone). Compare with the trained metrics -- a defect present here is a PRIOR problem; one that only appears af… |
| base_reward_weight | 1 |
| instant_success_rate | 0 |
| hard_clip_frac | 0 |
| objective_batch_std | 0.0054 |

Stage-flow measurements for the selected candidate:

| # | Stage | Occupancy | Reached | Entered | Handoff | Conversion | Authored success |
|---|---|---|---|---|---|---|---|
| 0 | approach_align | 0.9958 | 1 | 1 | 0.1084 | 0.1084 | 0.1797 |
| 1 | grasp_cage | 0.0042 | 0.2412 | 0.1084 | 0 | 0 | 0 |
| 2 | lift_hold | 0 | 0 | 0 | — | — | 0 |

The recorded structural stall was `grasp_cage` (index 1); terminal stage reached: no.

Candidate trajectory:

| iter | source | name | objective | frontier | wrist_driven | n_driven |
|---|---|---|---|---|---|---|
| 1 | explore | top_grasp_align_then_cage | 0.042383 | — | no | 21 |

## What the results mean and major discoveries

- The selection loop chose `top_grasp_align_then_cage` with objective 0.042383. Selection therefore established a best candidate relative to the tested pool; it did not by itself establish task mastery.
- Recorded success was zero while the associated lift/progress measurement was 0.5752. The candidate generated partial behavior under the run's diagnostics but did not satisfy the full success predicate in that evaluation.
- The stage-flow diagnostic localized the dominant structural bottleneck to `grasp_cage`. This is a measured control-flow finding from the authored program, useful for revision without treating the stage label as a framework-authored diagnosis.

## Limitations and reading guidance

- Metrics are reported under the names used by the historical environment and runner. Similar-looking metrics from different generations of the code are not assumed to be numerically interchangeable.
- A maximum observed in training is not treated as held-out performance. Terminal evaluation values are preferred whenever present.
- Missing summaries, empty metric streams, failed retries, and incomplete arms are preserved as negative evidence rather than silently dropped.
- Candidate names and stage names are LLM-authored run artifacts. Interpretations in this historical log do not alter framework prompts, defaults, diagnostics, or task logic.

## Artifact provenance

| Artifact type | Count |
|---|---|
| .json | 9 |
| .md | 5 |
| .txt | 5 |
| .pkl | 1 |

Primary evidence files:

- [`report.json`](../runs/prior_fable_20260704/report.json)

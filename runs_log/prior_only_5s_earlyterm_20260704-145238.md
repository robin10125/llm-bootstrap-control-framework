# Prior Only 5s Earlyterm

- **Run directory:** [`runs/prior_only_5s_earlyterm_20260704-145238`](../runs/prior_only_5s_earlyterm_20260704-145238)
- **Date:** 2026-07-04 14:52:38 (directory timestamp)
- **Status:** completed or completed-with-caveats
- **Experiment class:** agentic prior generation/selection study
- **Recorded task:** grasp and lift a 5cm cube off the table
- **Training metric rows recovered:** 0

## Abstract

This entry reconstructs `prior_only_5s_earlyterm_20260704-145238`, an agentic prior generation/selection study. The account is based only on artifacts that survive in the repository: machine-readable configuration, candidate programs, selection diagnostics, training metrics, evaluations, reports, and logs. The run targeted grasp and lift a 5cm cube off the table. Its purpose and comparisons are inferred from the recorded arm names and configuration where no authored abstract survives; those inferences are identified by this provenance-first framing rather than presented as new experimental facts.

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
| wall_hours | 0.081 |

### Arms and sub-runs represented by directories

`llm`, `prior_rollout_analysis`

## Results

### Recorded results: `prior_rollout_analysis/summary.json`

| Arm/interface | lift_max |
|---|---|
| rollout | — |
| performance | 0.164267 |
| stage_report | — |
| files | — |

### Recorded results: `report.json`

| Best-candidate field | Value |
|---|---|
| name | center_descend_envelop_hold_lift |
| source | explore |
| objective | 0.151172 |
| accounting.n_driven | 23 |
| accounting.n_unused_listed | 0 |
| accounting.wrist_driven | yes |

| Diagnostic | Value |
|---|---|
| success_rate | 0 |
| reach_rate | 0.1406 |
| grasp_rate | 0.1367 |
| grasp_lift_rate | 0.0625 |
| lift_reached_rate | 0.4922 |
| lift_max | 0.0654 |
| base_return | 1.5748 |
| train_return | 1.4004 |
| shaped_return | -0.1745 |
| action_abs_mean | 0.1272 |
| saturation_frac | 0 |
| note | behavior at PPO iteration 0: the prior acting through an UNTRAINED policy (network output ~0, so this is essentially the prior alone). Compare with the trained metrics -- a defect present here is a PRIOR problem; one that only appears af… |
| base_reward_weight | 1 |
| instant_success_rate | 0 |
| hard_clip_frac | 0 |

Stage-flow measurements for the selected candidate:

| # | Stage | Occupancy | Reached | Entered | Handoff | Conversion | Authored success |
|---|---|---|---|---|---|---|---|
| 0 | open_center | 0.3402 | 1 | 1 | 0.3633 | 0.3633 | 0.3438 |
| 1 | settle_descend | 0.0166 | 0.8945 | 0.4766 | 0.8945 | 1 | 0.8945 |
| 2 | envelop_close | 0.6432 | 1 | 1 | 0 | 0 | 0 |
| 3 | lift_carry | 0 | 0 | 0 | — | — | 0 |

The recorded structural stall was `envelop_close` (index 2); terminal stage reached: no.

Candidate trajectory:

| iter | source | name | objective | frontier | wrist_driven | n_driven |
|---|---|---|---|---|---|---|
| 1 | explore | center_descend_envelop_hold_lift | 0.151172 | — | yes | 23 |

## What the results mean and major discoveries

- The arm comparison is most informative when arms shared the same environment, seed, and budget. Where the configuration records only one seed, apparent gaps remain vulnerable to optimization variance.
- The selection loop chose `center_descend_envelop_hold_lift` with objective 0.151172. Selection therefore established a best candidate relative to the tested pool; it did not by itself establish task mastery.
- Recorded success was zero while the associated lift/progress measurement was 0.4922. The candidate generated partial behavior under the run's diagnostics but did not satisfy the full success predicate in that evaluation.
- The stage-flow diagnostic localized the dominant structural bottleneck to `envelop_close`. This is a measured control-flow finding from the authored program, useful for revision without treating the stage label as a framework-authored diagnosis.

## Limitations and reading guidance

- Metrics are reported under the names used by the historical environment and runner. Similar-looking metrics from different generations of the code are not assumed to be numerically interchangeable.
- A maximum observed in training is not treated as held-out performance. Terminal evaluation values are preferred whenever present.
- Missing summaries, empty metric streams, failed retries, and incomplete arms are preserved as negative evidence rather than silently dropped.
- Candidate names and stage names are LLM-authored run artifacts. Interpretations in this historical log do not alter framework prompts, defaults, diagnostics, or task logic.

## Artifact provenance

| Artifact type | Count |
|---|---|
| .json | 10 |
| .md | 5 |
| .txt | 5 |
| .csv | 3 |
| .html | 1 |
| .pkl | 1 |
| .py | 1 |

Primary evidence files:

- [`prior_rollout_analysis/summary.json`](../runs/prior_only_5s_earlyterm_20260704-145238/prior_rollout_analysis/summary.json)
- [`report.json`](../runs/prior_only_5s_earlyterm_20260704-145238/report.json)

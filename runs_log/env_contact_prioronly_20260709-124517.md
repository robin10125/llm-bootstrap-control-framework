# Env Contact Prioronly

- **Run directory:** [`runs/env_contact_prioronly_20260709-124517`](../runs/env_contact_prioronly_20260709-124517)
- **Date:** 2026-07-09 12:45:17 (directory timestamp)
- **Status:** completed or completed-with-caveats
- **Experiment class:** agentic prior generation/selection study
- **Recorded task:** grasp and lift a 5cm cube off the table
- **Training metric rows recovered:** 0

## Abstract

This entry reconstructs `env_contact_prioronly_20260709-124517`, an agentic prior generation/selection study. The account is based only on artifacts that survive in the repository: machine-readable configuration, candidate programs, selection diagnostics, training metrics, evaluations, reports, and logs. The run targeted grasp and lift a 5cm cube off the table. Its purpose and comparisons are inferred from the recorded arm names and configuration where no authored abstract survives; those inferences are identified by this provenance-first framing rather than presented as new experimental facts.

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
| wall_hours | 0.578 |

### Arms and sub-runs represented by directories

`llm`, `videos`

## Results

### Recorded results: `report.json`

| Best-candidate field | Value |
|---|---|
| name | guarded_descend_align_hysteresis_faster_z |
| source | refine:reachable_pitch_centered_pinch_lift |
| objective | 0.377107 |
| accounting.n_driven | 23 |
| accounting.n_unused_listed | 0 |
| accounting.wrist_driven | yes |

| Diagnostic | Value |
|---|---|
| graded_objective | 0.3771 |
| success_rate | 0 |
| reach_rate | 1 |
| grasp_rate | 0.2656 |
| grasp_lift_rate | 0 |
| lift_reached_rate | 0 |
| lift_max | 0.0014 |
| base_return | 63.6037 |
| train_return | 63.6515 |
| action_abs_mean | 0.0091 |
| note | behavior at PPO iteration 0: the prior acting through an UNTRAINED policy (network output ~0, so this is essentially the prior alone). Compare with the trained metrics -- a defect present here is a PRIOR problem; one that only appears af… |
| task_fitness | 0.5469 |
| stage_return | 0.0478 |
| prior_scale_mean | 0.9993 |

Stage-flow measurements for the selected candidate:

| # | Stage | Occupancy | Reached | Entered | Handoff | Conversion | Authored success |
|---|---|---|---|---|---|---|---|
| 0 | wrist_extend | 0.8447 | 1 | 1 | 1 | 1 | 1 |
| 1 | free_air_preshape | 0.0581 | 1 | 1 | 0.8359 | 0.8359 | 1 |
| 2 | lateral_align | 0.0601 | 0.8398 | 0.8359 | 0.8359 | 0.9953 | 0.8359 |
| 3 | guarded_descend | 0.0371 | 1 | 1 | 0 | 0 | 0 |
| 4 | balanced_light_contact | 0 | 0 | 0 | 0 | 0 | 0 |
| 5 | secure_grasp | 0 | 0 | 0 | 0 | 0 | 0 |
| 6 | load_test | 0 | 0 | 0 | 0 | 0 | 0 |
| 7 | lift_clear | 0 | 0 | 0 | 0 | 0 | 0 |
| 8 | hold_lifted | 0 | 0 | 0 | — | — | 0 |

The recorded structural stall was `guarded_descend` (index 3); terminal stage reached: no.

Candidate trajectory:

| iter | source | name | objective | frontier | wrist_driven | n_driven |
|---|---|---|---|---|---|---|
| 1 | explore | reachable_pitch_centered_pinch_lift | 0.315278 | 3 | yes | 23 |
| 2 | refine:reachable_pitch_centered_pinch_lift | guarded_descend_alignment_hysteresis | 0.33318 | 3 | yes | 23 |
| 3 | refine:reachable_pitch_centered_pinch_lift | hysteretic_descend_handoff | 0.331223 | 3 | yes | 23 |
| 4 | refine:reachable_pitch_centered_pinch_lift | guarded_descend_handoff_hold | 0.371087 | 3 | yes | 23 |
| 5 | refine:reachable_pitch_centered_pinch_lift | guarded_descend_align_hysteresis_faster_z | 0.377107 | 3 | yes | 23 |

## What the results mean and major discoveries

- The selection loop chose `guarded_descend_align_hysteresis_faster_z` with objective 0.377107. Selection therefore established a best candidate relative to the tested pool; it did not by itself establish task mastery.
- Recorded success was zero while the associated lift/progress measurement was 0. The candidate generated partial behavior under the run's diagnostics but did not satisfy the full success predicate in that evaluation.
- The stage-flow diagnostic localized the dominant structural bottleneck to `guarded_descend`. This is a measured control-flow finding from the authored program, useful for revision without treating the stage label as a framework-authored diagnosis.

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
| .html | 1 |
| .mp4 | 1 |
| .pkl | 1 |
| .png | 1 |

Primary evidence files:

- [`report.json`](../runs/env_contact_prioronly_20260709-124517/report.json)

# Shadowhand Recursive Codex Replay

- **Run directory:** [`famework_testing/runs/shadowhand_recursive_codex_replay`](../famework_testing/runs/shadowhand_recursive_codex_replay)
- **Date:** 2026-06-15 23:08:26 (summary.json manifest started_at)
- **Status:** completed or completed-with-caveats
- **Experiment class:** early interface/controller comparison
- **Recorded task:** lift
- **Training metric rows recovered:** 0

## Abstract

This entry reconstructs `shadowhand_recursive_codex_replay`, an early interface/controller comparison. The account is based only on artifacts that survive in the repository: machine-readable configuration, candidate programs, selection diagnostics, training metrics, evaluations, reports, and logs. The run targeted lift. Its purpose and comparisons are inferred from the recorded arm names and configuration where no authored abstract survives; those inferences are identified by this provenance-first framing rather than presented as new experimental facts.

## Experimental structure

No machine-readable configuration file survives at the run root. Structure below is reconstructed from reports, result files, subdirectory names, and logs.
### Arms and sub-runs represented by directories

`lift_s0_recursive_units_a0`, `lift_s0_recursive_units_a0_repair1`

## Results

### Recorded results: `lift_s0_recursive_units_a0/compiled_summary.json`

| Recorded field | Value |
|---|---|
| interface | recursive_units |
| action_shape | 100, 23 |
| metadata.target_steps | 100 |
| metadata.uses_reactive_executor | yes |
| metadata.recursive_units | yes |
| action_min | -0.788 |
| action_max | 1 |

### Recorded results: `lift_s0_recursive_units_a0/result.json`

| Recorded field | Value |
|---|---|
| task.name | lift |
| task.goal | raise the object above the table and keep it controlled |
| task.seed | 0 |
| task.episode_seconds | 2.5 |
| task.object_start | -0.0394, -0.0383, 0.025 |
| task.target_xy | — |
| task.metadata.height_threshold | 0.074966 |
| interface | recursive_units |
| seed | 0 |
| attempt | 0 |
| llm.source | codex |
| llm.ok | yes |
| llm.error | — |
| validation.ok | yes |
| validation.errors |  |
| validation.warnings |  |
| result.interface | recursive_units |
| result.task | lift |
| result.seed | 0 |
| result.success | no |
| result.score | 0.0143 |
| result.total_return | -15.2093 |
| result.final_object_x | -0.03765 |
| result.final_object_y | -0.039266 |
| result.final_object_z | 0.024992 |
| result.max_object_z | 0.0264 |
| result.errors |  |
| result.task_metrics | {"height_margin": -0.0486, "height_threshold": 0.075, "lift_delta": 0.0014, "max_object_z": 0.0264, "metric": "height", "score_note": "score is shaped progress, not absolute object height"} |
| result_metadata.target_steps | 100 |
| result_metadata.uses_reactive_executor | yes |
| result_metadata.recursive_units | yes |
| result_metadata.reactive_execution.online | yes |
| result_metadata.reactive_execution.steps_used | 100 |
| result_metadata.task_metrics.metric | height |
| result_metadata.task_metrics.height_threshold | 0.075 |
| result_metadata.task_metrics.max_object_z | 0.0264 |
| result_metadata.task_metrics.height_margin | -0.0486 |
| result_metadata.task_metrics.lift_delta | 0.0014 |
| result_metadata.task_metrics.score_note | score is shaped progress, not absolute object height |

### Recorded results: `lift_s0_recursive_units_a0_repair1/compiled_summary.json`

| Recorded field | Value |
|---|---|
| interface | recursive_units |
| action_shape | 100, 23 |
| metadata.target_steps | 100 |
| metadata.uses_reactive_executor | yes |
| metadata.recursive_units | yes |
| action_min | -1 |
| action_max | 1 |

### Recorded results: `lift_s0_recursive_units_a0_repair1/result.json`

| Recorded field | Value |
|---|---|
| task.name | lift |
| task.goal | raise the object above the table and keep it controlled |
| task.seed | 0 |
| task.episode_seconds | 2.5 |
| task.object_start | -0.0394, -0.0383, 0.025 |
| task.target_xy | — |
| task.metadata.height_threshold | 0.074966 |
| interface | recursive_units |
| seed | 0 |
| repair_attempt | 1 |
| parent_result.interface | recursive_units |
| parent_result.task | lift |
| parent_result.seed | 0 |
| parent_result.success | no |
| parent_result.score | 0.0143 |
| parent_result.total_return | -15.2093 |
| parent_result.final_object_x | -0.03765 |
| parent_result.final_object_y | -0.039266 |
| parent_result.final_object_z | 0.024992 |
| parent_result.max_object_z | 0.0264 |
| parent_result.errors |  |
| parent_result.task_metrics | {"height_margin": -0.0486, "height_threshold": 0.075, "lift_delta": 0.0014, "max_object_z": 0.0264, "metric": "height", "score_note": "score is shaped progress, not absolute object height"} |
| parent_result_metadata.target_steps | 100 |
| parent_result_metadata.uses_reactive_executor | yes |
| parent_result_metadata.recursive_units | yes |
| parent_result_metadata.reactive_execution.online | yes |
| parent_result_metadata.reactive_execution.steps_used | 100 |
| parent_result_metadata.task_metrics.metric | height |
| parent_result_metadata.task_metrics.height_threshold | 0.075 |
| parent_result_metadata.task_metrics.max_object_z | 0.0264 |
| parent_result_metadata.task_metrics.height_margin | -0.0486 |
| parent_result_metadata.task_metrics.lift_delta | 0.0014 |
| parent_result_metadata.task_metrics.score_note | score is shaped progress, not absolute object height |
| llm.source | codex |
| llm.ok | yes |
| llm.error | — |
| validation.ok | yes |
| validation.errors |  |
| validation.warnings |  |
| result.interface | recursive_units |
| result.task | lift |
| result.seed | 0 |
| result.success | no |
| result.score | 0.0183 |
| result.total_return | -14.6253 |
| result.final_object_x | -0.033346 |
| result.final_object_y | -0.045152 |
| result.final_object_z | 0.024993 |
| result.max_object_z | 0.0268 |
| result.errors |  |
| result.task_metrics | {"height_margin": -0.0482, "height_threshold": 0.075, "lift_delta": 0.0018, "max_object_z": 0.0268, "metric": "height", "score_note": "score is shaped progress, not absolute object height"} |
| result_metadata.target_steps | 100 |
| result_metadata.uses_reactive_executor | yes |
| result_metadata.recursive_units | yes |
| result_metadata.reactive_execution.online | yes |
| result_metadata.reactive_execution.steps_used | 100 |
| result_metadata.task_metrics.metric | height |
| result_metadata.task_metrics.height_threshold | 0.075 |
| result_metadata.task_metrics.max_object_z | 0.0268 |
| result_metadata.task_metrics.height_margin | -0.0482 |
| result_metadata.task_metrics.lift_delta | 0.0018 |
| result_metadata.task_metrics.score_note | score is shaped progress, not absolute object height |

### Recorded results: `summary.json`

| Arm/interface | success_rate | successes | mean_score | n |
|---|---|---|---|---|
| recursive_units | 0 | 0 | 0.0163 | 2 |

| Tag | Interface | Task | Seed | Success | Score | Return | Max object z | Errors |
|---|---|---|---|---|---|---|---|---|
| lift_s0_recursive_units_a0 | recursive_units | lift | 0 | no | 0.0143 | -15.2093 | 0.0264 |  |
| lift_s0_recursive_units_a0_repair1 | recursive_units | lift | 0 | no | 0.0183 | -14.6253 | 0.0268 |  |

## What the results mean and major discoveries

- Among the recorded arms, `recursive_units` had the highest held-out success (success_rate = 0). This is an ordering within this run, not a general ranking across experiments with different metric definitions or budgets.
- Among the recorded arms, `recursive_units` had the highest graded/task objective (mean_score = 0.0163). This is an ordering within this run, not a general ranking across experiments with different metric definitions or budgets.

## Limitations and reading guidance

- Metrics are reported under the names used by the historical environment and runner. Similar-looking metrics from different generations of the code are not assumed to be numerically interchangeable.
- A maximum observed in training is not treated as held-out performance. Terminal evaluation values are preferred whenever present.
- Missing summaries, empty metric streams, failed retries, and incomplete arms are preserved as negative evidence rather than silently dropped.
- Candidate names and stage names are LLM-authored run artifacts. Interpretations in this historical log do not alter framework prompts, defaults, diagnostics, or task logic.

## Artifact provenance

| Artifact type | Count |
|---|---|
| .json | 18 |
| .md | 12 |
| .txt | 12 |
| .npy | 2 |
| .csv | 1 |
| .mp4 | 1 |

Primary evidence files:

- [`report.md`](../famework_testing/runs/shadowhand_recursive_codex_replay/report.md)
- [`summary.json`](../famework_testing/runs/shadowhand_recursive_codex_replay/summary.json)

## Appendix: original authored run report

The following contemporaneous report is retained (with headings demoted) so that details and caveats are not lost in normalization.

### Source: [`report.md`](../famework_testing/runs/shadowhand_recursive_codex_replay/report.md)

## Interface Comparison

### Summary

- `recursive_units`: success_rate=0.0, mean_score=0.0163, n=2

### Runs

- `recursive_units` task=`lift` seed=0 success=False score=0.0143 errors=
- `recursive_units` task=`lift` seed=0 success=False score=0.0183 errors=

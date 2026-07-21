# Real Shadow Codex Lift S0

- **Run directory:** [`famework_testing/runs/real_shadow_codex_lift_s0`](../famework_testing/runs/real_shadow_codex_lift_s0)
- **Date:** 2026-06-15 13:26:15 (summary.json manifest started_at)
- **Status:** completed or completed-with-caveats
- **Experiment class:** early interface/controller comparison
- **Recorded task:** lift
- **Training metric rows recovered:** 0

## Abstract

This entry reconstructs `real_shadow_codex_lift_s0`, an early interface/controller comparison. The account is based only on artifacts that survive in the repository: machine-readable configuration, candidate programs, selection diagnostics, training metrics, evaluations, reports, and logs. The run targeted lift. Its purpose and comparisons are inferred from the recorded arm names and configuration where no authored abstract survives; those inferences are identified by this provenance-first framing rather than presented as new experimental facts.

## Experimental structure

No machine-readable configuration file survives at the run root. Structure below is reconstructed from reports, result files, subdirectory names, and logs.
### Arms and sub-runs represented by directories

`lift_s0_hybrid_a0`, `lift_s0_script_dsl_a0`, `lift_s0_waypoint_a0`

## Results

### Recorded results: `lift_s0_hybrid_a0/result.json`

| Recorded field | Value |
|---|---|
| task.name | lift |
| task.goal | raise the object above the table and keep it controlled |
| task.seed | 0 |
| task.episode_seconds | 2.5 |
| task.object_start | -0.0394, -0.0383, 0.025 |
| task.target_xy | — |
| task.metadata.height_threshold | 0.074966 |
| interface | hybrid |
| seed | 0 |
| attempt | 0 |
| llm.source | codex |
| llm.ok | yes |
| llm.error | — |
| validation.ok | no |
| validation.errors | block 1 has unknown op 'parallel', block 3 has unknown op 'parallel', block 4 has unknown op 'parallel' |
| validation.warnings |  |
| result.interface | hybrid |
| result.task | lift |
| result.seed | 0 |
| result.success | no |
| result.score | 0 |
| result.total_return | 0 |
| result.final_object_x | -0.039417 |
| result.final_object_y | -0.038329 |
| result.final_object_z | 0.024966 |
| result.max_object_z | 0.024966 |
| result.errors | block 1 has unknown op 'parallel'; block 3 has unknown op 'parallel'; block 4 has unknown op 'parallel' |

### Recorded results: `lift_s0_script_dsl_a0/compiled_summary.json`

| Recorded field | Value |
|---|---|
| interface | script_dsl |
| action_shape | 100, 23 |
| metadata.target_steps | 100 |
| action_min | -1 |
| action_max | 1 |

### Recorded results: `lift_s0_script_dsl_a0/result.json`

| Recorded field | Value |
|---|---|
| task.name | lift |
| task.goal | raise the object above the table and keep it controlled |
| task.seed | 0 |
| task.episode_seconds | 2.5 |
| task.object_start | -0.0394, -0.0383, 0.025 |
| task.target_xy | — |
| task.metadata.height_threshold | 0.074966 |
| interface | script_dsl |
| seed | 0 |
| attempt | 0 |
| llm.source | codex |
| llm.ok | yes |
| llm.error | — |
| validation.ok | yes |
| validation.errors |  |
| validation.warnings |  |
| result.interface | script_dsl |
| result.task | lift |
| result.seed | 0 |
| result.success | no |
| result.score | 0.0003 |
| result.total_return | -20.9307 |
| result.final_object_x | -0.039418 |
| result.final_object_y | -0.03833 |
| result.final_object_z | 0.024993 |
| result.max_object_z | 0.025 |
| result.errors |  |

### Recorded results: `lift_s0_waypoint_a0/compiled_summary.json`

| Recorded field | Value |
|---|---|
| interface | waypoint |
| action_shape | 100, 23 |
| metadata.source | bootstrapping.WaypointCompiler |
| metadata.task | lift |
| action_min | -0.08 |
| action_max | 1 |

### Recorded results: `lift_s0_waypoint_a0/result.json`

| Recorded field | Value |
|---|---|
| task.name | lift |
| task.goal | raise the object above the table and keep it controlled |
| task.seed | 0 |
| task.episode_seconds | 2.5 |
| task.object_start | -0.0394, -0.0383, 0.025 |
| task.target_xy | — |
| task.metadata.height_threshold | 0.074966 |
| interface | waypoint |
| seed | 0 |
| attempt | 0 |
| llm.source | codex |
| llm.ok | yes |
| llm.error | — |
| validation.ok | yes |
| validation.errors |  |
| validation.warnings |  |
| result.interface | waypoint |
| result.task | lift |
| result.seed | 0 |
| result.success | no |
| result.score | 0.0003 |
| result.total_return | -23.7696 |
| result.final_object_x | -0.039417 |
| result.final_object_y | -0.038329 |
| result.final_object_z | 0.024993 |
| result.max_object_z | 0.025 |
| result.errors |  |

### Recorded results: `summary.json`

| Arm/interface | success_rate | successes | mean_score | n |
|---|---|---|---|---|
| waypoint | 0 | 0 | 0.0003 | 1 |
| script_dsl | 0 | 0 | 0.0003 | 1 |
| hybrid | 0 | 0 | 0 | 1 |

| Tag | Interface | Task | Seed | Success | Score | Return | Max object z | Errors |
|---|---|---|---|---|---|---|---|---|
| lift_s0_waypoint_a0 | waypoint | lift | 0 | no | 0.0003 | -23.7696 | 0.025 |  |
| lift_s0_script_dsl_a0 | script_dsl | lift | 0 | no | 0.0003 | -20.9307 | 0.025 |  |
| lift_s0_hybrid_a0 | hybrid | lift | 0 | no | 0 | 0 | 0.024966 | block 1 has unknown op 'parallel'; block 3 has unknown op 'parallel'; block 4 has unknown op 'parallel' |

## What the results mean and major discoveries

- Among the recorded arms, `waypoint` had the highest held-out success (success_rate = 0). This is an ordering within this run, not a general ranking across experiments with different metric definitions or budgets.
- Among the recorded arms, `waypoint` had the highest graded/task objective (mean_score = 0.0003). This is an ordering within this run, not a general ranking across experiments with different metric definitions or budgets.
- The arm comparison is most informative when arms shared the same environment, seed, and budget. Where the configuration records only one seed, apparent gaps remain vulnerable to optimization variance.

## Limitations and reading guidance

- Metrics are reported under the names used by the historical environment and runner. Similar-looking metrics from different generations of the code are not assumed to be numerically interchangeable.
- A maximum observed in training is not treated as held-out performance. Terminal evaluation values are preferred whenever present.
- Missing summaries, empty metric streams, failed retries, and incomplete arms are preserved as negative evidence rather than silently dropped.
- Candidate names and stage names are LLM-authored run artifacts. Interpretations in this historical log do not alter framework prompts, defaults, diagnostics, or task logic.

## Artifact provenance

| Artifact type | Count |
|---|---|
| .json | 12 |
| .md | 7 |
| .txt | 6 |
| .csv | 1 |

Primary evidence files:

- [`report.md`](../famework_testing/runs/real_shadow_codex_lift_s0/report.md)
- [`summary.json`](../famework_testing/runs/real_shadow_codex_lift_s0/summary.json)

## Appendix: original authored run report

The following contemporaneous report is retained (with headings demoted) so that details and caveats are not lost in normalization.

### Source: [`report.md`](../famework_testing/runs/real_shadow_codex_lift_s0/report.md)

## Interface Comparison

### Summary

- `waypoint`: success_rate=0.0, mean_score=0.0003, n=1
- `script_dsl`: success_rate=0.0, mean_score=0.0003, n=1
- `hybrid`: success_rate=0.0, mean_score=0.0, n=1

### Runs

- `waypoint` task=`lift` seed=0 success=False score=0.0003 errors=
- `script_dsl` task=`lift` seed=0 success=False score=0.0003 errors=
- `hybrid` task=`lift` seed=0 success=False score=0.0 errors=block 1 has unknown op 'parallel'; block 3 has unknown op 'parallel'; block 4 has unknown op 'parallel'

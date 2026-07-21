# DSL Vs Freeform Debias

- **Run directory:** [`runs/dsl_vs_freeform_debias_20260630-120822`](../runs/dsl_vs_freeform_debias_20260630-120822)
- **Date:** 2026-06-30 12:08:22 (directory timestamp)
- **Status:** artifact-only or generation-only; no terminal evaluation recorded
- **Experiment class:** prior-representation study
- **Recorded task:** not explicitly recorded at the run root
- **Training metric rows recovered:** 0

## Abstract

This entry reconstructs `dsl_vs_freeform_debias_20260630-120822`, a prior-representation study. The account is based only on artifacts that survive in the repository: machine-readable configuration, candidate programs, selection diagnostics, training metrics, evaluations, reports, and logs. The run targeted not explicitly recorded at the run root. Its purpose and comparisons are inferred from the recorded arm names and configuration where no authored abstract survives; those inferences are identified by this provenance-first framing rather than presented as new experimental facts.

## Experimental structure

No machine-readable configuration file survives at the run root. Structure below is reconstructed from reports, result files, subdirectory names, and logs.
### Arms and sub-runs represented by directories

`llm`

## Results

### Recorded results: `dsl_scored.json`

| name | score.contact_gated_success | score.contact_conditioned_lift | score.contact_engagement | score.contacts_mean | score.fling_fraction | score.palm_obj_dist_min | score.saturation_frac | score.action_abs_mean | score.objective_score | coverage.n_addressed | coverage.n_total | coverage.wrist_used | program.mode | program.representation |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| opposed_thumb_index_precision | 0 | 0 | 0 | 0 | 0 | 0.062256 | 0 | 0 | -0.249026 | 23 | 23 | yes | stacked | dsl |
| tripod_pad_grasp | 0 | 0 | 0 | 0 | 0 | 0.027471 | 0 | 0 | -0.109883 | 23 | 23 | yes | stacked | dsl |
| four_finger_cage_with_thumb_stop | 0 | 0 | 0 | 0 | 0 | 0.063849 | 0 | 0 | -0.255394 | 23 | 23 | yes | stacked | dsl |

### Recorded results: `freeform_scored.json`

| name | score.contact_gated_success | score.contact_conditioned_lift | score.contact_engagement | score.contacts_mean | score.fling_fraction | score.palm_obj_dist_min | score.saturation_frac | score.action_abs_mean | score.objective_score | coverage.n_addressed | coverage.n_total | coverage.wrist_used | program.mode | program.representation |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| opposed_three_finger_pinch | 0 | 0.000003 | 0.001875 | 0.001875 | 0.023438 | 0.057131 | 0 | 0 | -0.801243 | 23 | 23 | yes | stacked | freeform |
| scoop_power_grasp | 0 | 0.000004 | 0.001875 | 0.002344 | 0.007812 | 0.047693 | 0 | 0 | -0.372382 | 23 | 23 | yes | stacked | freeform |
| side_clamp_then_lift | 0 | 0.000006 | 0.002266 | 0.002422 | 0.023438 | 0.059529 | 0 | 0 | -0.807871 | 23 | 23 | yes | stacked | freeform |

## What the results mean and major discoveries

- In `dsl_scored.json`, `tripod_pad_grasp` had the highest recorded open-loop objective score (-0.109883). All candidates should still be read against the contact, lift, displacement, and saturation measurements in the table; this prefilter score is not trained-policy success.
- In `freeform_scored.json`, `scoop_power_grasp` had the highest recorded open-loop objective score (-0.372382). All candidates should still be read against the contact, lift, displacement, and saturation measurements in the table; this prefilter score is not trained-policy success.

## Limitations and reading guidance

- Metrics are reported under the names used by the historical environment and runner. Similar-looking metrics from different generations of the code are not assumed to be numerically interchangeable.
- A maximum observed in training is not treated as held-out performance. Terminal evaluation values are preferred whenever present.
- Missing summaries, empty metric streams, failed retries, and incomplete arms are preserved as negative evidence rather than silently dropped.
- Candidate names and stage names are LLM-authored run artifacts. Interpretations in this historical log do not alter framework prompts, defaults, diagnostics, or task logic.

## Artifact provenance

| Artifact type | Count |
|---|---|
| .json | 7 |
| .md | 4 |
| .txt | 4 |

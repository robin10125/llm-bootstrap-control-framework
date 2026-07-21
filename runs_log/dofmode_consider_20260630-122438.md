# DOF-mode Consider

- **Run directory:** [`runs/dofmode_consider_20260630-122438`](../runs/dofmode_consider_20260630-122438)
- **Date:** 2026-06-30 12:24:38 (directory timestamp)
- **Status:** artifact-only or generation-only; no terminal evaluation recorded
- **Experiment class:** prior-representation study
- **Recorded task:** not explicitly recorded at the run root
- **Training metric rows recovered:** 0

## Abstract

This entry reconstructs `dofmode_consider_20260630-122438`, a prior-representation study. The account is based only on artifacts that survive in the repository: machine-readable configuration, candidate programs, selection diagnostics, training metrics, evaluations, reports, and logs. The run targeted not explicitly recorded at the run root. Its purpose and comparisons are inferred from the recorded arm names and configuration where no authored abstract survives; those inferences are identified by this provenance-first framing rather than presented as new experimental facts.

## Experimental structure

No machine-readable configuration file survives at the run root. Structure below is reconstructed from reports, result files, subdirectory names, and logs.
### Arms and sub-runs represented by directories

`llm`

## Results

### Recorded results: `dsl_scored.json`

| name | score.contact_gated_success | score.contact_conditioned_lift | score.contact_engagement | score.contacts_mean | score.fling_fraction | score.palm_obj_dist_min | score.saturation_frac | score.action_abs_mean | score.objective_score | accounting.n_driven | accounting.n_unused_listed | accounting.wrist_driven | program.mode | program.representation |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tripod_precision_pinch | 0 | 0 | 0 | 0 | 0 | 0.024812 | 0 | 0 | -0.099249 | 23 | 0 | yes | stacked | dsl |
| soft_enveloping_cage | 0 | 0 | 0 | 0 | 0 | 0.065536 | 0 | 0 | -0.262146 | 23 | 0 | yes | stacked | dsl |
| lateral_thumb_index_pinch | 0 | 0 | 0 | 0 | 0 | 0.023299 | 0 | 0 | -0.093198 | 23 | 0 | yes | stacked | dsl |

### Recorded results: `freeform_scored.json`

| name | score.contact_gated_success | score.contact_conditioned_lift | score.contact_engagement | score.contacts_mean | score.fling_fraction | score.palm_obj_dist_min | score.saturation_frac | score.action_abs_mean | score.objective_score | accounting.n_driven | accounting.n_unused_listed | accounting.wrist_driven | program.mode | program.representation |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pinch_then_vertical_lift | 0 | 0.000006 | 0.002266 | 0.002266 | 0.054688 | 0.061319 | 0 | 0 | -1.596428 | 23 | 0 | yes | stacked | freeform |
| three_finger_cradle | 0 | 0.000003 | 0.002266 | 0.002734 | 0.015625 | 0.053436 | 0 | 0 | -0.587943 | 23 | 0 | yes | stacked | freeform |
| opposed_side_grasp | 0 | 0.000005 | 0.002734 | 0.002813 | 0.015625 | 0.060102 | 0 | 0 | -0.611657 | 23 | 0 | yes | stacked | freeform |

## What the results mean and major discoveries

- In `dsl_scored.json`, `lateral_thumb_index_pinch` had the highest recorded open-loop objective score (-0.093198). All candidates should still be read against the contact, lift, displacement, and saturation measurements in the table; this prefilter score is not trained-policy success.
- In `freeform_scored.json`, `three_finger_cradle` had the highest recorded open-loop objective score (-0.587943). All candidates should still be read against the contact, lift, displacement, and saturation measurements in the table; this prefilter score is not trained-policy success.

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

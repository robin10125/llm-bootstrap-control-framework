# DOF-mode Encourage

- **Run directory:** [`runs/dofmode_encourage_20260630-123526`](../runs/dofmode_encourage_20260630-123526)
- **Date:** 2026-06-30 12:35:26 (directory timestamp)
- **Status:** artifact-only or generation-only; no terminal evaluation recorded
- **Experiment class:** prior-representation study
- **Recorded task:** not explicitly recorded at the run root
- **Training metric rows recovered:** 0

## Abstract

This entry reconstructs `dofmode_encourage_20260630-123526`, a prior-representation study. The account is based only on artifacts that survive in the repository: machine-readable configuration, candidate programs, selection diagnostics, training metrics, evaluations, reports, and logs. The run targeted not explicitly recorded at the run root. Its purpose and comparisons are inferred from the recorded arm names and configuration where no authored abstract survives; those inferences are identified by this provenance-first framing rather than presented as new experimental facts.

## Experimental structure

No machine-readable configuration file survives at the run root. Structure below is reconstructed from reports, result files, subdirectory names, and logs.
### Arms and sub-runs represented by directories

`llm`

## Results

### Recorded results: `dsl_scored.json`

| name | score.contact_gated_success | score.contact_conditioned_lift | score.contact_engagement | score.contacts_mean | score.fling_fraction | score.palm_obj_dist_min | score.saturation_frac | score.action_abs_mean | score.objective_score | accounting.n_driven | accounting.n_unused_listed | accounting.wrist_driven | program.mode | program.representation |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| precision_pinch_index_middle | 0 | 0 | 0 | 0 | 0 | 0.067618 | 0 | 0 | -0.270471 | 23 | 0 | yes | stacked | dsl |
| four_finger_cage_lift | 0 | 0 | 0 | 0 | 0 | 0.023714 | 0 | 0 | -0.094856 | 23 | 0 | yes | stacked | dsl |
| thumb_ring_little_scoop | 0 | 0 | 0 | 0 | 0 | 0.066569 | 0 | 0 | -0.266277 | 23 | 0 | yes | stacked | dsl |

### Recorded results: `freeform_scored.json`

| name | score.contact_gated_success | score.contact_conditioned_lift | score.contact_engagement | score.contacts_mean | score.fling_fraction | score.palm_obj_dist_min | score.saturation_frac | score.action_abs_mean | score.objective_score | accounting.n_driven | accounting.n_unused_listed | accounting.wrist_driven | program.mode | program.representation |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| centered_precision_pinch | 0 | 0.000002 | 0.000937 | 0.001094 | 0.015625 | 0.067219 | 0 | 0 | -0.652715 | 23 | 0 | yes | stacked | freeform |
| opposed_power_cage | 0 | 0.000002 | 0.001797 | 0.001875 | 0.007812 | 0.053047 | 0 | 0 | -0.394785 | 23 | 0 | yes | stacked | freeform |
| side_thumb_tripod_scoop | 0 | 0.000002 | 0.001094 | 0.001094 | 0.015625 | 0.05602 | 0 | 0 | -0.606992 | 23 | 0 | yes | stacked | freeform |

## What the results mean and major discoveries

- In `dsl_scored.json`, `four_finger_cage_lift` had the highest recorded open-loop objective score (-0.094856). All candidates should still be read against the contact, lift, displacement, and saturation measurements in the table; this prefilter score is not trained-policy success.
- In `freeform_scored.json`, `opposed_power_cage` had the highest recorded open-loop objective score (-0.394785). All candidates should still be read against the contact, lift, displacement, and saturation measurements in the table; this prefilter score is not trained-policy success.

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

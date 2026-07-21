# Conditional Vs Flat Video

- **Run directory:** [`famework_testing/runs/conditional_vs_flat_video`](../famework_testing/runs/conditional_vs_flat_video)
- **Date:** 2026-06-15 (earliest surviving artifact mtime (approximate))
- **Status:** completed or completed-with-caveats
- **Experiment class:** early interface/controller comparison
- **Recorded task:** lift
- **Training metric rows recovered:** 0

## Abstract

This entry reconstructs `conditional_vs_flat_video`, an early interface/controller comparison. The account is based only on artifacts that survive in the repository: machine-readable configuration, candidate programs, selection diagnostics, training metrics, evaluations, reports, and logs. The run targeted lift. Its purpose and comparisons are inferred from the recorded arm names and configuration where no authored abstract survives; those inferences are identified by this provenance-first framing rather than presented as new experimental facts.

## Experimental structure

No machine-readable configuration file survives at the run root. Structure below is reconstructed from reports, result files, subdirectory names, and logs.
## Results

### Recorded results: `summary.json`

| Arm/interface | frames | ok | video |
|---|---|---|---|
| validation | — | yes | — |
| conditional | 26 | — | runs/conditional_vs_flat_video/conditional_loops.mp4 |
| without_conditionals | 26 | — | runs/conditional_vs_flat_video/without_conditionals_flattened.mp4 |

| Variant | Success | Score | Return | Max object z | Online conditionals | Steps | Loops |
|---|---|---|---|---|---|---|---|
| conditional | no | 0.0002 | -3.2717 | 0.025 | yes | 25 | 3 |
| without_conditionals | no | 0.0002 | -2.732 | 0.025 | no | 25 | 0 |

## What the results mean and major discoveries

- Both the online-conditional and flattened variants failed with score 0.0002 and max object z 0.025. The three online loops therefore changed execution structure but produced no measured success or lift advantage in this episode; its return (-3.2717) was also below the flattened return (-2.732).
- The arm comparison is most informative when arms shared the same environment, seed, and budget. Where the configuration records only one seed, apparent gaps remain vulnerable to optimization variance.

## Limitations and reading guidance

- Metrics are reported under the names used by the historical environment and runner. Similar-looking metrics from different generations of the code are not assumed to be numerically interchangeable.
- A maximum observed in training is not treated as held-out performance. Terminal evaluation values are preferred whenever present.
- Missing summaries, empty metric streams, failed retries, and incomplete arms are preserved as negative evidence rather than silently dropped.
- Candidate names and stage names are LLM-authored run artifacts. Interpretations in this historical log do not alter framework prompts, defaults, diagnostics, or task logic.

## Artifact provenance

| Artifact type | Count |
|---|---|
| .json | 2 |
| .mp4 | 2 |

Primary evidence files:

- [`summary.json`](../famework_testing/runs/conditional_vs_flat_video/summary.json)

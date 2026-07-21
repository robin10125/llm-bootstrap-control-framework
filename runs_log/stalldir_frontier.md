# Stalldir Frontier

- **Run directory:** [`runs/stalldir_frontier`](../runs/stalldir_frontier)
- **Date:** 2026-07-02 (earliest surviving artifact mtime (approximate))
- **Status:** artifact-only or generation-only; no terminal evaluation recorded
- **Experiment class:** structural diagnostic/revision study
- **Recorded task:** not explicitly recorded at the run root
- **Training metric rows recovered:** 0

## Abstract

This entry reconstructs `stalldir_frontier`, a structural diagnostic/revision study. The account is based only on artifacts that survive in the repository: machine-readable configuration, candidate programs, selection diagnostics, training metrics, evaluations, reports, and logs. The run targeted not explicitly recorded at the run root. Its purpose and comparisons are inferred from the recorded arm names and configuration where no authored abstract survives; those inferences are identified by this provenance-first framing rather than presented as new experimental facts.

## Experimental structure

No machine-readable configuration file survives at the run root. Structure below is reconstructed from reports, result files, subdirectory names, and logs.
### Arms and sub-runs represented by directories

`llm`, `ppo`

## Results

### Recorded results: `frontier_report.json`

| Recorded field | Value |
|---|---|
| stop_reason | plateau |
| final.name | stage0_gate_handoff_and_decelerated_recover |
| final.objective | 0.0406 |
| final.stage_report.stage_names | recover_far_or_lost, funnel_contact_geometry, five_finger_clamp, lift_guarded, airborne_hold |
| final.stage_report.occupancy | 0.9698, 0.0157, 0.0145, 0, 0 |
| final.stage_report.reached_frac | 1, 0.2734, 0.4141, 0, 0 |
| final.stage_report.stall_stage | 2 |
| final.stage_report.stall_name | five_finger_clamp |
| final.stage_report.reaches_terminal | no |
| final.stage_report.stall_signal_trend.closure | 0.1884, 0.2186 |
| final.stage_report.stall_signal_trend.gripped | 0.0611, 0.075 |
| final.stage_report.stall_signal_trend.lift | 0.0124, 0.0111 |
| final.stage_report.stall_signal_trend.near | 0.3, 0.302 |
| final.stage_report.stall_signal_trend.obj_rel_x | -0.0095, 0.0117 |
| final.stage_report.stall_signal_trend.obj_rel_y | 0.006, -0.0026 |
| final.stage_report.stall_signal_trend.obj_rel_z | -0.0376, -0.0374 |
| final.stage_report.stall_signal_trend.palm_obj_dist | 0.057, 0.0568 |
| final.stage_report.stall_gate.expr | near*clip(1-gripped,0,1)*clip(closure+0.35,0,1)+0.15*(near>0.75) |
| final.stage_report.stall_gate.value_early_late | 0.1516, 0.1586 |
| final.stage_report.next_gate.index | 3 |
| final.stage_report.next_gate.name | lift_guarded |
| final.stage_report.next_gate.expr | 3*near*gripped*(lift<0.055) |
| final.stage_report.next_gate.signals | gripped, lift, near |
| final.stage_report.next_gate.value_early_late | 0.0541, 0.0675 |
| final.stage_report.self_lock | no |

## What the results mean and major discoveries

- The surviving artifacts document the attempted structure but do not contain enough terminal quantitative evidence for a strong result claim. Treat this entry as provenance and negative/unfinished evidence.

## Limitations and reading guidance

- Metrics are reported under the names used by the historical environment and runner. Similar-looking metrics from different generations of the code are not assumed to be numerically interchangeable.
- A maximum observed in training is not treated as held-out performance. Terminal evaluation values are preferred whenever present.
- Missing summaries, empty metric streams, failed retries, and incomplete arms are preserved as negative evidence rather than silently dropped.
- Candidate names and stage names are LLM-authored run artifacts. Interpretations in this historical log do not alter framework prompts, defaults, diagnostics, or task logic.

## Artifact provenance

| Artifact type | Count |
|---|---|
| .pkl | 21 |
| .json | 2 |
| .md | 1 |
| .txt | 1 |

Primary evidence files:

- [`frontier_report.json`](../runs/stalldir_frontier/frontier_report.json)

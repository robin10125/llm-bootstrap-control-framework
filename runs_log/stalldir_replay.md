# Stalldir Replay

- **Run directory:** [`runs/stalldir_replay`](../runs/stalldir_replay)
- **Date:** 2026-07-02 (earliest surviving artifact mtime (approximate))
- **Status:** artifact-only or generation-only; no terminal evaluation recorded
- **Experiment class:** structural diagnostic/revision study
- **Recorded task:** not explicitly recorded at the run root
- **Training metric rows recovered:** 0

## Abstract

This entry reconstructs `stalldir_replay`, a structural diagnostic/revision study. The account is based only on artifacts that survive in the repository: machine-readable configuration, candidate programs, selection diagnostics, training metrics, evaluations, reports, and logs. The run targeted not explicitly recorded at the run root. Its purpose and comparisons are inferred from the recorded arm names and configuration where no authored abstract survives; those inferences are identified by this provenance-first framing rather than presented as new experimental facts.

## Experimental structure

No machine-readable configuration file survives at the run root. Structure below is reconstructed from reports, result files, subdirectory names, and logs.
### Arms and sub-runs represented by directories

`llm`, `ppo`

## Results

### Recorded results: `base_diag.json`

| Recorded field | Value |
|---|---|
| trained_success | 0 |
| instant_success | 0 |
| reach_rate | 0.1836 |
| grasp_rate | 0 |
| lift_reached_rate | 0.4336 |
| lift_max | 0.0554 |
| mean_contacts | 0.1953 |
| mean_closure | 0.242 |
| obj_xy_drift | 0.7063 |
| saturation_frac | 0 |
| likely_failure_modes | never_reaches (palm not getting to object -> approach too weak/misaimed), flinging (object up without sustained contact -> non-prehensile) |
| stage_report.stage_names | recover_far_or_lost, funnel_contact_geometry, five_finger_clamp, lift_guarded, airborne_hold |
| stage_report.occupancy | 0.9994, 0.0005, 0, 0, 0 |
| stage_report.reached_frac | 1, 0.0234, 0.0039, 0, 0 |
| stage_report.stall_stage | 0 |
| stage_report.stall_name | recover_far_or_lost |
| stage_report.reaches_terminal | no |
| stage_report.stall_signal_trend.closure | 0.2029, 0.2215 |
| stage_report.stall_signal_trend.gripped | 0.0672, 0.0749 |
| stage_report.stall_signal_trend.lift | 0.0083, 0.0121 |
| stage_report.stall_signal_trend.near | 0.0274, 0.0069 |
| stage_report.stall_signal_trend.obj_rel_x | -0.0069, 0.0734 |
| stage_report.stall_signal_trend.obj_rel_y | -0.0013, -0.0149 |
| stage_report.stall_signal_trend.obj_rel_z | -0.0711, -0.0602 |
| stage_report.stall_signal_trend.palm_obj_dist | 0.1824, 0.4157 |
| stage_report.stall_gate.expr | 1.0-near+0.5*(1-gripped)*(lift<0.05) |
| stage_report.stall_gate.value_early_late | 1.4255, 1.4269 |
| stage_report.next_gate.index | 1 |
| stage_report.next_gate.name | funnel_contact_geometry |
| stage_report.next_gate.expr | near*(1-gripped)*clip(0.65-closure,0,1)+0.25*(near>0.35) |
| stage_report.next_gate.signals | closure, gripped, near |
| stage_report.next_gate.value_early_late | 0.0134, 0.003 |
| stage_report.self_lock | yes |

### Recorded results: `revised_diag.json`

| Recorded field | Value |
|---|---|
| trained_success | 0 |
| instant_success | 0 |
| reach_rate | 0.2031 |
| grasp_rate | 0 |
| lift_reached_rate | 0.4102 |
| lift_max | 0.0537 |
| mean_contacts | 0.2227 |
| mean_closure | 0.2301 |
| obj_xy_drift | 0.638 |
| saturation_frac | 0 |
| likely_failure_modes | never_reaches (palm not getting to object -> approach too weak/misaimed), flinging (object up without sustained contact -> non-prehensile) |
| stage_report.stage_names | recover_far_or_lost, funnel_contact_geometry, five_finger_clamp, lift_guarded, airborne_hold |
| stage_report.occupancy | 0.9698, 0.0157, 0.0145, 0, 0 |
| stage_report.reached_frac | 1, 0.2734, 0.4141, 0, 0 |
| stage_report.stall_stage | 2 |
| stage_report.stall_name | five_finger_clamp |
| stage_report.reaches_terminal | no |
| stage_report.stall_signal_trend.closure | 0.1884, 0.2186 |
| stage_report.stall_signal_trend.gripped | 0.0611, 0.075 |
| stage_report.stall_signal_trend.lift | 0.0124, 0.0111 |
| stage_report.stall_signal_trend.near | 0.3, 0.302 |
| stage_report.stall_signal_trend.obj_rel_x | -0.0095, 0.0117 |
| stage_report.stall_signal_trend.obj_rel_y | 0.006, -0.0026 |
| stage_report.stall_signal_trend.obj_rel_z | -0.0376, -0.0374 |
| stage_report.stall_signal_trend.palm_obj_dist | 0.057, 0.0568 |
| stage_report.stall_gate.expr | near*clip(1-gripped,0,1)*clip(closure+0.35,0,1)+0.15*(near>0.75) |
| stage_report.stall_gate.value_early_late | 0.1516, 0.1586 |
| stage_report.next_gate.index | 3 |
| stage_report.next_gate.name | lift_guarded |
| stage_report.next_gate.expr | 3*near*gripped*(lift<0.055) |
| stage_report.next_gate.signals | gripped, lift, near |
| stage_report.next_gate.value_early_late | 0.0541, 0.0675 |
| stage_report.self_lock | no |

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
| .pkl | 14 |
| .json | 4 |
| .md | 1 |
| .txt | 1 |

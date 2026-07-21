# Generation Only Envcontact 1seed

- **Run directory:** [`runs/generation_only_envcontact_1seed_20260708-1529`](../runs/generation_only_envcontact_1seed_20260708-1529)
- **Date:** 2026-07-08 15:29:00 (directory timestamp)
- **Status:** artifact-only or generation-only; no terminal evaluation recorded
- **Experiment class:** agentic prior generation/selection study
- **Recorded task:** not explicitly recorded at the run root
- **Training metric rows recovered:** 0

## Abstract

This entry reconstructs `generation_only_envcontact_1seed_20260708-1529`, an agentic prior generation/selection study. The account is based only on artifacts that survive in the repository: machine-readable configuration, candidate programs, selection diagnostics, training metrics, evaluations, reports, and logs. The run targeted not explicitly recorded at the run root. Its purpose and comparisons are inferred from the recorded arm names and configuration where no authored abstract survives; those inferences are identified by this provenance-first framing rather than presented as new experimental facts.

## Experimental structure

No machine-readable configuration file survives at the run root. Structure below is reconstructed from reports, result files, subdirectory names, and logs.
### Arms and sub-runs represented by directories

`llm`

## Results

### Recorded results: `raw_seeds.json`

| name | rationale | ir_version | signals.obj_speed | signals.obj_hspeed | signals.lat_err | signals.min_tip_z | signals.max_hand_c | signals.max_finger_c | signals.sum_finger_c | signals.max_env_c | signals.max_region_c | signals.no_env | signals.gentle_object | signals.force_under_cap | signals.wrist_margin | signals.done_wrist | signals.thumb_preshape_margin |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| reachable_tilt_opposing_contacts_lift | Use the reachable wrist extension limit first, then pre-shape thumb and long fingers in free space, center laterally, descend gently to first object contact, close into opposing thumb-plus-finger contact, stabilize force, then lift upwar… | 1 | sqrt(obj_vel_x*obj_vel_x + obj_vel_y*obj_vel_y + obj_vel_z*obj_vel_z) | sqrt(obj_vel_x*obj_vel_x + obj_vel_y*obj_vel_y) | sqrt(obj_rel_x*obj_rel_x + obj_rel_y*obj_rel_y) | min(min(rh_ffdistal_z,rh_mfdistal_z),min(min(rh_rfdistal_z,rh_lfdistal_z),rh_thdistal_z)) | max(c_thumb,max(c_index,max(c_middle,max(c_ring,max(c_little,c_palm))))) | max(c_index,max(c_middle,max(c_ring,c_little))) | c_index + c_middle + c_ring + c_little | max(env_c_thumb,max(env_c_index,max(env_c_middle,max(env_c_ring,max(env_c_little,env_c_palm))))) | max(c_thumb,max(c_index,max(c_middle,max(c_ring,max(c_little,c_palm))))) | clip((0.025 - max_env_c)/0.025,0,1) | clip((0.040 - obj_speed)/0.040,0,1) | clip((force_cap - max_region_c)/force_cap,0,1) | wrj1_ext - q_rh_A_WRJ1 | clip((wrist_margin + 0.015)/0.035,0,1) | min(min(q_rh_A_THJ4 - thumb4_pre,q_rh_A_THJ1 - thumb1_pre),min(0.020 - max_env_c,0.08 - c_thumb)) |

## What the results mean and major discoveries

- The generation-only run produced 1 candidate program(s). No rollout or training evaluation survives, so this establishes generation output, not behavioral quality.

## Limitations and reading guidance

- Metrics are reported under the names used by the historical environment and runner. Similar-looking metrics from different generations of the code are not assumed to be numerically interchangeable.
- A maximum observed in training is not treated as held-out performance. Terminal evaluation values are preferred whenever present.
- Missing summaries, empty metric streams, failed retries, and incomplete arms are preserved as negative evidence rather than silently dropped.
- Candidate names and stage names are LLM-authored run artifacts. Interpretations in this historical log do not alter framework prompts, defaults, diagnostics, or task logic.

## Artifact provenance

| Artifact type | Count |
|---|---|
| .json | 6 |
| .txt | 5 |
| .md | 4 |

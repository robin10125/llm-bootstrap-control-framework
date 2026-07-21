# Prior Only Speedlimit 1seed 4rev

- **Run directory:** [`runs/prior_only_speedlimit_1seed_4rev_20260708-144756`](../runs/prior_only_speedlimit_1seed_4rev_20260708-144756)
- **Date:** 2026-07-08 14:47:56 (directory timestamp)
- **Status:** completed or completed-with-caveats
- **Experiment class:** agentic prior generation/selection study
- **Recorded task:** grasp and lift a 5cm cube off the table
- **Training metric rows recovered:** 0

## Abstract

This entry reconstructs `prior_only_speedlimit_1seed_4rev_20260708-144756`, an agentic prior generation/selection study. The account is based only on artifacts that survive in the repository: machine-readable configuration, candidate programs, selection diagnostics, training metrics, evaluations, reports, and logs. The run targeted grasp and lift a 5cm cube off the table. Its purpose and comparisons are inferred from the recorded arm names and configuration where no authored abstract survives; those inferences are identified by this provenance-first framing rather than presented as new experimental facts.

## Experimental structure

No machine-readable configuration file survives at the run root. Structure below is reconstructed from reports, result files, subdirectory names, and logs.
### Selection-loop structure

| Field | Value |
|---|---|
| representation | freeform_staged |
| dof_mode | consider |
| arbiter | prior_only |
| budget | 6 |
| iters_used | 4 |
| n_seeds | 1 |
| wall_hours | 0.376 |

### Arms and sub-runs represented by directories

`llm`, `videos`

## Results

### Recorded results: `report.json`

| Best-candidate field | Value |
|---|---|
| name | stabilize_stage4_5_thumb_contact_with_descent_hold |
| source | refine:reachable_oblique_gentle_lift_ladder |
| objective | 0.320673 |
| accounting.n_driven | 23 |
| accounting.n_unused_listed | 0 |
| accounting.wrist_driven | yes |

| Diagnostic | Value |
|---|---|
| success_rate | 0 |
| reach_rate | 0.957 |
| grasp_rate | 0.207 |
| grasp_lift_rate | 0 |
| lift_reached_rate | 0.043 |
| lift_max | 0.0085 |
| base_return | 20.0382 |
| train_return | 20.0411 |
| shaped_return | 0.0029 |
| action_abs_mean | 0.0134 |
| saturation_frac | 0 |
| note | behavior at PPO iteration 0: the prior acting through an UNTRAINED policy (network output ~0, so this is essentially the prior alone). Compare with the trained metrics -- a defect present here is a PRIOR problem; one that only appears af… |
| base_reward_weight | 1 |
| instant_success_rate | 0 |
| hard_clip_frac | 0 |
| failure_rate | 0.1992 |
| calm_frac | 0.8008 |
| reach_rate_calm | 0.7578 |

Stage-flow measurements for the selected candidate:

| # | Stage | Occupancy | Reached | Entered | Handoff | Conversion | Authored success |
|---|---|---|---|---|---|---|---|
| 0 | establish_reachable_wrist_attitude | 0.0865 | 1 | 1 | 1 | 1 | 1 |
| 1 | pre_shape_open_opposition | 0.0359 | 1 | 1 | 0.9961 | 0.9961 | 1 |
| 2 | horizontal_centering_above_cube | 0.2183 | 0.9961 | 0.9961 | 0.7656 | 0.7686 | 0.7656 |
| 3 | controlled_descent_to_near_contact | 0.1077 | 1 | 1 | 0.9961 | 0.9961 | 1 |
| 4 | gentle_first_contact | 0.4632 | 1 | 0.9961 | 0.7891 | 0.7891 | 0.9102 |
| 5 | form_opposing_contacts | 0.0883 | 0.9258 | 0.7891 | 0.0039 | 0.0042 | 0 |
| 6 | settle_minimum_holding_force | 0 | 0.0078 | 0.0039 | 0 | 0 | 0 |
| 7 | lift_with_stable_grasp | 0 | 0.0039 | 0 | 0 | 0 | 0 |
| 8 | final_lift_hold | 0 | 0 | 0 | — | — | 0 |

The recorded structural stall was `form_opposing_contacts` (index 5); terminal stage reached: no.

Candidate trajectory:

| iter | source | name | objective | frontier | wrist_driven | n_driven |
|---|---|---|---|---|---|---|
| 1 | explore | reachable_oblique_gentle_lift_ladder | 0.187056 | — | yes | 23 |
| 2 | refine:reachable_oblique_gentle_lift_ladder | targeted_thumb_opposition_stage5 | 0.084646 | — | yes | 23 |
| 3 | refine:reachable_oblique_gentle_lift_ladder | stabilize_stage4_5_thumb_contact_with_descent_hold | 0.320673 | — | yes | 23 |
| 4 | refine:reachable_oblique_gentle_lift_ladder | thumb_swing_opposition_fix | 0.20657 | — | yes | 23 |

## What the results mean and major discoveries

- The selection loop chose `stabilize_stage4_5_thumb_contact_with_descent_hold` with objective 0.320673. Selection therefore established a best candidate relative to the tested pool; it did not by itself establish task mastery.
- Recorded success was zero while the associated lift/progress measurement was 0.043. The candidate generated partial behavior under the run's diagnostics but did not satisfy the full success predicate in that evaluation.
- The stage-flow diagnostic localized the dominant structural bottleneck to `form_opposing_contacts`. This is a measured control-flow finding from the authored program, useful for revision without treating the stage label as a framework-authored diagnosis.

## Limitations and reading guidance

- Metrics are reported under the names used by the historical environment and runner. Similar-looking metrics from different generations of the code are not assumed to be numerically interchangeable.
- A maximum observed in training is not treated as held-out performance. Terminal evaluation values are preferred whenever present.
- Missing summaries, empty metric streams, failed retries, and incomplete arms are preserved as negative evidence rather than silently dropped.
- Candidate names and stage names are LLM-authored run artifacts. Interpretations in this historical log do not alter framework prompts, defaults, diagnostics, or task logic.

## Artifact provenance

| Artifact type | Count |
|---|---|
| .json | 13 |
| .md | 10 |
| .txt | 10 |
| .html | 1 |
| .mp4 | 1 |
| .pkl | 1 |
| .png | 1 |

Primary evidence files:

- [`report.json`](../runs/prior_only_speedlimit_1seed_4rev_20260708-144756/report.json)

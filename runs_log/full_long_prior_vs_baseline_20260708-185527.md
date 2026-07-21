# Full Long Prior Vs Baseline

- **Run directory:** [`runs/full_long_prior_vs_baseline_20260708-185527`](../runs/full_long_prior_vs_baseline_20260708-185527)
- **Date:** 2026-07-08 18:55:27 (directory timestamp)
- **Status:** completed or completed-with-caveats
- **Experiment class:** agentic prior generation/selection study
- **Recorded task:** lift
- **Training metric rows recovered:** 3,080

## Abstract

This entry reconstructs `full_long_prior_vs_baseline_20260708-185527`, an agentic prior generation/selection study. The account is based only on artifacts that survive in the repository: machine-readable configuration, candidate programs, selection diagnostics, training metrics, evaluations, reports, and logs. The run targeted lift. Its purpose and comparisons are inferred from the recorded arm names and configuration where no authored abstract survives; those inferences are identified by this provenance-first framing rather than presented as new experimental facts.

## Experimental structure

#### Configuration: `ppo_baseline_no_prior_6h/config.json`

| Field | Recorded value |
|---|---|
| learner | long_fragmented_stage_ppo |
| task | lift |
| arm | baseline |
| seed | 0 |
| episode_seconds | 20 |
| env.horizon | 800 |
| env.fragment_steps | 100 |
| criteria.target_train_seconds | 21,600 |
| criteria.max_env_steps | — |
| criteria.iters | 10000000 |
| criteria.legacy_min_hours | 8 |
| criteria.legacy_plateau_hours | 2 |
| criteria.legacy_success_stop | 0.8 |
| ppo.iters | 10000000 |
| ppo.envs | 256 |
| ppo.eval_envs | 256 |
| ppo.fragment_steps | 100 |
| ppo.lr | 0.0003 |
| ppo.gamma | 0.99 |
| ppo.lam | 0.95 |
| ppo.hidden | 256, 256 |
| ppo.ent_coef | 0 |
| ppo.target_train_seconds | 21,600 |
| ppo.max_env_steps | — |
| ppo.checkpoint_every | 100 |
| ppo.eval_every | 25 |
| ppo.residual_action_scale | 1 |
| ppo.use_action_prior | no |
| ppo.learn_prior_scale | yes |
| ppo.prior_scale_mode | scalar |
| ppo.prior_scale_bias | 1 |
| ppo.prior_scale_gain | 1 |
| ppo.stage_reward_weight | 0 |
| ppo.stage_progress_weight | 1 |
| ppo.stage_completion_bonus | 0.05 |
| ppo.stage_success_temperature | 1 |
| ppo.stage_reward_clip | 0.5 |
| ppo.base_reward_weight | 1 |
| ppo.action_transform | tanh |
| ppo.warmup_compile | yes |

#### Configuration: `ppo_with_selected_prior_6h/config.json`

| Field | Recorded value |
|---|---|
| learner | long_fragmented_stage_ppo |
| task | lift |
| arm | freeform_encourage |
| seed | 0 |
| episode_seconds | 20 |
| env.horizon | 800 |
| env.fragment_steps | 100 |
| criteria.target_train_seconds | 21,600 |
| criteria.max_env_steps | — |
| criteria.iters | 10000000 |
| criteria.legacy_min_hours | 8 |
| criteria.legacy_plateau_hours | 2 |
| criteria.legacy_success_stop | 0.8 |
| ppo.iters | 10000000 |
| ppo.envs | 256 |
| ppo.eval_envs | 256 |
| ppo.fragment_steps | 100 |
| ppo.lr | 0.0003 |
| ppo.gamma | 0.99 |
| ppo.lam | 0.95 |
| ppo.hidden | 256, 256 |
| ppo.ent_coef | 0 |
| ppo.target_train_seconds | 21,600 |
| ppo.max_env_steps | — |
| ppo.checkpoint_every | 100 |
| ppo.eval_every | 25 |
| ppo.residual_action_scale | 1 |
| ppo.use_action_prior | yes |
| ppo.learn_prior_scale | yes |
| ppo.prior_scale_mode | scalar |
| ppo.prior_scale_bias | 1 |
| ppo.prior_scale_gain | 1 |
| ppo.stage_reward_weight | 1 |
| ppo.stage_progress_weight | 1 |
| ppo.stage_completion_bonus | 0.05 |
| ppo.stage_success_temperature | 1 |
| ppo.stage_reward_clip | 0.5 |
| ppo.base_reward_weight | 1 |
| ppo.action_transform | tanh |
| ppo.warmup_compile | yes |

### Arms and sub-runs represented by directories

`ppo_baseline_no_prior_6h`, `ppo_with_selected_prior_6h`, `selection_1seed_4rev`, `videos`

## Results

### Recorded results: `comparison.json`

| Recorded field | Value |
|---|---|
| experiment | /home/robin/Documents/agent-mini-script-control/llm-framework/runs/full_long_prior_vs_baseline_20260708-185527 |
| selection.dir | /home/robin/Documents/agent-mini-script-control/llm-framework/runs/full_long_prior_vs_baseline_20260708-185527/selection_1seed_4rev |
| selection.best_objective | 0.750724 |
| selection.selected_program | /home/robin/Documents/agent-mini-script-control/llm-framework/runs/full_long_prior_vs_baseline_20260708-185527/selection_1seed_4rev/best_program.json |
| selection.selection.reason | best_objective |
| selection.selection.objective_floor_fraction | 0.5 |
| selection.selection.objective_floor | 0.375362 |
| selection.selection.frontier_completion_fraction | 0.25 |
| selection.selection.selected_frontier | 1 |
| selection.selection.best_objective_frontier | 1 |
| selection.selection.best_objective.name | preshape_clearance_by_finger_lift |
| selection.selection.best_objective.source | refine:oblique_thumb_index_middle_lift |
| selection.selection.best_objective.objective | 0.750724 |
| selection.selection.best_objective.frontier | 1 |
| prior_ppo.dir | /home/robin/Documents/agent-mini-script-control/llm-framework/runs/full_long_prior_vs_baseline_20260708-185527/ppo_with_selected_prior_6h |
| prior_ppo.iters | 1530 |
| prior_ppo.best_iter | 1524 |
| prior_ppo.best_train_objective | 1.891866 |
| prior_ppo.eval_graded_objective | 0.841101 |
| prior_ppo.eval_success_rate | 0.140625 |
| prior_ppo.eval_task_fitness | 1.871091 |
| prior_ppo.eval_reach_rate | 1 |
| prior_ppo.eval_grasp_rate | 0.707031 |
| prior_ppo.eval_grasp_lift_rate | 0.113281 |
| prior_ppo.eval_lift_reached_rate | 0.140625 |
| prior_ppo.eval_lift_max | 0.029015 |
| prior_ppo.eval_train_return | 319.098 |
| prior_ppo.eval_base_return | 317.923 |
| prior_ppo.eval_stage_return | 1.175003 |
| prior_ppo.eval_prior_scale_mean | 0.826463 |
| prior_ppo.eval_action_abs_mean | 0.47868 |
| baseline_no_prior.dir | /home/robin/Documents/agent-mini-script-control/llm-framework/runs/full_long_prior_vs_baseline_20260708-185527/ppo_baseline_no_prior_6h |
| baseline_no_prior.iters | 1525 |
| baseline_no_prior.best_iter | 1524 |
| baseline_no_prior.best_train_objective | 5.592278 |
| baseline_no_prior.eval_graded_objective | 1.98976 |
| baseline_no_prior.eval_success_rate | 1 |
| baseline_no_prior.eval_task_fitness | 5.663913 |
| baseline_no_prior.eval_reach_rate | 1 |
| baseline_no_prior.eval_grasp_rate | 0.824219 |
| baseline_no_prior.eval_grasp_lift_rate | 0.824219 |
| baseline_no_prior.eval_lift_reached_rate | 1 |
| baseline_no_prior.eval_lift_max | 0.340702 |
| baseline_no_prior.eval_train_return | 757.406 |
| baseline_no_prior.eval_base_return | 757.406 |
| baseline_no_prior.eval_stage_return | 0 |
| baseline_no_prior.eval_prior_scale_mean | 0.890664 |
| baseline_no_prior.eval_action_abs_mean | 0.644135 |
| deltas_prior_minus_baseline.eval_graded_objective | -1.148659 |
| deltas_prior_minus_baseline.eval_success_rate | -0.859375 |

### Recorded results: `ppo_baseline_no_prior_6h/final_report.json`

| Run result | Value |
|---|---|
| stop_reason | fragmented-stage training budget exhausted |
| iters | 1525 |
| best_iter | 1524 |
| best_train_objective | 5.592278 |

| Evaluation metric | Value |
|---|---|
| eval_success_rate | 1 |
| eval_task_fitness | 5.663913 |
| eval_reach_rate | 1 |
| eval_grasp_rate | 0.824219 |
| eval_grasp_lift_rate | 0.824219 |
| eval_lift_reached_rate | 1 |
| eval_lift_max | 0.340702 |
| eval_train_return | 757.406 |
| eval_base_return | 757.406 |
| eval_stage_return | 0 |
| eval_prior_scale_mean | 0.890664 |
| eval_action_abs_mean | 0.644135 |
| eval_graded_objective | 1.98976 |

### Recorded results: `ppo_with_selected_prior_6h/final_report.json`

| Run result | Value |
|---|---|
| stop_reason | fragmented-stage training budget exhausted |
| iters | 1530 |
| best_iter | 1524 |
| best_train_objective | 1.891866 |

| Evaluation metric | Value |
|---|---|
| eval_success_rate | 0.140625 |
| eval_task_fitness | 1.871091 |
| eval_reach_rate | 1 |
| eval_grasp_rate | 0.707031 |
| eval_grasp_lift_rate | 0.113281 |
| eval_lift_reached_rate | 0.140625 |
| eval_lift_max | 0.029015 |
| eval_train_return | 319.098 |
| eval_base_return | 317.923 |
| eval_stage_return | 1.175003 |
| eval_prior_scale_mean | 0.826463 |
| eval_action_abs_mean | 0.47868 |
| eval_graded_objective | 0.841101 |

### Recorded results: `selection_1seed_4rev/report.json`

| Best-candidate field | Value |
|---|---|
| name | preshape_clearance_by_finger_lift |
| source | refine:oblique_thumb_index_middle_lift |
| objective | 0.750724 |
| accounting.n_driven | 23 |
| accounting.n_unused_listed | 0 |
| accounting.wrist_driven | yes |

| Diagnostic | Value |
|---|---|
| graded_objective | 0.7507 |
| success_rate | 0 |
| reach_rate | 1 |
| grasp_rate | 1 |
| grasp_lift_rate | 0 |
| lift_reached_rate | 0 |
| lift_max | 0.0035 |
| base_return | 61.1658 |
| train_return | 61.5407 |
| action_abs_mean | 0.046 |
| task_fitness | 0.6224 |
| stage_return | 0.375 |
| prior_scale_mean | 0.9998 |

Stage-flow measurements for the selected candidate:

| # | Stage | Occupancy | Reached | Entered | Handoff | Conversion | Authored success |
|---|---|---|---|---|---|---|---|
| 0 | wrist_extend | 0.1433 | 1 | 1 | 1 | 1 | 1 |
| 1 | preshape_open_opposition | 0.7948 | 1 | 1 | 0 | 0 | 0.0312 |
| 2 | center_xy | 0 | 0 | 0 | 0 | 0 | 0 |
| 3 | descend_near_contact | 0.0053 | 0.2617 | 0.2461 | 0.2617 | 1 | 0.2617 |
| 4 | light_opposing_contact | 0.0558 | 1 | 0.9922 | 0.0312 | 0.0312 | 0.1172 |
| 5 | close_retention | 0.0006 | 0.0781 | 0.0312 | 0.0039 | 0.05 | 0.0312 |
| 6 | settle_grasp | 0.0001 | 0.0195 | 0.0039 | 0.0039 | 0.2 | 0.0195 |
| 7 | lift_with_base | 0.0001 | 0.0312 | 0.0078 | 0 | 0 | 0 |
| 8 | hold_elevated | 0 | 0 | 0 | — | — | 0 |

The recorded structural stall was `preshape_open_opposition` (index 1); terminal stage reached: no.

Candidate trajectory:

| iter | source | name | objective | frontier | wrist_driven | n_driven |
|---|---|---|---|---|---|---|
| 1 | explore | oblique_thumb_index_middle_lift | 0.749043 | 1 | yes | 23 |
| 2 | refine:oblique_thumb_index_middle_lift | center_xy_entry_rescue | 0.747095 | 1 | yes | 23 |
| 3 | refine:oblique_thumb_index_middle_lift | center_xy_entry_clearance_relaxed | 0.747821 | 1 | yes | 23 |
| 4 | refine:oblique_thumb_index_middle_lift | preshape_clearance_real_margin | 0.738831 | 1 | yes | 23 |
| 5 | refine:oblique_thumb_index_middle_lift | preshape_clearance_by_finger_lift | 0.750724 | 1 | yes | 23 |

### Training-metric reconstruction

| Arm/sub-run | Rows | Last iter | Last env steps | Elapsed s | Last success | Best success | Last lift max | Last base return | Best graded/objective |
|---|---|---|---|---|---|---|---|---|---|
| ppo_baseline_no_prior_6h | 1525 | 1524 | 39040000 | 21,542.107 | 0.789062 | 0.953125 | — | 81.727264 | — |
| ppo_with_selected_prior_6h | 1530 | 1529 | 39168000 | 21,605.248 | 0.007812 | 0.183594 | — | 36.560936 | — |
| selection_1seed_4rev/ppo/iter1 | 3 | 2 | 76800 | 185.15 | 0.042969 | 0.042969 | — | 1.025847 | — |
| selection_1seed_4rev/ppo/iter2 | 6 | 5 | 153600 | 189.136 | 0.039062 | 0.089844 | — | 1.290321 | — |
| selection_1seed_4rev/ppo/iter3 | 5 | 4 | 128000 | 181.551 | 0.042969 | 0.097656 | — | 1.343761 | — |
| selection_1seed_4rev/ppo/iter4 | 5 | 4 | 128000 | 180.806 | 0.050781 | 0.054688 | — | 1.314999 | — |
| selection_1seed_4rev/ppo/iter5 | 6 | 5 | 153600 | 189.308 | 0.039062 | 0.058594 | — | 1.168362 | — |

These are training-log endpoints and maxima, not substitutes for held-out evaluation. They are included especially for interrupted runs whose terminal report is missing.

### Terminal and failure evidence from logs

- `experiment.log`: [done] selected obj=+0.7507 frontier=1 (refine:oblique_thumb_index_middle_lift 'preshape_clearance_by_finger_lift') after 5 iters -> /home/robin/Documents/agent-mini-script-control/llm-framework/runs/full_long_prior_vs_baseline_20260708-185527/selection_1seed_4rev/best_program.json
- `experiment.log`: [done] 1530 fragments, best_iter=1524, eval objective 0.841101 success 0.140625 -> /home/robin/Documents/agent-mini-script-control/llm-framework/runs/full_long_prior_vs_baseline_20260708-185527/ppo_with_selected_prior_6h/final_report.json
- `experiment.log`: [done] 1525 fragments, best_iter=1524, eval objective 1.98976 success 1.0 -> /home/robin/Documents/agent-mini-script-control/llm-framework/runs/full_long_prior_vs_baseline_20260708-185527/ppo_baseline_no_prior_6h/final_report.json

## What the results mean and major discoveries

- The prior-minus-baseline comparison reduced `eval_success_rate` by -0.859375. Because this is a direct within-run delta, it is the clearest headline comparison for this experiment.
- The selection loop chose `preshape_clearance_by_finger_lift` with objective 0.750724. Selection therefore established a best candidate relative to the tested pool; it did not by itself establish task mastery.
- Recorded success was zero while the associated lift/progress measurement was 0. The candidate generated partial behavior under the run's diagnostics but did not satisfy the full success predicate in that evaluation.
- The stage-flow diagnostic localized the dominant structural bottleneck to `preshape_open_opposition`. This is a measured control-flow finding from the authored program, useful for revision without treating the stage label as a framework-authored diagnosis.
- Across terminal sub-run reports, `ppo_baseline_no_prior_6h` recorded the strongest eval_task_fitness (5.663913).
- The best terminal success measurement was `ppo_baseline_no_prior_6h` at 1 (eval_success_rate).

## Limitations and reading guidance

- Metrics are reported under the names used by the historical environment and runner. Similar-looking metrics from different generations of the code are not assumed to be numerically interchangeable.
- A maximum observed in training is not treated as held-out performance. Terminal evaluation values are preferred whenever present.
- Missing summaries, empty metric streams, failed retries, and incomplete arms are preserved as negative evidence rather than silently dropped.
- Candidate names and stage names are LLM-authored run artifacts. Interpretations in this historical log do not alter framework prompts, defaults, diagnostics, or task logic.

## Artifact provenance

| Artifact type | Count |
|---|---|
| .pkl | 41 |
| .json | 21 |
| .md | 9 |
| .txt | 8 |
| .jsonl | 7 |
| .mp4 | 2 |
| .html | 1 |
| .log | 1 |
| .pid | 1 |
| .sh | 1 |

Primary evidence files:

- [`comparison.json`](../runs/full_long_prior_vs_baseline_20260708-185527/comparison.json)
- [`ppo_baseline_no_prior_6h/config.json`](../runs/full_long_prior_vs_baseline_20260708-185527/ppo_baseline_no_prior_6h/config.json)
- [`ppo_baseline_no_prior_6h/final_report.json`](../runs/full_long_prior_vs_baseline_20260708-185527/ppo_baseline_no_prior_6h/final_report.json)
- [`ppo_baseline_no_prior_6h/metrics.jsonl`](../runs/full_long_prior_vs_baseline_20260708-185527/ppo_baseline_no_prior_6h/metrics.jsonl)
- [`ppo_with_selected_prior_6h/config.json`](../runs/full_long_prior_vs_baseline_20260708-185527/ppo_with_selected_prior_6h/config.json)
- [`ppo_with_selected_prior_6h/final_report.json`](../runs/full_long_prior_vs_baseline_20260708-185527/ppo_with_selected_prior_6h/final_report.json)
- [`ppo_with_selected_prior_6h/metrics.jsonl`](../runs/full_long_prior_vs_baseline_20260708-185527/ppo_with_selected_prior_6h/metrics.jsonl)
- [`selection_1seed_4rev/ppo/iter1/metrics.jsonl`](../runs/full_long_prior_vs_baseline_20260708-185527/selection_1seed_4rev/ppo/iter1/metrics.jsonl)
- [`selection_1seed_4rev/ppo/iter2/metrics.jsonl`](../runs/full_long_prior_vs_baseline_20260708-185527/selection_1seed_4rev/ppo/iter2/metrics.jsonl)
- [`selection_1seed_4rev/ppo/iter3/metrics.jsonl`](../runs/full_long_prior_vs_baseline_20260708-185527/selection_1seed_4rev/ppo/iter3/metrics.jsonl)
- [`selection_1seed_4rev/ppo/iter4/metrics.jsonl`](../runs/full_long_prior_vs_baseline_20260708-185527/selection_1seed_4rev/ppo/iter4/metrics.jsonl)
- [`selection_1seed_4rev/ppo/iter5/metrics.jsonl`](../runs/full_long_prior_vs_baseline_20260708-185527/selection_1seed_4rev/ppo/iter5/metrics.jsonl)
- [`selection_1seed_4rev/report.json`](../runs/full_long_prior_vs_baseline_20260708-185527/selection_1seed_4rev/report.json)

## Appendix: original authored run report

The following contemporaneous report is retained (with headings demoted) so that details and caveats are not lost in normalization.

### Source: [`comparison.md`](../runs/full_long_prior_vs_baseline_20260708-185527/comparison.md)

## Prior vs Baseline PPO Comparison

- Selection: `/home/robin/Documents/agent-mini-script-control/llm-framework/runs/full_long_prior_vs_baseline_20260708-185527/selection_1seed_4rev`
- Prior PPO: `/home/robin/Documents/agent-mini-script-control/llm-framework/runs/full_long_prior_vs_baseline_20260708-185527/ppo_with_selected_prior_6h`
- Baseline PPO: `/home/robin/Documents/agent-mini-script-control/llm-framework/runs/full_long_prior_vs_baseline_20260708-185527/ppo_baseline_no_prior_6h`

### Final Evaluation

| metric | prior | baseline | delta |
|---|---:|---:|---:|
| best_iter | 1524 | 1524 | 0 |
| best_train_objective | 1.891866 | 5.592278 | -3.700412 |
| eval_action_abs_mean | 0.47868 | 0.644135 | -0.16545500000000002 |
| eval_base_return | 317.923498 | 757.405972 | -439.482474 |
| eval_graded_objective | 0.841101 | 1.98976 | -1.1486589999999999 |
| eval_grasp_lift_rate | 0.113281 | 0.824219 | -0.7109380000000001 |
| eval_grasp_rate | 0.707031 | 0.824219 | -0.11718800000000007 |
| eval_lift_max | 0.029015 | 0.340702 | -0.311687 |
| eval_lift_reached_rate | 0.140625 | 1.0 | -0.859375 |
| eval_prior_scale_mean | 0.826463 | 0.890664 | -0.06420100000000006 |
| eval_reach_rate | 1.0 | 1.0 | 0.0 |
| eval_stage_return | 1.175003 | 0.0 | 1.175003 |
| eval_success_rate | 0.140625 | 1.0 | -0.859375 |
| eval_task_fitness | 1.871091 | 5.663913 | -3.792822 |
| eval_train_return | 319.098499 | 757.405972 | -438.307473 |
| iters | 1530 | 1525 | 5 |

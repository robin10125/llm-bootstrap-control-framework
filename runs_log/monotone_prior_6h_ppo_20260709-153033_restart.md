# Monotone Prior 6h PPO Restart

- **Run directory:** [`runs/monotone_prior_6h_ppo_20260709-153033_restart`](../runs/monotone_prior_6h_ppo_20260709-153033_restart)
- **Date:** 2026-07-09 15:30:33 (directory timestamp)
- **Status:** completed or completed-with-caveats
- **Experiment class:** long-PPO training study
- **Recorded task:** lift
- **Training metric rows recovered:** 1,567

## Abstract

This entry reconstructs `monotone_prior_6h_ppo_20260709-153033_restart`, a long-PPO training study. The account is based only on artifacts that survive in the repository: machine-readable configuration, candidate programs, selection diagnostics, training metrics, evaluations, reports, and logs. The run targeted lift. Its purpose and comparisons are inferred from the recorded arm names and configuration where no authored abstract survives; those inferences are identified by this provenance-first framing rather than presented as new experimental facts.

## Experimental structure

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

`plots`, `videos`

## Results

### Recorded results: `final_report.json`

| Run result | Value |
|---|---|
| stop_reason | fragmented-stage training budget exhausted |
| iters | 1567 |
| best_iter | 1524 |
| best_train_objective | 2.131742 |

| Evaluation metric | Value |
|---|---|
| eval_success_rate | 0.5 |
| eval_task_fitness | 2.085886 |
| eval_reach_rate | 1 |
| eval_grasp_rate | 0.007812 |
| eval_grasp_lift_rate | 0.007812 |
| eval_lift_reached_rate | 0.5 |
| eval_lift_max | 0.055342 |
| eval_train_return | 307.744 |
| eval_base_return | 270.225 |
| eval_stage_return | 37.518423 |
| eval_prior_scale_mean | 0.798639 |
| eval_action_abs_mean | 0.54918 |
| eval_graded_objective | 0.750821 |

### Recorded results: `plots/rollout_trace/summary.json`

| Arm/interface | lift_max |
|---|---|
| 40003 | 0.065751 |
| 40005 | 0.071722 |

### Recorded results: `posthoc_stage_eval.json`

| Recorded field | Value |
|---|---|
| eval.eval_success_rate | 0.492188 |
| eval.eval_task_fitness | 2.090453 |
| eval.eval_reach_rate | 1 |
| eval.eval_grasp_rate | 0 |
| eval.eval_grasp_lift_rate | 0 |
| eval.eval_lift_reached_rate | 0.492188 |
| eval.eval_lift_max | 0.054871 |
| eval.eval_train_return | 295.793 |
| eval.eval_base_return | 256.989 |
| eval.eval_stage_return | 38.804729 |
| eval.eval_prior_scale_mean | 0.803763 |
| eval.eval_prior_scale_group_means.all | 0.803763 |
| eval.eval_action_abs_mean | 0.546092 |
| eval.eval_summary | 0.038147, 0.034036, 4.675781, 0.422146, 0.054871, 0.381569 |
| eval.eval_graded_objective | 0.73583 |
| stage_report_subset.stage_progression | monotone |
| stage_report_subset.stage_names | wrist_reachable_working_pitch, digit_and_thumb_preshape_clear, horizontal_alignment_above_cube, guarded_descent_to_side_contact_height, first_light_opposing_touch, balanced_capture_without_sliding, initial_lift_with_grasp_locked, raise_and_stabilize_clear_of_table |
| stage_report_subset.occupancy | 0.1021, 0.0177, 0.0729, 0.0367, 0.0333, 0.0209, 0.0226, 0.6937 |
| stage_report_subset.entered_frac | 1, 0.7148, 0.707, 0.5859, 0.4922, 0.4023, 0.4102, 0.8555 |
| stage_report_subset.handoff_frac | 0.7148, 0.707, 0.5859, 0.4922, 0.4023, 0.4102, 0.8555, — |
| stage_report_subset.stall_stage | — |
| stage_report_subset.stall_name | — |
| stage_report_subset.reaches_terminal | yes |
| stage_report_subset.raw_gate_reverse_frac | 0.9492, 0.4141, 0.043, 0, 0.0039, 0.0039, 0.0195, — |
| stage_report_subset.cursor_blocked_regression_frac | 0.5696 |
| stage_report_subset.time_report.rollout_seconds | 20 |
| stage_report_subset.time_report.per_stage_measured_seconds | 2.04, 0.35, 1.46, 0.73, 0.67, 0.42, 0.45, 13.87 |
| stage_report_subset.time_report.ended_in_stage_frac | 0, 0, 0.0547, 0.0195, 0.0352, 0.0156, 0.0195, 0.8555 |
| stage_report_subset.time_report.authored_est_seconds | 2.8, 2.5, 3, 2.4, 2, 2, 1.8, 2.5 |
| stage_report_subset.time_report.authored_total_est_seconds | 19 |
| stage_report_subset.time_report.fits_rollout | yes |
| stage_report_subset.time_report.measured_vs_est_ratio | 0.7, 0.1, 0.5, 0.3, 0.3, 0.2, 0.2, 5.5 |
| stage_report_subset.time_report.worst_overrun.stage | 7 |
| stage_report_subset.time_report.worst_overrun.name | raise_and_stabilize_clear_of_table |
| stage_report_subset.time_report.worst_overrun.ratio | 5.5 |
| stage_report_subset.time_report.worst_overrun.measured_seconds | 13.87 |
| stage_report_subset.time_report.worst_overrun.est_seconds | 2.5 |
| stage_report_subset.contact_forces.units | N |
| stage_report_subset.contact_forces.object_contact.regions | thumb, index, middle, ring, little, palm |
| stage_report_subset.contact_forces.environment_contact.regions | thumb, index, middle, ring, little, palm |

### Training-metric reconstruction

| Arm/sub-run | Rows | Last iter | Last env steps | Elapsed s | Last success | Best success | Last lift max | Last base return | Best graded/objective |
|---|---|---|---|---|---|---|---|---|---|
| . | 1567 | 1566 | 40115200 | 21,608.473 | 0 | 0.300781 | — | 28.587599 | — |

These are training-log endpoints and maxima, not substitutes for held-out evaluation. They are included especially for interrupted runs whose terminal report is missing.

### Terminal and failure evidence from logs

- `run.log`: [done] 1567 fragments, best_iter=1524, eval objective 0.750821 success 0.5 -> runs/monotone_prior_6h_ppo_20260709-153033_restart/final_report.json

## What the results mean and major discoveries

- The arm comparison is most informative when arms shared the same environment, seed, and budget. Where the configuration records only one seed, apparent gaps remain vulnerable to optimization variance.
- Across terminal sub-run reports, `monotone_prior_6h_ppo_20260709-153033_restart` recorded the strongest eval_task_fitness (2.085886).
- The best terminal success measurement was `monotone_prior_6h_ppo_20260709-153033_restart` at 0.5 (eval_success_rate).

## Limitations and reading guidance

- Metrics are reported under the names used by the historical environment and runner. Similar-looking metrics from different generations of the code are not assumed to be numerically interchangeable.
- A maximum observed in training is not treated as held-out performance. Terminal evaluation values are preferred whenever present.
- Missing summaries, empty metric streams, failed retries, and incomplete arms are preserved as negative evidence rather than silently dropped.
- Candidate names and stage names are LLM-authored run artifacts. Interpretations in this historical log do not alter framework prompts, defaults, diagnostics, or task logic.

## Artifact provenance

| Artifact type | Count |
|---|---|
| .pkl | 17 |
| .json | 7 |
| .png | 6 |
| .csv | 2 |
| .mp4 | 2 |
| .jsonl | 1 |
| .log | 1 |
| .txt | 1 |

Primary evidence files:

- [`config.json`](../runs/monotone_prior_6h_ppo_20260709-153033_restart/config.json)
- [`final_report.json`](../runs/monotone_prior_6h_ppo_20260709-153033_restart/final_report.json)
- [`metrics.jsonl`](../runs/monotone_prior_6h_ppo_20260709-153033_restart/metrics.jsonl)
- [`plots/rollout_trace/summary.json`](../runs/monotone_prior_6h_ppo_20260709-153033_restart/plots/rollout_trace/summary.json)

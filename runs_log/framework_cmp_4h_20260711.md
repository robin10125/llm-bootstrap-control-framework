# Framework Cmp 4h

- **Run directory:** [`runs/framework_cmp_4h_20260711`](../runs/framework_cmp_4h_20260711)
- **Date:** 2026-07-11 (directory timestamp)
- **Status:** completed or completed-with-caveats
- **Experiment class:** training-framework comparison
- **Recorded task:** lift
- **Training metric rows recovered:** 7,883

## Abstract

This entry reconstructs `framework_cmp_4h_20260711`, a training-framework comparison. The account is based only on artifacts that survive in the repository: machine-readable configuration, candidate programs, selection diagnostics, training metrics, evaluations, reports, and logs. The run targeted lift. Its purpose and comparisons are inferred from the recorded arm names and configuration where no authored abstract survives; those inferences are identified by this provenance-first framing rather than presented as new experimental facts.

## Experimental structure

#### Configuration: `clocked_autopilot/config.json`

| Field | Recorded value |
|---|---|
| learner | experimental_clocked_paths_ppo |
| progression | autopilot |
| task | lift |
| seed | 0 |
| env.horizon | 800 |
| env.control_dt | 0.025 |
| env.episode_seconds | 20 |
| env.fragment_steps | 100 |
| env.fragments_per_episode | 8 |
| ppo.iters | 100000 |
| ppo.envs | 256 |
| ppo.eval_envs | 256 |
| ppo.fragment_steps | 100 |
| ppo.lr | 0.0003 |
| ppo.gamma | 0.99 |
| ppo.lam | 0.95 |
| ppo.hidden | 256, 256 |
| ppo.ent_coef | 0 |
| ppo.target_train_seconds | 14,400 |
| ppo.max_env_steps | — |
| ppo.checkpoint_every | 0 |
| ppo.eval_every | 25 |
| ppo.base_reward_weight | 1 |
| ppo.warmup_compile | yes |
| ppo.control_dt | 0.025 |
| ppo.default_est_seconds | 2 |
| ppo.progression | autopilot |
| ppo.rate_max | 3 |
| ppo.min_dwell_steps | 5 |
| ppo.dwell_slack | 3 |
| ppo.hint_dwell_steps | 3 |
| ppo.recover_threshold | 0.5 |
| ppo.max_recoveries | 2 |
| ppo.recover_cooldown_steps | 20 |
| ppo.residual_scale | 1 |
| ppo.potential_weight | 0.5 |
| ppo.imitation_coef | 0.5 |
| ppo.imitation_anneal_iters | — |

#### Configuration: `clocked_learned/config.json`

| Field | Recorded value |
|---|---|
| learner | experimental_clocked_paths_ppo |
| progression | learned |
| task | lift |
| seed | 0 |
| env.horizon | 800 |
| env.control_dt | 0.025 |
| env.episode_seconds | 20 |
| env.fragment_steps | 100 |
| env.fragments_per_episode | 8 |
| ppo.iters | 100000 |
| ppo.envs | 256 |
| ppo.eval_envs | 256 |
| ppo.fragment_steps | 100 |
| ppo.lr | 0.0003 |
| ppo.gamma | 0.99 |
| ppo.lam | 0.95 |
| ppo.hidden | 256, 256 |
| ppo.ent_coef | 0 |
| ppo.target_train_seconds | 14,400 |
| ppo.max_env_steps | — |
| ppo.checkpoint_every | 0 |
| ppo.eval_every | 25 |
| ppo.base_reward_weight | 1 |
| ppo.warmup_compile | yes |
| ppo.control_dt | 0.025 |
| ppo.default_est_seconds | 2 |
| ppo.progression | learned |
| ppo.rate_max | 3 |
| ppo.min_dwell_steps | 5 |
| ppo.dwell_slack | 3 |
| ppo.hint_dwell_steps | 3 |
| ppo.recover_threshold | 0.5 |
| ppo.max_recoveries | 2 |
| ppo.recover_cooldown_steps | 20 |
| ppo.residual_scale | 1 |
| ppo.potential_weight | 0.5 |
| ppo.imitation_coef | 0.5 |
| ppo.imitation_anneal_iters | 250 |

#### Configuration: `critic_features/config.json`

| Field | Recorded value |
|---|---|
| learner | experimental_alt_method_ppo |
| method | critic_features |
| task | lift |
| seed | 0 |
| env.horizon | 800 |
| env.control_dt | 0.025 |
| env.episode_seconds | 20 |
| env.fragment_steps | 100 |
| env.fragments_per_episode | 8 |
| ppo.method | critic_features |
| ppo.iters | 100000 |
| ppo.envs | 256 |
| ppo.eval_envs | 256 |
| ppo.fragment_steps | 100 |
| ppo.lr | 0.0003 |
| ppo.gamma | 0.99 |
| ppo.lam | 0.95 |
| ppo.hidden | 256, 256 |
| ppo.ent_coef | 0 |
| ppo.target_train_seconds | 14,400 |
| ppo.max_env_steps | — |
| ppo.checkpoint_every | 0 |
| ppo.eval_every | 25 |
| ppo.base_reward_weight | 1 |
| ppo.warmup_compile | yes |
| ppo.proposal_prob | 0.3 |
| ppo.proposal_prob_final | 0 |
| ppo.proposal_anneal_iters | — |
| ppo.proposal_sigma | 0.1 |
| ppo.proposal_gate | none |
| ppo.proposal_offpolicy | ratio |
| ppo.warmup_frac | 0.5 |
| ppo.warmup_frac_final | 0 |
| ppo.warmup_anneal_iters | — |
| ppo.warmup_mode | uniform |
| ppo.stage_reward_weight | 1 |
| ppo.stage_progress_weight | 1 |
| ppo.stage_completion_bonus | 0.05 |
| ppo.stage_success_temperature | 1 |
| ppo.stage_reward_clip | 0.5 |
| ppo.potential_weight | 0.5 |
| ppo.potential_temp | 1 |
| ppo.aux_coef | 0 |
| ppo.kl_coef | 1 |
| ppo.kl_coef_final | 0 |
| ppo.kl_anneal_iters | — |
| ppo.kl_sigma_ref | 0.3 |
| ppo.kl_ref_clip | 3 |
| ppo.kl_target | — |
| ppo.kl_target_final | — |
| ppo.critic_stage_onehot | yes |
| ppo.critic_prior_action | yes |
| ppo.critic_prior_norms | yes |
| ppo.critic_success_margins | yes |

#### Configuration: `curriculum/config.json`

| Field | Recorded value |
|---|---|
| learner | experimental_alt_method_ppo |
| method | curriculum |
| task | lift |
| seed | 0 |
| env.horizon | 800 |
| env.control_dt | 0.025 |
| env.episode_seconds | 20 |
| env.fragment_steps | 100 |
| env.fragments_per_episode | 8 |
| ppo.method | curriculum |
| ppo.iters | 100000 |
| ppo.envs | 256 |
| ppo.eval_envs | 256 |
| ppo.fragment_steps | 100 |
| ppo.lr | 0.0003 |
| ppo.gamma | 0.99 |
| ppo.lam | 0.95 |
| ppo.hidden | 256, 256 |
| ppo.ent_coef | 0 |
| ppo.target_train_seconds | 14,400 |
| ppo.max_env_steps | — |
| ppo.checkpoint_every | 0 |
| ppo.eval_every | 25 |
| ppo.base_reward_weight | 1 |
| ppo.warmup_compile | yes |
| ppo.proposal_prob | 0.3 |
| ppo.proposal_prob_final | 0 |
| ppo.proposal_anneal_iters | — |
| ppo.proposal_sigma | 0.1 |
| ppo.proposal_gate | none |
| ppo.proposal_offpolicy | ratio |
| ppo.warmup_frac | 0.5 |
| ppo.warmup_frac_final | 0 |
| ppo.warmup_anneal_iters | 800 |
| ppo.warmup_mode | uniform |
| ppo.stage_reward_weight | 1 |
| ppo.stage_progress_weight | 1 |
| ppo.stage_completion_bonus | 0.05 |
| ppo.stage_success_temperature | 1 |
| ppo.stage_reward_clip | 0.5 |
| ppo.potential_weight | 0.5 |
| ppo.potential_temp | 1 |
| ppo.aux_coef | 0 |
| ppo.kl_coef | 1 |
| ppo.kl_coef_final | 0 |
| ppo.kl_anneal_iters | — |
| ppo.kl_sigma_ref | 0.3 |
| ppo.kl_ref_clip | 3 |
| ppo.kl_target | — |
| ppo.kl_target_final | — |
| ppo.critic_stage_onehot | yes |
| ppo.critic_prior_action | yes |
| ppo.critic_prior_norms | yes |
| ppo.critic_success_margins | yes |

#### Configuration: `kl_prior/config.json`

| Field | Recorded value |
|---|---|
| learner | experimental_alt_method_ppo |
| method | kl_prior |
| task | lift |
| seed | 0 |
| env.horizon | 800 |
| env.control_dt | 0.025 |
| env.episode_seconds | 20 |
| env.fragment_steps | 100 |
| env.fragments_per_episode | 8 |
| ppo.method | kl_prior |
| ppo.iters | 100000 |
| ppo.envs | 256 |
| ppo.eval_envs | 256 |
| ppo.fragment_steps | 100 |
| ppo.lr | 0.0003 |
| ppo.gamma | 0.99 |
| ppo.lam | 0.95 |
| ppo.hidden | 256, 256 |
| ppo.ent_coef | 0 |
| ppo.target_train_seconds | 14,400 |
| ppo.max_env_steps | — |
| ppo.checkpoint_every | 0 |
| ppo.eval_every | 25 |
| ppo.base_reward_weight | 1 |
| ppo.warmup_compile | yes |
| ppo.proposal_prob | 0.3 |
| ppo.proposal_prob_final | 0 |
| ppo.proposal_anneal_iters | — |
| ppo.proposal_sigma | 0.1 |
| ppo.proposal_gate | none |
| ppo.proposal_offpolicy | ratio |
| ppo.warmup_frac | 0.5 |
| ppo.warmup_frac_final | 0 |
| ppo.warmup_anneal_iters | — |
| ppo.warmup_mode | uniform |
| ppo.stage_reward_weight | 1 |
| ppo.stage_progress_weight | 1 |
| ppo.stage_completion_bonus | 0.05 |
| ppo.stage_success_temperature | 1 |
| ppo.stage_reward_clip | 0.5 |
| ppo.potential_weight | 0.5 |
| ppo.potential_temp | 1 |
| ppo.aux_coef | 0 |
| ppo.kl_coef | 1 |
| ppo.kl_coef_final | 0 |
| ppo.kl_anneal_iters | 250 |
| ppo.kl_sigma_ref | 0.3 |
| ppo.kl_ref_clip | 3 |
| ppo.kl_target | — |
| ppo.kl_target_final | — |
| ppo.critic_stage_onehot | yes |
| ppo.critic_prior_action | yes |
| ppo.critic_prior_norms | yes |
| ppo.critic_success_margins | yes |

#### Configuration: `no_prior_control/config.json`

| Field | Recorded value |
|---|---|
| learner | experimental_alt_method_ppo |
| method | value_shaping |
| task | lift |
| seed | 0 |
| env.horizon | 800 |
| env.control_dt | 0.025 |
| env.episode_seconds | 20 |
| env.fragment_steps | 100 |
| env.fragments_per_episode | 8 |
| ppo.method | value_shaping |
| ppo.iters | 100000 |
| ppo.envs | 256 |
| ppo.eval_envs | 256 |
| ppo.fragment_steps | 100 |
| ppo.lr | 0.0003 |
| ppo.gamma | 0.99 |
| ppo.lam | 0.95 |
| ppo.hidden | 256, 256 |
| ppo.ent_coef | 0 |
| ppo.target_train_seconds | 14,400 |
| ppo.max_env_steps | — |
| ppo.checkpoint_every | 0 |
| ppo.eval_every | 25 |
| ppo.base_reward_weight | 1 |
| ppo.warmup_compile | yes |
| ppo.proposal_prob | 0.3 |
| ppo.proposal_prob_final | 0 |
| ppo.proposal_anneal_iters | — |
| ppo.proposal_sigma | 0.1 |
| ppo.proposal_gate | none |
| ppo.proposal_offpolicy | ratio |
| ppo.warmup_frac | 0.5 |
| ppo.warmup_frac_final | 0 |
| ppo.warmup_anneal_iters | — |
| ppo.warmup_mode | uniform |
| ppo.stage_reward_weight | 0 |
| ppo.stage_progress_weight | 1 |
| ppo.stage_completion_bonus | 0.05 |
| ppo.stage_success_temperature | 1 |
| ppo.stage_reward_clip | 0.5 |
| ppo.potential_weight | 0 |
| ppo.potential_temp | 1 |
| ppo.aux_coef | 0 |
| ppo.kl_coef | 1 |
| ppo.kl_coef_final | 0 |
| ppo.kl_anneal_iters | — |
| ppo.kl_sigma_ref | 0.3 |
| ppo.kl_ref_clip | 3 |
| ppo.kl_target | — |
| ppo.kl_target_final | — |
| ppo.critic_stage_onehot | yes |
| ppo.critic_prior_action | yes |
| ppo.critic_prior_norms | yes |
| ppo.critic_success_margins | yes |

#### Configuration: `proposal/config.json`

| Field | Recorded value |
|---|---|
| learner | experimental_alt_method_ppo |
| method | proposal |
| task | lift |
| seed | 0 |
| env.horizon | 800 |
| env.control_dt | 0.025 |
| env.episode_seconds | 20 |
| env.fragment_steps | 100 |
| env.fragments_per_episode | 8 |
| ppo.method | proposal |
| ppo.iters | 100000 |
| ppo.envs | 256 |
| ppo.eval_envs | 256 |
| ppo.fragment_steps | 100 |
| ppo.lr | 0.0003 |
| ppo.gamma | 0.99 |
| ppo.lam | 0.95 |
| ppo.hidden | 256, 256 |
| ppo.ent_coef | 0 |
| ppo.target_train_seconds | 14,400 |
| ppo.max_env_steps | — |
| ppo.checkpoint_every | 0 |
| ppo.eval_every | 25 |
| ppo.base_reward_weight | 1 |
| ppo.warmup_compile | yes |
| ppo.proposal_prob | 0.3 |
| ppo.proposal_prob_final | 0 |
| ppo.proposal_anneal_iters | 800 |
| ppo.proposal_sigma | 0.1 |
| ppo.proposal_gate | none |
| ppo.proposal_offpolicy | ratio |
| ppo.warmup_frac | 0.5 |
| ppo.warmup_frac_final | 0 |
| ppo.warmup_anneal_iters | — |
| ppo.warmup_mode | uniform |
| ppo.stage_reward_weight | 1 |
| ppo.stage_progress_weight | 1 |
| ppo.stage_completion_bonus | 0.05 |
| ppo.stage_success_temperature | 1 |
| ppo.stage_reward_clip | 0.5 |
| ppo.potential_weight | 0.5 |
| ppo.potential_temp | 1 |
| ppo.aux_coef | 0 |
| ppo.kl_coef | 1 |
| ppo.kl_coef_final | 0 |
| ppo.kl_anneal_iters | — |
| ppo.kl_sigma_ref | 0.3 |
| ppo.kl_ref_clip | 3 |
| ppo.kl_target | — |
| ppo.kl_target_final | — |
| ppo.critic_stage_onehot | yes |
| ppo.critic_prior_action | yes |
| ppo.critic_prior_norms | yes |
| ppo.critic_success_margins | yes |

#### Configuration: `value_shaping/config.json`

| Field | Recorded value |
|---|---|
| learner | experimental_alt_method_ppo |
| method | value_shaping |
| task | lift |
| seed | 0 |
| env.horizon | 800 |
| env.control_dt | 0.025 |
| env.episode_seconds | 20 |
| env.fragment_steps | 100 |
| env.fragments_per_episode | 8 |
| ppo.method | value_shaping |
| ppo.iters | 100000 |
| ppo.envs | 256 |
| ppo.eval_envs | 256 |
| ppo.fragment_steps | 100 |
| ppo.lr | 0.0003 |
| ppo.gamma | 0.99 |
| ppo.lam | 0.95 |
| ppo.hidden | 256, 256 |
| ppo.ent_coef | 0 |
| ppo.target_train_seconds | 14,400 |
| ppo.max_env_steps | — |
| ppo.checkpoint_every | 0 |
| ppo.eval_every | 25 |
| ppo.base_reward_weight | 1 |
| ppo.warmup_compile | yes |
| ppo.proposal_prob | 0.3 |
| ppo.proposal_prob_final | 0 |
| ppo.proposal_anneal_iters | — |
| ppo.proposal_sigma | 0.1 |
| ppo.proposal_gate | none |
| ppo.proposal_offpolicy | ratio |
| ppo.warmup_frac | 0.5 |
| ppo.warmup_frac_final | 0 |
| ppo.warmup_anneal_iters | — |
| ppo.warmup_mode | uniform |
| ppo.stage_reward_weight | 1 |
| ppo.stage_progress_weight | 1 |
| ppo.stage_completion_bonus | 0.05 |
| ppo.stage_success_temperature | 1 |
| ppo.stage_reward_clip | 0.5 |
| ppo.potential_weight | 0.5 |
| ppo.potential_temp | 1 |
| ppo.aux_coef | 0.1 |
| ppo.kl_coef | 1 |
| ppo.kl_coef_final | 0 |
| ppo.kl_anneal_iters | — |
| ppo.kl_sigma_ref | 0.3 |
| ppo.kl_ref_clip | 3 |
| ppo.kl_target | — |
| ppo.kl_target_final | — |
| ppo.critic_stage_onehot | yes |
| ppo.critic_prior_action | yes |
| ppo.critic_prior_norms | yes |
| ppo.critic_success_margins | yes |

### Arms and sub-runs represented by directories

`clocked_autopilot`, `clocked_learned`, `critic_features`, `curriculum`, `kl_prior`, `no_prior_control`, `plots`, `proposal`, `value_shaping`

## Results

### Recorded results: `clocked_learned/final_report.json`

| Run result | Value |
|---|---|
| learner | experimental_clocked_paths_ppo |
| progression | learned |
| iters | 884 |
| best_iter | 399 |
| best_objective | 1.750121 |
| eval_objective | 1.265629 |
| eval_graded_objective | 1.265629 |
| eval_task_fitness | 1.717476 |
| eval_success_rate | 0.398438 |

| Evaluation metric | Value |
|---|---|
| eval_success_rate | 0.398438 |
| eval_task_fitness | 1.717476 |
| eval_reach_rate | 1 |
| eval_grasp_rate | 0.84375 |
| eval_grasp_lift_rate | 0.328125 |
| eval_lift_reached_rate | 0.398438 |
| eval_lift_max | 0.050641 |
| eval_train_return | 42.313862 |
| eval_base_return | 43.159407 |
| eval_shaping_return | -0.845544 |
| eval_forced_advance_frac | 0.001538 |
| eval_recover_frac | 0 |
| eval_graded_objective | 1.265629 |

### Recorded results: `critic_features/final_report.json`

| Run result | Value |
|---|---|
| method | critic_features |
| iters | 1025 |
| best_iter | 1024 |
| best_objective | 2.147188 |
| eval_objective | 1.517932 |
| eval_graded_objective | 1.517932 |
| eval_task_fitness | 2.088544 |
| eval_success_rate | 0.519531 |

| Evaluation metric | Value |
|---|---|
| eval_success_rate | 0.519531 |
| eval_task_fitness | 2.088544 |
| eval_reach_rate | 1 |
| eval_grasp_rate | 0.992188 |
| eval_grasp_lift_rate | 0.519531 |
| eval_lift_reached_rate | 0.519531 |
| eval_lift_max | 0.060793 |
| eval_train_return | 443.215 |
| eval_base_return | 443.215 |
| eval_shaping_return | 0 |
| eval_prior_disagreement | 0.469618 |
| eval_graded_objective | 1.517932 |

### Recorded results: `curriculum/final_report.json`

| Run result | Value |
|---|---|
| method | curriculum |
| iters | 1028 |
| best_iter | 474 |
| best_objective | 1.623383 |
| eval_objective | 1.00513 |
| eval_graded_objective | 1.00513 |
| eval_task_fitness | 1.56215 |
| eval_success_rate | 0.175781 |

| Evaluation metric | Value |
|---|---|
| eval_success_rate | 0.175781 |
| eval_task_fitness | 1.56215 |
| eval_reach_rate | 1 |
| eval_grasp_rate | 0.914062 |
| eval_grasp_lift_rate | 0.128906 |
| eval_lift_reached_rate | 0.175781 |
| eval_lift_max | 0.03906 |
| eval_train_return | 166.739 |
| eval_base_return | 166.739 |
| eval_shaping_return | 0 |
| eval_prior_disagreement | 0.417536 |
| eval_graded_objective | 1.00513 |

### Recorded results: `kl_prior/final_report.json`

| Run result | Value |
|---|---|
| method | kl_prior |
| iters | 1034 |
| best_iter | 699 |
| best_objective | 1.685467 |
| eval_objective | 0.507275 |
| eval_graded_objective | 0.507275 |
| eval_task_fitness | 1.665138 |
| eval_success_rate | 0.035156 |

| Evaluation metric | Value |
|---|---|
| eval_success_rate | 0.035156 |
| eval_task_fitness | 1.665138 |
| eval_reach_rate | 1 |
| eval_grasp_rate | 0.363281 |
| eval_grasp_lift_rate | 0.007812 |
| eval_lift_reached_rate | 0.035156 |
| eval_lift_max | 0.017889 |
| eval_train_return | 277.351 |
| eval_base_return | 277.351 |
| eval_shaping_return | 0 |
| eval_prior_disagreement | 0.195155 |
| eval_graded_objective | 0.507275 |

### Recorded results: `no_prior_control/final_report.json`

| Run result | Value |
|---|---|
| method | value_shaping |
| iters | 1000 |
| best_iter | 374 |
| best_objective | 1.560515 |
| eval_objective | 0.764635 |
| eval_graded_objective | 0.764635 |
| eval_task_fitness | 1.594346 |
| eval_success_rate | 0.226562 |

| Evaluation metric | Value |
|---|---|
| eval_success_rate | 0.226562 |
| eval_task_fitness | 1.594346 |
| eval_reach_rate | 1 |
| eval_grasp_rate | 0.382812 |
| eval_grasp_lift_rate | 0.105469 |
| eval_lift_reached_rate | 0.226562 |
| eval_lift_max | 0.04278 |
| eval_train_return | 46.10435 |
| eval_base_return | 46.10435 |
| eval_shaping_return | 0 |
| eval_prior_disagreement | 0.307899 |
| eval_graded_objective | 0.764635 |

### Recorded results: `proposal/final_report.json`

| Run result | Value |
|---|---|
| method | proposal |
| iters | 519 |
| best_iter | 174 |
| best_objective | 1.476086 |
| eval_objective | 0.912016 |
| eval_graded_objective | 0.912016 |
| eval_task_fitness | 1.45381 |
| eval_success_rate | 0.320312 |

| Evaluation metric | Value |
|---|---|
| eval_success_rate | 0.320312 |
| eval_task_fitness | 1.45381 |
| eval_reach_rate | 1 |
| eval_grasp_rate | 0.429688 |
| eval_grasp_lift_rate | 0.113281 |
| eval_lift_reached_rate | 0.320312 |
| eval_lift_max | 0.052225 |
| eval_train_return | 15.070441 |
| eval_base_return | 15.070441 |
| eval_shaping_return | 0 |
| eval_prior_disagreement | 0.375903 |
| eval_graded_objective | 0.912016 |

### Recorded results: `value_shaping/final_report.json`

| Run result | Value |
|---|---|
| method | value_shaping |
| iters | 1049 |
| best_iter | 899 |
| best_objective | 1.648404 |
| eval_objective | 0.553824 |
| eval_graded_objective | 0.553824 |
| eval_task_fitness | 1.667422 |
| eval_success_rate | 0 |

| Evaluation metric | Value |
|---|---|
| eval_success_rate | 0 |
| eval_task_fitness | 1.667422 |
| eval_reach_rate | 1 |
| eval_grasp_rate | 0.5625 |
| eval_grasp_lift_rate | 0 |
| eval_lift_reached_rate | 0 |
| eval_lift_max | 0.014451 |
| eval_train_return | 365.512 |
| eval_base_return | 366.056 |
| eval_shaping_return | -0.544702 |
| eval_prior_disagreement | 0.43036 |
| eval_graded_objective | 0.553824 |

### Training-metric reconstruction

| Arm/sub-run | Rows | Last iter | Last env steps | Elapsed s | Last success | Best success | Last lift max | Last base return | Best graded/objective |
|---|---|---|---|---|---|---|---|---|---|
| clocked_autopilot | 1344 | 671 | 17203200 | 10,989.583 | 0.074219 | 0.195312 | — | 20.430902 | 1.678691 |
| clocked_learned | 884 | 883 | 22630400 | 14,404.719 | 0.007812 | 0.261719 | — | 29.487652 | 1.750121 |
| critic_features | 1025 | 1024 | 26240000 | 14,383.516 | 0.007812 | 0.277344 | — | 10.653296 | 2.147188 |
| curriculum | 1028 | 1027 | 26316800 | 14,402.585 | 0.011719 | 0.136719 | — | 33.354813 | 1.623383 |
| kl_prior | 1034 | 1033 | 26470400 | 14,409.955 | 0.007812 | 0.121094 | — | 43.194984 | 1.685467 |
| no_prior_control | 1000 | 999 | 25600000 | 14,343.865 | 0.003906 | 0.183594 | — | 23.94421 | 1.560515 |
| proposal | 519 | 518 | 13286400 | 14,458.96 | 0 | 0.3125 | — | nan | 1.476086 |
| value_shaping | 1049 | 1048 | 26854400 | 14,408.221 | 0.015625 | 0.167969 | — | 30.183231 | 1.648404 |

These are training-log endpoints and maxima, not substitutes for held-out evaluation. They are included especially for interrupted runs whose terminal report is missing.

### Terminal and failure evidence from logs

- `baseline.log`: Traceback (most recent call last):
- `baseline.log`: ModuleNotFoundError: No module named 'policy_bias_lab'
- `clocked_learned.log`: [done] progression=learned 884 fragments, best_iter=399, graded_objective=1.265629 success=0.398438 -> runs/framework_cmp_4h_20260711/clocked_learned
- `critic_features.log`: [done] method=critic_features 1025 fragments, best_iter=1024, graded_objective=1.517932 success=0.519531 -> runs/framework_cmp_4h_20260711/critic_features
- `curriculum.log`: [done] method=curriculum 1028 fragments, best_iter=474, graded_objective=1.00513 success=0.175781 -> runs/framework_cmp_4h_20260711/curriculum
- `kl_prior.log`: [done] method=kl_prior 1034 fragments, best_iter=699, graded_objective=0.507275 success=0.035156 -> runs/framework_cmp_4h_20260711/kl_prior
- `no_prior_control.log`: [done] method=value_shaping 1000 fragments, best_iter=374, graded_objective=0.764635 success=0.226562 -> runs/framework_cmp_4h_20260711/no_prior_control
- `proposal.log`: [done] method=proposal 519 fragments, best_iter=174, graded_objective=0.912016 success=0.320312 -> runs/framework_cmp_4h_20260711/proposal
- `value_shaping.log`: [done] method=value_shaping 1049 fragments, best_iter=899, graded_objective=0.553824 success=0.0 -> runs/framework_cmp_4h_20260711/value_shaping

## What the results mean and major discoveries

- Across terminal sub-run reports, `critic_features` recorded the strongest eval_task_fitness (2.088544).
- The best terminal success measurement was `critic_features` at 0.519531 (eval_success_rate).

## Limitations and reading guidance

- Metrics are reported under the names used by the historical environment and runner. Similar-looking metrics from different generations of the code are not assumed to be numerically interchangeable.
- A maximum observed in training is not treated as held-out performance. Terminal evaluation values are preferred whenever present.
- Missing summaries, empty metric streams, failed retries, and incomplete arms are preserved as negative evidence rather than silently dropped.
- Candidate names and stage names are LLM-authored run artifacts. Interpretations in this historical log do not alter framework prompts, defaults, diagnostics, or task logic.

## Artifact provenance

| Artifact type | Count |
|---|---|
| .json | 23 |
| .pkl | 15 |
| .jsonl | 9 |
| .log | 9 |
| .png | 2 |

Primary evidence files:

- [`clocked_autopilot/config.json`](../runs/framework_cmp_4h_20260711/clocked_autopilot/config.json)
- [`clocked_autopilot/metrics.jsonl`](../runs/framework_cmp_4h_20260711/clocked_autopilot/metrics.jsonl)
- [`clocked_learned/config.json`](../runs/framework_cmp_4h_20260711/clocked_learned/config.json)
- [`clocked_learned/final_report.json`](../runs/framework_cmp_4h_20260711/clocked_learned/final_report.json)
- [`clocked_learned/metrics.jsonl`](../runs/framework_cmp_4h_20260711/clocked_learned/metrics.jsonl)
- [`critic_features/config.json`](../runs/framework_cmp_4h_20260711/critic_features/config.json)
- [`critic_features/final_report.json`](../runs/framework_cmp_4h_20260711/critic_features/final_report.json)
- [`critic_features/metrics.jsonl`](../runs/framework_cmp_4h_20260711/critic_features/metrics.jsonl)
- [`curriculum/config.json`](../runs/framework_cmp_4h_20260711/curriculum/config.json)
- [`curriculum/final_report.json`](../runs/framework_cmp_4h_20260711/curriculum/final_report.json)
- [`curriculum/metrics.jsonl`](../runs/framework_cmp_4h_20260711/curriculum/metrics.jsonl)
- [`kl_prior/config.json`](../runs/framework_cmp_4h_20260711/kl_prior/config.json)
- [`kl_prior/final_report.json`](../runs/framework_cmp_4h_20260711/kl_prior/final_report.json)
- [`kl_prior/metrics.jsonl`](../runs/framework_cmp_4h_20260711/kl_prior/metrics.jsonl)
- [`no_prior_control/config.json`](../runs/framework_cmp_4h_20260711/no_prior_control/config.json)
- [`no_prior_control/final_report.json`](../runs/framework_cmp_4h_20260711/no_prior_control/final_report.json)
- [`no_prior_control/metrics.jsonl`](../runs/framework_cmp_4h_20260711/no_prior_control/metrics.jsonl)
- [`proposal/config.json`](../runs/framework_cmp_4h_20260711/proposal/config.json)
- [`proposal/final_report.json`](../runs/framework_cmp_4h_20260711/proposal/final_report.json)
- [`proposal/metrics.jsonl`](../runs/framework_cmp_4h_20260711/proposal/metrics.jsonl)
- [`value_shaping/config.json`](../runs/framework_cmp_4h_20260711/value_shaping/config.json)
- [`value_shaping/final_report.json`](../runs/framework_cmp_4h_20260711/value_shaping/final_report.json)
- [`value_shaping/metrics.jsonl`](../runs/framework_cmp_4h_20260711/value_shaping/metrics.jsonl)

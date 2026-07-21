# Critic Useinfo 1h

- **Run directory:** [`runs/critic_useinfo_1h_20260713`](../runs/critic_useinfo_1h_20260713)
- **Date:** 2026-07-13 (directory timestamp)
- **Status:** completed or completed-with-caveats
- **Experiment class:** critic-feature study
- **Recorded task:** lift
- **Training metric rows recovered:** 1,483

## Abstract

This entry reconstructs `critic_useinfo_1h_20260713`, a critic-feature study. The account is based only on artifacts that survive in the repository: machine-readable configuration, candidate programs, selection diagnostics, training metrics, evaluations, reports, and logs. The run targeted lift. Its purpose and comparisons are inferred from the recorded arm names and configuration where no authored abstract survives; those inferences are identified by this provenance-first framing rather than presented as new experimental facts.

## Experimental structure

#### Configuration: `nouse_s0/config.json`

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
| ppo.target_train_seconds | 3,600 |
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

#### Configuration: `nouse_s1/config.json`

| Field | Recorded value |
|---|---|
| learner | experimental_alt_method_ppo |
| method | critic_features |
| task | lift |
| seed | 1 |
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
| ppo.target_train_seconds | 3,600 |
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

#### Configuration: `nouse_s2/config.json`

| Field | Recorded value |
|---|---|
| learner | experimental_alt_method_ppo |
| method | critic_features |
| task | lift |
| seed | 2 |
| extra_programs |  |
| critical_stages | — |
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
| ppo.target_train_seconds | 3,600 |
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
| ppo.critic_gate_values | no |
| ppo.critic_critical_actions | no |
| ppo.critical_stage_keywords | grasp, lift |

#### Configuration: `use_s0/config.json`

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
| ppo.target_train_seconds | 3,600 |
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

#### Configuration: `use_s1/config.json`

| Field | Recorded value |
|---|---|
| learner | experimental_alt_method_ppo |
| method | critic_features |
| task | lift |
| seed | 1 |
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
| ppo.target_train_seconds | 3,600 |
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

#### Configuration: `use_s2/config.json`

| Field | Recorded value |
|---|---|
| learner | experimental_alt_method_ppo |
| method | critic_features |
| task | lift |
| seed | 2 |
| extra_programs |  |
| critical_stages | — |
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
| ppo.target_train_seconds | 3,600 |
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
| ppo.critic_gate_values | no |
| ppo.critic_critical_actions | no |
| ppo.critical_stage_keywords | grasp, lift |

### Arms and sub-runs represented by directories

`nouse_s0`, `nouse_s1`, `nouse_s2`, `use_s0`, `use_s1`, `use_s2`

## Results

### Recorded results: `nouse_s0/final_report.json`

| Run result | Value |
|---|---|
| method | critic_features |
| iters | 249 |
| best_iter | 224 |
| best_objective | 1.463654 |
| eval_objective | 0.888854 |
| eval_graded_objective | 0.888854 |
| eval_task_fitness | 1.451514 |
| eval_success_rate | 0.101562 |

| Evaluation metric | Value |
|---|---|
| eval_success_rate | 0.101562 |
| eval_task_fitness | 1.451514 |
| eval_reach_rate | 1 |
| eval_grasp_rate | 0.914062 |
| eval_grasp_lift_rate | 0.085938 |
| eval_lift_reached_rate | 0.101562 |
| eval_lift_max | 0.030173 |
| eval_train_return | 64.985637 |
| eval_base_return | 64.985637 |
| eval_shaping_return | 0 |
| eval_prior_disagreement | 0.258852 |
| eval_graded_objective | 0.888854 |

### Recorded results: `nouse_s1/final_report.json`

| Run result | Value |
|---|---|
| method | critic_features |
| iters | 250 |
| best_iter | 224 |
| best_objective | 1.413873 |
| eval_objective | 0.831399 |
| eval_graded_objective | 0.831399 |
| eval_task_fitness | 1.386372 |
| eval_success_rate | 0.132812 |

| Evaluation metric | Value |
|---|---|
| eval_success_rate | 0.132812 |
| eval_task_fitness | 1.386372 |
| eval_reach_rate | 1 |
| eval_grasp_rate | 0.699219 |
| eval_grasp_lift_rate | 0.085938 |
| eval_lift_reached_rate | 0.132812 |
| eval_lift_max | 0.034109 |
| eval_train_return | 64.698383 |
| eval_base_return | 64.698383 |
| eval_shaping_return | 0 |
| eval_prior_disagreement | 0.260646 |
| eval_graded_objective | 0.831399 |

### Recorded results: `nouse_s2/final_report.json`

| Run result | Value |
|---|---|
| method | critic_features |
| iters | 245 |
| best_iter | 174 |
| best_objective | 1.323177 |
| eval_objective | 0.864617 |
| eval_graded_objective | 0.864617 |
| eval_task_fitness | 1.366723 |
| eval_success_rate | 0.144531 |

| Evaluation metric | Value |
|---|---|
| eval_success_rate | 0.144531 |
| eval_task_fitness | 1.366723 |
| eval_reach_rate | 1 |
| eval_grasp_rate | 0.722656 |
| eval_grasp_lift_rate | 0.058594 |
| eval_lift_reached_rate | 0.144531 |
| eval_lift_max | 0.036068 |
| eval_train_return | 42.004196 |
| eval_base_return | 42.004196 |
| eval_shaping_return | 0 |
| eval_prior_disagreement | 0.229561 |
| eval_graded_objective | 0.864617 |

### Recorded results: `use_s0/final_report.json`

| Run result | Value |
|---|---|
| method | critic_features |
| iters | 244 |
| best_iter | 124 |
| best_objective | 1.471636 |
| eval_objective | 0.915867 |
| eval_graded_objective | 0.915867 |
| eval_task_fitness | 1.479123 |
| eval_success_rate | 0.167969 |

| Evaluation metric | Value |
|---|---|
| eval_success_rate | 0.167969 |
| eval_task_fitness | 1.479123 |
| eval_reach_rate | 1 |
| eval_grasp_rate | 0.761719 |
| eval_grasp_lift_rate | 0.109375 |
| eval_lift_reached_rate | 0.167969 |
| eval_lift_max | 0.039269 |
| eval_train_return | 32.16772 |
| eval_base_return | 32.16772 |
| eval_shaping_return | 0 |
| eval_prior_disagreement | 0.17427 |
| eval_graded_objective | 0.915867 |

### Recorded results: `use_s1/final_report.json`

| Run result | Value |
|---|---|
| method | critic_features |
| iters | 248 |
| best_iter | 99 |
| best_objective | 1.421242 |
| eval_objective | 0.476112 |
| eval_graded_objective | 0.476112 |
| eval_task_fitness | 1.344054 |
| eval_success_rate | 0.191406 |

| Evaluation metric | Value |
|---|---|
| eval_success_rate | 0.191406 |
| eval_task_fitness | 1.344054 |
| eval_reach_rate | 1 |
| eval_grasp_rate | 0.046875 |
| eval_grasp_lift_rate | 0.015625 |
| eval_lift_reached_rate | 0.191406 |
| eval_lift_max | 0.038673 |
| eval_train_return | 25.087252 |
| eval_base_return | 25.087252 |
| eval_shaping_return | 0 |
| eval_prior_disagreement | 0.160718 |
| eval_graded_objective | 0.476112 |

### Recorded results: `use_s2/final_report.json`

| Run result | Value |
|---|---|
| method | critic_features |
| iters | 247 |
| best_iter | 224 |
| best_objective | 1.379538 |
| eval_objective | 0.926212 |
| eval_graded_objective | 0.926212 |
| eval_task_fitness | 1.41975 |
| eval_success_rate | 0.105469 |

| Evaluation metric | Value |
|---|---|
| eval_success_rate | 0.105469 |
| eval_task_fitness | 1.41975 |
| eval_reach_rate | 1 |
| eval_grasp_rate | 0.980469 |
| eval_grasp_lift_rate | 0.09375 |
| eval_lift_reached_rate | 0.105469 |
| eval_lift_max | 0.029632 |
| eval_train_return | 37.262847 |
| eval_base_return | 37.262847 |
| eval_shaping_return | 0 |
| eval_prior_disagreement | 0.243313 |
| eval_graded_objective | 0.926212 |

### Training-metric reconstruction

| Arm/sub-run | Rows | Last iter | Last env steps | Elapsed s | Last success | Best success | Last lift max | Last base return | Best graded/objective |
|---|---|---|---|---|---|---|---|---|---|
| nouse_s0 | 249 | 248 | 6374400 | 3,602.408 | 0.0625 | 0.171875 | — | 12.365909 | 1.463654 |
| nouse_s1 | 250 | 249 | 6400000 | 3,569.55 | 0.023438 | 0.15625 | — | 7.937101 | 1.413873 |
| nouse_s2 | 245 | 244 | 6272000 | 3,600.098 | 0.007812 | 0.160156 | — | 3.16818 | 1.323177 |
| use_s0 | 244 | 243 | 6246400 | 3,609.936 | 0.007812 | 0.160156 | — | 3.344837 | 1.471636 |
| use_s1 | 248 | 247 | 6348800 | 3,607.836 | 0 | 0.175781 | — | 1.347265 | 1.421242 |
| use_s2 | 247 | 246 | 6323200 | 3,604.737 | 0 | 0.160156 | — | 1.321555 | 1.379538 |

These are training-log endpoints and maxima, not substitutes for held-out evaluation. They are included especially for interrupted runs whose terminal report is missing.

### Terminal and failure evidence from logs

- `nouse_s0.log`: [done] method=critic_features 249 fragments, best_iter=224, graded_objective=0.888854 success=0.101562 -> runs/critic_useinfo_1h_20260713/nouse_s0
- `nouse_s1.log`: [done] method=critic_features 250 fragments, best_iter=224, graded_objective=0.831399 success=0.132812 -> runs/critic_useinfo_1h_20260713/nouse_s1
- `nouse_s2.log`: [done] method=critic_features 245 fragments, best_iter=174, graded_objective=0.864617 success=0.144531 -> runs/critic_useinfo_1h_20260713/nouse_s2
- `use_s0.log`: [done] method=critic_features 244 fragments, best_iter=124, graded_objective=0.915867 success=0.167969 -> runs/critic_useinfo_1h_20260713/use_s0
- `use_s1.log`: [done] method=critic_features 248 fragments, best_iter=99, graded_objective=0.476112 success=0.191406 -> runs/critic_useinfo_1h_20260713/use_s1
- `use_s2.log`: [done] method=critic_features 247 fragments, best_iter=224, graded_objective=0.926212 success=0.105469 -> runs/critic_useinfo_1h_20260713/use_s2

## What the results mean and major discoveries

- Across terminal sub-run reports, `use_s0` recorded the strongest eval_task_fitness (1.479123).
- The best terminal success measurement was `use_s1` at 0.191406 (eval_success_rate).

## Limitations and reading guidance

- Metrics are reported under the names used by the historical environment and runner. Similar-looking metrics from different generations of the code are not assumed to be numerically interchangeable.
- A maximum observed in training is not treated as held-out performance. Terminal evaluation values are preferred whenever present.
- Missing summaries, empty metric streams, failed retries, and incomplete arms are preserved as negative evidence rather than silently dropped.
- Candidate names and stage names are LLM-authored run artifacts. Interpretations in this historical log do not alter framework prompts, defaults, diagnostics, or task logic.

## Artifact provenance

| Artifact type | Count |
|---|---|
| .json | 18 |
| .pkl | 12 |
| .log | 7 |
| .jsonl | 6 |
| .md | 1 |

Primary evidence files:

- [`nouse_s0/config.json`](../runs/critic_useinfo_1h_20260713/nouse_s0/config.json)
- [`nouse_s0/final_report.json`](../runs/critic_useinfo_1h_20260713/nouse_s0/final_report.json)
- [`nouse_s0/metrics.jsonl`](../runs/critic_useinfo_1h_20260713/nouse_s0/metrics.jsonl)
- [`nouse_s1/config.json`](../runs/critic_useinfo_1h_20260713/nouse_s1/config.json)
- [`nouse_s1/final_report.json`](../runs/critic_useinfo_1h_20260713/nouse_s1/final_report.json)
- [`nouse_s1/metrics.jsonl`](../runs/critic_useinfo_1h_20260713/nouse_s1/metrics.jsonl)
- [`nouse_s2/config.json`](../runs/critic_useinfo_1h_20260713/nouse_s2/config.json)
- [`nouse_s2/final_report.json`](../runs/critic_useinfo_1h_20260713/nouse_s2/final_report.json)
- [`nouse_s2/metrics.jsonl`](../runs/critic_useinfo_1h_20260713/nouse_s2/metrics.jsonl)
- [`summary.md`](../runs/critic_useinfo_1h_20260713/summary.md)
- [`use_s0/config.json`](../runs/critic_useinfo_1h_20260713/use_s0/config.json)
- [`use_s0/final_report.json`](../runs/critic_useinfo_1h_20260713/use_s0/final_report.json)
- [`use_s0/metrics.jsonl`](../runs/critic_useinfo_1h_20260713/use_s0/metrics.jsonl)
- [`use_s1/config.json`](../runs/critic_useinfo_1h_20260713/use_s1/config.json)
- [`use_s1/final_report.json`](../runs/critic_useinfo_1h_20260713/use_s1/final_report.json)
- [`use_s1/metrics.jsonl`](../runs/critic_useinfo_1h_20260713/use_s1/metrics.jsonl)
- [`use_s2/config.json`](../runs/critic_useinfo_1h_20260713/use_s2/config.json)
- [`use_s2/final_report.json`](../runs/critic_useinfo_1h_20260713/use_s2/final_report.json)
- [`use_s2/metrics.jsonl`](../runs/critic_useinfo_1h_20260713/use_s2/metrics.jsonl)

## Appendix: original authored run report

The following contemporaneous report is retained (with headings demoted) so that details and caveats are not lost in normalization.

### Source: [`summary.md`](../runs/critic_useinfo_1h_20260713/summary.md)

## critic_features prior: use-info vs no-use-info (1h x 3 seeds)

| run | graded | success | fitness | best_iter/iters |
|---|---:|---:|---:|---:|
| nouse_s0 | 0.888854 | 0.101562 | 1.451514 | 224/249 |
| nouse_s1 | 0.831399 | 0.132812 | 1.386372 | 224/250 |
| nouse_s2 | 0.864617 | 0.144531 | 1.366723 | 174/245 |
| use_s0 | 0.915867 | 0.167969 | 1.479123 | 124/244 |
| use_s1 | 0.476112 | 0.191406 | 1.344054 | 99/248 |
| use_s2 | 0.926212 | 0.105469 | 1.41975 | 224/247 |

**nouse** (n=3): graded mean 0.862 (min 0.831, max 0.889); success mean 0.126 (min 0.102, max 0.145)

**use** (n=3): graded mean 0.773 (min 0.476, max 0.926); success mean 0.155 (min 0.105, max 0.191)

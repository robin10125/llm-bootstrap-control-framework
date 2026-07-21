# Prior Gen Study

- **Run directory:** [`runs/prior_gen_study_20260711`](../runs/prior_gen_study_20260711)
- **Date:** 2026-07-11 (directory timestamp)
- **Status:** completed or completed-with-caveats
- **Experiment class:** PPO bias/reward/prior experiment
- **Recorded task:** lift
- **Training metric rows recovered:** 2,205

## Abstract

This entry reconstructs `prior_gen_study_20260711`, a PPO bias/reward/prior experiment. The account is based only on artifacts that survive in the repository: machine-readable configuration, candidate programs, selection diagnostics, training metrics, evaluations, reports, and logs. The run targeted lift. Its purpose and comparisons are inferred from the recorded arm names and configuration where no authored abstract survives; those inferences are identified by this provenance-first framing rather than presented as new experimental facts.

## Experimental structure

#### Configuration: `run_clocked_nouse/config.json`

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
| ppo.target_train_seconds | 5,400 |
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
| ppo.imitation_anneal_iters | 350 |

#### Configuration: `run_clocked_use/config.json`

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
| ppo.target_train_seconds | 5,400 |
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
| ppo.imitation_anneal_iters | 350 |

#### Configuration: `run_kl_complex/config.json`

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
| ppo.target_train_seconds | 5,400 |
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
| ppo.kl_anneal_iters | 350 |
| ppo.kl_sigma_ref | 0.3 |
| ppo.kl_ref_clip | 3 |
| ppo.kl_target | — |
| ppo.kl_target_final | — |
| ppo.critic_stage_onehot | yes |
| ppo.critic_prior_action | yes |
| ppo.critic_prior_norms | yes |
| ppo.critic_success_margins | yes |

#### Configuration: `run_kl_nouse/config.json`

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
| ppo.target_train_seconds | 5,400 |
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
| ppo.kl_anneal_iters | 350 |
| ppo.kl_sigma_ref | 0.3 |
| ppo.kl_ref_clip | 3 |
| ppo.kl_target | — |
| ppo.kl_target_final | — |
| ppo.critic_stage_onehot | yes |
| ppo.critic_prior_action | yes |
| ppo.critic_prior_norms | yes |
| ppo.critic_success_margins | yes |

#### Configuration: `run_kl_simple/config.json`

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
| ppo.target_train_seconds | 5,400 |
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
| ppo.kl_anneal_iters | 350 |
| ppo.kl_sigma_ref | 0.3 |
| ppo.kl_ref_clip | 3 |
| ppo.kl_target | — |
| ppo.kl_target_final | — |
| ppo.critic_stage_onehot | yes |
| ppo.critic_prior_action | yes |
| ppo.critic_prior_norms | yes |
| ppo.critic_success_margins | yes |

#### Configuration: `run_kl_use/config.json`

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
| ppo.target_train_seconds | 5,400 |
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
| ppo.kl_anneal_iters | 350 |
| ppo.kl_sigma_ref | 0.3 |
| ppo.kl_ref_clip | 3 |
| ppo.kl_target | — |
| ppo.kl_target_final | — |
| ppo.critic_stage_onehot | yes |
| ppo.critic_prior_action | yes |
| ppo.critic_prior_norms | yes |
| ppo.critic_success_margins | yes |

### Arms and sub-runs represented by directories

`clocked_nouse`, `clocked_use`, `context`, `critic_complex`, `critic_simple`, `critic_use`, `kl_complex`, `kl_nouse`, `kl_simple`, `kl_use`, `run_clocked_nouse`, `run_clocked_use`, `run_kl_complex`, `run_kl_nouse`, `run_kl_simple`, `run_kl_use`

## Results

### Recorded results: `run_clocked_nouse/final_report.json`

| Run result | Value |
|---|---|
| learner | experimental_clocked_paths_ppo |
| progression | learned |
| iters | 325 |
| best_iter | 274 |
| best_objective | 1.659527 |
| eval_objective | 0.567308 |
| eval_graded_objective | 0.567308 |
| eval_task_fitness | 1.603675 |
| eval_success_rate | 0.144531 |

| Evaluation metric | Value |
|---|---|
| eval_success_rate | 0.144531 |
| eval_task_fitness | 1.603675 |
| eval_reach_rate | 1 |
| eval_grasp_rate | 0.226562 |
| eval_grasp_lift_rate | 0.046875 |
| eval_lift_reached_rate | 0.144531 |
| eval_lift_max | 0.033217 |
| eval_train_return | 59.848941 |
| eval_base_return | 60.294435 |
| eval_shaping_return | -0.445492 |
| eval_forced_advance_frac | 0.002495 |
| eval_recover_frac | 0 |
| eval_graded_objective | 0.567308 |

### Recorded results: `run_clocked_use/final_report.json`

| Run result | Value |
|---|---|
| learner | experimental_clocked_paths_ppo |
| progression | learned |
| iters | 325 |
| best_iter | 224 |
| best_objective | 1.471925 |
| eval_objective | 0.750089 |
| eval_graded_objective | 0.750089 |
| eval_task_fitness | 1.494673 |
| eval_success_rate | 0.128906 |

| Evaluation metric | Value |
|---|---|
| eval_success_rate | 0.128906 |
| eval_task_fitness | 1.494673 |
| eval_reach_rate | 1 |
| eval_grasp_rate | 0.570312 |
| eval_grasp_lift_rate | 0.066406 |
| eval_lift_reached_rate | 0.128906 |
| eval_lift_max | 0.041226 |
| eval_train_return | 39.14299 |
| eval_base_return | 39.64668 |
| eval_shaping_return | -0.503689 |
| eval_forced_advance_frac | 0.003682 |
| eval_recover_frac | 0 |
| eval_graded_objective | 0.750089 |

### Recorded results: `run_kl_complex/final_report.json`

| Run result | Value |
|---|---|
| method | kl_prior |
| iters | 399 |
| best_iter | 374 |
| best_objective | 1.358788 |
| eval_objective | 0.472686 |
| eval_graded_objective | 0.472686 |
| eval_task_fitness | 1.256579 |
| eval_success_rate | 0.097656 |

| Evaluation metric | Value |
|---|---|
| eval_success_rate | 0.097656 |
| eval_task_fitness | 1.256579 |
| eval_reach_rate | 1 |
| eval_grasp_rate | 0.175781 |
| eval_grasp_lift_rate | 0.023438 |
| eval_lift_reached_rate | 0.097656 |
| eval_lift_max | 0.028549 |
| eval_train_return | 64.336658 |
| eval_base_return | 64.336658 |
| eval_shaping_return | 0 |
| eval_prior_disagreement | 0.070947 |
| eval_graded_objective | 0.472686 |

### Recorded results: `run_kl_nouse/final_report.json`

| Run result | Value |
|---|---|
| method | kl_prior |
| iters | 379 |
| best_iter | 374 |
| best_objective | 1.407387 |
| eval_objective | 0.87825 |
| eval_graded_objective | 0.87825 |
| eval_task_fitness | 1.416116 |
| eval_success_rate | 0.097656 |

| Evaluation metric | Value |
|---|---|
| eval_success_rate | 0.097656 |
| eval_task_fitness | 1.416116 |
| eval_reach_rate | 1 |
| eval_grasp_rate | 0.898438 |
| eval_grasp_lift_rate | 0.085938 |
| eval_lift_reached_rate | 0.097656 |
| eval_lift_max | 0.029834 |
| eval_train_return | 85.016194 |
| eval_base_return | 85.016194 |
| eval_shaping_return | 0 |
| eval_prior_disagreement | 0.072401 |
| eval_graded_objective | 0.87825 |

### Recorded results: `run_kl_simple/final_report.json`

| Run result | Value |
|---|---|
| method | kl_prior |
| iters | 380 |
| best_iter | 374 |
| best_objective | 1.310157 |
| eval_objective | 0.662568 |
| eval_graded_objective | 0.662568 |
| eval_task_fitness | 1.304831 |
| eval_success_rate | 0.066406 |

| Evaluation metric | Value |
|---|---|
| eval_success_rate | 0.066406 |
| eval_task_fitness | 1.304831 |
| eval_reach_rate | 1 |
| eval_grasp_rate | 0.566406 |
| eval_grasp_lift_rate | 0.019531 |
| eval_lift_reached_rate | 0.066406 |
| eval_lift_max | 0.027159 |
| eval_train_return | 68.735552 |
| eval_base_return | 68.735552 |
| eval_shaping_return | 0 |
| eval_prior_disagreement | 0.095399 |
| eval_graded_objective | 0.662568 |

### Recorded results: `run_kl_use/final_report.json`

| Run result | Value |
|---|---|
| method | kl_prior |
| iters | 397 |
| best_iter | 374 |
| best_objective | 1.268556 |
| eval_objective | 0.748434 |
| eval_graded_objective | 0.748434 |
| eval_task_fitness | 1.241763 |
| eval_success_rate | 0.066406 |

| Evaluation metric | Value |
|---|---|
| eval_success_rate | 0.066406 |
| eval_task_fitness | 1.241763 |
| eval_reach_rate | 1 |
| eval_grasp_rate | 0.746094 |
| eval_grasp_lift_rate | 0.054688 |
| eval_lift_reached_rate | 0.066406 |
| eval_lift_max | 0.023365 |
| eval_train_return | 78.672706 |
| eval_base_return | 78.672706 |
| eval_shaping_return | 0 |
| eval_prior_disagreement | 0.082397 |
| eval_graded_objective | 0.748434 |

### Training-metric reconstruction

| Arm/sub-run | Rows | Last iter | Last env steps | Elapsed s | Last success | Best success | Last lift max | Last base return | Best graded/objective |
|---|---|---|---|---|---|---|---|---|---|
| run_clocked_nouse | 325 | 324 | 8320000 | 5,248.43 | 0.035156 | 0.214844 | — | 6.142939 | — |
| run_clocked_use | 325 | 324 | 8320000 | 5,383.708 | 0.015625 | 0.195312 | — | 4.682715 | — |
| run_kl_complex | 399 | 398 | 10214400 | 5,402.736 | 0 | 0.105469 | — | 2.23292 | 1.358788 |
| run_kl_nouse | 379 | 378 | 9702400 | 5,407.134 | 0.011719 | 0.113281 | — | 6.938906 | — |
| run_kl_simple | 380 | 379 | 9728000 | 5,402.739 | 0.011719 | 0.125 | — | 8.241873 | 1.310157 |
| run_kl_use | 397 | 396 | 10163200 | 5,400.762 | 0 | 0.101562 | — | 7.67852 | — |

These are training-log endpoints and maxima, not substitutes for held-out evaluation. They are included especially for interrupted runs whose terminal report is missing.

### Terminal and failure evidence from logs

- `run_clocked_nouse.log`: [done] progression=learned 325 fragments, best_iter=274, graded_objective=0.567308 success=0.144531 -> runs/prior_gen_study_20260711/run_clocked_nouse
- `run_clocked_use.log`: [done] progression=learned 325 fragments, best_iter=224, graded_objective=0.750089 success=0.128906 -> runs/prior_gen_study_20260711/run_clocked_use
- `run_kl_complex.log`: [done] method=kl_prior 399 fragments, best_iter=374, graded_objective=0.472686 success=0.097656 -> runs/prior_gen_study_20260711/run_kl_complex
- `run_kl_nouse.log`: [done] method=kl_prior 379 fragments, best_iter=374, graded_objective=0.87825 success=0.097656 -> runs/prior_gen_study_20260711/run_kl_nouse
- `run_kl_simple.log`: [done] method=kl_prior 380 fragments, best_iter=374, graded_objective=0.662568 success=0.066406 -> runs/prior_gen_study_20260711/run_kl_simple
- `run_kl_use.log`: [done] method=kl_prior 397 fragments, best_iter=374, graded_objective=0.748434 success=0.066406 -> runs/prior_gen_study_20260711/run_kl_use

## What the results mean and major discoveries

- Across terminal sub-run reports, `run_clocked_nouse` recorded the strongest eval_task_fitness (1.603675).
- The best terminal success measurement was `run_clocked_nouse` at 0.144531 (eval_success_rate).

## Limitations and reading guidance

- Metrics are reported under the names used by the historical environment and runner. Similar-looking metrics from different generations of the code are not assumed to be numerically interchangeable.
- A maximum observed in training is not treated as held-out performance. Terminal evaluation values are preferred whenever present.
- Missing summaries, empty metric streams, failed retries, and incomplete arms are preserved as negative evidence rather than silently dropped.
- Candidate names and stage names are LLM-authored run artifacts. Interpretations in this historical log do not alter framework prompts, defaults, diagnostics, or task logic.

## Artifact provenance

| Artifact type | Count |
|---|---|
| .json | 63 |
| .md | 27 |
| .txt | 25 |
| .pkl | 12 |
| .jsonl | 6 |
| .log | 6 |

Primary evidence files:

- [`STUDY_SUMMARY.md`](../runs/prior_gen_study_20260711/STUDY_SUMMARY.md)
- [`run_clocked_nouse/config.json`](../runs/prior_gen_study_20260711/run_clocked_nouse/config.json)
- [`run_clocked_nouse/final_report.json`](../runs/prior_gen_study_20260711/run_clocked_nouse/final_report.json)
- [`run_clocked_nouse/metrics.jsonl`](../runs/prior_gen_study_20260711/run_clocked_nouse/metrics.jsonl)
- [`run_clocked_use/config.json`](../runs/prior_gen_study_20260711/run_clocked_use/config.json)
- [`run_clocked_use/final_report.json`](../runs/prior_gen_study_20260711/run_clocked_use/final_report.json)
- [`run_clocked_use/metrics.jsonl`](../runs/prior_gen_study_20260711/run_clocked_use/metrics.jsonl)
- [`run_kl_complex/config.json`](../runs/prior_gen_study_20260711/run_kl_complex/config.json)
- [`run_kl_complex/final_report.json`](../runs/prior_gen_study_20260711/run_kl_complex/final_report.json)
- [`run_kl_complex/metrics.jsonl`](../runs/prior_gen_study_20260711/run_kl_complex/metrics.jsonl)
- [`run_kl_nouse/config.json`](../runs/prior_gen_study_20260711/run_kl_nouse/config.json)
- [`run_kl_nouse/final_report.json`](../runs/prior_gen_study_20260711/run_kl_nouse/final_report.json)
- [`run_kl_nouse/metrics.jsonl`](../runs/prior_gen_study_20260711/run_kl_nouse/metrics.jsonl)
- [`run_kl_simple/config.json`](../runs/prior_gen_study_20260711/run_kl_simple/config.json)
- [`run_kl_simple/final_report.json`](../runs/prior_gen_study_20260711/run_kl_simple/final_report.json)
- [`run_kl_simple/metrics.jsonl`](../runs/prior_gen_study_20260711/run_kl_simple/metrics.jsonl)
- [`run_kl_use/config.json`](../runs/prior_gen_study_20260711/run_kl_use/config.json)
- [`run_kl_use/final_report.json`](../runs/prior_gen_study_20260711/run_kl_use/final_report.json)
- [`run_kl_use/metrics.jsonl`](../runs/prior_gen_study_20260711/run_kl_use/metrics.jsonl)

## Appendix: original authored run report

The following contemporaneous report is retained (with headings demoted) so that details and caveats are not lost in normalization.

### Source: [`STUDY_SUMMARY.md`](../runs/prior_gen_study_20260711/STUDY_SUMMARY.md)

## Prior-generation study (2026-07-11): use-info and complexity

Controlled: one shared C0+C1 task-completion context (context/), representation freeform_staged,
task, spec, backend codex, 1 seed, 0 revisions, 90-min PPO per arm, seed 0.

### Phase 1 — accurate prior-USE info in the seed prompt

| arm | framework | use-info | graded | fitness | success | stages |
|---|---|---|---|---|---|---|
| kl_nouse | kl_prior | no | **0.878** | 1.416 | 0.098 | 8 |
| clocked_use | clocked learned | yes | 0.750 | 1.495 | 0.129 | 8 |
| kl_use | kl_prior | yes | 0.748 | 1.242 | 0.066 | 8 |
| clocked_nouse | clocked learned | no | 0.567 | 1.604 | 0.145 | 8 |

Use-info: helped clocked (+0.18 graded), hurt kl (-0.13). Mixed; fitness/success order clocked
arms higher than graded does.

### Phase 2 — simple vs complex emphasis (winning scheme: kl_prior, no use-info)

| arm | graded | fitness | success | stages | signals |
|---|---|---|---|---|---|
| kl_simple | 0.663 | 1.305 | 0.066 | 3 | 17 |
| kl_complex | 0.473 | 1.257 | 0.098 | 8 | 47 |

Simple > complex on this seed. The manipulation DID control structure (3 vs 8 stages; default
prompt always yields 8).

### Variance caveat (important)

kl_nouse (0.878) and kl_complex (0.473) are nearly the same scheme -- the default prompt already
produces 8 stages -- yet differ by 0.41 graded. Single-seed differences below ~0.4 in this study
are within generation+training noise; none of the manipulations shows an effect that clears that
bar. Additionally, all six runs were still improving at the 90-min cutoff (best_iter = final
eval in every run; kl base_return accelerates only after the KL anneal completes ~iter 350), so
these rankings are of truncated runs.

### Tentative conclusions

1. No evidence that use-info reliably helps; if anything the effect is framework-dependent.
2. Weak-signal preference for simpler priors under kl_prior; needs seeds to confirm.
3. The strongest lever observed was none of the manipulations -- it was training time and the
   KL anneal schedule. The 4h comparison (runs/framework_cmp_4h_20260711, kl anneal 250 iters)
   addresses both.
4. To make generation-scheme effects detectable: >=3 seeds per arm and full-length runs.

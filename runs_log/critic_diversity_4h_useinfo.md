# Critic Diversity 4h Useinfo

- **Run directory:** [`runs/critic_diversity_4h_useinfo`](../runs/critic_diversity_4h_useinfo)
- **Date:** 2026-07-14 (earliest surviving artifact mtime (approximate))
- **Status:** completed or completed-with-caveats
- **Experiment class:** critic-feature study
- **Recorded task:** lift
- **Training metric rows recovered:** 987

## Abstract

This entry reconstructs `critic_diversity_4h_useinfo`, a critic-feature study. The account is based only on artifacts that survive in the repository: machine-readable configuration, candidate programs, selection diagnostics, training metrics, evaluations, reports, and logs. The run targeted lift. Its purpose and comparisons are inferred from the recorded arm names and configuration where no authored abstract survives; those inferences are identified by this provenance-first framing rather than presented as new experimental facts.

## Experimental structure

#### Configuration: `parallel_s0/config.json`

| Field | Recorded value |
|---|---|
| learner | experimental_alt_method_ppo |
| method | critic_features |
| task | lift |
| seed | 0 |
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
| ppo.target_train_seconds | 7,200 |
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
| ppo.critic_gate_values | yes |
| ppo.critic_critical_actions | no |
| ppo.critical_stage_keywords | grasp, lift |

#### Configuration: `union_s0/config.json`

| Field | Recorded value |
|---|---|
| learner | experimental_alt_method_ppo |
| method | critic_features |
| task | lift |
| seed | 0 |
| extra_programs | runs/prior_gen_study_20260711/critic_simple/program.json, runs/prior_gen_study_20260711/critic_complex/program.json |
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
| ppo.target_train_seconds | 7,200 |
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

`parallel_s0`, `union_s0`

## Results

### Recorded results: `parallel_s0/final_report.json`

| Run result | Value |
|---|---|
| method | critic_features |
| iters | 500 |
| best_iter | 424 |
| best_objective | 1.536399 |
| eval_objective | 0.841646 |
| eval_graded_objective | 0.841646 |
| eval_task_fitness | 1.554699 |
| eval_success_rate | 0.148438 |

| Evaluation metric | Value |
|---|---|
| eval_success_rate | 0.148438 |
| eval_task_fitness | 1.554699 |
| eval_reach_rate | 1 |
| eval_grasp_rate | 0.679688 |
| eval_grasp_lift_rate | 0.113281 |
| eval_lift_reached_rate | 0.148438 |
| eval_lift_max | 0.036495 |
| eval_train_return | 100.743 |
| eval_base_return | 100.743 |
| eval_shaping_return | 0 |
| eval_prior_disagreement | 0.335554 |
| eval_graded_objective | 0.841646 |

### Recorded results: `union_s0/final_report.json`

| Run result | Value |
|---|---|
| method | critic_features |
| iters | 487 |
| best_iter | 474 |
| best_objective | 1.753225 |
| eval_objective | 0.987608 |
| eval_graded_objective | 0.987608 |
| eval_task_fitness | 1.723926 |
| eval_success_rate | 0.445312 |

| Evaluation metric | Value |
|---|---|
| eval_success_rate | 0.445312 |
| eval_task_fitness | 1.723926 |
| eval_reach_rate | 1 |
| eval_grasp_rate | 0.335938 |
| eval_grasp_lift_rate | 0.136719 |
| eval_lift_reached_rate | 0.445312 |
| eval_lift_max | 0.04353 |
| eval_train_return | 561.449 |
| eval_base_return | 561.449 |
| eval_shaping_return | 0 |
| eval_prior_disagreement | 0.326207 |
| eval_graded_objective | 0.987608 |

### Training-metric reconstruction

| Arm/sub-run | Rows | Last iter | Last env steps | Elapsed s | Last success | Best success | Last lift max | Last base return | Best graded/objective |
|---|---|---|---|---|---|---|---|---|---|
| parallel_s0 | 500 | 499 | 12800000 | 7,162.095 | 0.003906 | 0.160156 | — | 12.57954 | 1.536399 |
| union_s0 | 487 | 486 | 12467200 | 7,207.12 | 0.007812 | 0.167969 | — | 37.553444 | 1.753225 |

These are training-log endpoints and maxima, not substitutes for held-out evaluation. They are included especially for interrupted runs whose terminal report is missing.

### Terminal and failure evidence from logs

- `parallel_s0.log`: [done] method=critic_features 500 fragments, best_iter=424, graded_objective=0.841646 success=0.148438 -> runs/critic_diversity_4h_useinfo/parallel_s0
- `union_s0.log`: [done] method=critic_features 487 fragments, best_iter=474, graded_objective=0.987608 success=0.445312 -> runs/critic_diversity_4h_useinfo/union_s0

## What the results mean and major discoveries

- Across terminal sub-run reports, `union_s0` recorded the strongest eval_task_fitness (1.723926).
- The best terminal success measurement was `union_s0` at 0.445312 (eval_success_rate).

## Limitations and reading guidance

- Metrics are reported under the names used by the historical environment and runner. Similar-looking metrics from different generations of the code are not assumed to be numerically interchangeable.
- A maximum observed in training is not treated as held-out performance. Terminal evaluation values are preferred whenever present.
- Missing summaries, empty metric streams, failed retries, and incomplete arms are preserved as negative evidence rather than silently dropped.
- Candidate names and stage names are LLM-authored run artifacts. Interpretations in this historical log do not alter framework prompts, defaults, diagnostics, or task logic.

## Artifact provenance

| Artifact type | Count |
|---|---|
| .json | 8 |
| .pkl | 4 |
| .log | 3 |
| .jsonl | 2 |
| .md | 1 |

Primary evidence files:

- [`parallel_s0/config.json`](../runs/critic_diversity_4h_useinfo/parallel_s0/config.json)
- [`parallel_s0/final_report.json`](../runs/critic_diversity_4h_useinfo/parallel_s0/final_report.json)
- [`parallel_s0/metrics.jsonl`](../runs/critic_diversity_4h_useinfo/parallel_s0/metrics.jsonl)
- [`summary.md`](../runs/critic_diversity_4h_useinfo/summary.md)
- [`union_s0/config.json`](../runs/critic_diversity_4h_useinfo/union_s0/config.json)
- [`union_s0/final_report.json`](../runs/critic_diversity_4h_useinfo/union_s0/final_report.json)
- [`union_s0/metrics.jsonl`](../runs/critic_diversity_4h_useinfo/union_s0/metrics.jsonl)

## Appendix: original authored run report

The following contemporaneous report is retained (with headings demoted) so that details and caveats are not lost in normalization.

### Source: [`summary.md`](../runs/critic_diversity_4h_useinfo/summary.md)

## critic-feature diversity study, use-info programs (2 arms x 1 seed x 2h)

union = critic_use+critic_simple+critic_complex feature union (complexity diversity)
parallel = critic_use + parallel gate values (cursor-free)

| run | graded | success | fitness | best_iter/iters |
|---|---:|---:|---:|---:|
| parallel_s0 | 0.841646 | 0.148438 | 1.554699 | 424/500 |
| union_s0 | 0.987608 | 0.445312 | 1.723926 | 474/487 |

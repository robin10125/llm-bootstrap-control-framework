# Arm Lift Only

- **Run directory:** [`runs/arm_lift_only`](../runs/arm_lift_only)
- **Date:** 2026-07-03 (earliest surviving artifact mtime (approximate))
- **Status:** completed or completed-with-caveats
- **Experiment class:** long-PPO training study
- **Recorded task:** lift
- **Training metric rows recovered:** 1,012

## Abstract

This entry reconstructs `arm_lift_only`, a long-PPO training study. The account is based only on artifacts that survive in the repository: machine-readable configuration, candidate programs, selection diagnostics, training metrics, evaluations, reports, and logs. The run targeted lift. Its purpose and comparisons are inferred from the recorded arm names and configuration where no authored abstract survives; those inferences are identified by this provenance-first framing rather than presented as new experimental facts.

## Experimental structure

| Field | Recorded value |
|---|---|
| learner | long_ppo |
| task | lift |
| arm | freeform_encourage |
| seed | 0 |
| reward_mode | lift_only |
| reward_env_overrides.w_reach | 0 |
| reward_env_overrides.w_finger | 0 |
| reward_env_overrides.w_close | 0 |
| reward_env_overrides.w_contact | 0 |
| reward_env_overrides.w_hold | 0 |
| reward_contrib_names | approach_potential, closure_progress, contact_progress, empty_squeeze_penalty, hand_gate_mean |
| criteria.min_hours | 99 |
| criteria.plateau_hours | 2 |
| criteria.plateau_eps | 0.00001 |
| criteria.success_stop | 0.8 |
| criteria.success_window | 10 |
| criteria.max_hours | 2 |
| ppo.iters | 10000000 |
| ppo.envs | 256 |
| ppo.lr | 0.0003 |
| ppo.gamma | 0.99 |
| ppo.lam | 0.95 |
| ppo.hidden | 256, 256 |
| ppo.ent_coef | 0 |
| ppo.supervised_steps | 80 |
| ppo.supervised_batch | 128 |
| ppo.supervised_lr | 0.001 |
| ppo.bc_critic_pretrain | yes |
| ppo.bc_rollout_states | yes |
| ppo.bc_kl_coef | 0 |
| ppo.bc_kl_anneal_iters | 200 |
| ppo.checkpoint_count | 0 |
| ppo.target_train_seconds | — |
| ppo.max_env_steps | — |
| ppo.action_transform | tanh |
| ppo.saturation_penalty | 0 |
| ppo.saturation_threshold | 0.98 |
| ppo.prior_logit_clip | 0.95 |
| ppo.action_target_reward_weight | 0 |
| ppo.success_hold_seconds | 0.5 |
| ppo.success_lift_threshold | 0.05 |
| ppo.warmup_compile | no |

## Results

### Recorded results: `final_report.json`

| Run result | Value |
|---|---|
| stop_reason | max run time reached (2.0h) |
| wall_hours | 2.001 |
| iters | 1011 |
| best_metric | 0.000399 |
| best_train_success | 0 |
| best_iter | 10 |

| Evaluation metric | Value |
|---|---|
| eval_base_return | 0.014761 |
| eval_base_reward_weight | 1 |
| eval_shaped_return | -0.154986 |
| eval_train_return | -0.140224 |
| eval_success_rate | 0 |
| eval_instant_success_rate | 0 |
| eval_reach_rate | 0.28125 |
| eval_grasp_rate | 0.109375 |
| eval_grasp_lift_rate | 0.042969 |
| eval_lift_reached_rate | 0.328125 |
| eval_lift_max | 0.050541 |
| eval_hard_clip_frac | 0 |
| eval_saturation_frac | 0 |
| eval_action_abs_mean | 0.136995 |

### Training-metric reconstruction

| Arm/sub-run | Rows | Last iter | Last env steps | Elapsed s | Last success | Best success | Last lift max | Last base return | Best graded/objective |
|---|---|---|---|---|---|---|---|---|---|
| freeform_encourage | 1012 | 1011 | 25907200 | 7,202.33 | 0 | 0 | 0.000027 | 0 | — |

These are training-log endpoints and maxima, not substitutes for held-out evaluation. They are included especially for interrupted runs whose terminal report is missing.

## What the results mean and major discoveries

- The best terminal success measurement was `arm_lift_only` at 0 (eval_success_rate).

## Limitations and reading guidance

- Metrics are reported under the names used by the historical environment and runner. Similar-looking metrics from different generations of the code are not assumed to be numerically interchangeable.
- A maximum observed in training is not treated as held-out performance. Terminal evaluation values are preferred whenever present.
- Missing summaries, empty metric streams, failed retries, and incomplete arms are preserved as negative evidence rather than silently dropped.
- Candidate names and stage names are LLM-authored run artifacts. Interpretations in this historical log do not alter framework prompts, defaults, diagnostics, or task logic.

## Artifact provenance

| Artifact type | Count |
|---|---|
| .json | 3 |
| .pkl | 2 |
| .jsonl | 1 |
| .png | 1 |

Primary evidence files:

- [`config.json`](../runs/arm_lift_only/config.json)
- [`final_report.json`](../runs/arm_lift_only/final_report.json)
- [`metrics.jsonl`](../runs/arm_lift_only/metrics.jsonl)

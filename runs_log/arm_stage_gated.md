# Arm Stage Gated

- **Run directory:** [`runs/arm_stage_gated`](../runs/arm_stage_gated)
- **Date:** 2026-07-03 (earliest surviving artifact mtime (approximate))
- **Status:** partial/interrupted; training metrics survive but no complete run-level report
- **Experiment class:** long-PPO training study
- **Recorded task:** lift
- **Training metric rows recovered:** 358

## Abstract

This entry reconstructs `arm_stage_gated`, a long-PPO training study. The account is based only on artifacts that survive in the repository: machine-readable configuration, candidate programs, selection diagnostics, training metrics, evaluations, reports, and logs. The run targeted lift. Its purpose and comparisons are inferred from the recorded arm names and configuration where no authored abstract survives; those inferences are identified by this provenance-first framing rather than presented as new experimental facts.

## Experimental structure

| Field | Recorded value |
|---|---|
| learner | long_ppo |
| task | lift |
| arm | freeform_encourage |
| seed | 0 |
| reward_mode | stage_gated |
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

### Training-metric reconstruction

| Arm/sub-run | Rows | Last iter | Last env steps | Elapsed s | Last success | Best success | Last lift max | Last base return | Best graded/objective |
|---|---|---|---|---|---|---|---|---|---|
| freeform_encourage | 358 | 357 | 9164800 | 4,173.259 | 0 | 0 | 0.012841 | 0.002355 | — |

These are training-log endpoints and maxima, not substitutes for held-out evaluation. They are included especially for interrupted runs whose terminal report is missing.

## What the results mean and major discoveries

- Only training metrics survive; their largest recorded success value was 0 for `freeform_encourage`. Because there is no terminal held-out report, this is progress evidence rather than a defensible final result.
- The missing terminal artifact is itself important: this attempt cannot support a comparative scientific conclusion. Its value is operational—showing what was launched, how far it progressed, and where the evidence trail ends.

## Limitations and reading guidance

- Metrics are reported under the names used by the historical environment and runner. Similar-looking metrics from different generations of the code are not assumed to be numerically interchangeable.
- A maximum observed in training is not treated as held-out performance. Terminal evaluation values are preferred whenever present.
- Missing summaries, empty metric streams, failed retries, and incomplete arms are preserved as negative evidence rather than silently dropped.
- Candidate names and stage names are LLM-authored run artifacts. Interpretations in this historical log do not alter framework prompts, defaults, diagnostics, or task logic.

## Artifact provenance

| Artifact type | Count |
|---|---|
| .json | 2 |
| .jsonl | 1 |
| .pkl | 1 |
| .png | 1 |

Primary evidence files:

- [`config.json`](../runs/arm_stage_gated/config.json)
- [`metrics.jsonl`](../runs/arm_stage_gated/metrics.jsonl)

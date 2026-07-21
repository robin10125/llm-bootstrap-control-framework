# Monotone Prior 6h PPO

- **Run directory:** [`runs/monotone_prior_6h_ppo_20260709-153033`](../runs/monotone_prior_6h_ppo_20260709-153033)
- **Date:** 2026-07-09 15:30:33 (directory timestamp)
- **Status:** partial/interrupted; training metrics survive but no complete run-level report
- **Experiment class:** long-PPO training study
- **Recorded task:** lift
- **Training metric rows recovered:** 150

## Abstract

This entry reconstructs `monotone_prior_6h_ppo_20260709-153033`, a long-PPO training study. The account is based only on artifacts that survive in the repository: machine-readable configuration, candidate programs, selection diagnostics, training metrics, evaluations, reports, and logs. The run targeted lift. Its purpose and comparisons are inferred from the recorded arm names and configuration where no authored abstract survives; those inferences are identified by this provenance-first framing rather than presented as new experimental facts.

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

## Results

### Training-metric reconstruction

| Arm/sub-run | Rows | Last iter | Last env steps | Elapsed s | Last success | Best success | Last lift max | Last base return | Best graded/objective |
|---|---|---|---|---|---|---|---|---|---|
| . | 150 | 149 | 3840000 | 3,007.104 | 0 | 0.15625 | — | 0.663837 | — |

These are training-log endpoints and maxima, not substitutes for held-out evaluation. They are included especially for interrupted runs whose terminal report is missing.

## What the results mean and major discoveries

- Only training metrics survive; their largest recorded success value was 0.15625 for `.`. Because there is no terminal held-out report, this is progress evidence rather than a defensible final result.
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
| .pkl | 2 |
| .jsonl | 1 |
| .log | 1 |
| .txt | 1 |

Primary evidence files:

- [`config.json`](../runs/monotone_prior_6h_ppo_20260709-153033/config.json)
- [`metrics.jsonl`](../runs/monotone_prior_6h_ppo_20260709-153033/metrics.jsonl)

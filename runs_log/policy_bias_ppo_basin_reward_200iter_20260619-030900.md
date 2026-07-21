# Policy Bias PPO Basin Reward 200iter

- **Run directory:** [`runs/policy_bias_ppo_basin_reward_200iter_20260619-030900`](../runs/policy_bias_ppo_basin_reward_200iter_20260619-030900)
- **Date:** 2026-06-19 03:09:00 (directory timestamp)
- **Status:** completed or completed-with-caveats
- **Experiment class:** PPO bias/reward/prior experiment
- **Recorded task:** lift
- **Training metric rows recovered:** 200

## Abstract

This entry reconstructs `policy_bias_ppo_basin_reward_200iter_20260619-030900`, a PPO bias/reward/prior experiment. The account is based only on artifacts that survive in the repository: machine-readable configuration, candidate programs, selection diagnostics, training metrics, evaluations, reports, and logs. The run targeted lift. Its purpose and comparisons are inferred from the recorded arm names and configuration where no authored abstract survives; those inferences are identified by this provenance-first framing rather than presented as new experimental facts.

## Experimental structure

| Field | Recorded value |
|---|---|
| learner | ppo |
| tasks | lift |
| arms | reward |
| arm_mechanisms.reward.reward | yes |
| arm_mechanisms.reward.action_prior | no |
| arm_mechanisms.reward.exploration | no |
| arm_mechanisms.reward.supervised_init | no |
| seeds | 0 |
| env.obs_size | 89 |
| env.action_size | 23 |
| env.horizon | 100 |
| env.frame_skip | 2 |
| env.actuators | rh_A_WRJ2, rh_A_WRJ1, rh_A_THJ5, rh_A_THJ4, rh_A_THJ3, rh_A_THJ2, rh_A_THJ1, rh_A_FFJ4, rh_A_FFJ3, rh_A_FFJ0, rh_A_MFJ4, rh_A_MFJ3, rh_A_MFJ0, rh_A_RFJ4, rh_A_RFJ3, rh_A_RFJ0, rh_A_LFJ5, rh_A_LFJ4, rh_A_LFJ3, rh_A_LFJ0, base_x, base_y, base_z |
| env.tasks.lift.objective | raise object above starting height |
| env.tasks.lift.success | lift > 0.05m |
| ppo.iters | 200 |
| ppo.envs | 1024 |
| ppo.lr | 0.0003 |
| ppo.gamma | 0.99 |
| ppo.lam | 0.95 |
| ppo.hidden | 128, 128 |
| ppo.ent_coef | 0 |
| ppo.supervised_steps | 80 |
| ppo.supervised_batch | 128 |
| ppo.supervised_lr | 0.001 |
| ppo.checkpoint_count | 5 |
| ppo.target_train_seconds | — |
| llm_backend | fixture |
| llm_model | — |

### Arms and sub-runs represented by directories

`lift_s0_reward`

## Results

### Recorded results: `lift_s0_reward/eval.json`

| Recorded field | Value |
|---|---|
| task | lift |
| seed | 0 |
| arm | reward |
| eval_base_return | 72.595726 |
| eval_shaped_return | 0.730859 |
| eval_train_return | 73.326584 |
| eval_success_rate | 0.894531 |
| eval_lift_max | 0.262312 |
| eval_summary | 0.10073, 0.048691, 0.016602, 0.201659, 0.262312, 2.468667 |

### Recorded results: `summary.json`

| Arm/interface | eval_success_rate | eval_lift_max | eval_base_return | eval_train_return | eval_shaped_return |
|---|---|---|---|---|---|
| reward | 0.894531 | 0.262312 | 72.595726 | 73.326584 | 0.730859 |

### Training-metric reconstruction

| Arm/sub-run | Rows | Last iter | Last env steps | Elapsed s | Last success | Best success | Last lift max | Last base return | Best graded/objective |
|---|---|---|---|---|---|---|---|---|---|
| reward | 200 | 199 | — | 8,259.733 | 0.772461 | 0.804688 | 0.157932 | 41.290787 | — |

These are training-log endpoints and maxima, not substitutes for held-out evaluation. They are included especially for interrupted runs whose terminal report is missing.

## What the results mean and major discoveries

- Among the recorded arms, `reward` had the highest held-out success (eval_success_rate = 0.894531). This is an ordering within this run, not a general ranking across experiments with different metric definitions or budgets.

## Limitations and reading guidance

- Metrics are reported under the names used by the historical environment and runner. Similar-looking metrics from different generations of the code are not assumed to be numerically interchangeable.
- A maximum observed in training is not treated as held-out performance. Terminal evaluation values are preferred whenever present.
- Missing summaries, empty metric streams, failed retries, and incomplete arms are preserved as negative evidence rather than silently dropped.
- Candidate names and stage names are LLM-authored run artifacts. Interpretations in this historical log do not alter framework prompts, defaults, diagnostics, or task logic.

## Artifact provenance

| Artifact type | Count |
|---|---|
| .pkl | 7 |
| .json | 4 |
| .csv | 1 |
| .jsonl | 1 |
| .log | 1 |
| .txt | 1 |

Primary evidence files:

- [`config.json`](../runs/policy_bias_ppo_basin_reward_200iter_20260619-030900/config.json)
- [`metrics.jsonl`](../runs/policy_bias_ppo_basin_reward_200iter_20260619-030900/metrics.jsonl)
- [`summary.json`](../runs/policy_bias_ppo_basin_reward_200iter_20260619-030900/summary.json)

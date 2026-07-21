# Policy Bias PPO Shadow Lift 1h Isolation

- **Run directory:** [`runs/policy_bias_ppo_shadow_lift_1h_isolation_20260618-004640`](../runs/policy_bias_ppo_shadow_lift_1h_isolation_20260618-004640)
- **Date:** 2026-06-18 00:46:40 (directory timestamp)
- **Status:** completed or completed-with-caveats
- **Experiment class:** PPO bias/reward/prior experiment
- **Recorded task:** lift
- **Training metric rows recovered:** 91

## Abstract

This entry reconstructs `policy_bias_ppo_shadow_lift_1h_isolation_20260618-004640`, a PPO bias/reward/prior experiment. The account is based only on artifacts that survive in the repository: machine-readable configuration, candidate programs, selection diagnostics, training metrics, evaluations, reports, and logs. The run targeted lift. Its purpose and comparisons are inferred from the recorded arm names and configuration where no authored abstract survives; those inferences are identified by this provenance-first framing rather than presented as new experimental facts.

## Experimental structure

| Field | Recorded value |
|---|---|
| learner | ppo |
| tasks | lift |
| arms | baseline, reward, action_prior, exploration, supervised_init |
| arm_mechanisms.baseline.reward | no |
| arm_mechanisms.baseline.action_prior | no |
| arm_mechanisms.baseline.exploration | no |
| arm_mechanisms.baseline.supervised_init | no |
| arm_mechanisms.reward.reward | yes |
| arm_mechanisms.reward.action_prior | no |
| arm_mechanisms.reward.exploration | no |
| arm_mechanisms.reward.supervised_init | no |
| arm_mechanisms.action_prior.reward | no |
| arm_mechanisms.action_prior.action_prior | yes |
| arm_mechanisms.action_prior.exploration | no |
| arm_mechanisms.action_prior.supervised_init | no |
| arm_mechanisms.exploration.reward | no |
| arm_mechanisms.exploration.action_prior | no |
| arm_mechanisms.exploration.exploration | yes |
| arm_mechanisms.exploration.supervised_init | no |
| arm_mechanisms.supervised_init.reward | no |
| arm_mechanisms.supervised_init.action_prior | no |
| arm_mechanisms.supervised_init.exploration | no |
| arm_mechanisms.supervised_init.supervised_init | yes |
| seeds | 0 |
| env.obs_size | 89 |
| env.action_size | 23 |
| env.horizon | 100 |
| env.frame_skip | 2 |
| env.actuators | rh_A_WRJ2, rh_A_WRJ1, rh_A_THJ5, rh_A_THJ4, rh_A_THJ3, rh_A_THJ2, rh_A_THJ1, rh_A_FFJ4, rh_A_FFJ3, rh_A_FFJ0, rh_A_MFJ4, rh_A_MFJ3, rh_A_MFJ0, rh_A_RFJ4, rh_A_RFJ3, rh_A_RFJ0, rh_A_LFJ5, rh_A_LFJ4, rh_A_LFJ3, rh_A_LFJ0, base_x, base_y, base_z |
| env.tasks.lift.objective | raise object above starting height |
| env.tasks.lift.success | lift > 0.05m |
| ppo.iters | 100000 |
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
| ppo.target_train_seconds | 720 |
| llm_backend | codex |
| llm_model | — |

### Arms and sub-runs represented by directories

`lift_s0_action_prior`, `lift_s0_baseline`, `lift_s0_exploration`, `lift_s0_reward`, `lift_s0_supervised_init`

## Results

### Recorded results: `lift_s0_action_prior/eval.json`

| Recorded field | Value |
|---|---|
| task | lift |
| seed | 0 |
| arm | action_prior |
| eval_return | -25.051395 |
| eval_success_rate | 0.172852 |
| eval_lift_max | 0.030742 |
| eval_summary | 0.092912, 0.041862, 0.148438, 0.768675, 0.030742, 0.282269 |

### Recorded results: `lift_s0_baseline/eval.json`

| Recorded field | Value |
|---|---|
| task | lift |
| seed | 0 |
| arm | baseline |
| eval_return | -2.928156 |
| eval_success_rate | 0.151367 |
| eval_lift_max | 0.033579 |
| eval_summary | 0.060222, 0.038775, 0.073242, 0.215679, 0.033579, 0.291714 |

### Recorded results: `lift_s0_exploration/eval.json`

| Recorded field | Value |
|---|---|
| task | lift |
| seed | 0 |
| arm | exploration |
| eval_return | -4.262866 |
| eval_success_rate | 0.118164 |
| eval_lift_max | 0.030324 |
| eval_summary | 0.061373, 0.038621, 0.0625, 0.28993, 0.030324, 0.225793 |

### Recorded results: `lift_s0_reward/eval.json`

| Recorded field | Value |
|---|---|
| task | lift |
| seed | 0 |
| arm | reward |
| eval_return | -284.68 |
| eval_success_rate | 0.072266 |
| eval_lift_max | 0.01674 |
| eval_summary | 0.08224, 0.039172, 0.027344, 0.308053, 0.01674, 0.110773 |

### Recorded results: `lift_s0_supervised_init/eval.json`

| Recorded field | Value |
|---|---|
| task | lift |
| seed | 0 |
| arm | supervised_init |
| eval_return | -8.981997 |
| eval_success_rate | 0.209961 |
| eval_lift_max | 0.039268 |
| eval_summary | 0.056661, 0.036763, 0.234375, 0.843581, 0.039268, 0.451875 |

### Recorded results: `summary.json`

| Arm/interface | eval_success_rate | eval_lift_max |
|---|---|---|
| action_prior | 0.172852 | 0.030742 |
| baseline | 0.151367 | 0.033579 |
| exploration | 0.118164 | 0.030324 |
| reward | 0.072266 | 0.01674 |
| supervised_init | 0.209961 | 0.039268 |

### Training-metric reconstruction

| Arm/sub-run | Rows | Last iter | Last env steps | Elapsed s | Last success | Best success | Last lift max | Last base return | Best graded/objective |
|---|---|---|---|---|---|---|---|---|---|
| baseline | 18 | 17 | — | 753.833 | 0.266602 | 0.266602 | 0.043549 | — | — |
| reward | 18 | 17 | — | 749.69 | 0.105469 | 0.137695 | 0.026847 | — | — |
| action_prior | 19 | 18 | — | 745.724 | 0.085938 | 0.09375 | 0.018407 | — | — |
| exploration | 18 | 17 | — | 755.129 | 0.211914 | 0.211914 | 0.039395 | — | — |
| supervised_init | 18 | 17 | — | 737.912 | 0.261719 | 0.272461 | 0.040278 | — | — |

These are training-log endpoints and maxima, not substitutes for held-out evaluation. They are included especially for interrupted runs whose terminal report is missing.

## What the results mean and major discoveries

- Among the recorded arms, `supervised_init` had the highest held-out success (eval_success_rate = 0.209961). This is an ordering within this run, not a general ranking across experiments with different metric definitions or budgets.
- The arm comparison is most informative when arms shared the same environment, seed, and budget. Where the configuration records only one seed, apparent gaps remain vulnerable to optimization variance.

## Limitations and reading guidance

- Metrics are reported under the names used by the historical environment and runner. Similar-looking metrics from different generations of the code are not assumed to be numerically interchangeable.
- A maximum observed in training is not treated as held-out performance. Terminal evaluation values are preferred whenever present.
- Missing summaries, empty metric streams, failed retries, and incomplete arms are preserved as negative evidence rather than silently dropped.
- Candidate names and stage names are LLM-authored run artifacts. Interpretations in this historical log do not alter framework prompts, defaults, diagnostics, or task logic.

## Artifact provenance

| Artifact type | Count |
|---|---|
| .pkl | 35 |
| .json | 8 |
| .csv | 1 |
| .jsonl | 1 |

Primary evidence files:

- [`config.json`](../runs/policy_bias_ppo_shadow_lift_1h_isolation_20260618-004640/config.json)
- [`metrics.jsonl`](../runs/policy_bias_ppo_shadow_lift_1h_isolation_20260618-004640/metrics.jsonl)
- [`summary.json`](../runs/policy_bias_ppo_shadow_lift_1h_isolation_20260618-004640/summary.json)

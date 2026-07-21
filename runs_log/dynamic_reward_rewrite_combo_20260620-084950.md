# Dynamic Reward Rewrite Combo

- **Run directory:** [`runs/dynamic_reward_rewrite_combo_20260620-084950`](../runs/dynamic_reward_rewrite_combo_20260620-084950)
- **Date:** 2026-06-20 08:49:50 (directory timestamp)
- **Status:** completed or completed-with-caveats
- **Experiment class:** PPO bias/reward/prior experiment
- **Recorded task:** lift
- **Training metric rows recovered:** 1,062

## Abstract

This entry reconstructs `dynamic_reward_rewrite_combo_20260620-084950`, a PPO bias/reward/prior experiment. The account is based only on artifacts that survive in the repository: machine-readable configuration, candidate programs, selection diagnostics, training metrics, evaluations, reports, and logs. The run targeted lift. Its purpose and comparisons are inferred from the recorded arm names and configuration where no authored abstract survives; those inferences are identified by this provenance-first framing rather than presented as new experimental facts.

## Experimental structure

| Field | Recorded value |
|---|---|
| learner | ppo_dynamic_reward |
| tasks | lift |
| arms | baseline, reward_action_prior, reward_supervised_init |
| arm_mechanisms.baseline.reward | no |
| arm_mechanisms.baseline.action_prior | no |
| arm_mechanisms.baseline.exploration | no |
| arm_mechanisms.baseline.supervised_init | no |
| arm_mechanisms.reward_action_prior.reward | yes |
| arm_mechanisms.reward_action_prior.action_prior | yes |
| arm_mechanisms.reward_action_prior.exploration | no |
| arm_mechanisms.reward_action_prior.supervised_init | no |
| arm_mechanisms.reward_supervised_init.reward | yes |
| arm_mechanisms.reward_supervised_init.action_prior | no |
| arm_mechanisms.reward_supervised_init.exploration | no |
| arm_mechanisms.reward_supervised_init.supervised_init | yes |
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
| ppo.target_train_seconds | 14,400 |
| ppo.action_transform | tanh |
| ppo.saturation_penalty | 0 |
| ppo.saturation_threshold | 0.98 |
| ppo.prior_logit_clip | 0.95 |
| ppo.action_target_reward_weight | 0 |
| dynamic_reward.cheap_checkup_steps | 50 |
| dynamic_reward.deep_checkup_seconds | 7,200 |
| dynamic_reward.max_template_weight | 1.25 |
| dynamic_reward.allow_reward_rewrite | yes |
| dynamic_reward.rewrite_fraction | 0.5 |
| dynamic_reward.previous_run_dir | runs/dynamic_reward_combo_20260619-174924 |
| dynamic_reward.xla_cache | /home/robin/Documents/agent-mini-script-control/llm-framework/.xla_cache |
| llm_backend | codex |
| llm_model | — |

### Arms and sub-runs represented by directories

`lift_s0_baseline`, `lift_s0_reward_action_prior`, `lift_s0_reward_supervised_init`, `llm_initial_bias`, `plots`

## Results

### Recorded results: `lift_s0_baseline/eval.json`

| Recorded field | Value |
|---|---|
| task | lift |
| seed | 0 |
| arm | baseline |
| eval_base_return | 31.322052 |
| eval_shaped_return | 0 |
| eval_reward_template_returns | 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 |
| eval_train_return | 31.322052 |
| eval_success_rate | 0.625977 |
| eval_lift_max | 0.076248 |
| eval_hard_clip_frac | 0 |
| eval_saturation_frac | 0.000185 |
| eval_action_abs_mean | 0.273553 |
| eval_summary | 0.058546, 0.033232, 0.348633, 0.310239, 0.076248, 0.618349 |

### Recorded results: `lift_s0_reward_action_prior/eval.json`

| Recorded field | Value |
|---|---|
| task | lift |
| seed | 0 |
| arm | reward_action_prior |
| eval_base_return | 77.309769 |
| eval_shaped_return | -0.072056 |
| eval_reward_template_returns | -0.030625, -0.037322, -0.000002, 0.000003, 0, 0.001988, -0.002385, 0, 0, 0, 0, 0, 0, 0, 0, 0 |
| eval_train_return | 77.237701 |
| eval_success_rate | 0.892578 |
| eval_lift_max | 0.280235 |
| eval_hard_clip_frac | 0 |
| eval_saturation_frac | 0.000445 |
| eval_action_abs_mean | 0.353013 |
| eval_summary | 0.09984, 0.049056, 0.024414, 0.270398, 0.280235, 2.735118 |

### Recorded results: `lift_s0_reward_supervised_init/eval.json`

| Recorded field | Value |
|---|---|
| task | lift |
| seed | 0 |
| arm | reward_supervised_init |
| eval_base_return | 55.131584 |
| eval_shaped_return | -0.14395 |
| eval_reward_template_returns | -0.098521, -0.042899, 0.000027, 0.000017, 0, 0.002089, -0.007309, 0, 0, 0, 0, 0, 0, 0, 0, 0 |
| eval_train_return | 54.987633 |
| eval_success_rate | 0.829102 |
| eval_lift_max | 0.183835 |
| eval_hard_clip_frac | 0 |
| eval_saturation_frac | 0.00252 |
| eval_action_abs_mean | 0.353611 |
| eval_summary | 0.08609, 0.041969, 0.063477, 0.259638, 0.183835, 1.936172 |

### Recorded results: `summary.json`

| Arm/interface | eval_success_rate | eval_lift_max | eval_base_return | eval_train_return | eval_shaped_return | eval_action_abs_mean | eval_saturation_frac |
|---|---|---|---|---|---|---|---|
| baseline | 0.625977 | 0.076248 | 31.322052 | 31.322052 | 0 | 0.273553 | 0.000185 |
| reward_action_prior | 0.892578 | 0.280235 | 77.309769 | 77.237701 | -0.072056 | 0.353013 | 0.000445 |
| reward_supervised_init | 0.829102 | 0.183835 | 55.131584 | 54.987633 | -0.14395 | 0.353611 | 0.00252 |

### Training-metric reconstruction

| Arm/sub-run | Rows | Last iter | Last env steps | Elapsed s | Last success | Best success | Last lift max | Last base return | Best graded/objective |
|---|---|---|---|---|---|---|---|---|---|
| baseline | 338 | 337 | — | 7,230.796 | 0.525391 | 0.552734 | 0.066737 | 20.168411 | — |
| reward_action_prior | 368 | 367 | — | 7,225.641 | 0.879883 | 0.911133 | 0.283074 | 73.208031 | — |
| reward_supervised_init | 356 | 355 | — | 7,208.581 | 0.766602 | 0.80957 | 0.157261 | 47.031723 | — |

These are training-log endpoints and maxima, not substitutes for held-out evaluation. They are included especially for interrupted runs whose terminal report is missing.

## What the results mean and major discoveries

- Among the recorded arms, `reward_action_prior` had the highest held-out success (eval_success_rate = 0.892578). This is an ordering within this run, not a general ranking across experiments with different metric definitions or budgets.
- The arm comparison is most informative when arms shared the same environment, seed, and budget. Where the configuration records only one seed, apparent gaps remain vulnerable to optimization variance.

## Limitations and reading guidance

- Metrics are reported under the names used by the historical environment and runner. Similar-looking metrics from different generations of the code are not assumed to be numerically interchangeable.
- A maximum observed in training is not treated as held-out performance. Terminal evaluation values are preferred whenever present.
- Missing summaries, empty metric streams, failed retries, and incomplete arms are preserved as negative evidence rather than silently dropped.
- Candidate names and stage names are LLM-authored run artifacts. Interpretations in this historical log do not alter framework prompts, defaults, diagnostics, or task logic.

## Artifact provenance

| Artifact type | Count |
|---|---|
| .json | 44 |
| .pkl | 24 |
| .md | 5 |
| .txt | 4 |
| .svg | 2 |
| .csv | 1 |
| .jsonl | 1 |
| .log | 1 |
| .tsv | 1 |

Primary evidence files:

- [`REPORT.md`](../runs/dynamic_reward_rewrite_combo_20260620-084950/REPORT.md)
- [`config.json`](../runs/dynamic_reward_rewrite_combo_20260620-084950/config.json)
- [`metrics.jsonl`](../runs/dynamic_reward_rewrite_combo_20260620-084950/metrics.jsonl)
- [`summary.json`](../runs/dynamic_reward_rewrite_combo_20260620-084950/summary.json)

## Appendix: original authored run report

The following contemporaneous report is retained (with headings demoted) so that details and caveats are not lost in normalization.

### Source: [`REPORT.md`](../runs/dynamic_reward_rewrite_combo_20260620-084950/REPORT.md)

## Dynamic Reward Combination Experiment

Created: 2026-06-20 21:06:55

### Configuration

- Arms: `baseline,reward_action_prior,reward_supervised_init`
- Tasks: `lift`
- Target seconds per arm: `14400.0`
- Cheap checkup steps: `50`
- Deep checkup seconds: `7200.0`
- Reward rewrite: `True`
- Rewrite fraction: `0.5`
- Action transform: `tanh`

### Summary

```json
{
  "baseline": {
    "eval_success_rate": 0.625977,
    "eval_base_return": 31.322052,
    "eval_shaped_return": 0.0,
    "eval_train_return": 31.322052,
    "eval_lift_max": 0.076248,
    "eval_hard_clip_frac": 0.0,
    "eval_saturation_frac": 0.000185,
    "eval_action_abs_mean": 0.273553,
    "n_eval": 1
  },
  "reward_action_prior": {
    "eval_success_rate": 0.892578,
    "eval_base_return": 77.309769,
    "eval_shaped_return": -0.072056,
    "eval_train_return": 77.237701,
    "eval_lift_max": 0.280235,
    "eval_hard_clip_frac": 0.0,
    "eval_saturation_frac": 0.000445,
    "eval_action_abs_mean": 0.353013,
    "n_eval": 1
  },
  "reward_supervised_init": {
    "eval_success_rate": 0.829102,
    "eval_base_return": 55.131584,
    "eval_shaped_return": -0.14395,
    "eval_train_return": 54.987633,
    "eval_lift_max": 0.183835,
    "eval_hard_clip_frac": 0.0,
    "eval_saturation_frac": 0.00252,
    "eval_action_abs_mean": 0.353611,
    "n_eval": 1
  }
}
```

### Notes

Reward coach prompts intentionally omit env reward, train return, base return, and PPO losses.
Rejected and deep-checkup proposals are stored under each arm's `reward_checkups/` directory.

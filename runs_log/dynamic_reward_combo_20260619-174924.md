# Dynamic Reward Combo

- **Run directory:** [`runs/dynamic_reward_combo_20260619-174924`](../runs/dynamic_reward_combo_20260619-174924)
- **Date:** 2026-06-19 17:49:24 (directory timestamp)
- **Status:** completed or completed-with-caveats
- **Experiment class:** PPO bias/reward/prior experiment
- **Recorded task:** lift
- **Training metric rows recovered:** 1,126

## Abstract

This entry reconstructs `dynamic_reward_combo_20260619-174924`, a PPO bias/reward/prior experiment. The account is based only on artifacts that survive in the repository: machine-readable configuration, candidate programs, selection diagnostics, training metrics, evaluations, reports, and logs. The run targeted lift. Its purpose and comparisons are inferred from the recorded arm names and configuration where no authored abstract survives; those inferences are identified by this provenance-first framing rather than presented as new experimental facts.

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
| dynamic_reward.xla_cache | /home/robin/Documents/agent-mini-script-control/llm-framework/.xla_cache |
| llm_backend | codex |
| llm_model | — |

### Arms and sub-runs represented by directories

`lift_s0_baseline`, `lift_s0_reward_action_prior`, `lift_s0_reward_supervised_init`

## Results

### Recorded results: `lift_s0_baseline/eval.json`

| Recorded field | Value |
|---|---|
| task | lift |
| seed | 0 |
| arm | baseline |
| eval_base_return | 58.798714 |
| eval_shaped_return | 0 |
| eval_reward_template_returns | 0, 0, 0, 0, 0, 0, 0, 0 |
| eval_train_return | 58.798714 |
| eval_success_rate | 0.870117 |
| eval_lift_max | 0.171179 |
| eval_hard_clip_frac | 0 |
| eval_saturation_frac | 0.000372 |
| eval_action_abs_mean | 0.336908 |
| eval_summary | 0.088683, 0.045072, 0.027344, 0.399318, 0.171179, 1.528761 |

### Recorded results: `lift_s0_reward_action_prior/eval.json`

| Recorded field | Value |
|---|---|
| task | lift |
| seed | 0 |
| arm | reward_action_prior |
| eval_base_return | 436.842 |
| eval_shaped_return | 0.397982 |
| eval_reward_template_returns | 0.397982, -0.029635, 0.000033, 0.000199, 0, 0.023946, 0.014063, -1.188962 |
| eval_train_return | 437.24 |
| eval_success_rate | 0.981445 |
| eval_lift_max | 0.339693 |
| eval_hard_clip_frac | 0 |
| eval_saturation_frac | 0.011798 |
| eval_action_abs_mean | 0.669626 |
| eval_summary | 0.03329, 0.031105, 0.438477, 0.87902, 0.339693, 0.595669 |

### Recorded results: `lift_s0_reward_supervised_init/eval.json`

| Recorded field | Value |
|---|---|
| task | lift |
| seed | 0 |
| arm | reward_supervised_init |
| eval_base_return | 164.89 |
| eval_shaped_return | -0.08073 |
| eval_reward_template_returns | -0.08073, -0.036118, 0.000002, 0.000031, 0, 0.014156, -0.005457, -0.386425 |
| eval_train_return | 164.81 |
| eval_success_rate | 1 |
| eval_lift_max | 0.495984 |
| eval_hard_clip_frac | 0 |
| eval_saturation_frac | 0.006181 |
| eval_action_abs_mean | 0.530177 |
| eval_summary | 0.06021, 0.045534, 0.040039, 0.723532, 0.495984, 2.033602 |

### Recorded results: `summary.json`

| Arm/interface | eval_success_rate | eval_lift_max | eval_base_return | eval_train_return | eval_shaped_return | eval_action_abs_mean | eval_saturation_frac |
|---|---|---|---|---|---|---|---|
| baseline | 0.870117 | 0.171179 | 58.798714 | 58.798714 | 0 | 0.336908 | 0.000372 |
| reward_action_prior | 0.981445 | 0.339693 | 436.842 | 437.24 | 0.397982 | 0.669626 | 0.011798 |
| reward_supervised_init | 1 | 0.495984 | 164.89 | 164.81 | -0.08073 | 0.530177 | 0.006181 |

### Training-metric reconstruction

| Arm/sub-run | Rows | Last iter | Last env steps | Elapsed s | Last success | Best success | Last lift max | Last base return | Best graded/objective |
|---|---|---|---|---|---|---|---|---|---|
| baseline | 362 | 361 | — | 14,407.464 | 0.814453 | 0.824219 | 0.136751 | 45.851997 | — |
| reward_action_prior | 387 | 386 | — | 14,434.656 | 0.953125 | 0.962891 | 0.356524 | 373.756 | — |
| reward_supervised_init | 377 | 376 | — | 14,419.068 | 0.970703 | 0.977539 | 0.457355 | 148.831 | — |

These are training-log endpoints and maxima, not substitutes for held-out evaluation. They are included especially for interrupted runs whose terminal report is missing.

## What the results mean and major discoveries

- Among the recorded arms, `reward_supervised_init` had the highest held-out success (eval_success_rate = 1). This is an ordering within this run, not a general ranking across experiments with different metric definitions or budgets.
- The arm comparison is most informative when arms shared the same environment, seed, and budget. Where the configuration records only one seed, apparent gaps remain vulnerable to optimization variance.

## Limitations and reading guidance

- Metrics are reported under the names used by the historical environment and runner. Similar-looking metrics from different generations of the code are not assumed to be numerically interchangeable.
- A maximum observed in training is not treated as held-out performance. Terminal evaluation values are preferred whenever present.
- Missing summaries, empty metric streams, failed retries, and incomplete arms are preserved as negative evidence rather than silently dropped.
- Candidate names and stage names are LLM-authored run artifacts. Interpretations in this historical log do not alter framework prompts, defaults, diagnostics, or task logic.

## Artifact provenance

| Artifact type | Count |
|---|---|
| .json | 37 |
| .pkl | 21 |
| .txt | 5 |
| .md | 4 |
| .csv | 1 |
| .jsonl | 1 |
| .log | 1 |

Primary evidence files:

- [`REPORT.md`](../runs/dynamic_reward_combo_20260619-174924/REPORT.md)
- [`config.json`](../runs/dynamic_reward_combo_20260619-174924/config.json)
- [`metrics.jsonl`](../runs/dynamic_reward_combo_20260619-174924/metrics.jsonl)
- [`summary.json`](../runs/dynamic_reward_combo_20260619-174924/summary.json)

## Appendix: original authored run report

The following contemporaneous report is retained (with headings demoted) so that details and caveats are not lost in normalization.

### Source: [`REPORT.md`](../runs/dynamic_reward_combo_20260619-174924/REPORT.md)

## Dynamic Reward Combination Experiment

Created: 2026-06-20 06:02:27

### Configuration

- Arms: `baseline,reward_action_prior,reward_supervised_init`
- Tasks: `lift`
- Target seconds per arm: `14400.0`
- Cheap checkup steps: `50`
- Deep checkup seconds: `7200.0`
- Action transform: `tanh`

### Summary

```json
{
  "baseline": {
    "eval_success_rate": 0.870117,
    "eval_base_return": 58.798714,
    "eval_shaped_return": 0.0,
    "eval_train_return": 58.798714,
    "eval_lift_max": 0.171179,
    "eval_hard_clip_frac": 0.0,
    "eval_saturation_frac": 0.000372,
    "eval_action_abs_mean": 0.336908,
    "n_eval": 1
  },
  "reward_action_prior": {
    "eval_success_rate": 0.981445,
    "eval_base_return": 436.841614,
    "eval_shaped_return": 0.397982,
    "eval_train_return": 437.239563,
    "eval_lift_max": 0.339693,
    "eval_hard_clip_frac": 0.0,
    "eval_saturation_frac": 0.011798,
    "eval_action_abs_mean": 0.669626,
    "n_eval": 1
  },
  "reward_supervised_init": {
    "eval_success_rate": 1.0,
    "eval_base_return": 164.890381,
    "eval_shaped_return": -0.08073,
    "eval_train_return": 164.809662,
    "eval_lift_max": 0.495984,
    "eval_hard_clip_frac": 0.0,
    "eval_saturation_frac": 0.006181,
    "eval_action_abs_mean": 0.530177,
    "n_eval": 1
  }
}
```

### Notes

Reward coach prompts intentionally omit env reward, train return, base return, and PPO losses.
Rejected and deep-checkup proposals are stored under each arm's `reward_checkups/` directory.

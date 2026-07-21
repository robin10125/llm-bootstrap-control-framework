# Pareto Fixed Prior Shaped Reward 12h Nowarmup

- **Run directory:** [`runs/pareto_fixed_prior_shaped_reward_12h_nowarmup_20260622-203237`](../runs/pareto_fixed_prior_shaped_reward_12h_nowarmup_20260622-203237)
- **Date:** 2026-06-22 20:32:37 (directory timestamp)
- **Status:** completed or completed-with-caveats
- **Experiment class:** PPO bias/reward/prior experiment
- **Recorded task:** lift
- **Training metric rows recovered:** 1,044

## Abstract

This entry reconstructs `pareto_fixed_prior_shaped_reward_12h_nowarmup_20260622-203237`, a PPO bias/reward/prior experiment. The account is based only on artifacts that survive in the repository: machine-readable configuration, candidate programs, selection diagnostics, training metrics, evaluations, reports, and logs. The run targeted lift. Its purpose and comparisons are inferred from the recorded arm names and configuration where no authored abstract survives; those inferences are identified by this provenance-first framing rather than presented as new experimental facts.

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
| ppo.success_hold_seconds | 0.5 |
| ppo.success_lift_threshold | 0.05 |
| ppo.warmup_compile | no |
| dynamic_reward.cheap_checkup_steps | 50 |
| dynamic_reward.deep_checkup_seconds | 7,200 |
| dynamic_reward.max_template_weight | 1.25 |
| dynamic_reward.min_base_reward_weight | 0 |
| dynamic_reward.post_new_reward_checkup_steps | 10 |
| dynamic_reward.post_new_reward_fast_checkups | 3 |
| dynamic_reward.allow_reward_rewrite | yes |
| dynamic_reward.pre_run_reward_analysis | yes |
| dynamic_reward.rewrite_fraction | 0.5 |
| dynamic_reward.previous_run_dir | — |
| dynamic_reward.xla_cache | /home/robin/Documents/agent-mini-script-control/llm-framework/.xla_cache |
| dynamic_action_prior.llm_action_prior | no |
| dynamic_action_prior.pareto_action_prior | no |
| dynamic_action_prior.action_prior_candidates | 5 |
| dynamic_action_prior.pareto_supervised_init | no |
| dynamic_action_prior.supervised_candidates | 5 |
| dynamic_action_prior.selection_envs | 128 |
| dynamic_action_prior.selection_steps | — |
| dynamic_action_prior.selection_seed | 0 |
| dynamic_action_prior.checkup_steps | 0 |
| dynamic_action_prior.max_checkups | 3 |
| dynamic_action_prior.max_weight | 0.6 |
| llm_backend | codex |
| llm_model | — |

### Arms and sub-runs represented by directories

`graphs`, `lift_s0_baseline`, `lift_s0_reward_action_prior`, `lift_s0_reward_supervised_init`, `pre_run_reward_analysis`

## Results

### Recorded results: `lift_s0_baseline/eval.json`

| Recorded field | Value |
|---|---|
| task | lift |
| seed | 0 |
| arm | baseline |
| eval_base_return | 61.414513 |
| eval_base_reward_weight | 1 |
| eval_shaped_return | 0 |
| eval_reward_template_returns | 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 |
| eval_action_prior_weights | 0.22, 0.2, 0.18, 0.1, 0.24, 0.18 |
| eval_train_return | 61.414513 |
| eval_success_rate | 0.268555 |
| eval_instant_success_rate | 0.799805 |
| eval_lift_max | 0.17236 |
| eval_hard_clip_frac | 0 |
| eval_saturation_frac | 0.00138 |
| eval_action_abs_mean | 0.295245 |
| eval_summary | 0.055716, 0.044345, 0.021484, 0.216736, 0.17236, 1.536656 |

### Recorded results: `lift_s0_reward_action_prior/eval.json`

| Recorded field | Value |
|---|---|
| task | lift |
| seed | 0 |
| arm | reward_action_prior |
| eval_base_return | 183.869 |
| eval_base_reward_weight | 0.35 |
| eval_shaped_return | -0.464694 |
| eval_reward_template_returns | 0.103957, -0.023964, 0.000019, 0.000081, 0, 0.010427, 0.004108, -0.200746, 0.104688, 0, -0.516207, -0.04584, -0.020891, 0, 0, 0 |
| eval_action_prior_weights | 0.22, 0.2, 0.18, 0.1, 0.24, 0.18 |
| eval_train_return | 63.889442 |
| eval_success_rate | 0.453125 |
| eval_instant_success_rate | 0.637695 |
| eval_lift_max | 0.195064 |
| eval_hard_clip_frac | 0 |
| eval_saturation_frac | 0.000006 |
| eval_action_abs_mean | 0.408233 |
| eval_summary | 0.050966, 0.036159, 0.126953, 0.659101, 0.195064, 0.397492 |

### Recorded results: `lift_s0_reward_supervised_init/eval.json`

| Recorded field | Value |
|---|---|
| task | lift |
| seed | 0 |
| arm | reward_supervised_init |
| eval_base_return | 36.497444 |
| eval_base_reward_weight | 0.4 |
| eval_shaped_return | -0.614108 |
| eval_reward_template_returns | -0.135866, -0.048913, 0.000038, 0.000027, 0, 0.006726, -0.00744, 0, 0.051069, 0.000439, 0.000425, -2.05423, -0.203481, 0, 0, 0 |
| eval_action_prior_weights | 0.22, 0.2, 0.18, 0.1, 0.24, 0.18 |
| eval_train_return | 13.984871 |
| eval_success_rate | 0.075195 |
| eval_instant_success_rate | 0.785156 |
| eval_lift_max | 0.103619 |
| eval_hard_clip_frac | 0 |
| eval_saturation_frac | 0.000153 |
| eval_action_abs_mean | 0.340544 |
| eval_summary | 0.101804, 0.041975, 0.061523, 0.371542, 0.103619, 1.015383 |

### Recorded results: `summary.json`

| Arm/interface | eval_success_rate | eval_instant_success_rate | eval_lift_max | eval_base_return | eval_train_return | eval_shaped_return | eval_action_abs_mean | eval_saturation_frac |
|---|---|---|---|---|---|---|---|---|
| baseline | 0.268555 | 0.799805 | 0.17236 | 61.414513 | 61.414513 | 0 | 0.295245 | 0.00138 |
| reward_action_prior | 0.453125 | 0.637695 | 0.195064 | 183.869 | 63.889442 | -0.464694 | 0.408233 | 0.000006 |
| reward_supervised_init | 0.075195 | 0.785156 | 0.103619 | 36.497444 | 13.984871 | -0.614108 | 0.340544 | 0.000153 |

### Training-metric reconstruction

| Arm/sub-run | Rows | Last iter | Last env steps | Elapsed s | Last success | Best success | Last lift max | Last base return | Best graded/objective |
|---|---|---|---|---|---|---|---|---|---|
| baseline | 344 | 343 | — | 14,406.829 | 0.130859 | 0.140625 | 0.114438 | 39.709412 | — |
| reward_action_prior | 347 | 346 | — | 7,208.2 | 0.354492 | 0.519531 | 0.159557 | 133.637 | — |
| reward_supervised_init | 353 | 352 | — | 7,202.625 | 0.05957 | 0.074219 | 0.098418 | 31.859348 | — |

These are training-log endpoints and maxima, not substitutes for held-out evaluation. They are included especially for interrupted runs whose terminal report is missing.

## What the results mean and major discoveries

- Among the recorded arms, `reward_action_prior` had the highest held-out success (eval_success_rate = 0.453125). This is an ordering within this run, not a general ranking across experiments with different metric definitions or budgets.
- The arm comparison is most informative when arms shared the same environment, seed, and budget. Where the configuration records only one seed, apparent gaps remain vulnerable to optimization variance.

## Limitations and reading guidance

- Metrics are reported under the names used by the historical environment and runner. Similar-looking metrics from different generations of the code are not assumed to be numerically interchangeable.
- A maximum observed in training is not treated as held-out performance. Terminal evaluation values are preferred whenever present.
- Missing summaries, empty metric streams, failed retries, and incomplete arms are preserved as negative evidence rather than silently dropped.
- Candidate names and stage names are LLM-authored run artifacts. Interpretations in this historical log do not alter framework prompts, defaults, diagnostics, or task logic.

## Artifact provenance

| Artifact type | Count |
|---|---|
| .json | 69 |
| .pkl | 23 |
| .md | 11 |
| .svg | 11 |
| .txt | 10 |
| .csv | 2 |
| .jsonl | 1 |

Primary evidence files:

- [`REPORT.md`](../runs/pareto_fixed_prior_shaped_reward_12h_nowarmup_20260622-203237/REPORT.md)
- [`config.json`](../runs/pareto_fixed_prior_shaped_reward_12h_nowarmup_20260622-203237/config.json)
- [`metrics.jsonl`](../runs/pareto_fixed_prior_shaped_reward_12h_nowarmup_20260622-203237/metrics.jsonl)
- [`summary.json`](../runs/pareto_fixed_prior_shaped_reward_12h_nowarmup_20260622-203237/summary.json)

## Appendix: original authored run report

The following contemporaneous report is retained (with headings demoted) so that details and caveats are not lost in normalization.

### Source: [`REPORT.md`](../runs/pareto_fixed_prior_shaped_reward_12h_nowarmup_20260622-203237/REPORT.md)

## Dynamic Reward Combination Experiment

Created: 2026-06-23 08:42:17

### Configuration

- Arms: `baseline,reward_action_prior,reward_supervised_init`
- Tasks: `lift`
- Target seconds per arm: `14400.0`
- Cheap checkup steps: `50`
- Deep checkup seconds: `7200.0`
- Post-new-reward checkup steps: `10`
- Post-new-reward fast checkups: `3`
- Reward rewrite: `True`
- Rewrite fraction: `0.5`
- LLM action prior: `False`
- Action prior checkup steps: `0`
- Action transform: `tanh`

### Summary

```json
{
  "baseline": {
    "eval_success_rate": 0.268555,
    "eval_instant_success_rate": 0.799805,
    "eval_base_return": 61.414513,
    "eval_shaped_return": 0.0,
    "eval_train_return": 61.414513,
    "eval_lift_max": 0.17236,
    "eval_hard_clip_frac": 0.0,
    "eval_saturation_frac": 0.00138,
    "eval_action_abs_mean": 0.295245,
    "n_eval": 1
  },
  "reward_action_prior": {
    "eval_success_rate": 0.453125,
    "eval_instant_success_rate": 0.637695,
    "eval_base_return": 183.868958,
    "eval_shaped_return": -0.464694,
    "eval_train_return": 63.889442,
    "eval_lift_max": 0.195064,
    "eval_hard_clip_frac": 0.0,
    "eval_saturation_frac": 6e-06,
    "eval_action_abs_mean": 0.408233,
    "n_eval": 1
  },
  "reward_supervised_init": {
    "eval_success_rate": 0.075195,
    "eval_instant_success_rate": 0.785156,
    "eval_base_return": 36.497444,
    "eval_shaped_return": -0.614108,
    "eval_train_return": 13.984871,
    "eval_lift_max": 0.103619,
    "eval_hard_clip_frac": 0.0,
    "eval_saturation_frac": 0.000153,
    "eval_action_abs_mean": 0.340544,
    "n_eval": 1
  }
}
```

### Notes

Reward coach prompts intentionally omit env reward, train return, base return, and PPO losses.
Rejected and deep-checkup proposals are stored under each arm's `reward_checkups/` directory.

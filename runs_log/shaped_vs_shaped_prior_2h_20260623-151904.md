# Shaped Vs Shaped Prior 2h

- **Run directory:** [`runs/shaped_vs_shaped_prior_2h_20260623-151904`](../runs/shaped_vs_shaped_prior_2h_20260623-151904)
- **Date:** 2026-06-23 15:19:04 (directory timestamp)
- **Status:** completed or completed-with-caveats
- **Experiment class:** PPO bias/reward/prior experiment
- **Recorded task:** lift
- **Training metric rows recovered:** 344

## Abstract

This entry reconstructs `shaped_vs_shaped_prior_2h_20260623-151904`, a PPO bias/reward/prior experiment. The account is based only on artifacts that survive in the repository: machine-readable configuration, candidate programs, selection diagnostics, training metrics, evaluations, reports, and logs. The run targeted lift. Its purpose and comparisons are inferred from the recorded arm names and configuration where no authored abstract survives; those inferences are identified by this provenance-first framing rather than presented as new experimental facts.

## Experimental structure

| Field | Recorded value |
|---|---|
| learner | ppo_dynamic_reward |
| tasks | lift |
| arms | reward_action_prior, reward |
| arm_mechanisms.reward_action_prior.reward | yes |
| arm_mechanisms.reward_action_prior.action_prior | yes |
| arm_mechanisms.reward_action_prior.exploration | no |
| arm_mechanisms.reward_action_prior.supervised_init | no |
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
| ppo.target_train_seconds | 7,200 |
| ppo.max_env_steps | — |
| ppo.action_transform | tanh |
| ppo.saturation_penalty | 0 |
| ppo.saturation_threshold | 0.98 |
| ppo.prior_logit_clip | 0.95 |
| ppo.action_target_reward_weight | 0 |
| ppo.success_hold_seconds | 0.5 |
| ppo.success_lift_threshold | 0.05 |
| ppo.warmup_compile | yes |
| dynamic_reward.cheap_checkup_steps | 50 |
| dynamic_reward.deep_checkup_seconds | 3,600 |
| dynamic_reward.max_template_weight | 1.25 |
| dynamic_reward.min_base_reward_weight | 0 |
| dynamic_reward.post_new_reward_checkup_steps | 10 |
| dynamic_reward.post_new_reward_fast_checkups | 3 |
| dynamic_reward.allow_reward_rewrite | yes |
| dynamic_reward.pre_run_reward_analysis | yes |
| dynamic_reward.initial_reward_program | yes |
| dynamic_reward.anneal_shaping | yes |
| dynamic_reward.shaping_anneal_start_fraction | 0.7 |
| dynamic_reward.max_env_steps | — |
| dynamic_reward.efficiency_success_threshold | 0.2 |
| dynamic_reward.rewrite_fraction | 0.5 |
| dynamic_reward.previous_run_dir | — |
| dynamic_reward.xla_cache | /home/robin/Documents/agent-mini-script-control/llm-framework/.xla_cache |
| dynamic_action_prior.llm_action_prior | no |
| dynamic_action_prior.pareto_action_prior | yes |
| dynamic_action_prior.action_prior_candidates | 5 |
| dynamic_action_prior.pareto_supervised_init | no |
| dynamic_action_prior.supervised_candidates | 5 |
| dynamic_action_prior.selection_envs | 128 |
| dynamic_action_prior.selection_steps | — |
| dynamic_action_prior.selection_seed | 0 |
| dynamic_action_prior.checkup_steps | 0 |
| dynamic_action_prior.max_checkups | 3 |
| dynamic_action_prior.max_weight | 0.6 |
| llm_backend | claude-code |
| llm_model | — |

### Arms and sub-runs represented by directories

`lift_s0_reward`, `lift_s0_reward_action_prior`, `llm_initial_bias`, `pareto_action_prior`, `pre_run_reward_analysis`

## Results

### Recorded results: `lift_s0_reward/eval.json`

| Recorded field | Value |
|---|---|
| task | lift |
| seed | 0 |
| arm | reward |
| eval_base_return | -3.024993 |
| eval_base_reward_weight | 0.301266 |
| eval_shaped_return | 0.064877 |
| eval_reward_template_returns | 0.021965, -0.023936, 0.000006, 0.000047, 0.000001, 0.004579, 0.003297, 0, 0.048546, 0.000586, 0, 0, -4.10125, -2.579309, 0, 0 |
| eval_action_prior_weights | 0.32, 0.16, 0.22, 0.12 |
| eval_train_return | -0.846451 |
| eval_success_rate | 0.010742 |
| eval_instant_success_rate | 0.102539 |
| eval_lift_max | 0.024096 |
| eval_hard_clip_frac | 0 |
| eval_saturation_frac | 0 |
| eval_action_abs_mean | 0.237359 |
| eval_summary | 0.074346, 0.043142, 0.025391, 0.260583, 0.024096, 0.156197 |
| efficiency_success_threshold | 0.2 |
| steps_to_success_threshold | — |
| reached_success_threshold | no |
| success_step_auc | 0.022592 |
| total_env_steps | 17305600 |

### Recorded results: `lift_s0_reward_action_prior/eval.json`

| Recorded field | Value |
|---|---|
| task | lift |
| seed | 0 |
| arm | reward_action_prior |
| eval_base_return | 3.87551 |
| eval_base_reward_weight | 0.513287 |
| eval_shaped_return | -0.000139 |
| eval_reward_template_returns | 0.02212, -0.02386, 0, 0.00006, 0, 0.006135, 0.009725, 0, 0.000163, -0.009694, -0.22428, 0, 0, 0, 0, 0 |
| eval_action_prior_weights | 0.32, 0.16, 0.22, 0.12 |
| eval_train_return | 1.989111 |
| eval_success_rate | 0.033203 |
| eval_instant_success_rate | 0.15625 |
| eval_lift_max | 0.037614 |
| eval_hard_clip_frac | 0 |
| eval_saturation_frac | 0.000011 |
| eval_action_abs_mean | 0.232027 |
| eval_summary | 0.091221, 0.039632, 0.027344, 0.171308, 0.037614, 0.143041 |
| efficiency_success_threshold | 0.2 |
| steps_to_success_threshold | — |
| reached_success_threshold | no |
| success_step_auc | 0.006482 |
| total_env_steps | 17920000 |

### Recorded results: `summary.json`

| Arm/interface | eval_success_rate | eval_instant_success_rate | eval_lift_max | eval_base_return | eval_train_return | eval_shaped_return | eval_action_abs_mean | eval_saturation_frac | mean_total_env_steps |
|---|---|---|---|---|---|---|---|---|---|
| reward | 0.010742 | 0.102539 | 0.024096 | -3.024993 | -0.846451 | 0.064877 | 0.237359 | 0 | 17,305,600 |
| reward_action_prior | 0.033203 | 0.15625 | 0.037614 | 3.87551 | 1.989111 | -0.000139 | 0.232027 | 0.000011 | 17,920,000 |

### Training-metric reconstruction

| Arm/sub-run | Rows | Last iter | Last env steps | Elapsed s | Last success | Best success | Last lift max | Last base return | Best graded/objective |
|---|---|---|---|---|---|---|---|---|---|
| reward_action_prior | 175 | 174 | 17920000 | 3,637.959 | 0.025391 | 0.025391 | 0.029096 | -2.379854 | — |
| reward | 169 | 168 | 17305600 | 3,608.644 | 0.009766 | 0.075195 | 0.030971 | -2.339577 | — |

These are training-log endpoints and maxima, not substitutes for held-out evaluation. They are included especially for interrupted runs whose terminal report is missing.

## What the results mean and major discoveries

- Among the recorded arms, `reward_action_prior` had the highest held-out success (eval_success_rate = 0.033203). This is an ordering within this run, not a general ranking across experiments with different metric definitions or budgets.
- The arm comparison is most informative when arms shared the same environment, seed, and budget. Where the configuration records only one seed, apparent gaps remain vulnerable to optimization variance.

## Limitations and reading guidance

- Metrics are reported under the names used by the historical environment and runner. Similar-looking metrics from different generations of the code are not assumed to be numerically interchangeable.
- A maximum observed in training is not treated as held-out performance. Terminal evaluation values are preferred whenever present.
- Missing summaries, empty metric streams, failed retries, and incomplete arms are preserved as negative evidence rather than silently dropped.
- Candidate names and stage names are LLM-authored run artifacts. Interpretations in this historical log do not alter framework prompts, defaults, diagnostics, or task logic.

## Artifact provenance

| Artifact type | Count |
|---|---|
| .json | 62 |
| .md | 16 |
| .pkl | 16 |
| .txt | 15 |
| .csv | 1 |
| .jsonl | 1 |

Primary evidence files:

- [`REPORT.md`](../runs/shaped_vs_shaped_prior_2h_20260623-151904/REPORT.md)
- [`config.json`](../runs/shaped_vs_shaped_prior_2h_20260623-151904/config.json)
- [`metrics.jsonl`](../runs/shaped_vs_shaped_prior_2h_20260623-151904/metrics.jsonl)
- [`summary.json`](../runs/shaped_vs_shaped_prior_2h_20260623-151904/summary.json)

## Appendix: original authored run report

The following contemporaneous report is retained (with headings demoted) so that details and caveats are not lost in normalization.

### Source: [`REPORT.md`](../runs/shaped_vs_shaped_prior_2h_20260623-151904/REPORT.md)

## Dynamic Reward Combination Experiment

Created: 2026-06-23 19:42:42

### Configuration

- Arms: `reward_action_prior,reward`
- Tasks: `lift`
- Target seconds per arm: `7200.0`
- Cheap checkup steps: `50`
- Deep checkup seconds: `3600.0`
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
  "reward": {
    "eval_success_rate": 0.010742,
    "eval_instant_success_rate": 0.102539,
    "eval_base_return": -3.024993,
    "eval_shaped_return": 0.064877,
    "eval_train_return": -0.846451,
    "eval_lift_max": 0.024096,
    "eval_hard_clip_frac": 0.0,
    "eval_saturation_frac": 0.0,
    "eval_action_abs_mean": 0.237359,
    "n_eval": 1,
    "efficiency_success_threshold": 0.2,
    "reached_success_threshold_frac": 0.0,
    "mean_steps_to_success_threshold": null,
    "mean_success_step_auc": 0.022592,
    "mean_total_env_steps": 17305600.0
  },
  "reward_action_prior": {
    "eval_success_rate": 0.033203,
    "eval_instant_success_rate": 0.15625,
    "eval_base_return": 3.87551,
    "eval_shaped_return": -0.000139,
    "eval_train_return": 1.989111,
    "eval_lift_max": 0.037614,
    "eval_hard_clip_frac": 0.0,
    "eval_saturation_frac": 1.1e-05,
    "eval_action_abs_mean": 0.232027,
    "n_eval": 1,
    "efficiency_success_threshold": 0.2,
    "reached_success_threshold_frac": 0.0,
    "mean_steps_to_success_threshold": null,
    "mean_success_step_auc": 0.006482,
    "mean_total_env_steps": 17920000.0
  }
}
```

### Notes

Reward coach prompts intentionally omit env reward, train return, base return, and PPO losses.
Rejected and deep-checkup proposals are stored under each arm's `reward_checkups/` directory.

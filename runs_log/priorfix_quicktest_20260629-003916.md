# Priorfix Quicktest

- **Run directory:** [`runs/priorfix_quicktest_20260629-003916`](../runs/priorfix_quicktest_20260629-003916)
- **Date:** 2026-06-29 00:39:16 (directory timestamp)
- **Status:** completed or completed-with-caveats
- **Experiment class:** PPO bias/reward/prior experiment
- **Recorded task:** lift
- **Training metric rows recovered:** 63

## Abstract

This entry reconstructs `priorfix_quicktest_20260629-003916`, a PPO bias/reward/prior experiment. The account is based only on artifacts that survive in the repository: machine-readable configuration, candidate programs, selection diagnostics, training metrics, evaluations, reports, and logs. The run targeted lift. Its purpose and comparisons are inferred from the recorded arm names and configuration where no authored abstract survives; those inferences are identified by this provenance-first framing rather than presented as new experimental facts.

## Experimental structure

| Field | Recorded value |
|---|---|
| learner | ppo_dynamic_reward |
| tasks | lift |
| arms | reward_action_prior |
| arm_mechanisms.reward_action_prior.reward | yes |
| arm_mechanisms.reward_action_prior.action_prior | yes |
| arm_mechanisms.reward_action_prior.exploration | no |
| arm_mechanisms.reward_action_prior.supervised_init | no |
| seeds | 0 |
| env.obs_size | 89 |
| env.action_size | 23 |
| env.horizon | 100 |
| env.frame_skip | 2 |
| env.actuators | rh_A_WRJ2, rh_A_WRJ1, rh_A_THJ5, rh_A_THJ4, rh_A_THJ3, rh_A_THJ2, rh_A_THJ1, rh_A_FFJ4, rh_A_FFJ3, rh_A_FFJ0, rh_A_MFJ4, rh_A_MFJ3, rh_A_MFJ0, rh_A_RFJ4, rh_A_RFJ3, rh_A_RFJ0, rh_A_LFJ5, rh_A_LFJ4, rh_A_LFJ3, rh_A_LFJ0, base_x, base_y, base_z |
| env.tasks.lift.objective | raise object above starting height |
| env.tasks.lift.success | lift > 0.05m |
| ppo.iters | 100000 |
| ppo.envs | 256 |
| ppo.lr | 0.0003 |
| ppo.gamma | 0.99 |
| ppo.lam | 0.95 |
| ppo.hidden | 128, 128 |
| ppo.ent_coef | 0 |
| ppo.supervised_steps | 80 |
| ppo.supervised_batch | 128 |
| ppo.supervised_lr | 0.001 |
| ppo.bc_critic_pretrain | yes |
| ppo.bc_rollout_states | yes |
| ppo.bc_kl_coef | 0.5 |
| ppo.bc_kl_anneal_iters | 200 |
| ppo.checkpoint_count | 5 |
| ppo.target_train_seconds | 700 |
| ppo.max_env_steps | — |
| ppo.action_transform | tanh |
| ppo.saturation_penalty | 0 |
| ppo.saturation_threshold | 0.98 |
| ppo.prior_logit_clip | 0.95 |
| ppo.action_target_reward_weight | 0 |
| ppo.success_hold_seconds | 0.5 |
| ppo.success_lift_threshold | 0.05 |
| ppo.warmup_compile | yes |
| dynamic_reward.cheap_checkup_steps | 1000000000 |
| dynamic_reward.deep_checkup_seconds | 7,200 |
| dynamic_reward.max_template_weight | 1.25 |
| dynamic_reward.min_base_reward_weight | 0.5 |
| dynamic_reward.post_new_reward_checkup_steps | 10 |
| dynamic_reward.post_new_reward_fast_checkups | 3 |
| dynamic_reward.allow_reward_rewrite | no |
| dynamic_reward.pre_run_reward_analysis | no |
| dynamic_reward.initial_reward_program | no |
| dynamic_reward.freeze_reward_shaping | yes |
| dynamic_reward.anneal_shaping | no |
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
| llm_backend | fixture |
| llm_model | — |

### Arms and sub-runs represented by directories

`lift_s0_reward_action_prior`, `pareto_action_prior`

## Results

### Recorded results: `lift_s0_reward_action_prior/eval.json`

| Recorded field | Value |
|---|---|
| task | lift |
| seed | 0 |
| arm | reward_action_prior |
| best_checkpoint_iter | 0 |
| best_train_success | 0 |
| eval_base_return | 3.274767 |
| eval_base_reward_weight | 1 |
| eval_shaped_return | 0.202959 |
| eval_reward_template_returns | 0.202959, -0.023971, 0, 0, 0, 0.014806, 0.022122, -0.000068, 0, 0, 0, 0, 0, 0, 0, 0 |
| eval_action_prior_weights | 0.266, 0.306, 0.306, 0.255, 0.13 |
| eval_train_return | 3.477726 |
| eval_success_rate | 0 |
| eval_instant_success_rate | 0 |
| eval_reach_rate | 0 |
| eval_grasp_rate | 0 |
| eval_lift_reached_rate | 0.042969 |
| eval_lift_max | 0.016861 |
| eval_hard_clip_frac | 0 |
| eval_saturation_frac | 0 |
| eval_action_abs_mean | 0.112999 |
| eval_summary | 0.035739, 0.039689, 0, 0.496584, 0.016861, 0.131163 |
| efficiency_success_threshold | 0.2 |
| steps_to_success_threshold | — |
| reached_success_threshold | no |
| success_step_auc | 0 |
| total_env_steps | 1612800 |

### Recorded results: `summary.json`

| Arm/interface | eval_success_rate | eval_instant_success_rate | eval_lift_max | eval_base_return | eval_train_return | eval_shaped_return | eval_action_abs_mean | eval_saturation_frac | mean_total_env_steps |
|---|---|---|---|---|---|---|---|---|---|
| reward_action_prior | 0 | 0 | 0.016861 | 3.274767 | 3.477726 | 0.202959 | 0.112999 | 0 | 1,612,800 |

### Training-metric reconstruction

| Arm/sub-run | Rows | Last iter | Last env steps | Elapsed s | Last success | Best success | Last lift max | Last base return | Best graded/objective |
|---|---|---|---|---|---|---|---|---|---|
| reward_action_prior | 63 | 62 | 1612800 | 709.159 | 0 | 0 | 0.013949 | 6.00054 | — |

These are training-log endpoints and maxima, not substitutes for held-out evaluation. They are included especially for interrupted runs whose terminal report is missing.

## What the results mean and major discoveries

- Among the recorded arms, `reward_action_prior` had the highest held-out success (eval_success_rate = 0). This is an ordering within this run, not a general ranking across experiments with different metric definitions or budgets.

## Limitations and reading guidance

- Metrics are reported under the names used by the historical environment and runner. Similar-looking metrics from different generations of the code are not assumed to be numerically interchangeable.
- A maximum observed in training is not treated as held-out performance. Terminal evaluation values are preferred whenever present.
- Missing summaries, empty metric streams, failed retries, and incomplete arms are preserved as negative evidence rather than silently dropped.
- Candidate names and stage names are LLM-authored run artifacts. Interpretations in this historical log do not alter framework prompts, defaults, diagnostics, or task logic.

## Artifact provenance

| Artifact type | Count |
|---|---|
| .json | 14 |
| .pkl | 8 |
| .csv | 1 |
| .jsonl | 1 |
| .md | 1 |

Primary evidence files:

- [`REPORT.md`](../runs/priorfix_quicktest_20260629-003916/REPORT.md)
- [`config.json`](../runs/priorfix_quicktest_20260629-003916/config.json)
- [`metrics.jsonl`](../runs/priorfix_quicktest_20260629-003916/metrics.jsonl)
- [`summary.json`](../runs/priorfix_quicktest_20260629-003916/summary.json)

## Appendix: original authored run report

The following contemporaneous report is retained (with headings demoted) so that details and caveats are not lost in normalization.

### Source: [`REPORT.md`](../runs/priorfix_quicktest_20260629-003916/REPORT.md)

## Dynamic Reward Combination Experiment

Created: 2026-06-29 00:55:58

### Configuration

- Arms: `reward_action_prior`
- Tasks: `lift`
- Target seconds per arm: `700.0`
- Cheap checkup steps: `1000000000`
- Deep checkup seconds: `7200.0`
- Post-new-reward checkup steps: `10`
- Post-new-reward fast checkups: `3`
- Reward rewrite: `False`
- Rewrite fraction: `0.5`
- LLM action prior: `False`
- Action prior checkup steps: `0`
- Action transform: `tanh`

### Summary

```json
{
  "reward_action_prior": {
    "eval_success_rate": 0.0,
    "eval_instant_success_rate": 0.0,
    "eval_base_return": 3.274767,
    "eval_shaped_return": 0.202959,
    "eval_train_return": 3.477726,
    "eval_lift_max": 0.016861,
    "eval_hard_clip_frac": 0.0,
    "eval_saturation_frac": 0.0,
    "eval_action_abs_mean": 0.112999,
    "n_eval": 1,
    "efficiency_success_threshold": 0.2,
    "reached_success_threshold_frac": 0.0,
    "mean_steps_to_success_threshold": null,
    "mean_success_step_auc": 0.0,
    "mean_total_env_steps": 1612800.0
  }
}
```

### Notes

Reward coach prompts intentionally omit env reward, train return, base return, and PPO losses.
Rejected and deep-checkup proposals are stored under each arm's `reward_checkups/` directory.

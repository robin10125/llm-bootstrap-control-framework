# Diagnostic No Warmup 32env

- **Run directory:** [`runs/diagnostic_no_warmup_32env_20260622-202913`](../runs/diagnostic_no_warmup_32env_20260622-202913)
- **Date:** 2026-06-22 20:29:13 (directory timestamp)
- **Status:** completed or completed-with-caveats
- **Experiment class:** PPO bias/reward/prior experiment
- **Recorded task:** lift
- **Training metric rows recovered:** 1

## Abstract

This entry reconstructs `diagnostic_no_warmup_32env_20260622-202913`, a PPO bias/reward/prior experiment. The account is based only on artifacts that survive in the repository: machine-readable configuration, candidate programs, selection diagnostics, training metrics, evaluations, reports, and logs. The run targeted lift. Its purpose and comparisons are inferred from the recorded arm names and configuration where no authored abstract survives; those inferences are identified by this provenance-first framing rather than presented as new experimental facts.

## Experimental structure

| Field | Recorded value |
|---|---|
| learner | ppo_dynamic_reward |
| tasks | lift |
| arms | baseline |
| arm_mechanisms.baseline.reward | no |
| arm_mechanisms.baseline.action_prior | no |
| arm_mechanisms.baseline.exploration | no |
| arm_mechanisms.baseline.supervised_init | no |
| seeds | 0 |
| env.obs_size | 89 |
| env.action_size | 23 |
| env.horizon | 100 |
| env.frame_skip | 2 |
| env.actuators | rh_A_WRJ2, rh_A_WRJ1, rh_A_THJ5, rh_A_THJ4, rh_A_THJ3, rh_A_THJ2, rh_A_THJ1, rh_A_FFJ4, rh_A_FFJ3, rh_A_FFJ0, rh_A_MFJ4, rh_A_MFJ3, rh_A_MFJ0, rh_A_RFJ4, rh_A_RFJ3, rh_A_RFJ0, rh_A_LFJ5, rh_A_LFJ4, rh_A_LFJ3, rh_A_LFJ0, base_x, base_y, base_z |
| env.tasks.lift.objective | raise object above starting height |
| env.tasks.lift.success | lift > 0.05m |
| ppo.iters | 1 |
| ppo.envs | 32 |
| ppo.lr | 0.0003 |
| ppo.gamma | 0.99 |
| ppo.lam | 0.95 |
| ppo.hidden | 128, 128 |
| ppo.ent_coef | 0 |
| ppo.supervised_steps | 80 |
| ppo.supervised_batch | 128 |
| ppo.supervised_lr | 0.001 |
| ppo.checkpoint_count | 0 |
| ppo.target_train_seconds | 1 |
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
| dynamic_reward.allow_reward_rewrite | no |
| dynamic_reward.pre_run_reward_analysis | no |
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
| llm_backend | fixture |
| llm_model | — |

### Arms and sub-runs represented by directories

`lift_s0_baseline`

## Results

### Recorded results: `lift_s0_baseline/eval.json`

| Recorded field | Value |
|---|---|
| task | lift |
| seed | 0 |
| arm | baseline |
| eval_base_return | -15.413185 |
| eval_base_reward_weight | 1 |
| eval_shaped_return | 0 |
| eval_reward_template_returns | 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 |
| eval_action_prior_weights | 0.22, 0.2, 0.18, 0.1, 0.24, 0.18 |
| eval_train_return | -15.413185 |
| eval_success_rate | 0 |
| eval_instant_success_rate | 0 |
| eval_lift_max | 0.000376 |
| eval_hard_clip_frac | 0 |
| eval_saturation_frac | 0 |
| eval_action_abs_mean | 0.004452 |
| eval_summary | 0.116228, 0.058552, 0, 0.206599, 0.000376, 0.001755 |

### Recorded results: `summary.json`

| Arm/interface | eval_success_rate | eval_instant_success_rate | eval_lift_max | eval_base_return | eval_train_return | eval_shaped_return | eval_action_abs_mean | eval_saturation_frac |
|---|---|---|---|---|---|---|---|---|
| baseline | 0 | 0 | 0.000376 | -15.413185 | -15.413185 | 0 | 0.004452 | 0 |

### Training-metric reconstruction

| Arm/sub-run | Rows | Last iter | Last env steps | Elapsed s | Last success | Best success | Last lift max | Last base return | Best graded/objective |
|---|---|---|---|---|---|---|---|---|---|
| baseline | 1 | 0 | — | 22.868 | 0 | 0 | 0.005178 | -28.91449 | — |

These are training-log endpoints and maxima, not substitutes for held-out evaluation. They are included especially for interrupted runs whose terminal report is missing.

## What the results mean and major discoveries

- Among the recorded arms, `baseline` had the highest held-out success (eval_success_rate = 0). This is an ordering within this run, not a general ranking across experiments with different metric definitions or budgets.

## Limitations and reading guidance

- Metrics are reported under the names used by the historical environment and runner. Similar-looking metrics from different generations of the code are not assumed to be numerically interchangeable.
- A maximum observed in training is not treated as held-out performance. Terminal evaluation values are preferred whenever present.
- Missing summaries, empty metric streams, failed retries, and incomplete arms are preserved as negative evidence rather than silently dropped.
- Candidate names and stage names are LLM-authored run artifacts. Interpretations in this historical log do not alter framework prompts, defaults, diagnostics, or task logic.

## Artifact provenance

| Artifact type | Count |
|---|---|
| .json | 9 |
| .csv | 1 |
| .jsonl | 1 |
| .md | 1 |
| .pkl | 1 |

Primary evidence files:

- [`REPORT.md`](../runs/diagnostic_no_warmup_32env_20260622-202913/REPORT.md)
- [`config.json`](../runs/diagnostic_no_warmup_32env_20260622-202913/config.json)
- [`metrics.jsonl`](../runs/diagnostic_no_warmup_32env_20260622-202913/metrics.jsonl)
- [`summary.json`](../runs/diagnostic_no_warmup_32env_20260622-202913/summary.json)

## Appendix: original authored run report

The following contemporaneous report is retained (with headings demoted) so that details and caveats are not lost in normalization.

### Source: [`REPORT.md`](../runs/diagnostic_no_warmup_32env_20260622-202913/REPORT.md)

## Dynamic Reward Combination Experiment

Created: 2026-06-22 20:32:15

### Configuration

- Arms: `baseline`
- Tasks: `lift`
- Target seconds per arm: `1.0`
- Cheap checkup steps: `50`
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
  "baseline": {
    "eval_success_rate": 0.0,
    "eval_instant_success_rate": 0.0,
    "eval_base_return": -15.413185,
    "eval_shaped_return": 0.0,
    "eval_train_return": -15.413185,
    "eval_lift_max": 0.000376,
    "eval_hard_clip_frac": 0.0,
    "eval_saturation_frac": 0.0,
    "eval_action_abs_mean": 0.004452,
    "n_eval": 1
  }
}
```

### Notes

Reward coach prompts intentionally omit env reward, train return, base return, and PPO losses.
Rejected and deep-checkup proposals are stored under each arm's `reward_checkups/` directory.

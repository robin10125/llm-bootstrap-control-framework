# Action Prior General Prompt 2h

- **Run directory:** [`runs/action_prior_general_prompt_2h_20260622-125750`](../runs/action_prior_general_prompt_2h_20260622-125750)
- **Date:** 2026-06-22 12:57:50 (directory timestamp)
- **Status:** completed or completed-with-caveats
- **Experiment class:** PPO bias/reward/prior experiment
- **Recorded task:** lift
- **Training metric rows recovered:** 168

## Abstract

This entry reconstructs `action_prior_general_prompt_2h_20260622-125750`, a PPO bias/reward/prior experiment. The account is based only on artifacts that survive in the repository: machine-readable configuration, candidate programs, selection diagnostics, training metrics, evaluations, reports, and logs. The run targeted lift. Its purpose and comparisons are inferred from the recorded arm names and configuration where no authored abstract survives; those inferences are identified by this provenance-first framing rather than presented as new experimental facts.

## Experimental structure

| Field | Recorded value |
|---|---|
| learner | ppo_dynamic_reward |
| tasks | lift |
| arms | action_prior |
| arm_mechanisms.action_prior.reward | no |
| arm_mechanisms.action_prior.action_prior | yes |
| arm_mechanisms.action_prior.exploration | no |
| arm_mechanisms.action_prior.supervised_init | no |
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
| ppo.action_transform | tanh |
| ppo.saturation_penalty | 0 |
| ppo.saturation_threshold | 0.98 |
| ppo.prior_logit_clip | 0.95 |
| ppo.action_target_reward_weight | 0 |
| ppo.success_hold_seconds | 0.5 |
| ppo.success_lift_threshold | 0.05 |
| dynamic_reward.cheap_checkup_steps | 50 |
| dynamic_reward.deep_checkup_seconds | 7,200 |
| dynamic_reward.max_template_weight | 1.25 |
| dynamic_reward.min_base_reward_weight | 0 |
| dynamic_reward.post_new_reward_checkup_steps | 10 |
| dynamic_reward.post_new_reward_fast_checkups | 3 |
| dynamic_reward.allow_reward_rewrite | no |
| dynamic_reward.pre_run_reward_analysis | yes |
| dynamic_reward.rewrite_fraction | 0.5 |
| dynamic_reward.previous_run_dir | — |
| dynamic_reward.xla_cache | /home/robin/Documents/agent-mini-script-control/llm-framework/.xla_cache |
| dynamic_action_prior.llm_action_prior | yes |
| dynamic_action_prior.checkup_steps | 50 |
| dynamic_action_prior.max_checkups | 3 |
| dynamic_action_prior.max_weight | 0.6 |
| llm_backend | codex |
| llm_model | — |

### Arms and sub-runs represented by directories

`lift_s0_action_prior`, `llm_action_prior`, `llm_initial_bias`, `pre_run_reward_analysis`

## Results

### Recorded results: `lift_s0_action_prior/eval.json`

| Recorded field | Value |
|---|---|
| task | lift |
| seed | 0 |
| arm | action_prior |
| eval_base_return | 2.116148 |
| eval_base_reward_weight | 1 |
| eval_shaped_return | 0 |
| eval_reward_template_returns | 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 |
| eval_action_prior_weights | 0.24, 0.2, 0.22, 0.21, 0.2, 0.13, 0.11, 0.06 |
| eval_train_return | 2.116148 |
| eval_success_rate | 0.008789 |
| eval_instant_success_rate | 0.176758 |
| eval_lift_max | 0.036141 |
| eval_hard_clip_frac | 0 |
| eval_saturation_frac | 0.000101 |
| eval_action_abs_mean | 0.225717 |
| eval_summary | 0.053105, 0.038335, 0.106445, 0.30712, 0.036141, 0.391712 |

### Recorded results: `summary.json`

| Arm/interface | eval_success_rate | eval_instant_success_rate | eval_lift_max | eval_base_return | eval_train_return | eval_shaped_return | eval_action_abs_mean | eval_saturation_frac |
|---|---|---|---|---|---|---|---|---|
| action_prior | 0.008789 | 0.176758 | 0.036141 | 2.116148 | 2.116148 | 0 | 0.225717 | 0.000101 |

### Training-metric reconstruction

| Arm/sub-run | Rows | Last iter | Last env steps | Elapsed s | Last success | Best success | Last lift max | Last base return | Best graded/objective |
|---|---|---|---|---|---|---|---|---|---|
| action_prior | 168 | 167 | — | 7,214.741 | 0.013672 | 0.019531 | 0.041061 | 1.679945 | — |

These are training-log endpoints and maxima, not substitutes for held-out evaluation. They are included especially for interrupted runs whose terminal report is missing.

## What the results mean and major discoveries

- Among the recorded arms, `action_prior` had the highest held-out success (eval_success_rate = 0.008789). This is an ordering within this run, not a general ranking across experiments with different metric definitions or budgets.

## Limitations and reading guidance

- Metrics are reported under the names used by the historical environment and runner. Similar-looking metrics from different generations of the code are not assumed to be numerically interchangeable.
- A maximum observed in training is not treated as held-out performance. Terminal evaluation values are preferred whenever present.
- Missing summaries, empty metric streams, failed retries, and incomplete arms are preserved as negative evidence rather than silently dropped.
- Candidate names and stage names are LLM-authored run artifacts. Interpretations in this historical log do not alter framework prompts, defaults, diagnostics, or task logic.

## Artifact provenance

| Artifact type | Count |
|---|---|
| .json | 27 |
| .md | 8 |
| .pkl | 7 |
| .txt | 7 |
| .csv | 1 |
| .jsonl | 1 |

Primary evidence files:

- [`REPORT.md`](../runs/action_prior_general_prompt_2h_20260622-125750/REPORT.md)
- [`config.json`](../runs/action_prior_general_prompt_2h_20260622-125750/config.json)
- [`metrics.jsonl`](../runs/action_prior_general_prompt_2h_20260622-125750/metrics.jsonl)
- [`summary.json`](../runs/action_prior_general_prompt_2h_20260622-125750/summary.json)

## Appendix: original authored run report

The following contemporaneous report is retained (with headings demoted) so that details and caveats are not lost in normalization.

### Source: [`REPORT.md`](../runs/action_prior_general_prompt_2h_20260622-125750/REPORT.md)

## Dynamic Reward Combination Experiment

Created: 2026-06-22 15:04:49

### Configuration

- Arms: `action_prior`
- Tasks: `lift`
- Target seconds per arm: `7200.0`
- Cheap checkup steps: `50`
- Deep checkup seconds: `7200.0`
- Post-new-reward checkup steps: `10`
- Post-new-reward fast checkups: `3`
- Reward rewrite: `False`
- Rewrite fraction: `0.5`
- LLM action prior: `True`
- Action prior checkup steps: `50`
- Action transform: `tanh`

### Summary

```json
{
  "action_prior": {
    "eval_success_rate": 0.008789,
    "eval_instant_success_rate": 0.176758,
    "eval_base_return": 2.116148,
    "eval_shaped_return": 0.0,
    "eval_train_return": 2.116148,
    "eval_lift_max": 0.036141,
    "eval_hard_clip_frac": 0.0,
    "eval_saturation_frac": 0.000101,
    "eval_action_abs_mean": 0.225717,
    "n_eval": 1
  }
}
```

### Notes

Reward coach prompts intentionally omit env reward, train return, base return, and PPO losses.
Rejected and deep-checkup proposals are stored under each arm's `reward_checkups/` directory.

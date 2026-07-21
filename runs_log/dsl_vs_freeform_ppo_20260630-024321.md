# DSL Vs Freeform PPO

- **Run directory:** [`runs/dsl_vs_freeform_ppo_20260630-024321`](../runs/dsl_vs_freeform_ppo_20260630-024321)
- **Date:** 2026-06-30 02:43:21 (directory timestamp)
- **Status:** completed or completed-with-caveats
- **Experiment class:** prior-representation study
- **Recorded task:** lift
- **Training metric rows recovered:** 319

## Abstract

This entry reconstructs `dsl_vs_freeform_ppo_20260630-024321`, a prior-representation study. The account is based only on artifacts that survive in the repository: machine-readable configuration, candidate programs, selection diagnostics, training metrics, evaluations, reports, and logs. The run targeted lift. Its purpose and comparisons are inferred from the recorded arm names and configuration where no authored abstract survives; those inferences are identified by this provenance-first framing rather than presented as new experimental facts.

## Experimental structure

| Field | Recorded value |
|---|---|
| learner | ppo_dynamic_reward |
| tasks | lift |
| arms | dsl_stacked, freeform_stacked |
| arm_mechanisms.dsl_stacked.reward | yes |
| arm_mechanisms.dsl_stacked.action_prior | yes |
| arm_mechanisms.dsl_stacked.exploration | no |
| arm_mechanisms.dsl_stacked.supervised_init | no |
| arm_mechanisms.freeform_stacked.reward | yes |
| arm_mechanisms.freeform_stacked.action_prior | yes |
| arm_mechanisms.freeform_stacked.exploration | no |
| arm_mechanisms.freeform_stacked.supervised_init | no |
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
| ppo.target_train_seconds | 1,700 |
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

`lift_s0_dsl_stacked`, `lift_s0_freeform_stacked`

## Results

### Recorded results: `lift_s0_dsl_stacked/eval.json`

| Recorded field | Value |
|---|---|
| task | lift |
| seed | 0 |
| arm | dsl_stacked |
| best_checkpoint_iter | 0 |
| best_train_success | 0 |
| eval_base_return | 0.586917 |
| eval_base_reward_weight | 1 |
| eval_shaped_return | 0.206884 |
| eval_reward_template_returns | 0.206884, -0.019335, 0, 0, 0, 0.006111, 0.001447, -0.000026, 0, 0, 0, 0, 0, 0, 0, 0 |
| eval_action_prior_weights | 1, 1, 1 |
| eval_train_return | 0.7938 |
| eval_success_rate | 0 |
| eval_instant_success_rate | 0 |
| eval_reach_rate | 0 |
| eval_grasp_rate | 0 |
| eval_lift_reached_rate | 0.050781 |
| eval_lift_max | 0.016736 |
| eval_hard_clip_frac | 0 |
| eval_saturation_frac | 0 |
| eval_action_abs_mean | 0.148125 |
| eval_summary | 0.024715, 0.046603, 0, 0.275376, 0.016736, 0.108637 |
| efficiency_success_threshold | 0.2 |
| steps_to_success_threshold | — |
| reached_success_threshold | no |
| success_step_auc | 0 |
| total_env_steps | 4172800 |

### Recorded results: `lift_s0_freeform_stacked/eval.json`

| Recorded field | Value |
|---|---|
| task | lift |
| seed | 0 |
| arm | freeform_stacked |
| best_checkpoint_iter | 11 |
| best_train_success | 0 |
| eval_base_return | 2.924451 |
| eval_base_reward_weight | 1 |
| eval_shaped_return | -0.006621 |
| eval_reward_template_returns | -0.006621, -0.02628, 0.000007, 0.000125, 0, 0.009439, 0.007179, 0, 0, 0, 0, 0, 0, 0, 0, 0 |
| eval_action_prior_weights | 1, 1, 1 |
| eval_train_return | 2.91783 |
| eval_success_rate | 0 |
| eval_instant_success_rate | 0 |
| eval_reach_rate | 0.117188 |
| eval_grasp_rate | 0 |
| eval_lift_reached_rate | 0.164062 |
| eval_lift_max | 0.040474 |
| eval_hard_clip_frac | 0 |
| eval_saturation_frac | 0 |
| eval_action_abs_mean | 0.121209 |
| eval_summary | 0.061836, 0.036852, 0.125, 0.414425, 0.040474, 0.414964 |
| efficiency_success_threshold | 0.2 |
| steps_to_success_threshold | — |
| reached_success_threshold | no |
| success_step_auc | 0 |
| total_env_steps | 3993600 |

### Recorded results: `summary.json`

| Arm/interface | eval_success_rate | eval_instant_success_rate | eval_lift_max | eval_base_return | eval_train_return | eval_shaped_return | eval_action_abs_mean | eval_saturation_frac | mean_total_env_steps |
|---|---|---|---|---|---|---|---|---|---|
| dsl_stacked | 0 | 0 | 0.016736 | 0.586917 | 0.7938 | 0.206884 | 0.148125 | 0 | 4,172,800 |
| freeform_stacked | 0 | 0 | 0.040474 | 2.924451 | 2.91783 | -0.006621 | 0.121209 | 0 | 3,993,600 |

### Training-metric reconstruction

| Arm/sub-run | Rows | Last iter | Last env steps | Elapsed s | Last success | Best success | Last lift max | Last base return | Best graded/objective |
|---|---|---|---|---|---|---|---|---|---|
| dsl_stacked | 163 | 162 | 4172800 | 1,709.232 | 0 | 0 | 0.003548 | 7.663902 | — |
| freeform_stacked | 156 | 155 | 3993600 | 1,706.354 | 0 | 0 | 0.005243 | 7.556446 | — |

These are training-log endpoints and maxima, not substitutes for held-out evaluation. They are included especially for interrupted runs whose terminal report is missing.

## What the results mean and major discoveries

- Among the recorded arms, `freeform_stacked` had the highest held-out success (eval_success_rate = 0). This is an ordering within this run, not a general ranking across experiments with different metric definitions or budgets.
- The arm comparison is most informative when arms shared the same environment, seed, and budget. Where the configuration records only one seed, apparent gaps remain vulnerable to optimization variance.

## Limitations and reading guidance

- Metrics are reported under the names used by the historical environment and runner. Similar-looking metrics from different generations of the code are not assumed to be numerically interchangeable.
- A maximum observed in training is not treated as held-out performance. Terminal evaluation values are preferred whenever present.
- Missing summaries, empty metric streams, failed retries, and incomplete arms are preserved as negative evidence rather than silently dropped.
- Candidate names and stage names are LLM-authored run artifacts. Interpretations in this historical log do not alter framework prompts, defaults, diagnostics, or task logic.

## Artifact provenance

| Artifact type | Count |
|---|---|
| .json | 18 |
| .pkl | 16 |
| .csv | 1 |
| .jsonl | 1 |
| .md | 1 |

Primary evidence files:

- [`REPORT.md`](../runs/dsl_vs_freeform_ppo_20260630-024321/REPORT.md)
- [`config.json`](../runs/dsl_vs_freeform_ppo_20260630-024321/config.json)
- [`metrics.jsonl`](../runs/dsl_vs_freeform_ppo_20260630-024321/metrics.jsonl)
- [`summary.json`](../runs/dsl_vs_freeform_ppo_20260630-024321/summary.json)

## Appendix: original authored run report

The following contemporaneous report is retained (with headings demoted) so that details and caveats are not lost in normalization.

### Source: [`REPORT.md`](../runs/dsl_vs_freeform_ppo_20260630-024321/REPORT.md)

## Dynamic Reward Combination Experiment

Created: 2026-06-30 03:45:14

### Configuration

- Arms: `dsl_stacked,freeform_stacked`
- Tasks: `lift`
- Target seconds per arm: `1700.0`
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
  "dsl_stacked": {
    "eval_success_rate": 0.0,
    "eval_instant_success_rate": 0.0,
    "eval_base_return": 0.586917,
    "eval_shaped_return": 0.206884,
    "eval_train_return": 0.7938,
    "eval_lift_max": 0.016736,
    "eval_hard_clip_frac": 0.0,
    "eval_saturation_frac": 0.0,
    "eval_action_abs_mean": 0.148125,
    "n_eval": 1,
    "efficiency_success_threshold": 0.2,
    "reached_success_threshold_frac": 0.0,
    "mean_steps_to_success_threshold": null,
    "mean_success_step_auc": 0.0,
    "mean_total_env_steps": 4172800.0
  },
  "freeform_stacked": {
    "eval_success_rate": 0.0,
    "eval_instant_success_rate": 0.0,
    "eval_base_return": 2.924451,
    "eval_shaped_return": -0.006621,
    "eval_train_return": 2.91783,
    "eval_lift_max": 0.040474,
    "eval_hard_clip_frac": 0.0,
    "eval_saturation_frac": 0.0,
    "eval_action_abs_mean": 0.121209,
    "n_eval": 1,
    "efficiency_success_threshold": 0.2,
    "reached_success_threshold_frac": 0.0,
    "mean_steps_to_success_threshold": null,
    "mean_success_step_auc": 0.0,
    "mean_total_env_steps": 3993600.0
  }
}
```

### Notes

Reward coach prompts intentionally omit env reward, train return, base return, and PPO losses.
Rejected and deep-checkup proposals are stored under each arm's `reward_checkups/` directory.

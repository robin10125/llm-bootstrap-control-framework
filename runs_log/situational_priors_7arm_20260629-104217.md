# Situational Priors 7arm

- **Run directory:** [`runs/situational_priors_7arm_20260629-104217`](../runs/situational_priors_7arm_20260629-104217)
- **Date:** 2026-06-29 10:42:17 (directory timestamp)
- **Status:** completed or completed-with-caveats
- **Experiment class:** long-PPO training study
- **Recorded task:** lift
- **Training metric rows recovered:** 2,393

## Abstract

This entry reconstructs `situational_priors_7arm_20260629-104217`, a long-PPO training study. The account is based only on artifacts that survive in the repository: machine-readable configuration, candidate programs, selection diagnostics, training metrics, evaluations, reports, and logs. The run targeted lift. Its purpose and comparisons are inferred from the recorded arm names and configuration where no authored abstract survives; those inferences are identified by this provenance-first framing rather than presented as new experimental facts.

## Experimental structure

| Field | Recorded value |
|---|---|
| learner | ppo_dynamic_reward |
| tasks | lift |
| arms | prior_monolithic, prior_gate_soft, prior_gate_subgoal, prior_gate_options, prior_gate_stacked, prior_reactive_law, prior_dmp |
| arm_mechanisms.prior_monolithic.reward | yes |
| arm_mechanisms.prior_monolithic.action_prior | yes |
| arm_mechanisms.prior_monolithic.exploration | no |
| arm_mechanisms.prior_monolithic.supervised_init | no |
| arm_mechanisms.prior_gate_soft.reward | yes |
| arm_mechanisms.prior_gate_soft.action_prior | yes |
| arm_mechanisms.prior_gate_soft.exploration | no |
| arm_mechanisms.prior_gate_soft.supervised_init | no |
| arm_mechanisms.prior_gate_subgoal.reward | yes |
| arm_mechanisms.prior_gate_subgoal.action_prior | yes |
| arm_mechanisms.prior_gate_subgoal.exploration | no |
| arm_mechanisms.prior_gate_subgoal.supervised_init | no |
| arm_mechanisms.prior_gate_options.reward | yes |
| arm_mechanisms.prior_gate_options.action_prior | yes |
| arm_mechanisms.prior_gate_options.exploration | no |
| arm_mechanisms.prior_gate_options.supervised_init | no |
| arm_mechanisms.prior_gate_stacked.reward | yes |
| arm_mechanisms.prior_gate_stacked.action_prior | yes |
| arm_mechanisms.prior_gate_stacked.exploration | no |
| arm_mechanisms.prior_gate_stacked.supervised_init | no |
| arm_mechanisms.prior_reactive_law.reward | yes |
| arm_mechanisms.prior_reactive_law.action_prior | yes |
| arm_mechanisms.prior_reactive_law.exploration | no |
| arm_mechanisms.prior_reactive_law.supervised_init | no |
| arm_mechanisms.prior_dmp.reward | yes |
| arm_mechanisms.prior_dmp.action_prior | yes |
| arm_mechanisms.prior_dmp.exploration | no |
| arm_mechanisms.prior_dmp.supervised_init | no |
| seeds | 0 |
| prior_program_arms.prior_monolithic.mode | monolithic |
| prior_program_arms.prior_gate_soft.mode | gated |
| prior_program_arms.prior_gate_soft.discipline | soft |
| prior_program_arms.prior_gate_subgoal.mode | gated |
| prior_program_arms.prior_gate_subgoal.discipline | subgoal |
| prior_program_arms.prior_gate_options.mode | gated |
| prior_program_arms.prior_gate_options.discipline | options |
| prior_program_arms.prior_gate_stacked.mode | gated |
| prior_program_arms.prior_gate_stacked.discipline | stacked |
| prior_program_arms.prior_reactive_law.mode | reactive_law |
| prior_program_arms.prior_dmp.mode | dmp |
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
| ppo.checkpoint_count | 8 |
| ppo.target_train_seconds | 3,600 |
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

### Arms and sub-runs represented by directories

`checkpoint_videos`, `figures`, `lift_s0_prior_dmp`, `lift_s0_prior_gate_options`, `lift_s0_prior_gate_soft`, `lift_s0_prior_gate_stacked`, `lift_s0_prior_gate_subgoal`, `lift_s0_prior_monolithic`, `lift_s0_prior_reactive_law`

## Results

### Recorded results: `lift_s0_prior_dmp/eval.json`

| Recorded field | Value |
|---|---|
| task | lift |
| seed | 0 |
| arm | prior_dmp |
| best_checkpoint_iter | 8 |
| best_train_success | 0 |
| eval_base_return | 4.075987 |
| eval_base_reward_weight | 1 |
| eval_shaped_return | 0.153549 |
| eval_reward_template_returns | 0.153549, -0.014742, 0, 0.000006, 0, 0.008235, 0.011367, -0.043702, 0, 0, 0, 0, 0, 0, 0, 0 |
| eval_action_prior_weights | 1, 0.3, 0.8 |
| eval_train_return | 4.229536 |
| eval_success_rate | 0 |
| eval_instant_success_rate | 0 |
| eval_reach_rate | 0.011719 |
| eval_grasp_rate | 0.011719 |
| eval_lift_reached_rate | 0.132812 |
| eval_lift_max | 0.031278 |
| eval_hard_clip_frac | 0 |
| eval_saturation_frac | 0 |
| eval_action_abs_mean | 0.08369 |
| eval_summary | 0.079043, 0.038527, 0.011719, 0.576789, 0.031278, 0.212254 |
| efficiency_success_threshold | 0.2 |
| steps_to_success_threshold | — |
| reached_success_threshold | no |
| success_step_auc | 0 |
| total_env_steps | 8806400 |

### Recorded results: `lift_s0_prior_gate_options/eval.json`

| Recorded field | Value |
|---|---|
| task | lift |
| seed | 0 |
| arm | prior_gate_options |
| best_checkpoint_iter | 0 |
| best_train_success | 0 |
| eval_base_return | 2.936446 |
| eval_base_reward_weight | 1 |
| eval_shaped_return | 0.109502 |
| eval_reward_template_returns | 0.109502, -0.020271, 0, 0, 0, 0.005007, 0.018386, -0.008172, 0, 0, 0, 0, 0, 0, 0, 0 |
| eval_action_prior_weights | 1, 1, 1 |
| eval_train_return | 3.045948 |
| eval_success_rate | 0 |
| eval_instant_success_rate | 0 |
| eval_reach_rate | 0.003906 |
| eval_grasp_rate | 0 |
| eval_lift_reached_rate | 0.054688 |
| eval_lift_max | 0.015218 |
| eval_hard_clip_frac | 0 |
| eval_saturation_frac | 0 |
| eval_action_abs_mean | 0.04403 |
| eval_summary | 0.055329, 0.040477, 0.003906, 0.434542, 0.015218, 0.114636 |
| efficiency_success_threshold | 0.2 |
| steps_to_success_threshold | — |
| reached_success_threshold | no |
| success_step_auc | 0 |
| total_env_steps | 8934400 |

### Recorded results: `lift_s0_prior_gate_soft/eval.json`

| Recorded field | Value |
|---|---|
| task | lift |
| seed | 0 |
| arm | prior_gate_soft |
| best_checkpoint_iter | 10 |
| best_train_success | 0 |
| eval_base_return | 4.807348 |
| eval_base_reward_weight | 1 |
| eval_shaped_return | 0.104172 |
| eval_reward_template_returns | 0.104172, -0.019841, 0.000018, 0.000042, 0, 0.007549, 0.01718, -0.033793, 0, 0, 0, 0, 0, 0, 0, 0 |
| eval_action_prior_weights | 1, 1, 1 |
| eval_train_return | 4.91152 |
| eval_success_rate | 0 |
| eval_instant_success_rate | 0 |
| eval_reach_rate | 0.054688 |
| eval_grasp_rate | 0.054688 |
| eval_lift_reached_rate | 0.136719 |
| eval_lift_max | 0.025247 |
| eval_hard_clip_frac | 0 |
| eval_saturation_frac | 0 |
| eval_action_abs_mean | 0.100855 |
| eval_summary | 0.070292, 0.038549, 0.058594, 0.582284, 0.025247, 0.202821 |
| efficiency_success_threshold | 0.2 |
| steps_to_success_threshold | — |
| reached_success_threshold | no |
| success_step_auc | 0 |
| total_env_steps | 8652800 |

### Recorded results: `lift_s0_prior_gate_stacked/eval.json`

| Recorded field | Value |
|---|---|
| task | lift |
| seed | 0 |
| arm | prior_gate_stacked |
| best_checkpoint_iter | 13 |
| best_train_success | 0 |
| eval_base_return | 5.021127 |
| eval_base_reward_weight | 1 |
| eval_shaped_return | 0.157558 |
| eval_reward_template_returns | 0.157558, -0.013869, 0, 0.000043, 0, 0.009954, 0.019867, -0.028682, 0, 0, 0, 0, 0, 0, 0, 0 |
| eval_action_prior_weights | 1, 1, 1 |
| eval_train_return | 5.178685 |
| eval_success_rate | 0 |
| eval_instant_success_rate | 0 |
| eval_reach_rate | 0.105469 |
| eval_grasp_rate | 0.046875 |
| eval_lift_reached_rate | 0.097656 |
| eval_lift_max | 0.020188 |
| eval_hard_clip_frac | 0 |
| eval_saturation_frac | 0 |
| eval_action_abs_mean | 0.099037 |
| eval_summary | 0.076892, 0.036675, 0.113281, 0.559195, 0.020188, 0.154417 |
| efficiency_success_threshold | 0.2 |
| steps_to_success_threshold | — |
| reached_success_threshold | no |
| success_step_auc | 0 |
| total_env_steps | 9011200 |

### Recorded results: `lift_s0_prior_gate_subgoal/eval.json`

| Recorded field | Value |
|---|---|
| task | lift |
| seed | 0 |
| arm | prior_gate_subgoal |
| best_checkpoint_iter | 12 |
| best_train_success | 0 |
| eval_base_return | 4.995108 |
| eval_base_reward_weight | 1 |
| eval_shaped_return | 0.114127 |
| eval_reward_template_returns | 0.114127, -0.019105, 0, 0.000031, 0, 0.00811, 0.018506, -0.005363, 0, 0, 0, 0, 0, 0, 0, 0 |
| eval_action_prior_weights | 1, 1, 1 |
| eval_train_return | 5.109234 |
| eval_success_rate | 0 |
| eval_instant_success_rate | 0 |
| eval_reach_rate | 0.039062 |
| eval_grasp_rate | 0.039062 |
| eval_lift_reached_rate | 0.113281 |
| eval_lift_max | 0.020758 |
| eval_hard_clip_frac | 0 |
| eval_saturation_frac | 0 |
| eval_action_abs_mean | 0.101756 |
| eval_summary | 0.071252, 0.036912, 0.039062, 0.552976, 0.020758, 0.161319 |
| efficiency_success_threshold | 0.2 |
| steps_to_success_threshold | — |
| reached_success_threshold | no |
| success_step_auc | 0 |
| total_env_steps | 8883200 |

### Recorded results: `lift_s0_prior_monolithic/eval.json`

| Recorded field | Value |
|---|---|
| task | lift |
| seed | 0 |
| arm | prior_monolithic |
| best_checkpoint_iter | 4 |
| best_train_success | 0 |
| eval_base_return | 3.417606 |
| eval_base_reward_weight | 1 |
| eval_shaped_return | -0.001555 |
| eval_reward_template_returns | -0.001555, -0.0343, 0, 0, 0, 0.017525, 0.001803, -0.39454, 0, 0, 0, 0, 0, 0, 0, 0 |
| eval_action_prior_weights | 0.5, 0.3, 0.5, 0.45, 0.4, 0.4, 0.2, 0.2, 0.2, 0, 0, 0 |
| eval_train_return | 3.416051 |
| eval_success_rate | 0 |
| eval_instant_success_rate | 0 |
| eval_reach_rate | 0.003906 |
| eval_grasp_rate | 0.003906 |
| eval_lift_reached_rate | 0.089844 |
| eval_lift_max | 0.029694 |
| eval_hard_clip_frac | 0 |
| eval_saturation_frac | 0 |
| eval_action_abs_mean | 0.271709 |
| eval_summary | 0.039311, 0.040749, 0.003906, 0.698736, 0.029694, 0.462118 |
| efficiency_success_threshold | 0.2 |
| steps_to_success_threshold | — |
| reached_success_threshold | no |
| success_step_auc | 0 |
| total_env_steps | 8780800 |

### Recorded results: `lift_s0_prior_reactive_law/eval.json`

| Recorded field | Value |
|---|---|
| task | lift |
| seed | 0 |
| arm | prior_reactive_law |
| best_checkpoint_iter | 319 |
| best_train_success | 0 |
| eval_base_return | 71.073975 |
| eval_base_reward_weight | 1 |
| eval_shaped_return | 0.331049 |
| eval_reward_template_returns | 0.331049, -0.000622, 0.001797, 0.0033, 0.026509, 0.025195, 0.035822, -0.643008, 0, 0, 0, 0, 0, 0, 0, 0 |
| eval_action_prior_weights | 1, 0.6, 0.8, 0.6 |
| eval_train_return | 71.405014 |
| eval_success_rate | 0 |
| eval_instant_success_rate | 0.925781 |
| eval_reach_rate | 0.941406 |
| eval_grasp_rate | 0.941406 |
| eval_lift_reached_rate | 1 |
| eval_lift_max | 0.081486 |
| eval_hard_clip_frac | 0 |
| eval_saturation_frac | 0.000953 |
| eval_action_abs_mean | 0.340457 |
| eval_summary | 0.037297, 0.026625, 0.945312, 0.719949, 0.081486, 0.058447 |
| efficiency_success_threshold | 0.2 |
| steps_to_success_threshold | — |
| reached_success_threshold | no |
| success_step_auc | 0 |
| total_env_steps | 8192000 |

### Recorded results: `summary.json`

| Arm/interface | eval_success_rate | eval_instant_success_rate | eval_lift_max | eval_base_return | eval_train_return | eval_shaped_return | eval_action_abs_mean | eval_saturation_frac | mean_total_env_steps |
|---|---|---|---|---|---|---|---|---|---|
| prior_dmp | 0 | 0 | 0.031278 | 4.075987 | 4.229536 | 0.153549 | 0.08369 | 0 | 8,806,400 |
| prior_gate_options | 0 | 0 | 0.015218 | 2.936446 | 3.045948 | 0.109502 | 0.04403 | 0 | 8,934,400 |
| prior_gate_soft | 0 | 0 | 0.025247 | 4.807348 | 4.91152 | 0.104172 | 0.100855 | 0 | 8,652,800 |
| prior_gate_stacked | 0 | 0 | 0.020188 | 5.021127 | 5.178685 | 0.157558 | 0.099037 | 0 | 9,011,200 |
| prior_gate_subgoal | 0 | 0 | 0.020758 | 4.995108 | 5.109234 | 0.114127 | 0.101756 | 0 | 8,883,200 |
| prior_monolithic | 0 | 0 | 0.029694 | 3.417606 | 3.416051 | -0.001555 | 0.271709 | 0 | 8,780,800 |
| prior_reactive_law | 0 | 0.925781 | 0.081486 | 71.073975 | 71.405014 | 0.331049 | 0.340457 | 0.000953 | 8,192,000 |

### Training-metric reconstruction

| Arm/sub-run | Rows | Last iter | Last env steps | Elapsed s | Last success | Best success | Last lift max | Last base return | Best graded/objective |
|---|---|---|---|---|---|---|---|---|---|
| prior_monolithic | 343 | 342 | 8780800 | 3,606.864 | 0 | 0 | 0.012755 | 12.077486 | — |
| prior_gate_soft | 338 | 337 | 8652800 | 3,605.343 | 0 | 0 | 0.010133 | 11.026793 | — |
| prior_gate_subgoal | 347 | 346 | 8883200 | 3,608.955 | 0 | 0 | 0.003124 | 10.791459 | — |
| prior_gate_options | 349 | 348 | 8934400 | 3,609.335 | 0 | 0 | 0.003054 | 8.944071 | — |
| prior_gate_stacked | 352 | 351 | 9011200 | 3,606.563 | 0 | 0 | 0.014988 | 12.918411 | — |
| prior_reactive_law | 320 | 319 | 8192000 | 3,603.094 | 0 | 0 | 0.077911 | 35.936249 | — |
| prior_dmp | 344 | 343 | 8806400 | 3,605.999 | 0 | 0 | 0.0041 | 9.358808 | — |

These are training-log endpoints and maxima, not substitutes for held-out evaluation. They are included especially for interrupted runs whose terminal report is missing.

## What the results mean and major discoveries

- Among the recorded arms, `prior_reactive_law` had the highest held-out success (eval_success_rate = 0). This is an ordering within this run, not a general ranking across experiments with different metric definitions or budgets.
- The arm comparison is most informative when arms shared the same environment, seed, and budget. Where the configuration records only one seed, apparent gaps remain vulnerable to optimization variance.

## Limitations and reading guidance

- Metrics are reported under the names used by the historical environment and runner. Similar-looking metrics from different generations of the code are not assumed to be numerically interchangeable.
- A maximum observed in training is not treated as held-out performance. Terminal evaluation values are preferred whenever present.
- Missing summaries, empty metric streams, failed retries, and incomplete arms are preserved as negative evidence rather than silently dropped.
- Candidate names and stage names are LLM-authored run artifacts. Interpretations in this historical log do not alter framework prompts, defaults, diagnostics, or task logic.

## Artifact provenance

| Artifact type | Count |
|---|---|
| .pkl | 77 |
| .json | 61 |
| .mp4 | 7 |
| .png | 2 |
| .csv | 1 |
| .jsonl | 1 |
| .md | 1 |

Primary evidence files:

- [`REPORT.md`](../runs/situational_priors_7arm_20260629-104217/REPORT.md)
- [`config.json`](../runs/situational_priors_7arm_20260629-104217/config.json)
- [`metrics.jsonl`](../runs/situational_priors_7arm_20260629-104217/metrics.jsonl)
- [`summary.json`](../runs/situational_priors_7arm_20260629-104217/summary.json)

## Appendix: original authored run report

The following contemporaneous report is retained (with headings demoted) so that details and caveats are not lost in normalization.

### Source: [`REPORT.md`](../runs/situational_priors_7arm_20260629-104217/REPORT.md)

## Dynamic Reward Combination Experiment

Created: 2026-06-29 17:56:18

### Configuration

- Arms: `prior_monolithic,prior_gate_soft,prior_gate_subgoal,prior_gate_options,prior_gate_stacked,prior_reactive_law,prior_dmp`
- Tasks: `lift`
- Target seconds per arm: `3600.0`
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
  "prior_dmp": {
    "eval_success_rate": 0.0,
    "eval_instant_success_rate": 0.0,
    "eval_base_return": 4.075987,
    "eval_shaped_return": 0.153549,
    "eval_train_return": 4.229536,
    "eval_lift_max": 0.031278,
    "eval_hard_clip_frac": 0.0,
    "eval_saturation_frac": 0.0,
    "eval_action_abs_mean": 0.08369,
    "n_eval": 1,
    "efficiency_success_threshold": 0.2,
    "reached_success_threshold_frac": 0.0,
    "mean_steps_to_success_threshold": null,
    "mean_success_step_auc": 0.0,
    "mean_total_env_steps": 8806400.0
  },
  "prior_gate_options": {
    "eval_success_rate": 0.0,
    "eval_instant_success_rate": 0.0,
    "eval_base_return": 2.936446,
    "eval_shaped_return": 0.109502,
    "eval_train_return": 3.045948,
    "eval_lift_max": 0.015218,
    "eval_hard_clip_frac": 0.0,
    "eval_saturation_frac": 0.0,
    "eval_action_abs_mean": 0.04403,
    "n_eval": 1,
    "efficiency_success_threshold": 0.2,
    "reached_success_threshold_frac": 0.0,
    "mean_steps_to_success_threshold": null,
    "mean_success_step_auc": 0.0,
    "mean_total_env_steps": 8934400.0
  },
  "prior_gate_soft": {
    "eval_success_rate": 0.0,
    "eval_instant_success_rate": 0.0,
    "eval_base_return": 4.807348,
    "eval_shaped_return": 0.104172,
    "eval_train_return": 4.91152,
    "eval_lift_max": 0.025247,
    "eval_hard_clip_frac": 0.0,
    "eval_saturation_frac": 0.0,
    "eval_action_abs_mean": 0.100855,
    "n_eval": 1,
    "efficiency_success_threshold": 0.2,
    "reached_success_threshold_frac": 0.0,
    "mean_steps_to_success_threshold": null,
    "mean_success_step_auc": 0.0,
    "mean_total_env_steps": 8652800.0
  },
  "prior_gate_stacked": {
    "eval_success_rate": 0.0,
    "eval_instant_success_rate": 0.0,
    "eval_base_return": 5.021127,
    "eval_shaped_return": 0.157558,
    "eval_train_return": 5.178685,
    "eval_lift_max": 0.020188,
    "eval_hard_clip_frac": 0.0,
    "eval_saturation_frac": 0.0,
    "eval_action_abs_mean": 0.099037,
    "n_eval": 1,
    "efficiency_success_threshold": 0.2,
    "reached_success_threshold_frac": 0.0,
    "mean_steps_to_success_threshold": null,
    "mean_success_step_auc": 0.0,
    "mean_total_env_steps": 9011200.0
  },
  "prior_gate_subgoal": {
    "eval_success_rate": 0.0,
    "eval_instant_success_rate": 0.0,
    "eval_base_return": 4.995108,
    "eval_shaped_return": 0.114127,
    "eval_train_return": 5.109234,
    "eval_lift_max": 0.020758,
    "eval_hard_clip_frac": 0.0,
    "eval_saturation_frac": 0.0,
    "eval_action_abs_mean": 0.101756,
    "n_eval": 1,
    "efficiency_success_threshold": 0.2,
    "reached_success_threshold_frac": 0.0,
    "mean_steps_to_success_threshold": null,
    "mean_success_step_auc": 0.0,
    "mean_total_env_steps": 8883200.0
  },
  "prior_monolithic": {
    "eval_success_rate": 0.0,
    "eval_instant_success_rate": 0.0,
    "eval_base_return": 3.417606,
    "eval_shaped_return": -0.001555,
    "eval_train_return": 3.416051,
    "eval_lift_max": 0.029694,
    "eval_hard_clip_frac": 0.0,
    "eval_saturation_frac": 0.0,
    "eval_action_abs_mean": 0.271709,
    "n_eval": 1,
    "efficiency_success_threshold": 0.2,
    "reached_success_threshold_frac": 0.0,
    "mean_steps_to_success_threshold": null,
    "mean_success_step_auc": 0.0,
    "mean_total_env_steps": 8780800.0
  },
  "prior_reactive_law": {
    "eval_success_rate": 0.0,
    "eval_instant_success_rate": 0.925781,
    "eval_base_return": 71.073975,
    "eval_shaped_return": 0.331049,
    "eval_train_return": 71.405014,
    "eval_lift_max": 0.081486,
    "eval_hard_clip_frac": 0.0,
    "eval_saturation_frac": 0.000953,
    "eval_action_abs_mean": 0.340457,
    "n_eval": 1,
    "efficiency_success_threshold": 0.2,
    "reached_success_threshold_frac": 0.0,
    "mean_steps_to_success_threshold": null,
    "mean_success_step_auc": 0.0,
    "mean_total_env_steps": 8192000.0
  }
}
```

### Notes

Reward coach prompts intentionally omit env reward, train return, base return, and PPO losses.
Rejected and deep-checkup proposals are stored under each arm's `reward_checkups/` directory.

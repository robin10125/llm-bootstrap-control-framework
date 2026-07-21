# Pareto Fixed Prior Shaped Reward 12h

- **Run directory:** [`runs/pareto_fixed_prior_shaped_reward_12h_20260622-181701`](../runs/pareto_fixed_prior_shaped_reward_12h_20260622-181701)
- **Date:** 2026-06-22 18:17:01 (directory timestamp)
- **Status:** artifact-only or generation-only; no terminal evaluation recorded
- **Experiment class:** PPO bias/reward/prior experiment
- **Recorded task:** lift
- **Training metric rows recovered:** 0

## Abstract

This entry reconstructs `pareto_fixed_prior_shaped_reward_12h_20260622-181701`, a PPO bias/reward/prior experiment. The account is based only on artifacts that survive in the repository: machine-readable configuration, candidate programs, selection diagnostics, training metrics, evaluations, reports, and logs. The run targeted lift. Its purpose and comparisons are inferred from the recorded arm names and configuration where no authored abstract survives; those inferences are identified by this provenance-first framing rather than presented as new experimental facts.

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
| dynamic_action_prior.pareto_action_prior | yes |
| dynamic_action_prior.action_prior_candidates | 5 |
| dynamic_action_prior.pareto_supervised_init | yes |
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

`lift_s0_baseline`, `llm_initial_bias`, `pareto_action_prior`, `pareto_supervised_init`, `pre_run_reward_analysis`

## Results

No decodable run-level result JSON or non-empty training metric stream survives.

## What the results mean and major discoveries

- The surviving artifacts document the attempted structure but do not contain enough terminal quantitative evidence for a strong result claim. Treat this entry as provenance and negative/unfinished evidence.

## Limitations and reading guidance

- Metrics are reported under the names used by the historical environment and runner. Similar-looking metrics from different generations of the code are not assumed to be numerically interchangeable.
- A maximum observed in training is not treated as held-out performance. Terminal evaluation values are preferred whenever present.
- Missing summaries, empty metric streams, failed retries, and incomplete arms are preserved as negative evidence rather than silently dropped.
- Candidate names and stage names are LLM-authored run artifacts. Interpretations in this historical log do not alter framework prompts, defaults, diagnostics, or task logic.

## Artifact provenance

| Artifact type | Count |
|---|---|
| .json | 22 |
| .md | 7 |
| .txt | 7 |
| .jsonl | 1 |

Primary evidence files:

- [`config.json`](../runs/pareto_fixed_prior_shaped_reward_12h_20260622-181701/config.json)
- [`metrics.jsonl`](../runs/pareto_fixed_prior_shaped_reward_12h_20260622-181701/metrics.jsonl)

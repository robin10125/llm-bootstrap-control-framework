# Saturation Strategy Compare

- **Run directory:** [`runs/saturation_strategy_compare_20260619-134650`](../runs/saturation_strategy_compare_20260619-134650)
- **Date:** 2026-06-19 13:46:50 (directory timestamp)
- **Status:** completed or completed-with-caveats
- **Experiment class:** action-output transform comparison
- **Recorded task:** not explicitly recorded at the run root
- **Training metric rows recovered:** 142

## Abstract

This entry reconstructs `saturation_strategy_compare_20260619-134650`, an action-output transform comparison. The account is based only on artifacts that survive in the repository: machine-readable configuration, candidate programs, selection diagnostics, training metrics, evaluations, reports, and logs. The run targeted not explicitly recorded at the run root. Its purpose and comparisons are inferred from the recorded arm names and configuration where no authored abstract survives; those inferences are identified by this provenance-first framing rather than presented as new experimental facts.

## Experimental structure

#### Configuration: `raw_none/config.json`

| Field | Recorded value |
|---|---|
| learner | ppo |
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
| ppo.target_train_seconds | 1,800 |
| ppo.action_transform | raw |
| ppo.saturation_penalty | 0 |
| ppo.saturation_threshold | 0.98 |
| ppo.prior_logit_clip | 0.95 |
| llm_backend | fake |
| llm_model | — |

#### Configuration: `raw_penalty/config.json`

| Field | Recorded value |
|---|---|
| learner | ppo |
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
| ppo.target_train_seconds | 1,800 |
| ppo.action_transform | raw |
| ppo.saturation_penalty | 0.1 |
| ppo.saturation_threshold | 0.98 |
| ppo.prior_logit_clip | 0.95 |
| llm_backend | fake |
| llm_model | — |

#### Configuration: `tanh/config.json`

| Field | Recorded value |
|---|---|
| learner | ppo |
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
| ppo.target_train_seconds | 1,800 |
| ppo.action_transform | tanh |
| ppo.saturation_penalty | 0 |
| ppo.saturation_threshold | 0.98 |
| ppo.prior_logit_clip | 0.95 |
| llm_backend | fake |
| llm_model | — |

### Arms and sub-runs represented by directories

`raw_none`, `raw_penalty`, `tanh`

## Results

### Recorded results: `raw_none/lift_s0_action_prior/eval.json`

| Recorded field | Value |
|---|---|
| task | lift |
| seed | 0 |
| arm | action_prior |
| eval_base_return | -14.126237 |
| eval_shaped_return | 0 |
| eval_train_return | -14.126237 |
| eval_success_rate | 0.196289 |
| eval_lift_max | 0.050161 |
| eval_hard_clip_frac | 0.256408 |
| eval_saturation_frac | 0.304417 |
| eval_action_abs_mean | 0.614073 |
| eval_summary | 0.0929, 0.043918, 0.098633, 0.795434, 0.050161, 0.330848 |

### Recorded results: `raw_none/summary.json`

| Arm/interface | eval_success_rate | eval_lift_max | eval_base_return | eval_train_return | eval_shaped_return | eval_action_abs_mean | eval_saturation_frac |
|---|---|---|---|---|---|---|---|
| action_prior | 0.196289 | 0.050161 | -14.126237 | -14.126237 | 0 | 0.614073 | 0.304417 |

### Recorded results: `raw_penalty/lift_s0_action_prior/eval.json`

| Recorded field | Value |
|---|---|
| task | lift |
| seed | 0 |
| arm | action_prior |
| eval_base_return | -11.82022 |
| eval_shaped_return | 0 |
| eval_train_return | -11.82022 |
| eval_success_rate | 0.230469 |
| eval_lift_max | 0.053597 |
| eval_hard_clip_frac | 0.20634 |
| eval_saturation_frac | 0.256971 |
| eval_action_abs_mean | 0.605066 |
| eval_summary | 0.092414, 0.042977, 0.101562, 0.790091, 0.053597, 0.398773 |

### Recorded results: `raw_penalty/summary.json`

| Arm/interface | eval_success_rate | eval_lift_max | eval_base_return | eval_train_return | eval_shaped_return | eval_action_abs_mean | eval_saturation_frac |
|---|---|---|---|---|---|---|---|
| action_prior | 0.230469 | 0.053597 | -11.82022 | -11.82022 | 0 | 0.605066 | 0.256971 |

### Recorded results: `tanh/lift_s0_action_prior/eval.json`

| Recorded field | Value |
|---|---|
| task | lift |
| seed | 0 |
| arm | action_prior |
| eval_base_return | 2.344067 |
| eval_shaped_return | 0 |
| eval_train_return | 2.344067 |
| eval_success_rate | 0.313477 |
| eval_lift_max | 0.094524 |
| eval_hard_clip_frac | 0 |
| eval_saturation_frac | 0.000308 |
| eval_action_abs_mean | 0.616944 |
| eval_summary | 0.086952, 0.048073, 0.016602, 0.731475, 0.094524, 0.3981 |

### Recorded results: `tanh/summary.json`

| Arm/interface | eval_success_rate | eval_lift_max | eval_base_return | eval_train_return | eval_shaped_return | eval_action_abs_mean | eval_saturation_frac |
|---|---|---|---|---|---|---|---|
| action_prior | 0.313477 | 0.094524 | 2.344067 | 2.344067 | 0 | 0.616944 | 0.000308 |

### Training-metric reconstruction

| Arm/sub-run | Rows | Last iter | Last env steps | Elapsed s | Last success | Best success | Last lift max | Last base return | Best graded/objective |
|---|---|---|---|---|---|---|---|---|---|
| action_prior | 142 | 47 | — | 1,832.878 | 0.193359 | 0.217773 | 0.045265 | -16.606194 | — |

These are training-log endpoints and maxima, not substitutes for held-out evaluation. They are included especially for interrupted runs whose terminal report is missing.

## What the results mean and major discoveries

- Among the recorded arms, `action_prior` had the highest held-out success (eval_success_rate = 0.313477). This is an ordering within this run, not a general ranking across experiments with different metric definitions or budgets.
- The arm comparison is most informative when arms shared the same environment, seed, and budget. Where the configuration records only one seed, apparent gaps remain vulnerable to optimization variance.

## Limitations and reading guidance

- Metrics are reported under the names used by the historical environment and runner. Similar-looking metrics from different generations of the code are not assumed to be numerically interchangeable.
- A maximum observed in training is not treated as held-out performance. Terminal evaluation values are preferred whenever present.
- Missing summaries, empty metric streams, failed retries, and incomplete arms are preserved as negative evidence rather than silently dropped.
- Candidate names and stage names are LLM-authored run artifacts. Interpretations in this historical log do not alter framework prompts, defaults, diagnostics, or task logic.

## Artifact provenance

| Artifact type | Count |
|---|---|
| .pkl | 21 |
| .json | 12 |
| .csv | 3 |
| .jsonl | 3 |
| .md | 1 |

Primary evidence files:

- [`REPORT.md`](../runs/saturation_strategy_compare_20260619-134650/REPORT.md)
- [`raw_none/config.json`](../runs/saturation_strategy_compare_20260619-134650/raw_none/config.json)
- [`raw_none/metrics.jsonl`](../runs/saturation_strategy_compare_20260619-134650/raw_none/metrics.jsonl)
- [`raw_none/summary.json`](../runs/saturation_strategy_compare_20260619-134650/raw_none/summary.json)
- [`raw_penalty/config.json`](../runs/saturation_strategy_compare_20260619-134650/raw_penalty/config.json)
- [`raw_penalty/metrics.jsonl`](../runs/saturation_strategy_compare_20260619-134650/raw_penalty/metrics.jsonl)
- [`raw_penalty/summary.json`](../runs/saturation_strategy_compare_20260619-134650/raw_penalty/summary.json)
- [`tanh/config.json`](../runs/saturation_strategy_compare_20260619-134650/tanh/config.json)
- [`tanh/metrics.jsonl`](../runs/saturation_strategy_compare_20260619-134650/tanh/metrics.jsonl)
- [`tanh/summary.json`](../runs/saturation_strategy_compare_20260619-134650/tanh/summary.json)

## Appendix: original authored run report

The following contemporaneous report is retained (with headings demoted) so that details and caveats are not lost in normalization.

### Source: [`REPORT.md`](../runs/saturation_strategy_compare_20260619-134650/REPORT.md)

## Clipping Strategy Comparison

Date: 2026-06-19

### Purpose

Test whether action saturation is materially harming the action-prior PPO policy, and compare three ways of handling bounded actions:

1. Raw Gaussian actions with environment hard clipping.
2. Raw Gaussian actions with a saturation penalty.
3. Tanh-squashed Gaussian actions with corrected PPO log-probabilities.

The motivation was that the action prior can push the actor mean toward action limits. When the environment clips the final action, PPO models the unclipped Gaussian sample while the physics receives the clipped action. This can create a mismatch between the policy distribution and the executed behavior.

### Setup

Run directory:

`runs/saturation_strategy_compare_20260619-134650`

Task:

`lift`

Arm:

`action_prior`

Environment:

- Robot: Shadow Hand MJX environment
- Observation size: 89
- Action size: 23
- Horizon: 100 control steps
- Control timestep: 0.025 s
- Episode length: 2.5 s
- Parallel environments: 1024
- Eval environments: 1024

PPO:

- Network: 128, 128
- Learning rate: 3e-4
- Gamma: 0.99
- Lambda: 0.95
- Entropy coefficient: 0.0
- Training budget: 1800 s per strategy, not including compile/warmup
- Checkpoints: 5

Saturation metrics:

- `hard_clip_frac`: fraction of action dimensions where the policy output exceeded `[-1, 1]` before env clipping.
- `saturation_frac`: fraction of executed action dimensions with absolute value at least `0.98`.
- `action_abs_mean`: mean absolute executed action magnitude.

### Strategies

#### Raw None

Raw Gaussian PPO action:

```text
action = mean + std * noise
executed_action = clip(action, -1, 1)
```

No explicit penalty for saturation.

#### Raw Penalty

Same raw Gaussian policy as above, with an auxiliary penalty for frequent near-limit actions:

```text
penalty = 0.10 * mean(max(abs(executed_action) - 0.98, 0) / 0.02)
```

The environment still hard-clips actions.

#### Tanh Squashed

Bounded policy action:

```text
raw = mean + std * noise
action = tanh(raw)
```

PPO log-probability includes the tanh change-of-variables correction. The LLM action prior is added in pre-squash space by applying `atanh` to the clipped prior:

```text
mean = neural_policy(obs) + atanh(clamp(prior, -0.95, 0.95))
action = tanh(mean + std * noise)
```

The environment hard clip remains as a safety fallback.

### Results

Held-out eval:

| Strategy | PPO Iters | Success | Base Return | Lift Max | Hard Clip Frac | Saturation Frac | Action Abs Mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| Raw none | 47 | 0.196289 | -14.126237 | 0.050161 | 0.256408 | 0.304417 | 0.614073 |
| Raw penalty | 47 | 0.230469 | -11.820220 | 0.053597 | 0.206340 | 0.256971 | 0.605066 |
| Tanh squashed | 48 | 0.313477 | 2.344067 | 0.094524 | 0.000000 | 0.000308 | 0.616944 |

Final training rows:

| Strategy | Success | Base Return | Lift Max | Hard Clip Frac | Saturation Frac | Action Abs Mean |
|---|---:|---:|---:|---:|---:|---:|
| Raw none | 0.167969 | -17.532223 | 0.038396 | 0.328161 | 0.338930 | 0.646434 |
| Raw penalty | 0.143555 | -20.048532 | 0.033236 | 0.316955 | 0.327727 | 0.638407 |
| Tanh squashed | 0.193359 | -16.606194 | 0.045265 | 0.000000 | 0.125504 | 0.701199 |

### Interpretation

The raw action-prior policy showed substantial saturation. In held-out eval, roughly 25.6% of raw action dimensions exceeded the valid action range, and roughly 30.4% of executed action dimensions were near the saturation threshold.

The saturation penalty reduced saturation, but only modestly. Held-out hard clipping fell from 0.256408 to 0.206340, and held-out saturation fell from 0.304417 to 0.256971. Performance improved slightly, but the policy remained heavily saturated.

The tanh-squashed policy was clearly better in this run. It eliminated hard clipping by construction and reduced held-out near-limit saturation to 0.000308. It also improved task performance:

- Success increased from 0.196289 to 0.313477 versus raw none.
- Base return increased from -14.126237 to 2.344067.
- Lift max increased from 0.050161 to 0.094524.

The training-time tanh saturation metric was still 0.125504 because stochastic samples can land near the tanh limits during exploration. However, deterministic held-out evaluation had almost no saturation.

### Conclusion

For this experiment, tanh-squashed actions are the preferred default. They better align the PPO policy distribution with the bounded action actually sent to the environment, remove the hard-clipping mismatch, and improved held-out performance.

The environment hard clip should remain as a final safety fallback, but it should not be the primary bounding mechanism used during learning.

### Limitations

This was a single-seed comparison on the `action_prior` arm only. The result is strong enough to justify making tanh squashing the default, but a technical report should label it as preliminary until repeated across seeds and tested with combined arms such as `reward_action_prior` and `full`.

The tanh policy changes the effective semantics of action-prior magnitudes. Priors are now interpreted in pre-squash space, so very large priors can still push actions close to the limits. The current implementation clips prior values to `[-0.95, 0.95]` before `atanh`.

Compile/warmup time was long in the fresh JAX/MJX environment. Future runs should add explicit compile-phase timing and use a persistent XLA cache so the measured training budget is easier to audit.

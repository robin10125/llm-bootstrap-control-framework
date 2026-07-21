# Policy Bias PPO Shadow Lift 200iter

- **Run directory:** [`runs/policy_bias_ppo_shadow_lift_200iter_20260618-123903`](../runs/policy_bias_ppo_shadow_lift_200iter_20260618-123903)
- **Date:** 2026-06-18 12:39:03 (directory timestamp)
- **Status:** completed or completed-with-caveats
- **Experiment class:** PPO bias/reward/prior experiment
- **Recorded task:** lift
- **Training metric rows recovered:** 1,000

## Abstract

This entry reconstructs `policy_bias_ppo_shadow_lift_200iter_20260618-123903`, a PPO bias/reward/prior experiment. The account is based only on artifacts that survive in the repository: machine-readable configuration, candidate programs, selection diagnostics, training metrics, evaluations, reports, and logs. The run targeted lift. Its purpose and comparisons are inferred from the recorded arm names and configuration where no authored abstract survives; those inferences are identified by this provenance-first framing rather than presented as new experimental facts.

## Experimental structure

| Field | Recorded value |
|---|---|
| learner | ppo |
| tasks | lift |
| arms | baseline, reward, action_prior, exploration, supervised_init |
| arm_mechanisms.baseline.reward | no |
| arm_mechanisms.baseline.action_prior | no |
| arm_mechanisms.baseline.exploration | no |
| arm_mechanisms.baseline.supervised_init | no |
| arm_mechanisms.reward.reward | yes |
| arm_mechanisms.reward.action_prior | no |
| arm_mechanisms.reward.exploration | no |
| arm_mechanisms.reward.supervised_init | no |
| arm_mechanisms.action_prior.reward | no |
| arm_mechanisms.action_prior.action_prior | yes |
| arm_mechanisms.action_prior.exploration | no |
| arm_mechanisms.action_prior.supervised_init | no |
| arm_mechanisms.exploration.reward | no |
| arm_mechanisms.exploration.action_prior | no |
| arm_mechanisms.exploration.exploration | yes |
| arm_mechanisms.exploration.supervised_init | no |
| arm_mechanisms.supervised_init.reward | no |
| arm_mechanisms.supervised_init.action_prior | no |
| arm_mechanisms.supervised_init.exploration | no |
| arm_mechanisms.supervised_init.supervised_init | yes |
| seeds | 0 |
| env.obs_size | 89 |
| env.action_size | 23 |
| env.horizon | 100 |
| env.frame_skip | 2 |
| env.actuators | rh_A_WRJ2, rh_A_WRJ1, rh_A_THJ5, rh_A_THJ4, rh_A_THJ3, rh_A_THJ2, rh_A_THJ1, rh_A_FFJ4, rh_A_FFJ3, rh_A_FFJ0, rh_A_MFJ4, rh_A_MFJ3, rh_A_MFJ0, rh_A_RFJ4, rh_A_RFJ3, rh_A_RFJ0, rh_A_LFJ5, rh_A_LFJ4, rh_A_LFJ3, rh_A_LFJ0, base_x, base_y, base_z |
| env.tasks.lift.objective | raise object above starting height |
| env.tasks.lift.success | lift > 0.05m |
| ppo.iters | 200 |
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
| ppo.target_train_seconds | — |
| llm_backend | fixture |
| llm_model | — |

### Arms and sub-runs represented by directories

`checkpoint_videos`, `checkpoint_videos_smoke`, `lift_s0_action_prior`, `lift_s0_baseline`, `lift_s0_exploration`, `lift_s0_reward`, `lift_s0_supervised_init`

## Results

### Recorded results: `lift_s0_action_prior/eval.json`

| Recorded field | Value |
|---|---|
| task | lift |
| seed | 0 |
| arm | action_prior |
| eval_base_return | 212.488 |
| eval_shaped_return | 0 |
| eval_train_return | 212.488 |
| eval_success_rate | 0.956055 |
| eval_lift_max | 0.439774 |
| eval_summary | 0.042358, 0.040873, 0.03125, 0.834152, 0.439774, 1.86546 |

### Recorded results: `lift_s0_baseline/eval.json`

| Recorded field | Value |
|---|---|
| task | lift |
| seed | 0 |
| arm | baseline |
| eval_base_return | 20.22802 |
| eval_shaped_return | 0 |
| eval_train_return | 20.22802 |
| eval_success_rate | 0.617188 |
| eval_lift_max | 0.085193 |
| eval_summary | 0.067318, 0.035619, 0.280273, 0.251877, 0.085193, 1.005599 |

### Recorded results: `lift_s0_exploration/eval.json`

| Recorded field | Value |
|---|---|
| task | lift |
| seed | 0 |
| arm | exploration |
| eval_base_return | 47.972065 |
| eval_shaped_return | 0 |
| eval_train_return | 47.972065 |
| eval_success_rate | 0.828125 |
| eval_lift_max | 0.159126 |
| eval_summary | 0.086142, 0.041297, 0.06543, 0.283692, 0.159126, 1.934819 |

### Recorded results: `lift_s0_reward/eval.json`

| Recorded field | Value |
|---|---|
| task | lift |
| seed | 0 |
| arm | reward |
| eval_base_return | 41.753387 |
| eval_shaped_return | -0.13385 |
| eval_train_return | 41.619537 |
| eval_success_rate | 0.780273 |
| eval_lift_max | 0.134684 |
| eval_summary | 0.080214, 0.039277, 0.117188, 0.239394, 0.134684, 1.609619 |

### Recorded results: `lift_s0_supervised_init/eval.json`

| Recorded field | Value |
|---|---|
| task | lift |
| seed | 0 |
| arm | supervised_init |
| eval_base_return | 127.711 |
| eval_shaped_return | 0 |
| eval_train_return | 127.711 |
| eval_success_rate | 0.999023 |
| eval_lift_max | 0.393314 |
| eval_summary | 0.07031, 0.044551, 0.104492, 0.697632, 0.393314, 2.717785 |

### Recorded results: `summary.json`

| Arm/interface | eval_success_rate | eval_lift_max | eval_base_return | eval_train_return | eval_shaped_return |
|---|---|---|---|---|---|
| action_prior | 0.956055 | 0.439774 | 212.488 | 212.488 | 0 |
| baseline | 0.617188 | 0.085193 | 20.22802 | 20.22802 | 0 |
| exploration | 0.828125 | 0.159126 | 47.972065 | 47.972065 | 0 |
| reward | 0.780273 | 0.134684 | 41.753387 | 41.619537 | -0.13385 |
| supervised_init | 0.999023 | 0.393314 | 127.711 | 127.711 | 0 |

### Training-metric reconstruction

| Arm/sub-run | Rows | Last iter | Last env steps | Elapsed s | Last success | Best success | Last lift max | Last base return | Best graded/objective |
|---|---|---|---|---|---|---|---|---|---|
| baseline | 200 | 199 | — | 8,690.222 | 0.59375 | 0.607422 | 0.080626 | 16.432192 | — |
| reward | 200 | 199 | — | 8,486.458 | 0.703125 | 0.713867 | 0.109199 | 30.112476 | — |
| action_prior | 200 | 199 | — | 7,848.905 | 0.887695 | 0.887695 | 0.305541 | 143.278 | — |
| exploration | 200 | 199 | — | 8,314.363 | 0.686523 | 0.72168 | 0.112608 | 30.846779 | — |
| supervised_init | 200 | 199 | — | 8,047.692 | 0.969727 | 0.969727 | 0.362606 | 114.543 | — |

These are training-log endpoints and maxima, not substitutes for held-out evaluation. They are included especially for interrupted runs whose terminal report is missing.

## What the results mean and major discoveries

- Among the recorded arms, `supervised_init` had the highest held-out success (eval_success_rate = 0.999023). This is an ordering within this run, not a general ranking across experiments with different metric definitions or budgets.
- The arm comparison is most informative when arms shared the same environment, seed, and budget. Where the configuration records only one seed, apparent gaps remain vulnerable to optimization variance.

## Limitations and reading guidance

- Metrics are reported under the names used by the historical environment and runner. Similar-looking metrics from different generations of the code are not assumed to be numerically interchangeable.
- A maximum observed in training is not treated as held-out performance. Terminal evaluation values are preferred whenever present.
- Missing summaries, empty metric streams, failed retries, and incomplete arms are preserved as negative evidence rather than silently dropped.
- Candidate names and stage names are LLM-authored run artifacts. Interpretations in this historical log do not alter framework prompts, defaults, diagnostics, or task logic.

## Artifact provenance

| Artifact type | Count |
|---|---|
| .mp4 | 55 |
| .json | 40 |
| .pkl | 35 |
| .csv | 1 |
| .jsonl | 1 |
| .log | 1 |
| .md | 1 |
| .txt | 1 |

Primary evidence files:

- [`REPORT.md`](../runs/policy_bias_ppo_shadow_lift_200iter_20260618-123903/REPORT.md)
- [`config.json`](../runs/policy_bias_ppo_shadow_lift_200iter_20260618-123903/config.json)
- [`metrics.jsonl`](../runs/policy_bias_ppo_shadow_lift_200iter_20260618-123903/metrics.jsonl)
- [`summary.json`](../runs/policy_bias_ppo_shadow_lift_200iter_20260618-123903/summary.json)

## Appendix: original authored run report

The following contemporaneous report is retained (with headings demoted) so that details and caveats are not lost in normalization.

### Source: [`REPORT.md`](../runs/policy_bias_ppo_shadow_lift_200iter_20260618-123903/REPORT.md)

## PPO Bias Isolation Report

### Run

- Directory: `runs/policy_bias_ppo_shadow_lift_200iter_20260618-123903`
- Task: Shadow Hand lift
- Seeds: `0`
- PPO iterations: `200` per arm
- Parallel environments: `1024`
- Episode length: `2.5s`
- Control dt: `0.025s`
- Network: `128, 128`
- Arms: `baseline`, `reward`, `action_prior`, `exploration`, `supervised_init`
- Total metric rows: `1000`
- Completed: `2026-06-19 00:18:34 EDT`

### Eval Results

| Arm | Success | Base Return | Shaped Return | Lift Max |
|---|---:|---:|---:|---:|
| baseline | `0.617188` | `20.228020` | `0.000000` | `0.085193` |
| reward | `0.780273` | `41.753387` | `-0.133850` | `0.134684` |
| exploration | `0.828125` | `47.972065` | `0.000000` | `0.159126` |
| action_prior | `0.956055` | `212.487762` | `0.000000` | `0.439774` |
| supervised_init | `0.999023` | `127.710983` | `0.000000` | `0.393314` |

### Comparison To Baseline

| Arm | Success Delta | Return Multiple | Lift Multiple |
|---|---:|---:|---:|
| reward | `+0.163085` | `2.06x` | `1.58x` |
| exploration | `+0.210937` | `2.37x` | `1.87x` |
| action_prior | `+0.338867` | `10.50x` | `5.16x` |
| supervised_init | `+0.381835` | `6.31x` | `4.62x` |

### Final Training Rows

| Arm | Success | Base Return | Lift Max | KL | Entropy |
|---|---:|---:|---:|---:|---:|
| baseline | `0.593750` | `16.432192` | `0.080626` | `0.005370` | `18.521824` |
| reward | `0.703125` | `30.112476` | `0.109199` | `0.004842` | `18.313332` |
| exploration | `0.686523` | `30.846779` | `0.112608` | `0.006010` | `11.952831` |
| action_prior | `0.887695` | `143.277557` | `0.305541` | `0.002619` | `19.620310` |
| supervised_init | `0.969727` | `114.542953` | `0.362606` | `0.004107` | `19.097942` |

### Notes

- All bias arms outperformed baseline on held-out eval success.
- `supervised_init` had the highest eval success: `0.999023`.
- `action_prior` had the highest eval return and lift.
- Reward shaping improved performance, but the shaped reward contribution was small after overlap removal.
- This is a single-seed, single-task result. Multi-seed repeats are needed before treating the ordering as robust.

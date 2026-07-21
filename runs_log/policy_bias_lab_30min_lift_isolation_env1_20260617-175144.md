# Policy Bias Lab 30min Lift Isolation Env1

- **Run directory:** [`runs/policy_bias_lab_30min_lift_isolation_env1_20260617-175144`](../runs/policy_bias_lab_30min_lift_isolation_env1_20260617-175144)
- **Date:** 2026-06-17 17:51:44 (directory timestamp)
- **Status:** completed or completed-with-caveats
- **Experiment class:** agentic prior generation/selection study
- **Recorded task:** lift
- **Training metric rows recovered:** 153,504

## Abstract

This entry reconstructs `policy_bias_lab_30min_lift_isolation_env1_20260617-175144`, an agentic prior generation/selection study. The account is based only on artifacts that survive in the repository: machine-readable configuration, candidate programs, selection diagnostics, training metrics, evaluations, reports, and logs. The run targeted lift. Its purpose and comparisons are inferred from the recorded arm names and configuration where no authored abstract survives; those inferences are identified by this provenance-first framing rather than presented as new experimental facts.

## Experimental structure

| Field | Recorded value |
|---|---|
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
| env.horizon | 1 |
| env.actuators | rh_A_WRJ2, rh_A_WRJ1, rh_A_THJ5, rh_A_THJ4, rh_A_THJ3, rh_A_THJ2, rh_A_THJ1, rh_A_FFJ4, rh_A_FFJ3, rh_A_FFJ0, rh_A_MFJ4, rh_A_MFJ3, rh_A_MFJ0, rh_A_RFJ4, rh_A_RFJ3, rh_A_RFJ0, rh_A_LFJ5, rh_A_LFJ4, rh_A_LFJ3, rh_A_LFJ0, base_x, base_y, base_z |
| env.tasks.lift.objective | raise object above starting height |
| env.tasks.lift.success | lift > 0.05m |
| es.generations | 1000000 |
| es.population | 1 |
| es.population_batch | 1 |
| es.envs | 1 |
| es.sigma | 0.04 |
| es.lr | 0.03 |
| es.elite_frac | 0.5 |
| es.supervised_steps | 80 |
| es.supervised_batch | 128 |
| es.supervised_lr | 0.001 |
| es.target_train_seconds | 1,800 |
| llm_backend | codex |
| llm_model | — |

### Arms and sub-runs represented by directories

`lift_s0_action_prior`, `lift_s0_baseline`, `lift_s0_exploration`, `lift_s0_reward`, `lift_s0_supervised_init`

## Results

### Recorded results: `lift_s0_action_prior/eval.json`

| Recorded field | Value |
|---|---|
| task | lift |
| seed | 0 |
| arm | action_prior |
| eval_fitness | -0.195236 |
| eval_success_rate | 0 |
| eval_summary.palm_obj_dist | 0.193104 |
| eval_summary.min_finger_dist | 0.112708 |
| eval_summary.n_contacts | 0 |
| eval_summary.closure | 0.276875 |
| eval_summary.lift | 0.00002 |
| eval_summary.obj_xy_disp | 0 |

### Recorded results: `lift_s0_baseline/eval.json`

| Recorded field | Value |
|---|---|
| task | lift |
| seed | 0 |
| arm | baseline |
| eval_fitness | -0.19904 |
| eval_success_rate | 0 |
| eval_summary.palm_obj_dist | 0.193005 |
| eval_summary.min_finger_dist | 0.112614 |
| eval_summary.n_contacts | 0 |
| eval_summary.closure | 0.250187 |
| eval_summary.lift | 0.00002 |
| eval_summary.obj_xy_disp | 0 |

### Recorded results: `lift_s0_exploration/eval.json`

| Recorded field | Value |
|---|---|
| task | lift |
| seed | 0 |
| arm | exploration |
| eval_fitness | -0.19904 |
| eval_success_rate | 0 |
| eval_summary.palm_obj_dist | 0.193005 |
| eval_summary.min_finger_dist | 0.112614 |
| eval_summary.n_contacts | 0 |
| eval_summary.closure | 0.250187 |
| eval_summary.lift | 0.00002 |
| eval_summary.obj_xy_disp | 0 |

### Recorded results: `lift_s0_reward/eval.json`

| Recorded field | Value |
|---|---|
| task | lift |
| seed | 0 |
| arm | reward |
| eval_fitness | -4.502122 |
| eval_success_rate | 0 |
| eval_summary.palm_obj_dist | 0.193005 |
| eval_summary.min_finger_dist | 0.112614 |
| eval_summary.n_contacts | 0 |
| eval_summary.closure | 0.250187 |
| eval_summary.lift | 0.00002 |
| eval_summary.obj_xy_disp | 0 |

### Recorded results: `lift_s0_supervised_init/eval.json`

| Recorded field | Value |
|---|---|
| task | lift |
| seed | 0 |
| arm | supervised_init |
| eval_fitness | -0.198979 |
| eval_success_rate | 0 |
| eval_summary.palm_obj_dist | 0.192997 |
| eval_summary.min_finger_dist | 0.112609 |
| eval_summary.n_contacts | 0 |
| eval_summary.closure | 0.250441 |
| eval_summary.lift | 0.00002 |
| eval_summary.obj_xy_disp | 0 |

### Recorded results: `summary.json`

| Arm/interface | eval_success_rate | eval_fitness |
|---|---|---|
| action_prior | 0 | -0.195236 |
| baseline | 0 | -0.19904 |
| exploration | 0 | -0.19904 |
| reward | 0 | -4.502122 |
| supervised_init | 0 | -0.198979 |

### Training-metric reconstruction

| Arm/sub-run | Rows | Last iter | Last env steps | Elapsed s | Last success | Best success | Last lift max | Last base return | Best graded/objective |
|---|---|---|---|---|---|---|---|---|---|
| baseline | 30951 | — | — | — | — | — | — | — | — |
| reward | 31162 | — | — | — | — | — | — | — | — |
| action_prior | 29784 | — | — | — | — | — | — | — | — |
| exploration | 30592 | — | — | — | — | — | — | — | — |
| supervised_init | 31015 | — | — | — | — | — | — | — | — |

These are training-log endpoints and maxima, not substitutes for held-out evaluation. They are included especially for interrupted runs whose terminal report is missing.

## What the results mean and major discoveries

- Among the recorded arms, `supervised_init` had the highest held-out success (eval_success_rate = 0). This is an ordering within this run, not a general ranking across experiments with different metric definitions or budgets.
- Among the recorded arms, `action_prior` had the highest graded/task objective (eval_fitness = -0.195236). This is an ordering within this run, not a general ranking across experiments with different metric definitions or budgets.
- The arm comparison is most informative when arms shared the same environment, seed, and budget. Where the configuration records only one seed, apparent gaps remain vulnerable to optimization variance.

## Limitations and reading guidance

- Metrics are reported under the names used by the historical environment and runner. Similar-looking metrics from different generations of the code are not assumed to be numerically interchangeable.
- A maximum observed in training is not treated as held-out performance. Terminal evaluation values are preferred whenever present.
- Missing summaries, empty metric streams, failed retries, and incomplete arms are preserved as negative evidence rather than silently dropped.
- Candidate names and stage names are LLM-authored run artifacts. Interpretations in this historical log do not alter framework prompts, defaults, diagnostics, or task logic.

## Artifact provenance

| Artifact type | Count |
|---|---|
| .pkl | 30 |
| .json | 8 |
| .csv | 1 |
| .jsonl | 1 |
| .md | 1 |

Primary evidence files:

- [`config.json`](../runs/policy_bias_lab_30min_lift_isolation_env1_20260617-175144/config.json)
- [`metrics.jsonl`](../runs/policy_bias_lab_30min_lift_isolation_env1_20260617-175144/metrics.jsonl)
- [`report.md`](../runs/policy_bias_lab_30min_lift_isolation_env1_20260617-175144/report.md)
- [`summary.json`](../runs/policy_bias_lab_30min_lift_isolation_env1_20260617-175144/summary.json)

## Appendix: original authored run report

The following contemporaneous report is retained (with headings demoted) so that details and caveats are not lost in normalization.

### Source: [`report.md`](../runs/policy_bias_lab_30min_lift_isolation_env1_20260617-175144/report.md)

## Policy Bias Lab

### Summary

- `action_prior`: auc=0.0 eval_success=0.0 eval_fitness=-0.195236 n=1
- `baseline`: auc=0.0 eval_success=0.0 eval_fitness=-0.19904 n=1
- `exploration`: auc=0.0 eval_success=0.0 eval_fitness=-0.19904 n=1
- `reward`: auc=0.0 eval_success=0.0 eval_fitness=-4.502122 n=1
- `supervised_init`: auc=0.0 eval_success=0.0 eval_fitness=-0.198979 n=1

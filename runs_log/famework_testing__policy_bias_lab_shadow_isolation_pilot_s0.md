# Policy Bias Lab Shadow Isolation Pilot S0

- **Run directory:** [`famework_testing/runs/policy_bias_lab_shadow_isolation_pilot_s0`](../famework_testing/runs/policy_bias_lab_shadow_isolation_pilot_s0)
- **Date:** 2026-06-17 (earliest surviving artifact mtime (approximate))
- **Status:** completed or completed-with-caveats
- **Experiment class:** early policy-bias isolation experiment
- **Recorded task:** lift, push, stabilize
- **Training metric rows recovered:** 15

## Abstract

This entry reconstructs `policy_bias_lab_shadow_isolation_pilot_s0`, an early policy-bias isolation experiment. The account is based only on artifacts that survive in the repository: machine-readable configuration, candidate programs, selection diagnostics, training metrics, evaluations, reports, and logs. The run targeted lift, push, stabilize. Its purpose and comparisons are inferred from the recorded arm names and configuration where no authored abstract survives; those inferences are identified by this provenance-first framing rather than presented as new experimental facts.

## Experimental structure

| Field | Recorded value |
|---|---|
| tasks | lift, push, stabilize |
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
| env.tasks.push.objective | move object horizontally from start |
| env.tasks.push.success | obj_xy_disp > 0.06m |
| env.tasks.stabilize.objective | avoid object drift while maintaining control |
| env.tasks.stabilize.success | obj_xy_disp < 0.035m |
| es.generations | 1 |
| es.population | 2 |
| es.population_batch | 1 |
| es.envs | 2 |
| es.sigma | 0.04 |
| es.lr | 0.03 |
| es.elite_frac | 0.5 |
| es.supervised_steps | 1 |
| es.supervised_batch | 2 |
| es.supervised_lr | 0.001 |
| llm_backend | codex |
| llm_model | — |

### Arms and sub-runs represented by directories

`lift_s0_action_prior`, `lift_s0_baseline`, `lift_s0_exploration`, `lift_s0_reward`, `lift_s0_supervised_init`, `push_s0_action_prior`, `push_s0_baseline`, `push_s0_exploration`, `push_s0_reward`, `push_s0_supervised_init`, `stabilize_s0_action_prior`, `stabilize_s0_baseline`, `stabilize_s0_exploration`, `stabilize_s0_reward`, `stabilize_s0_supervised_init`

## Results

### Recorded results: `lift_s0_action_prior/eval.json`

| Recorded field | Value |
|---|---|
| task | lift |
| seed | 0 |
| arm | action_prior |
| eval_fitness | -0.226343 |
| eval_success_rate | 0 |
| eval_summary.palm_obj_dist | 0.203289 |
| eval_summary.min_finger_dist | 0.128168 |
| eval_summary.n_contacts | 0 |
| eval_summary.closure | 0.203979 |
| eval_summary.lift | 0.00002 |
| eval_summary.obj_xy_disp | 0 |

### Recorded results: `lift_s0_baseline/eval.json`

| Recorded field | Value |
|---|---|
| task | lift |
| seed | 0 |
| arm | baseline |
| eval_fitness | -0.226402 |
| eval_success_rate | 0 |
| eval_summary.palm_obj_dist | 0.203324 |
| eval_summary.min_finger_dist | 0.128217 |
| eval_summary.n_contacts | 0 |
| eval_summary.closure | 0.203979 |
| eval_summary.lift | 0.00002 |
| eval_summary.obj_xy_disp | 0 |

### Recorded results: `lift_s0_exploration/eval.json`

| Recorded field | Value |
|---|---|
| task | lift |
| seed | 0 |
| arm | exploration |
| eval_fitness | -0.225738 |
| eval_success_rate | 0 |
| eval_summary.palm_obj_dist | 0.202922 |
| eval_summary.min_finger_dist | 0.127692 |
| eval_summary.n_contacts | 0 |
| eval_summary.closure | 0.203978 |
| eval_summary.lift | 0.00002 |
| eval_summary.obj_xy_disp | 0 |

### Recorded results: `lift_s0_reward/eval.json`

| Recorded field | Value |
|---|---|
| task | lift |
| seed | 0 |
| arm | reward |
| eval_fitness | -4.998878 |
| eval_success_rate | 0 |
| eval_summary.palm_obj_dist | 0.203324 |
| eval_summary.min_finger_dist | 0.128217 |
| eval_summary.n_contacts | 0 |
| eval_summary.closure | 0.203979 |
| eval_summary.lift | 0.00002 |
| eval_summary.obj_xy_disp | 0 |

### Recorded results: `lift_s0_supervised_init/eval.json`

| Recorded field | Value |
|---|---|
| task | lift |
| seed | 0 |
| arm | supervised_init |
| eval_fitness | -0.170907 |
| eval_success_rate | 0 |
| eval_summary.palm_obj_dist | 0.183322 |
| eval_summary.min_finger_dist | 0.106537 |
| eval_summary.n_contacts | 0 |
| eval_summary.closure | 0.327244 |
| eval_summary.lift | 0.00002 |
| eval_summary.obj_xy_disp | 0 |

### Recorded results: `push_s0_action_prior/eval.json`

| Recorded field | Value |
|---|---|
| task | push |
| seed | 0 |
| arm | action_prior |
| eval_fitness | -0.074024 |
| eval_success_rate | 0 |
| eval_summary.palm_obj_dist | 0.185061 |
| eval_summary.min_finger_dist | 0.107159 |
| eval_summary.n_contacts | 0 |
| eval_summary.closure | 0.213544 |
| eval_summary.lift | 0.00002 |
| eval_summary.obj_xy_disp | 0 |

### Recorded results: `push_s0_baseline/eval.json`

| Recorded field | Value |
|---|---|
| task | push |
| seed | 0 |
| arm | baseline |
| eval_fitness | -0.073842 |
| eval_success_rate | 0 |
| eval_summary.palm_obj_dist | 0.184605 |
| eval_summary.min_finger_dist | 0.106714 |
| eval_summary.n_contacts | 0 |
| eval_summary.closure | 0.212735 |
| eval_summary.lift | 0.00002 |
| eval_summary.obj_xy_disp | 0 |

### Recorded results: `push_s0_exploration/eval.json`

| Recorded field | Value |
|---|---|
| task | push |
| seed | 0 |
| arm | exploration |
| eval_fitness | -0.074503 |
| eval_success_rate | 0 |
| eval_summary.palm_obj_dist | 0.186258 |
| eval_summary.min_finger_dist | 0.108471 |
| eval_summary.n_contacts | 0 |
| eval_summary.closure | 0.212287 |
| eval_summary.lift | 0.00002 |
| eval_summary.obj_xy_disp | 0 |

### Recorded results: `push_s0_reward/eval.json`

| Recorded field | Value |
|---|---|
| task | push |
| seed | 0 |
| arm | reward |
| eval_fitness | -4.853805 |
| eval_success_rate | 0 |
| eval_summary.palm_obj_dist | 0.203324 |
| eval_summary.min_finger_dist | 0.128217 |
| eval_summary.n_contacts | 0 |
| eval_summary.closure | 0.203979 |
| eval_summary.lift | 0.00002 |
| eval_summary.obj_xy_disp | 0 |

### Recorded results: `push_s0_supervised_init/eval.json`

| Recorded field | Value |
|---|---|
| task | push |
| seed | 0 |
| arm | supervised_init |
| eval_fitness | -0.073329 |
| eval_success_rate | 0 |
| eval_summary.palm_obj_dist | 0.183322 |
| eval_summary.min_finger_dist | 0.106537 |
| eval_summary.n_contacts | 0 |
| eval_summary.closure | 0.327237 |
| eval_summary.lift | 0.00002 |
| eval_summary.obj_xy_disp | 0 |

### Recorded results: `stabilize_s0_action_prior/eval.json`

| Recorded field | Value |
|---|---|
| task | stabilize |
| seed | 0 |
| arm | action_prior |
| eval_fitness | -0.092522 |
| eval_success_rate | 1 |
| eval_summary.palm_obj_dist | 0.185043 |
| eval_summary.min_finger_dist | 0.10714 |
| eval_summary.n_contacts | 0 |
| eval_summary.closure | 0.213545 |
| eval_summary.lift | 0.00002 |
| eval_summary.obj_xy_disp | 0 |

### Recorded results: `stabilize_s0_baseline/eval.json`

| Recorded field | Value |
|---|---|
| task | stabilize |
| seed | 0 |
| arm | baseline |
| eval_fitness | -0.092293 |
| eval_success_rate | 1 |
| eval_summary.palm_obj_dist | 0.184586 |
| eval_summary.min_finger_dist | 0.106694 |
| eval_summary.n_contacts | 0 |
| eval_summary.closure | 0.212738 |
| eval_summary.lift | 0.00002 |
| eval_summary.obj_xy_disp | 0 |

### Recorded results: `stabilize_s0_exploration/eval.json`

| Recorded field | Value |
|---|---|
| task | stabilize |
| seed | 0 |
| arm | exploration |
| eval_fitness | -0.093114 |
| eval_success_rate | 1 |
| eval_summary.palm_obj_dist | 0.186227 |
| eval_summary.min_finger_dist | 0.108438 |
| eval_summary.n_contacts | 0 |
| eval_summary.closure | 0.212295 |
| eval_summary.lift | 0.00002 |
| eval_summary.obj_xy_disp | 0 |

### Recorded results: `stabilize_s0_reward/eval.json`

| Recorded field | Value |
|---|---|
| task | stabilize |
| seed | 0 |
| arm | reward |
| eval_fitness | -4.874138 |
| eval_success_rate | 1 |
| eval_summary.palm_obj_dist | 0.203324 |
| eval_summary.min_finger_dist | 0.128217 |
| eval_summary.n_contacts | 0 |
| eval_summary.closure | 0.203979 |
| eval_summary.lift | 0.00002 |
| eval_summary.obj_xy_disp | 0 |

### Recorded results: `stabilize_s0_supervised_init/eval.json`

| Recorded field | Value |
|---|---|
| task | stabilize |
| seed | 0 |
| arm | supervised_init |
| eval_fitness | -0.091661 |
| eval_success_rate | 1 |
| eval_summary.palm_obj_dist | 0.183322 |
| eval_summary.min_finger_dist | 0.106537 |
| eval_summary.n_contacts | 0 |
| eval_summary.closure | 0.327239 |
| eval_summary.lift | 0.00002 |
| eval_summary.obj_xy_disp | 0 |

### Recorded results: `summary.json`

| Arm/interface | eval_success_rate | eval_fitness |
|---|---|---|
| action_prior | 0.333333 | -0.130963 |
| baseline | 0.333333 | -0.130846 |
| exploration | 0.333333 | -0.131118 |
| reward | 0.333333 | -4.90894 |
| supervised_init | 0.333333 | -0.111966 |

### Training-metric reconstruction

| Arm/sub-run | Rows | Last iter | Last env steps | Elapsed s | Last success | Best success | Last lift max | Last base return | Best graded/objective |
|---|---|---|---|---|---|---|---|---|---|
| baseline | 3 | — | — | — | — | — | — | — | — |
| reward | 3 | — | — | — | — | — | — | — | — |
| action_prior | 3 | — | — | — | — | — | — | — | — |
| exploration | 3 | — | — | — | — | — | — | — | — |
| supervised_init | 3 | — | — | — | — | — | — | — | — |

These are training-log endpoints and maxima, not substitutes for held-out evaluation. They are included especially for interrupted runs whose terminal report is missing.

## What the results mean and major discoveries

- Among the recorded arms, `supervised_init` had the highest held-out success (eval_success_rate = 0.333333). This is an ordering within this run, not a general ranking across experiments with different metric definitions or budgets.
- Among the recorded arms, `supervised_init` had the highest graded/task objective (eval_fitness = -0.111966). This is an ordering within this run, not a general ranking across experiments with different metric definitions or budgets.
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
| .pkl | 15 |
| .csv | 1 |
| .jsonl | 1 |
| .md | 1 |

Primary evidence files:

- [`config.json`](../famework_testing/runs/policy_bias_lab_shadow_isolation_pilot_s0/config.json)
- [`metrics.jsonl`](../famework_testing/runs/policy_bias_lab_shadow_isolation_pilot_s0/metrics.jsonl)
- [`report.md`](../famework_testing/runs/policy_bias_lab_shadow_isolation_pilot_s0/report.md)
- [`summary.json`](../famework_testing/runs/policy_bias_lab_shadow_isolation_pilot_s0/summary.json)

## Appendix: original authored run report

The following contemporaneous report is retained (with headings demoted) so that details and caveats are not lost in normalization.

### Source: [`report.md`](../famework_testing/runs/policy_bias_lab_shadow_isolation_pilot_s0/report.md)

## Policy Bias Lab

### Summary

- `action_prior`: auc=1.0 eval_success=0.333333 eval_fitness=-0.130963 n=3
- `baseline`: auc=1.0 eval_success=0.333333 eval_fitness=-0.130846 n=3
- `exploration`: auc=1.0 eval_success=0.333333 eval_fitness=-0.131118 n=3
- `reward`: auc=1.0 eval_success=0.333333 eval_fitness=-4.90894 n=3
- `supervised_init`: auc=1.0 eval_success=0.333333 eval_fitness=-0.111966 n=3

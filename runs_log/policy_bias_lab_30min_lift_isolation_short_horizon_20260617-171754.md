# Policy Bias Lab 30min Lift Isolation Short Horizon

- **Run directory:** [`runs/policy_bias_lab_30min_lift_isolation_short_horizon_20260617-171754`](../runs/policy_bias_lab_30min_lift_isolation_short_horizon_20260617-171754)
- **Date:** 2026-06-17 17:17:54 (directory timestamp)
- **Status:** artifact-only or generation-only; no terminal evaluation recorded
- **Experiment class:** agentic prior generation/selection study
- **Recorded task:** lift
- **Training metric rows recovered:** 0

## Abstract

This entry reconstructs `policy_bias_lab_30min_lift_isolation_short_horizon_20260617-171754`, an agentic prior generation/selection study. The account is based only on artifacts that survive in the repository: machine-readable configuration, candidate programs, selection diagnostics, training metrics, evaluations, reports, and logs. The run targeted lift. Its purpose and comparisons are inferred from the recorded arm names and configuration where no authored abstract survives; those inferences are identified by this provenance-first framing rather than presented as new experimental facts.

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
| env.horizon | 20 |
| env.actuators | rh_A_WRJ2, rh_A_WRJ1, rh_A_THJ5, rh_A_THJ4, rh_A_THJ3, rh_A_THJ2, rh_A_THJ1, rh_A_FFJ4, rh_A_FFJ3, rh_A_FFJ0, rh_A_MFJ4, rh_A_MFJ3, rh_A_MFJ0, rh_A_RFJ4, rh_A_RFJ3, rh_A_RFJ0, rh_A_LFJ5, rh_A_LFJ4, rh_A_LFJ3, rh_A_LFJ0, base_x, base_y, base_z |
| env.tasks.lift.objective | raise object above starting height |
| env.tasks.lift.success | lift > 0.05m |
| es.generations | 100000 |
| es.population | 4 |
| es.population_batch | 1 |
| es.envs | 64 |
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

`lift_s0_baseline`

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
| .json | 2 |
| .jsonl | 1 |
| .pkl | 1 |

Primary evidence files:

- [`config.json`](../runs/policy_bias_lab_30min_lift_isolation_short_horizon_20260617-171754/config.json)
- [`metrics.jsonl`](../runs/policy_bias_lab_30min_lift_isolation_short_horizon_20260617-171754/metrics.jsonl)

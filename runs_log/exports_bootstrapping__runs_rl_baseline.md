# Runs Rl Baseline

- **Run artifact:** [`exports/export_experiment/source/bootstrapping/runs_rl_baseline.log`](../exports/export_experiment/source/bootstrapping/runs_rl_baseline.log)
- **Date:** 2026-06-12 23:51:15 (artifact mtime)
- **Status:** completed
- **Experiment class:** early bootstrapping PPO training run
- **Recorded task:** lift-like object-raising task (inferred from the logged `lift` metric)
- **Training metric rows recovered:** 23

## Abstract

This standalone log records one arm of the early bootstrapping PPO study. It preserved 23 progress checkpoints and a terminal result. The configuration and quantitative trajectory below are transcribed from the log rather than reconstructed from code defaults.

## Experimental structure

- `env: obs=27 act=5 horizon=100 frame_skip=5 | envs=512 iters=150`
- `arm=rl_only logging to runs_rl/rl_only`

## Results

| Iteration | Return | Success | Lift | KL |
|---|---|---|---|---|
| 0 | -22.85 | 0 | 0.003 | 0.006 |
| 149 | 385.42 | 1 | 0.28 | 0.0102 |

Best logged checkpoint by success then return: iteration 149, success 1, return 385.42, lift 0.28.

Terminal line: success 1.000, return 385.42.

## What the results mean and major discoveries

- The arm progressed from success 0 at iteration 0 to terminal success 1.000. This establishes convergence in this recorded single run, not a variance-controlled advantage over another arm.

## Limitations and reading guidance

- This is a standalone console log without the saved parameter directory it references.
- Only sampled checkpoints are available; intermediate behavior and held-out evaluation are absent.
- Comparisons with later Shadow Hand runs are invalid because the log records a different 27-observation, 5-action environment.

## Artifact provenance

- Source console log: [`exports/export_experiment/source/bootstrapping/runs_rl_baseline.log`](../exports/export_experiment/source/bootstrapping/runs_rl_baseline.log)
- All tabulated values were parsed from `it ...` and `final ...` lines in that file.

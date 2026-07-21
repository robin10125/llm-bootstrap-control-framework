# Runs Rl S0

- **Run artifact:** [`exports/export_experiment/source/bootstrapping/runs_rl_s0.log`](../exports/export_experiment/source/bootstrapping/runs_rl_s0.log)
- **Date:** 2026-06-12 22:54:45 (artifact mtime)
- **Status:** failed/aborted attempt; no training rows or terminal result survive
- **Experiment class:** early bootstrapping PPO training run
- **Recorded task:** lift-like object-raising task (inferred from the logged `lift` metric)
- **Training metric rows recovered:** 0

## Abstract

This standalone log records one arm of the early bootstrapping PPO study. It preserved 0 progress checkpoints and no terminal result. The configuration and quantitative trajectory below are transcribed from the log rather than reconstructed from code defaults.

## Experimental structure

The log contains only an initialization warning; no environment, arm, or optimizer configuration was recorded.

## Results

No training iterations were logged.

## What the results mean and major discoveries

- The surviving file contains no experimental outcome. It documents an attempted launch only and cannot support a performance conclusion.

## Limitations and reading guidance

- This is a standalone console log without the saved parameter directory it references.
- Only sampled checkpoints are available; intermediate behavior and held-out evaluation are absent.
- Comparisons with later Shadow Hand runs are invalid because the log records a different 27-observation, 5-action environment.

## Artifact provenance

- Source console log: [`exports/export_experiment/source/bootstrapping/runs_rl_s0.log`](../exports/export_experiment/source/bootstrapping/runs_rl_s0.log)
- All tabulated values were parsed from `it ...` and `final ...` lines in that file.

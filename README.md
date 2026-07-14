# llm-framework

Experiments in LLM-authored action interfaces and inductive biases for closed-loop robot control.
The repository is self-contained: shared environment/training code lives in `experiment_runtime/`,
Shadow Hand assets live in `experiment_runtime/assets/`, and every command uses the local `.venv`.
No sibling `bootstrapping` or `hand-manipulation` checkout is loaded at runtime.

## Setup

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[test]'
```

For NVIDIA execution, install the JAX CUDA extra compatible with the machine after creating the
venv, for example `.venv/bin/python -m pip install -U 'jax[cuda12]'`. See
[`docs/dependencies.md`](docs/dependencies.md) for system requirements and validation commands.

## Layout

- `experiment_runtime/`: repository-local MJX environment, compact PPO primitives, LLM backend,
  waypoint compiler, eval-vector implementation, and robot assets.
- `famework_testing/llm_framework/`: the general LLM control-interface package (historical source
  directory name retained for now).
- `policy_bias_lab/`: active policy-bias framework, CLI runners, training code, experiments,
  reports, and quarantined legacy implementations.
- `runs/`: generated experiment outputs.

## Framework smoke run

```bash
.venv/bin/python -m llm_framework.experiments.compare_interfaces \
  --interfaces optimal_bias_plain,optimal_bias_guided \
  --backend mock --tasks lift --seeds 0 --out runs/optimal_bias_mock_smoke
```

The editable install exposes `llm_framework`, `policy_bias_lab`, and `experiment_runtime` without
manually setting `PYTHONPATH`.

# llm-framework

This is an experiment to create a functional framework for LLMs to control robot agents directly by outputting the robot actions.  This is being used in another experiment where I am using LLM policies to bootstrap robot control policies by providing demos.

## Optimal-Bias Experiment

Two recursive Shadow-hand interfaces test whether a reasoning prepass can bias policy generation toward better action basins:

- `optimal_bias_plain`: asks for an optimal approach and inductive biases before policy generation.
- `optimal_bias_guided`: adds abstraction-level prioritization, pedagogical explanation, similar-task adaptation, reward-shaping hints, policy-initialization hints, and iteration rules.

Example smoke run:

```bash
../bootstrapping/.venv/bin/python -m llm_framework.experiments.compare_interfaces \
  --interfaces optimal_bias_plain,optimal_bias_guided \
  --backend mock --tasks lift --seeds 0 --out runs/optimal_bias_mock_smoke
```

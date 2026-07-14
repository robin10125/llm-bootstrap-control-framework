# Policy Bias Lab

A task-agnostic framework for testing whether LLM-authored inductive biases steer a closed-loop
robot policy toward better action basins. Environment/robot data comes from
`experiment_runtime.environment`; task definitions are injected from `tasks.py`; derived signals,
stages, gates, channels, probes, and evals are authored by the LLM during each run.

## Layout

- `cli/`: active selection, PPO, dashboard, and rendering commands.
- `training/`: fragmented-stage PPO and the candidate arbiter.
- `experiments/`: focused ablation runners that are not part of the default workflow.
- `experimental/`: in-progress alternative methods and studies.
- `prompts/`: task-neutral prompt templates.
- `docs/`: design and workflow documentation.
- `reports/`: experiment findings, including the DSL-vs-freeform preliminary.
- `legacy/`: preserved, runnable historical implementations; see `legacy/README.md`.

## Active workflow

Select a model-authored staged prior:

```bash
.venv/bin/python -m policy_bias_lab.cli.agentic_selection \
  --out runs/agentic_selection --rep freeform_staged --budget 10
```

Train a selected program:

```bash
.venv/bin/python -m policy_bias_lab.cli.prior_ppo \
  --arms freeform_encourage \
  --prior-program-arm freeform_encourage=runs/agentic_selection/best_program.json \
  --out runs/prior_ppo
```

Run long training with one fixed program:

```bash
.venv/bin/python -m policy_bias_lab.cli.long_ppo \
  --program runs/agentic_selection/best_program.json \
  --out runs/long_ppo
```

A minimal baseline invocation uses the same active trainer:

```bash
.venv/bin/python -m policy_bias_lab.cli.prior_ppo \
  --arms baseline --iters 1 --envs 16 --eval-envs 16 \
  --target-arm-seconds 1 --episode-seconds 0.1 --out runs/policy_bias_smoke
```

Environment initialization can take roughly four minutes because of JAX/XLA. Long PPO jobs should
run sequentially on the single GPU.

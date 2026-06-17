# Policy Bias Lab

Standalone experiment for testing whether LLM-authored inductive biases can steer a
closed-loop Shadow Hand policy model toward better action basins during exploration.

This folder does not import `llm_framework`. It uses only the Shadow MJX environment
boundary from `../bootstrapping/mjx_env.py`.

## Arms

- `baseline`: ES with task fitness only
- `reward`: ES with LLM-shaped auxiliary rewards
- `action_prior`: ES with LLM-derived action priors during rollout
- `exploration`: ES with LLM-derived exploration covariance scaling
- `supervised_init`: behavior-cloning initialization from LLM-derived target rules
- `reward_action_prior`: reward shaping plus action priors
- `full`: reward shaping plus action priors plus supervised initialization

## Smoke

```bash
../bootstrapping/.venv/bin/python -m policy_bias_lab.run_experiment \
  --llm-backend fixture --smoke --tasks lift --arms baseline,reward \
  --out runs/policy_bias_lab_smoke
```

## Real Run

```bash
../bootstrapping/.venv/bin/python -m policy_bias_lab.run_experiment \
  --llm-backend codex --tasks lift,push,stabilize \
  --arms baseline,reward,action_prior,exploration,supervised_init \
  --seeds 0,1,2 --generations 80 --population 64 --envs 64 \
  --out runs/policy_bias_lab_shadow_isolation
```

Add `reward_action_prior,full` to `--arms` only when testing combined effects after
the isolated comparisons.

For Shadow/MJX memory safety, ES candidates are evaluated one at a time over a vmapped
environment batch. This avoids compiling a `candidate x env` physics graph, which can
exceed 8 GB GPU memory even when the final environment state tensors are small.

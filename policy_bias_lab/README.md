# Policy Bias Lab

Standalone experiment for testing whether LLM-authored inductive biases can steer a
closed-loop Shadow Hand policy model toward better action basins during exploration.

This folder does not import `llm_framework`. It uses only the Shadow MJX environment
boundary from `../bootstrapping/mjx_env.py`.

## Arms

- `baseline`: PPO with the environment reward only
- `reward`: PPO with LLM-shaped auxiliary rewards
- `action_prior`: PPO with LLM-derived action priors added to the actor mean
- `exploration`: PPO with LLM-derived action-group exploration std scaling
- `supervised_init`: behavior-cloning initialization from LLM-derived target rules
- `reward_action_prior`: reward shaping plus action priors
- `full`: reward shaping plus action priors plus supervised initialization

## Smoke

```bash
../bootstrapping/.venv/bin/python -m policy_bias_lab.run_ppo_experiment \
  --llm-backend fixture --tasks lift --arms baseline,reward \
  --iters 1 --envs 16 --eval-envs 16 --episode-seconds 0.1 \
  --out runs/policy_bias_ppo_smoke
```

## Real Run

```bash
../bootstrapping/.venv/bin/python -m policy_bias_lab.run_ppo_experiment \
  --llm-backend codex --tasks lift \
  --arms baseline,reward,action_prior,exploration,supervised_init \
  --seeds 0 --iters 300 --envs 1024 --eval-envs 1024 \
  --episode-seconds 2.5 --control-dt 0.025 \
  --out runs/policy_bias_ppo_shadow_lift
```

Add `reward_action_prior,full` to `--arms` only when testing combined effects after
the isolated comparisons.

`run_experiment.py` is retained as the legacy ES runner. It is useful for narrow
optimizer diagnostics, but it is not the right learner for the main actor-critic
experiment because PPO learns from one policy collecting many parallel rollouts.

Shadow/MJX memory safety now comes from `../bootstrapping/mjx_env.py`: large colliding
distal fingertip meshes are replaced with primitive collision proxies while visual
meshes, joints, actuators, tendons, and object dynamics remain intact. This allows PPO
to run thousands of parallel Shadow environments on an 8 GB GPU.

Reward-shaping terms are compiled as bounded potential-progress terms, not absolute
per-step rewards. For the lift task, overlapping env-reward terms (`palm_obj_dist`,
`closure`, `lift`) and contradictory terms such as maximizing `obj_xy_disp` are discarded.
Metrics report `base_return`, `shaped_return`, and `train_return`; compare arms with the
held-out base return and success/lift metrics.

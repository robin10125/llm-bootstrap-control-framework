# Fragmented Stage PPO Experiment

This folder contains an isolated prototype for long-horizon PPO with action priors.

What it changes:

- Long episodes can be 10-20s while PPO updates on shorter fragments such as 50-100 control steps.
- Simulator state is carried from one fragment to the next until the full episode horizon is tiled.
- Dense shaping comes from each staged prior's own `success` expression, using progress in that expression's sigmoid potential plus a small threshold-crossing bonus.
- The policy emits extra control outputs that map to a `[0, 1]` prior **strength** scale via `clip(bias + gain*out, 0, 1)` (defaults `bias=gain=1`), so it starts at exactly `1.0` (full prior) at init and the controller can scale it all the way down to `0` when the residual policy no longer needs it. This scales only the prior's action strength -- the prior's gates, limits, and stage conditions are unaffected.
  - `--prior-scale-mode` sets the granularity: `group` by default (one knob per **semantic actuator group**, mechanically derived from the robot actuator names), `scalar` (one knob for the whole prior, mainly for replay compatibility), or `per_joint` (one knob per actuator; most expressive, weakest prior). Per-group means are reported in `eval_prior_scale_group_means` and each training row's `prior_scale_group_means`.

Run example:

```bash
.venv/bin/python -m policy_bias_lab.experimental.run_fragmented_stage_ppo \
  --out runs/frag_stage_test \
  --program runs/agentic1/best_program.json \
  --episode-seconds 20 \
  --fragment-steps 100 \
  --envs 128 \
  --eval-envs 128
```

`--fragment-steps` must divide `env.horizon` exactly. For the default `control_dt=0.025`, 20s gives
800 steps, so 50, 80, 100, 160, 200, and 400 are valid fragment sizes.

The main experiment code is not changed. If this works, the integration path is to move the
fragment collector, stage-success reward function, and prior-scale actor head into `ppo_bias.py`
behind explicit config flags.

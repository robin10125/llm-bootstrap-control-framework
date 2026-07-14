# bootstrapping

An experiment in LLM supervision of robot control policies, to accelerate learning new
skills.

## Thesis

When a person learns a new motor skill, the slow, deliberate machinery of the brain does
the early work: attention, focus, and frontal-cortex reasoning guide each attempt,
evaluate what went wrong, and adjust the next try. With practice, control transfers to
faster, cheaper circuits and the skill becomes muscle memory. Deliberate reasoning then
steps back, intervening only to correct or refine.

This project tests the analogous arrangement for robots, with an LLM playing the role of
deliberate reasoning and a small neural policy playing the role of muscle memory. Two
questions:

1. Can LLMs bootstrap robot control policies? An LLM reasons about the scene and the goal
   and proposes a control plan; plans that succeed become training data for the neural
   policy, so it learns from goal-directed demonstrations instead of pure random
   exploration.
2. Can robot control be further refined by LLMs making residual corrections? Once the
   trained policy is competent, the LLM stops authoring from scratch and starts editing:
   it inspects the policy's rollout and proposes a better trajectory, which feeds back into
   training.

If both hold, the expensive reasoning system is engaged only where it adds value — early
acquisition and later refinement — while routine execution runs on a fast learned policy
with no LLM in the control loop.

## How the experiment is built

The control policy is a **closed-loop neural network trained with PPO**: every control
step it observes the scene (joint state, object pose, fingertip positions) and emits
per-joint position-target adjustments. The episode is a real multi-step MDP, trained **at
scale on the GPU with MJX** (MuJoCo-XLA) across hundreds-to-thousands of parallel robots.

The LLM operates the *same* control interface through a **waypoint plan** — a few via-poses
with timings — which a compiler expands into the exact per-step action stream the policy
outputs. So an LLM demonstration is directly comparable to, and directly clonable by, the
policy. The compiler is pluggable, so other LLM control representations can be swapped in.

The bootstrapping loop (one training process; the LLM never blocks the GPU):

1. PPO collects parallel rollouts and updates the policy from reward.
2. An **always-busy async LLM worker** (background thread, fixed call budget) is fed fresh
   scenes; whenever idle it starts the next query. Its plans are scored in MJX against the
   current policy.
3. A plan that beats the policy is cloned: its `(observation, action)` pairs enter a demo
   buffer that drives a **behavior-cloning auxiliary loss** alongside PPO.
4. Once the policy is competent, the worker switches from authoring to **residual
   correction**: it sees the policy's rollout and proposes a fix.

The control policy is tunable after the fact through this same correction channel — the
LLM (or a human) inspects the movement and adjusts it in whatever way works best.

## Arms and evaluation

The headline comparison is the **same PPO procedure with and without LLM supervision**,
under matched env/seed budgets:

- `rl_only` — pure RL, no LLM (the control).
- `rl_<backend>` — PPO + LLM behavior-cloning (and later residual corrections).

Policies are evaluated on **held-out object positions** with success rate (lift-and-hold)
and lift height, reported with bootstrap 95% confidence intervals (`eval.py`). LLM dollar
and token cost is a first-class result axis — the thesis is that the LLM is the expensive
system, engaged sparingly.

First embodiment/task: a parallel-jaw gripper lifting a cube. The env layer is generic
over actuators so the 24-DOF Shadow Hand reuses it once its mesh scene is MJX-portable.

## Layout

- `mjx_env.py` — batched MJX env (reset/step/reward/obs, object randomization).
- `ppo.py` — flax actor-critic, GAE, clipped PPO update (+ BC auxiliary-loss hook).
- `waypoints.py` — LLM waypoint-plan compiler + MJX plan rollout (scoring + demos).
- `llm_worker.py` — always-busy async LLM worker, budget, scratch/residual prompts.
- `llm_supervision.py` — main-thread glue: feed / score / clone, demo buffer, BC batches.
- `train.py` — training loop and arms (`--llm none` = pure RL).
- `eval.py` — held-out comparison with bootstrap CIs.
- `llm_backend.py` — claude-code / codex / anthropic backends (cost + tokens logged).
- `rl_redesign.md` — design rationale and status.
- `legacy/` — the prior open-loop primitive-schedule / CEM experiment (superseded).

Setup: `python -m venv .venv && .venv/bin/pip install "jax[cuda12]" mujoco mujoco-mjx
flax optax` (needs a CUDA GPU).

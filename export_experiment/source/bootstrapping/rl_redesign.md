# LLM-Bootstrapped Closed-Loop RL Control (MJX redesign)

## Context

The prior experiment framed control as an open-loop *primitive schedule* chosen once per
episode — a contextual bandit optimized with CEM. That design deliberately avoided
gradient RL ("the episode is a single schedule, so PPO/SAC-style multistep machinery does
not apply"). This redesign changes the framing: control becomes a **closed-loop, per-step
joint policy** so that real RL applies, trained **at scale on the GPU with MJX**, with an
LLM acting as the deliberate-reasoning system that supplies demonstrations and, later,
residual corrections.

The brain analogy is unchanged: the small RL net is muscle memory; the LLM is the slow,
expensive frontal-cortex reasoning that is engaged only where it adds value (early
acquisition and later correction), never in the fast control loop.

## Decisions (locked with the user)

- **RL stack:** MJX (MuJoCo-XLA) + JAX on GPU; thousands of parallel envs for the gripper,
  hundreds for the Shadow Hand later. Custom PPO (flax/optax). Target hardware: RTX 2070
  (8 GB), so batch size is a tuned/configurable parameter, not a fixed "thousands".
- **Old design:** superseded. Move the schedule/CEM files to `legacy/` (archive, not
  delete) so prior results stay reproducible.
- **LLM control interface:** the LLM emits **waypoints** (sparse via-poses + timings) that
  a compiler expands into the *same* per-step joint-target stream the RL net outputs. The
  compiler is behind an interface so other methods (raw per-step, primitive language) can
  be swapped in later.
- **Supervision loop:** PPO + an **auxiliary DAgger-BC loss**. Plus a **pure-RL control
  arm** (no LLM) for comparison. The LLM runs as an **always-busy async worker** with a
  **call budget (~100 this stage)**: it pulls a fresh sample whenever it is idle, never
  blocks the trainer, and only a small subset of states ever get LLM input. Two modes:
  **from-scratch waypoints** early, **residual correction** once the RL policy is decent
  (the LLM looks at a rollout and corrects it however works best). Accepted demos
  (LLM beats current policy) enter a demo buffer that feeds the BC loss.
- **First target:** gripper + lift on the 2070, env layer kept generic to port to Shadow.

## Architecture

```
                 ┌───────────────────────────────────────┐
                 │ MJX env (JAX, batched N envs on GPU)   │
   PPO rollouts  │  reset / step / reward / obs           │
 ◄───────────────┤  domain randomization (pos/size/mass)  │
                 └───────────────────────────────────────┘
        │ obs, action (per-joint target deltas)
        ▼
 ┌──────────────┐   GAE + PPO update    ┌──────────────────┐
 │ policy/value │◄──────────────────────┤  trainer (train.py)│
 │  net (flax)  │   + BC loss on demos  └──────────────────┘
 └──────────────┘            ▲
                             │ sampled demos (obs → target stream)
                 ┌───────────┴────────────┐
                 │ demo buffer            │
                 └───────────▲────────────┘
                             │ accepted (LLM beats policy)
            ┌────────────────┴─────────────────┐
            │ async LLM worker (llm_worker.py)  │
            │  - pull idle → query LLM          │
            │  - scratch waypoints | residual   │
            │  - compile → CPU-sim → score      │
            │  - budget cap (~100 calls)        │
            └───────────────────────────────────┘
```

### Components / files

- `mjx_env.py` — functional MJX env for the gripper-lift task: `reset(key, setup)`,
  `step(state, action)`, reward, observation vector; batched with `jax.vmap`; domain
  randomization of object position/size/mass/friction. A possibly MJX-tweaked copy of the
  scene (pyramidal cone / condim 3 if elliptic/condim-4 are unsupported).
- `action_space.py` — per-joint position-target deltas around a nominal pose, squashed to
  actuator `ctrlrange`. Gripper action dim = 5 (base_x/y/z, g_left, g_right). The same
  vector is what the waypoint compiler produces, so LLM demos are directly cloneable.
- `waypoints.py` — waypoint-plan schema + compiler to a per-step target stream (linear/
  spline interpolation between via-poses with timings). Behind a `Compiler` interface.
- `ppo.py` — flax MLP policy (Gaussian) + value head; GAE; clipped PPO update in optax.
- `llm_worker.py` — async worker (thread/process) that reuses `llm_backend.py`
  (codex/claude-code), builds scratch and residual prompts, parses waypoint JSON, simulates
  candidates on CPU MuJoCo (reuse `ShadowPolicyRunner`-style stepping or a single env),
  scores vs the current policy, and pushes accepted demos. Enforces the call budget and
  logs every prompt/completion/meta.
- `train.py` — main loop: collect MJX rollouts → PPO update (+ BC loss for LLM arms) →
  drain accepted demos from the worker. Arms: `rl_only`, `rl_llm_bc`, `rl_llm_residual`.
  Flags for batch size, seeds, LLM backend/model, call budget, BC coefficient.
- `eval.py` — held-out setup evaluation; success rate (held lift) + lift height with
  bootstrap CIs over seeds; cost/token reporting.
- Reuse: `llm_backend.py` (unchanged), `models/gripper_scene.xml` (source for the MJX
  scene), keyframe poses for the nominal action center.

## MJX risks to verify before building

- Scene must load under MJX: `option cone="elliptic"` and `condim="4"` may need to become
  pyramidal / condim 3. Cylinder collisions (`g_mount`) are limited in older MJX — it is
  non-load-bearing and can be made `contype=0 conaffinity=0` if needed.
- 8 GB VRAM bounds the env batch; measure max stable N for the gripper.
- LLM candidate simulation runs on **CPU MuJoCo** (outside JAX jit) to keep the worker
  simple and the training jit clean.

## Phasing

- **Phase 0 (in progress):** install JAX-CUDA/MJX/flax/optax in a fresh project venv;
  smoke-test that the gripper scene loads and steps in MJX on the 2070.
- **Phase 1:** `mjx_env` + `ppo` + `train.py` pure-RL arm; train a gripper that lifts.
- **Phase 2:** `waypoints` compiler + `llm_worker` (scratch mode) + DAgger-BC; LLM arm.
- **Phase 3:** residual-correction LLM mode once the policy is competent.
- **Phase 4:** eval harness + stats; expose the tunable correction interface; then begin
  the Shadow Hand (mesh) port.

## Test run results (gripper-lift, codex/gpt-5.5, 2026-06-12/13)

First real end-to-end run on the RTX 2070. 512 parallel envs, 150 PPO iters, seed 0,
matched across arms; LLM budget 100 codex/gpt-5.5 calls.

Training success rate by iteration:

| iter | rl_only | rl_codex (bc=0.5, annealed) | rl_codex (bc=1.0) |
|------|---------|------------------------------|--------------------|
| 10   | 0.16    | —                            | —                  |
| 20   | 0.55    | —                            | —                  |
| 30   | 0.85    | 0.50                         | 0.46               |
| 60   | 0.98    | 0.65                         | (stopped)          |
| 90   | ~1.00   | 0.87                         | —                  |
| 150  | 1.00    | 0.99                         | —                  |

Held-out eval (n=512, bootstrap 95% CI):

| object range      | rl_only            | rl_codex (tuned)        |
|-------------------|--------------------|-------------------------|
| xy=0.04 (train)   | 1.000              | 1.000                   |
| xy=0.08 (2x wide) | 1.000              | 0.982 [0.971, 0.992]    |

Findings:

1. **The full loop works.** codex emits waypoint plans -> compiled to per-step actions ->
   scored in MJX -> accepted only if they beat the current policy -> behavior-cloned.
   100 calls, 0 errors, 18 plans accepted (1800 demos). Accepts stopped near iter 30 once
   the policy surpassed the open-loop LLM plans — the designed self-regulation, and the
   natural trigger to switch to residual mode (Phase 3).
2. **On this task the LLM does not help; it mildly hurts.** Pure PPO gets first lifts by
   iter ~10 and 100% by iter ~60. BC toward ~60%-success open-loop demos anchors the policy
   below RL's own trajectory. `bc_coef=1.0` over-weights BC and caps progress; annealing
   `0.5 -> 0` by iter ~90 lets PPO recover to ~baseline. Pure RL is still fastest and
   generalizes slightly better (100% vs 98.2% at 2x range).
3. **Why:** gripper-lift is too easy and its reward is dense, so RL needs no help getting
   initial reward — exactly the regime where demonstrations add least.

Implications / next steps:

- To expose LLM benefit, move to a regime where RL alone struggles for initial traction:
  sparse/terminal-only reward, much wider randomization + perturbations, or the 24-DOF
  Shadow Hand. That is where goal-directed demonstrations should pay off.
- Make annealed `bc_coef` the default (done: 0.5, anneal 0.6).

## Implemented after the first run (reward-weighted BC + residual mode)

- **Reward-weighted bootstrap (advantage-weighted BC).** Each accepted demo is stored with
  a weight equal to its plan's return margin over the policy (`DemoBuffer.add(..., weight)`).
  The BC loss is `-(w * logπ(a_demo|s)).mean()` with `w = softmax-ish exp(adv/temp)` on the
  batch-normalized margins (`ppo.ppo_loss`, `PPOConfig.bc_weight_temp`, flag
  `--bc-weight-temp`, default 1.0; `<=0` = uniform BC). So demos that beat the policy by more
  pull harder, and marginal demos pull less — a partial fix for plain forward-KL BC averaging
  toward mediocre demos.
- **LLM residual-correction mode (Phase 3).** The worker switches from `scratch` to
  `residual` once training success exceeds `--residual-after` (0.3). In residual mode the LLM
  is shown the policy's own rollout (state trace sampled ~every 0.3 s) plus the outcome
  (lifted? peak object height) and asked to fix the failure; the corrected plan is scored and
  cloned like any other demo. This is the accept-rate-collapse handoff: the LLM goes from
  author to corrector once the policy surpasses from-scratch plans.

## Shadow Hand port + residual lifecycle (2026-06-13)

### Shadow Hand in MJX on the 2070

The 23-DOF Shadow Hand (`make_env("shadow")`, scene `hand-manipulation/.../scene_cube.xml`) now
loads and steps in MJX through the *same* `MjxEnv`/PPO/waypoint/supervisor stack as the gripper.
Three MJX-portability gaps were patched generically in `load_model` (so the gripper is
untouched):

- **cylinder collisions** are unimplemented for several partners in MJX 3.9 (notably
  cylinder<->mesh, which the Shadow finger/wrist collision cylinders need). Every *colliding*
  cylinder is converted to a capsule of the same (radius, half-length).
- **all-pairs self-collision blows up memory.** With every geom at `contype=conaffinity=1`,
  `put_model` took ~420 s and `reset` OOM'd (a single constraint matmul wanted 8.7 GiB > 8 GB).
  `EnvConfig.grasp_geom_substr` keeps only the hand's grasping geoms (palm + finger
  segments + thumb) collidable, in a contype/conaffinity group that collides with the cube and
  table but **not itself**. This drops load to ~125 s and fits N=64 on the 2070.
- **visual-only meshes** must never be promoted to collidable (some aren't convex-decomposable
  — `mesh.convex` raises). The allowlist only ever keeps/drops collisions on already-colliding
  geoms.

The single waypoint `open` scalar is generalized from the gripper's 1-DOF finger to the whole
hand: the env derives open/closed actuator targets from the `open hand` / `close hand`
keyframes (mapped through each actuator's transmission via MuJoCo `actuator_length`, which
handles the coupled-tendon finger actuators), and `open` interpolates the hand between them.
The reward's "close when near" term and the LLM prompts are likewise embodiment-aware.

**Throughput / memory (RTX 2070, N=64):** ~1.4k physics-steps/s with a fused `scan` rollout at
`physics_dt=0.01` (2 substeps); an N=64 episode rolls out in ~9 s. N=256 OOMs. So Shadow runs
at small batch only — consistent with the gripper being ~12x faster.

**Open issue — the grasp itself is unsolved.** A scripted descend+close+lift expert only nudges
the cube ~0.02 m (success needs 0.05 m): a five-fingered top-down power grasp does not reliably
cage a 5 cm tabletop cube (no thumb opposition from below; fingertip/palm contact only). The
*pipeline* runs end-to-end on the hand, but demonstrating an LLM-bootstrapping *win* needs the
manipulation tuned first (approach pose / finger spread / thumb opposition, or an easier object
such as a graspable handle or a larger/edged object), or a simpler Shadow sub-task.

### Residual-correction lifecycle (switch on outperform, discard, give up)

`Supervisor` now runs a latching three-state lifecycle, driven by whether the LLM is actually
beating the policy (accepts are gated on succeeding *and* out-returning the policy on the same
reset — `_drain`):

- **scratch** — author plans from scratch. Stay here while they help.
- **scratch -> residual** when the policy **outperforms the LLM's scratch demos**: the recent
  scratch accept-rate collapses below `switch_rate` (after `switch_min_attempts`), with the old
  `training_success >= residual_after` as a fallback trigger.
- **residual** — correct the policy's own rollout. A correction that fails to beat the policy is
  **thrown out** (not cloned) and counts against the helpfulness rate.
- **residual -> off** when residual corrections are **consistently unhelpful**: the recent
  residual accept-rate stays at/below `give_up_rate` after `give_up_min_attempts`. The worker
  then stops being fed, so no more LLM budget is spent.

Validated deterministically (controlled rollout returns, no LLM cost): the supervisor walks
scratch -> residual -> off exactly as the policy first lags, then beats, the LLM. Reward-weighted
BC (advantage-weighted, from the first run) is unchanged and still applies to accepted demos.

## Verification

- MJX smoke test: load scene, `jax.jit` a step, vmap a batched rollout, assert run on GPU.
- Phase 1: lift success climbs well above random; report a learning curve.
- Phase 2: budget respected; demos logged with prompt/completion; demos accepted only when
  they beat the current policy; BC loss measurably pulls the LLM arm.
- Final: `rl_only` vs `rl_llm_bc` vs `rl_llm_residual` with bootstrap CIs over seeds, plus
  LLM dollar/token cost as a first-class axis.

# Toward a Publishable Experiment

Problems with the current setup and what to change, ordered by how much each
threatens the claim.

## Status (2026-06-12)

- Section 1 (real LLM calls): mechanism done. `llm_backend.py` provides
  `claude-code` (headless Claude Code, cost/tokens captured per call), `codex`,
  and `anthropic` backends; both experiment scripts take `--llm` / `--llm-model`,
  log every prompt/completion/meta to the run directory, record the source of
  every supervision policy (real vs `mock_fallback`), and report aggregate cost.
  Remaining: actually rerun the experiments with a real backend, ideally two
  model families.
- Section 2 (policy restructure): done. `sequence_policy.py` defines
  variable-length token sequences (discrete primitive head, continuous param
  heads, stop head, until flag) with lossless structural round-trip from
  schedule JSON; `SeqPolicyNet` is an autoregressive BC net. `policy_spaces.py`
  exposes `--policy-space template|sequence` in both experiment scripts, so the
  old fixed template survives as the easy-mode comparison.
- Section 3 (primitives): done. Runner v2 adds `until` sensor-conditioned step
  termination (duration becomes a timeout), `move_delta`, object-centric
  `approach_object`, continuous `grasp` closure, object shape/size/mass/friction
  variation via XML rewrite, and mid-episode velocity perturbations.
  `shadow_primitives.json` is at version 2.
- Section 4 (confounds): done. Per-(seed, arm) rng streams; `--ablation` adds
  init-mean/init-std control arms plus a best-of-random-init arm; residual
  prompts condition on the best and worst per-setup rollouts; lift success is
  a held lift (0.75 s hold), and `success_rate` is each task's binary check.
- Section 5 (scale/statistics): mechanics done. Bootstrap 95% CIs over seeds,
  paired per-seed comparisons vs `normal_rl_cem`, default 10 seeds, candidate
  evaluation parallelized; the bootstrap loop trains BC across randomized
  setups and evaluates held-out. Remaining: run at scale.
- Section 6 (baselines): done. `--baselines` adds `scripted_expert`, `llm_only`
  (fresh LLM schedule per test setup, no learning), `normal_rl_bonus_cem`
  (budget-compensated CEM), and `reinforce` (ES-style REINFORCE over the same
  latent space; the episode is a single schedule, so PPO/SAC-style multistep RL
  does not apply to this contextual-bandit setting). The bootstrap script gains
  `--expert-mode auto|scratch|residual` for supervision-mode ablations, and
  `--llm none` is the BC-without-LLM control.
- Section 7 (task suite): done. `tasks.py` registers `lift`, `lift_perturbed`
  (mid-episode velocity kick), `place`, and `push`, each with setup samplers,
  binary success checks, shaped scores, and scripted experts;
  `--object-set varied` randomizes shape/size/mass/friction. Second embodiment:
  `models/gripper_scene.xml` + `gripper_policy_runner.py`, a floating-base
  parallel-jaw gripper driven by the same primitive language
  (`--embodiment gripper`, sequence space only).
- Section 8 (framing/related work): writing task, not started.

Measured caveats to carry into any writeup:

- The stored Codex lift demonstration is a knife-edge policy: it succeeds at the
  centered object position but fails at 1–2 cm offsets, and fails even at the
  center if its transit timing is slowed. The scripted Shadow experts inherit
  this brittleness (the gripper experts are robust: 4/4 on lift and perturbed
  lift across randomized positions). This is honest evidence for the motivation
  — single demonstrations are fragile and robustness must come from the
  learning loop — but it means Shadow-side mock runs bootstrap from weak
  demonstrations, and headline Shadow results need real LLM supervision runs.
- Remaining for publication: scale runs (10+ seeds, two model families, real
  backends) across the task × embodiment matrix, and the related-work section.

## 1. The "LLM" arms are mostly not an LLM

The `codex_with_residual_cem` arm in `shadow_fair_compare_training.py` draws its
residual candidates from `local_residual_policy()`, a hand-written heuristic cycling
through three fixed parameter sets. The scratch expert is also hand-coded. As it
stands, the result tests "hand-tuned residual edits help CEM," not "LLM residual
suggestions help CEM."

Fixes:

- Every arm labeled LLM uses real model calls. Log every prompt and completion.
- Run at least two model families so the claim is not model-specific.
- Report token counts and dollar cost as a first-class result axis. The thesis is
  that the LLM is the expensive system; cost must appear in the results, not just
  rollout counts.

## 2. The policy parameterization gives away the answer

`decode_policy` maps a 12-dim latent into a fixed schedule template
(open, approach, descend, pre-grasp, grasp, wait, lift). The skill structure is
hard-coded; CEM only tunes 12 continuous parameters. Consequences:

- The "normal RL" baseline already knows how to grasp. Any bootstrap advantage
  reduces to "better init on a 12-dim landscape."
- `encode_policy` projects whatever the LLM writes back into the same template,
  discarding step order, step count, and extra primitives. The LLM's comparative
  advantage, proposing structure, is compiled away before it can matter.

Fix: make the policy a variable-length sequence over the primitive vocabulary,
with a discrete primitive head and continuous parameter heads (as already sketched
in `rl_bootstrap_plan.md`). The LLM's structural choices then survive into the
learned policy, and the baseline has to discover structure itself.

## 3. Restructure the primitives

Current vocabulary (`set_base`, 8 named `hand_pose` keyframes, `wait`) is fine for
v1 but limits claims:

- Replace fixed `duration_s` with sensor-conditioned termination
  ("execute until `contacts.hand_object_count >= 2` or timeout"). Fixed durations
  make every policy open-loop and brittle.
- Add relative motion (`move_delta`) and object-centric primitives
  (`approach(object, offset)`). Object-relative primitives are what make
  generalization claims meaningful.
- Make grasping continuous (closure fraction or per-finger-group synergies)
  instead of an 8-way enum. Keep named keyframes as LLM-facing macros.

Note: the observation is just object xyz and `decode_policy` already centers on
the object position, so held-out-position generalization is largely built into
the parameterization. Harder generalization axes are needed (see tasks).

## 4. Confounds in the comparison

- Arms differ in CEM init std (1.35 vs 0.45) and init mean, so improvement
  cannot be attributed to the Codex init alone. Ablate: Codex-init with wide std,
  zero-init with narrow std, and a best-of-N-random init given the same rollout
  budget the LLM consumed.
- Arms within a seed share one rng consumed sequentially, so they see different
  random streams. Give each arm its own identically seeded generator.
- Residuals always reference `train_setups[0]`'s rollout; condition them on the
  setup they will be evaluated on.
- Headline metric should be binary: lifted above threshold and held for N
  seconds. Report the shaped score secondarily.

## 5. Scale and statistics

3 seeds, 5 generations, population 8, 3 train / 5 test positions in a 2.5 cm
radius is a smoke test. Needed:

- 10–20 seeds per arm; learning curves with 95% CIs (seeds as the unit);
  bootstrap CIs or a nonparametric test on arm differences; success rates with
  binomial CIs.
- Wider position randomization plus randomized object orientation, size, mass,
  and friction.
- Train the BC policy across randomized setups. The current bootstrap loop
  trains on a single fixed setup, so the MLP memorizes one observation.

## 6. Missing baselines

- Scripted expert alone.
- LLM-only, no learning (re-query every episode); establishes the cost the
  learned policy saves.
- BC on the scripted expert with no LLM; isolates whether LLM authorship
  matters or just demonstrations.
- Plain CEM given the LLM's rollout budget in addition to its own.
- A standard RL baseline (PPO/SAC) on the same primitive action space.
- Ablations: scratch-only vs residual-only vs both; number of LLM calls.

## 7. Task suite

One cube lift cannot support the thesis. Ladder, in rough order of effort:

- Object variation: sphere, cylinder, thin plate, irregular meshes; varied size,
  mass, friction. Thin plates and small spheres require different hand poses, so
  structural choices matter.
- New verbs: place at target, push to goal region, reorient cube to a target
  face, handover between positions. Reorientation exercises residual corrections.
- Perturbation/recovery: knock the object mid-episode; tests whether residual
  supervision teaches recovery rather than parameter polish.
- A second embodiment (parallel-jaw gripper, different scene). Cheapest single
  credibility boost.

Strongest result format: a tasks-by-arms matrix showing where bootstrapping
helps, where residuals help, and at least one place where they do not.

## 8. Framing and related work

Position against Code-as-Policies, Eureka, Language-to-Rewards, VoxPoser, and
residual policy learning (Silver et al., Johannink et al.). Eureka is the
comparison reviewers will reach for: there the LLM shapes the objective; here it
supplies demonstrations and residual actions. State plainly that policies are
primitive schedules, not torque control, and treat it as a scoping decision,
with sensor-conditioned primitives as the answer to "isn't this open-loop?"

## Priority order

1. Real LLM calls everywhere.
2. Variable-length policy with discrete/continuous heads so structure survives.
3. Fix the CEM confounds.
4. Scale seeds and randomization with proper statistics.
5. More tasks plus the missing baselines.
6. Second embodiment.

Items 1–4 make a defensible workshop paper; 5–6 are what gets past conference
reviewers.

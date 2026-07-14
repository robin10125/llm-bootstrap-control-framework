# Plan: LLM-authored task rewards + progress curriculum for judging partial policies

> **Status: implemented (2026-06-15).** Modules `eval_metrics.py`, `reward_spec.py`,
> `curriculum.py`; integrated into `mjx_env.py` (eval vector + stage + reward modes),
> `llm_supervision.py` (progress-dominance acceptance + `author_reward_spec`),
> `llm_worker.py` (reward-authoring/reflection prompts), `train.py`
> (`--reward {builtin,default,llm,<expr>}`, `--curriculum`, `--reward-reflect`). All four
> phases are in. See the "Implementation notes" at the end.

## Why

Today the system judges an LLM demo with a **binary success gate** (`llm_supervision._drain`):

```python
accepted = plan_succ and (not pol_succ or margin > 0.0)   # margin on episodic return
```

On a hard task (the Shadow Hand cube lift) *nothing succeeds* — neither the policy nor the
LLM — so nothing is ever accepted and the bootstrap loop is vacuous. Yet some plans are clearly
**more helpful**: they move the palm closer, get 1 then 2 then 3 fingers onto the cube, nudge it
up 0.03 m. The binary gate throws all of that away.

Two changes fix this, and they reinforce each other:

1. **Judge by progress, not just success.** Measure *how far along the task* a rollout got, so a
   helpful-but-incomplete plan can beat the policy and be cloned.
2. **Let the LLM decompose the task into a curriculum of milestones** (reach → contact 1 finger
   → 3 fingers → cage → lift small → lift to height), each with its own dense feedback. The same
   milestones (a) give the LLM a richer yardstick for ranking its own demos and (b) give PPO a
   denser, staged training signal instead of a sparse terminal reward.

The LLM authors the reward/curriculum; the RL agent consumes it; the supervisor uses it to
accept partial-credit demos. This is Eureka-style reward synthesis fused with automatic
curriculum and our existing scratch→residual→give-up demo loop.

## Core abstraction: a structured eval vector

Everything keys off a per-step **eval vector** of ground-truth, task-relevant measurements that
the env computes from sim state (never from the policy's self-report). It is the shared
vocabulary the LLM writes rewards/milestones against and the judge ranks rollouts with.

For the Shadow cube lift, the eval vector includes:

| field                | meaning                                                        |
|----------------------|----------------------------------------------------------------|
| `palm_obj_dist`      | ‖grasp_site − object‖ (reach)                                  |
| `finger_obj_dist[5]` | each fingertip (distal) → object distance                      |
| `n_contacts`         | # fingertips within a contact threshold of the cube (0..5)     |
| `closure`            | mean hand-actuator progress open→closed (already in `_reward`) |
| `lift`               | object_z − obj_start_z                                         |
| `obj_xy_disp`        | how far the cube slid (penalize knocking it away)              |
| `success`            | lift > success_height (the north-star, kept separate)          |

Implementation: extend `EnvState.metrics` (already carries `lift`/`success`/`reach`). Fingertip
distances come from `data.xpos`/`site_xpos` of the distal bodies vs `object_bid` (proximity is
jit-trivial; true contact via `data.contact` is fiddlier in MJX, so default to a distance
threshold and treat MJX contacts as an optional refinement). `n_contacts` = count of
`finger_obj_dist < contact_eps`. All jit-friendly.

New file `eval_metrics.py` computes the eval vector from `mjx.Data`; `mjx_env._observe`/`_reward`
call into it so obs, reward, and judging all read the same numbers.

## Component 1 — reward spec (what the LLM produces)

The LLM does **not** emit free-form Python in the hot loop first. Stage it for safety + jit:

- **Tier A (default, jit-safe):** the LLM emits a **JSON reward/curriculum spec** — a list of
  terms and milestones drawn from a fixed, audited library of operators over eval-vector fields
  (e.g. `neg_dist(palm_obj_dist)`, `ge(n_contacts, k)`, `clip(lift, 0, h)`, `exp_kernel(field,
  scale)`), each with a weight. A trusted compiler (`reward_spec.py`) turns the spec into a pure
  `jp` function `reward(eval) -> float`. The LLM controls *structure and weights*, not arbitrary
  code, so it can't break jit or escape the sandbox.
- **Tier B (opt-in):** free-form codex Python reward, executed in a restricted sandbox (AST
  allowlist: no imports/attributes/builtins beyond `jp` + eval fields; compiled into restricted
  globals; smoke-run for finiteness/shape). Only enabled behind a flag, and still validated by
  the same checks as Tier A.

Spec schema (Tier A):

```json
{
  "milestones": [
    {"name": "reach",     "predicate": {"op": "lt", "field": "palm_obj_dist", "thr": 0.06},
     "reward": [{"op": "neg_dist", "field": "palm_obj_dist", "w": 1.0}]},
    {"name": "touch1",    "predicate": {"op": "ge", "field": "n_contacts", "thr": 1},
     "reward": [{"op": "linear", "field": "n_contacts", "w": 0.5}]},
    {"name": "grip3",     "predicate": {"op": "ge", "field": "n_contacts", "thr": 3},
     "reward": [{"op": "linear", "field": "n_contacts", "w": 1.0},
                {"op": "linear", "field": "closure",    "w": 0.5}]},
    {"name": "liftlow",   "predicate": {"op": "gt", "field": "lift", "thr": 0.02},
     "reward": [{"op": "clip", "field": "lift", "lo": 0, "hi": 0.05, "w": 12.0}]},
    {"name": "success",   "predicate": {"op": "gt", "field": "lift", "thr": 0.05},
     "reward": [{"op": "const", "w": 5.0}]}
  ],
  "penalties": [{"op": "linear", "field": "obj_xy_disp", "w": -0.5}]
}
```

## Component 2 — curriculum controller (`curriculum.py`)

- **Cumulative potential-based shaping**, not switching. Total reward = penalties + Σ over
  milestones up to (current_stage + 1) of that milestone's reward, with each milestone's bonus
  gated by reaching the previous one's predicate. Cumulative + potential-based means advancing
  the curriculum never makes earlier progress un-rewarded (avoids catastrophic forgetting and
  reward oscillation), and shaping provably doesn't move the optimum if each term is written as a
  potential difference (recommended where feasible).
- **Advancement rule:** track, across the training batch, the fraction of envs that satisfy each
  milestone predicate. When stage k is met by ≥ `advance_frac` (e.g. 0.6) of envs, expose stage
  k+1. **Stall guard:** if a stage doesn't advance within N iters, flag it back to the LLM for a
  reward/milestone revision (Component 4).
- The controller's per-rollout output is `(milestone_level, intra_stage_progress)` — the basis
  for judging.

## Component 3 — progress-dominance demo acceptance (the key judging change)

Replace the binary gate in `_drain` with **lexicographic progress dominance**, computed from the
eval summaries of the plan rollout vs the policy rollout on the **same reset**:

```
plan beats policy  iff  (plan.milestone_level, plan.intra_progress)
                         >  (policy.milestone_level, policy.intra_progress)
```

So a plan that gets 3 fingers on the cube and lifts 0.03 m beats a policy that only reaches —
even though neither "succeeds". Accept on dominance; **weight the cloned demo by the progress
margin** (extends the existing advantage-weighted BC — `bc_weight_temp`). This is exactly "focus
on improving the policy, not on maximizing a sparse reward": demos are ranked by *relative
improvement along the curriculum*, and partial wins are first-class.

Ties into the lifecycle unchanged: scratch→residual switch still fires when the policy stops
being dominated by scratch demos; give-up still fires when residuals stop dominating. The signal
is just richer than before, so it moves on hard tasks instead of being stuck at 0.

## Component 4 — LLM authoring + reflection loop

Reuse the async worker + budget. Three new prompt roles in `llm_worker.py`:

- **decompose:** given the eval-vector API (field names, meanings, ranges) and the task goal,
  emit the milestone curriculum JSON. (Once per task, refreshable.)
- **author/tune reward:** given the milestone and eval API, emit/adjust the reward spec terms +
  weights for a stage.
- **reflect (Eureka-style):** after a short training window, feed the LLM the per-milestone
  achievement fractions, true-success rate, and reward-component statistics; ask it to revise the
  spec (re-weight, split a stalled milestone, fix a term that correlates poorly with true
  progress). This closes the loop: reward code → train → measure → revise.

The true-success metric is always tracked **separately** as the north star and is never authored
by the LLM, so a reward the LLM games is caught by flat/declining true success and rejected on
the next reflection.

## Anti-gaming / validity guardrails

- Eval metrics come from ground-truth sim state, not policy output.
- Potential-based / cumulative shaping so shaping can't invert the true objective.
- Every authored reward is validated before adoption: finite & correct shape on smoke rollouts,
  and a **correlation check** — optimizing it must raise milestone achievement / true success,
  else it's reverted (reward reflection).
- Tier-B free-form code runs only in the AST-allowlisted sandbox.
- Curriculum monotonic: a later stage's bonus is gated on the earlier predicate, so the agent
  can't skip the hard part by farming an early term.

## Files

- `eval_metrics.py` — structured eval vector from `mjx.Data` (new).
- `reward_spec.py` — spec schema + safe compiler to a `jp` reward fn + validators (new).
- `curriculum.py` — cumulative potential combiner + advancement/stall controller (new).
- `mjx_env.py` — `_reward` consumes a compiled reward fn; metrics carry the eval vector.
- `llm_supervision.py` — `_drain` uses progress-dominance; margin = progress margin.
- `llm_worker.py` — decompose / author-reward / reflect prompt builders + parsers.
- `train.py` — flags: `--reward {builtin,llm_spec,llm_code}`, `--curriculum`, advance/stall params.

## Phasing (each phase independently useful)

1. **Eval vector + hand-written curriculum + progress-dominance acceptance.** No LLM codegen
   yet; hand-author the Shadow grasp milestones. Immediately unblocks "nothing accepted on
   Shadow" and gives PPO dense staged reward — testable on its own.
2. **LLM decomposes the curriculum** (milestone JSON) from the eval API; compare to hand-written.
3. **LLM authors/tunes the reward spec** (Tier A) per stage, with the reflection loop.
4. **Tier-B sandboxed free-form reward code**, gated and validated.

## How this answers the two asks

- *Judge helpful-but-incomplete LLM policies:* progress-dominance acceptance (Component 3) over
  the structured eval vector — partial credit by milestone + intra-stage progress.
- *Improve the policy rather than chase sparse reward:* demos are ranked by **relative
  improvement along the curriculum**, and the curriculum (Components 1–2) converts a sparse
  terminal task into dense staged feedback for both the LLM's judgment and PPO's gradient — the
  "1→2→3→all fingers" decomposition the user described, authored by the LLM.

## Implementation notes (2026-06-15)

What shipped, mapped to the plan:

- **Eval vector** (`eval_metrics.py`): 6 fields — `palm_obj_dist`, `min_finger_dist`,
  `n_contacts`, `closure`, `lift`, `obj_xy_disp` — each with an episode reduction (min/max).
  Computed in `MjxEnv._eval` from `mjx.Data` (fingertip bodies via `EnvConfig.fingertip_bodies`;
  contact = within `contact_eps`). Carried per-step in `EnvState.metrics["eval"]`.
- **Reward spec** (`reward_spec.py`): Tier-A operator library (`const/linear/neg_dist/proximity/
  clip/` indicator) + JSON validator + `compile_spec` -> jit `reward_fn(eval, stage)`. Tier-B
  `compile_expr` compiles a single AST-allowlisted arithmetic expression (blocks imports/calls
  outside `clip/exp/abs/min/max/where`). Default hand-written gripper + shadow curricula.
- **Curriculum** (`curriculum.py`): cumulative gated shaping (env unlocks milestones 0..stage+1,
  each gated by the previous predicate holding that step — `stage` is a traced arg, no recompile
  on advance); `CurriculumController` advances when ≥`advance_frac` of the batch clears the stage
  (with patience) and flags stalls; `progress_score` = gated milestone level + fraction toward
  the next, for judging.
- **Progress-dominance acceptance** (`llm_supervision._drain`): with a spec attached, a demo is
  accepted iff its `progress_score` beats the policy's on the same reset (margin weights the
  advantage-weighted BC). Without a spec, the old binary success+return gate is used unchanged.
- **LLM authoring + reflection** (`llm_worker.build_reward_prompt`/`parse_reward_spec`,
  `llm_supervision.author_reward_spec`): one validated LLM call emits/revises the spec;
  `train.py --reward llm` authors at startup, `--reward-reflect N` revises every N iters from
  per-milestone achievement + true success (recompiles the env reward). True success is tracked
  separately and never authored by the LLM (anti-gaming north star).

Verified: spec compile/validate/sandbox; progress_score orders a 3-finger partial lift above a
mere approach; curriculum advances on the gripper (`reach`->`grip` as the batch clears it);
progress-dominance accepts helpful demos via a mock LLM; LLM authoring parses+validates+compiles
via a mock backend; gripper builtin path + the scratch->residual->off lifecycle unchanged.

**The key result** — a pure-RL Shadow run with `--reward default --curriculum` (N=64, 25 iters,
no LLM) shows the dense partial-credit signal the binary metric hides: the `reach`-milestone
clear-fraction climbs 0.05 -> 0.52 and mean gated-milestone level 0.05 -> 0.55 while true success
is still only ~0.08. So "got closer / got a finger on / lifted a little" is now measured,
rewarded, and usable to rank LLM demos — exactly the goal. (The Shadow grasp itself is still
unsolved, so the curriculum held at stage 0 over 25 iters; longer training / grasp tuning would
let it advance.)

Not live-tested (opt-in, costs real LLM calls + a Shadow recompile): the `--reward-reflect`
in-loop revision and `--reward llm` against a real backend — both wired and unit-tested with mock
backends.

## Open questions to settle before building

- Contact counting: distance-threshold proxy (jit-trivial) vs true MJX `data.contact` (more
  faithful, trickier under vmap/jit). Start with the proxy.
- Potential-based vs raw shaped terms: pure potential-based is safest but constrains the term
  library; allow raw terms with the correlation guard as the safety net.
- Curriculum representation in the jitted env: pass `current_stage` as a traced scalar and mask
  milestone bonuses, vs recompile per stage. Masking avoids recompiles (preferred on the 2070,
  where compiles are expensive).

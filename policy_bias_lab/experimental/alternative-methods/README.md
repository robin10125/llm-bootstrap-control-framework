# Alternative prior-influence methods

Five experimental ways for an LLM-authored prior program to influence PPO **without sitting in
the final policy's action path**. The baseline (`policy_bias_lab.ppo_bias`) composes
`env_action = residual + prior_scale * prior`; every method here trains a **pure neural policy**
(the deterministic evaluation runs the network alone, prior fully off), and the prior enters only
through exploration, state visitation, reward, or the critic.

All methods share the same fragmented-PPO skeleton, env, and arbiter-parity evaluation
(`task_graded_objective` on the same `eval_summary` columns) as the baseline, so
`final_report.json:eval_graded_objective` is directly comparable across arms and to
prior-generation scores. Task knowledge enters ONLY via the prior program JSON.

## Files

- `alt_methods_ppo.py` — shared trainer: split actor/critic nets, fragment collector with
  behavior-override + train-mask, masked PPO update, deterministic eval.
- `run_alt_method.py` — CLI runner
  (`--method proposal|curriculum|value_shaping|critic_features|kl_prior`).

## Methods

### 1. `proposal` — prior as exploration proposal
Rollout steps execute, with probability `p`, a sample `clip(prior + sigma*noise)` instead of the
policy's own sample. The stored action is the **executed** one, scored under the current neural
distribution, so PPO's clipped importance ratio absorbs the off-policy data
(`--proposal-offpolicy ratio`); `--proposal-offpolicy mask` instead drops those steps from the
loss (pure state-visitation variant). `p` anneals linearly (`--proposal-prob` →
`--proposal-prob-final` over `--proposal-anneal-iters`). `--proposal-gate low_value` proposes
only in states whose critic value is below the batch median (low-confidence proxy).

### 2. `curriculum` — prior as state-visitation shaper
Each episode begins with a per-env warmup prefix driven **purely by the prior** (its staged
cursor advances normally); warmup steps are masked out of the PPO loss, so the policy learns only
from its own actions in the states the prior reached. `--warmup-mode uniform` draws each env's
warmup length U{0..max} so batches mix every stage depth; the max anneals from `--warmup-frac`
of the horizon to `--warmup-frac-final`. Because warmup is an episode prefix, masked steps never
contaminate the GAE advantages of trained steps.

### 3. `value_shaping` — prior as value/reward information
No action prior at all. Rewards added to the env reward:
- authored per-stage `success` progress + completion bonus (same machinery/flags as the baseline
  stage reward: `--stage-reward-weight`, `--stage-completion-bonus`, ...);
- a potential-based term `potential_weight * (gamma*phi(s') - phi(s))` with
  `phi(s) = mean_k sigmoid(success_k / potential_temp)` — pays for reaching states the prior's
  ladder considers accomplished, cannot change the optimal policy (potential-based);
- optional auxiliary critic head regressing `phi` (`--aux-coef 0.1`), giving the critic a dense
  prior-defined progress target without touching the reward.

### 4. `critic_features` — prior diagnostics as critic input
Actor input/output unchanged (raw obs → action): the policy interface is explicitly NOT extended.
The critic additionally receives: stage-cursor one-hot, the prior's suggested action vector,
prior action norm, prior–policy disagreement norm, and per-stage success margins
(each toggleable via `--no-critic-*` flags). The critic can value states where the prior sees
structure while the actor stays fully expressive.

### 5. `kl_prior` — prior as a KL-regularized exploration direction
The prior defines a reference distribution: a pre-tanh Gaussian centered at `atanh(prior(s))`
with width `--kl-sigma-ref` (the "instruction cone"). The PPO loss adds
`beta * KL(pi(.|s) || pi_ref(.|s))` per state, so the policy — and therefore its own exploration
sampling — is pulled toward the instruction everywhere, but diverges at exactly the states where
the reward advantage of diverging outbids the price `beta`. "Clearly harmful" needs no detector:
it is the reward signal outbidding the KL cost, per state and per actuator, continuously.
Two schedules:
- **anneal**: `beta` goes `--kl-coef` → `--kl-coef-final` linearly (default), so early training
  hugs the instruction and the asymptotic optimum is the unregularized one;
- **budget servo** (`--kl-target`, optionally → `--kl-target-final`): `beta` is adjusted
  multiplicatively each iteration so the measured mean KL tracks an annealed divergence budget —
  set a small initial target and a large final one to *grow* permitted divergence over training.
`metrics.jsonl` logs `prior_kl` (measured divergence, all methods) and `kl_coef` (current price).

## Running an experiment

From `llm-framework/` with the project venv:

```bash
PY=.venv/bin/python
RUN=policy_bias_lab/experimental/alternative-methods/run_alt_method.py
PROG=runs/agentic_v4_20260704/best_program.json   # any freeform_staged best_program.json

$PY $RUN --method proposal        --out runs/alt_proposal_$(date +%m%d)  --program $PROG \
    --proposal-prob 0.3 --proposal-sigma 0.1
$PY $RUN --method curriculum      --out runs/alt_curric_$(date +%m%d)    --program $PROG \
    --warmup-frac 0.5 --warmup-mode uniform
$PY $RUN --method value_shaping   --out runs/alt_value_$(date +%m%d)     --program $PROG \
    --potential-weight 0.5 --aux-coef 0.1
$PY $RUN --method critic_features --out runs/alt_critic_$(date +%m%d)    --program $PROG
$PY $RUN --method kl_prior        --out runs/alt_klprior_$(date +%m%d)   --program $PROG \
    --kl-coef 1.0 --kl-sigma-ref 0.3
```

Baseline arm for comparison (prior in the action path):

```bash
$PY policy_bias_lab/experimental/run_fragmented_stage_ppo.py \
    --out runs/alt_baseline_$(date +%m%d) --program $PROG
```

A no-prior control is `--method value_shaping --stage-reward-weight 0 --potential-weight 0`
(pure PPO on the env reward with the identical network/optimizer/eval).

Per-iteration diagnostics land in `metrics.jsonl` (`prior_used_frac`, `train_mask_frac`,
`prior_disagreement`, `shaping_return`, annealed knob values); the headline comparison number is
`final_report.json:eval_graded_objective`.

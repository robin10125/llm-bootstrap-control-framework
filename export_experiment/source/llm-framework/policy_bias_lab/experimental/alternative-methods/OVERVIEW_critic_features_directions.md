# Critic-features directions: prior quality, composition, annealing, and speed

Status: design overview, 2026-07-13. Companion to `alt_methods_ppo.py` (`method=critic_features`)
and the prior-generation study (`prior-generation-study/`). Five questions, each with reasoning,
a concrete suggestion, feasibility, and related work.

The mechanism under discussion: the LLM program feeds the CRITIC ONLY, as per-step features
(stage-cursor one-hot, prior action, prior norm + policy-prior disagreement, per-stage success
margins). The single causal path is: features → better value fit → more accurate/lower-variance
GAE advantages → sharper policy gradient. The actor, the reward, and therefore the optimal
policy are untouched.

---

## 1. Do bad priors have a pronounced net negative? Minimal-clean vs extensive-with-bad-parts

### Reasoning

The harm model for a bad *feature* is categorically milder than for a bad *action prior*:

- **Useless features** (uncorrelated noise, saturated margins): cost is statistical, not
  behavioral. Extra input dimensions dilute critic capacity and add estimation variance; the
  critic learns near-zero weights at roughly the cost of some extra samples. Bounded, decaying.
- **Confounded features**: correlated with return under the current policy for the wrong reason
  (e.g., a margin that tracks "arm moved a lot", which correlates with reward early but not
  late). These transiently bias the value function exactly when the policy distribution shifts —
  the critic chases a moving spurious correlate. This is the worst case, and it is still
  self-correcting: the critic re-fits, and the actor never consumed the feature directly.
- **A misleading cursor** (gates that advance spuriously): the stage one-hot then injects wrong
  *history* information. Actively wrong beats absent, but the critic can down-weight a block that
  fails to predict returns, given enough data.

So the theoretical asymmetry is: **good features have unbounded upside (they supply exactly the
progress/history signal the critic otherwise learns slowly or can't recover from obs at all),
bad features have bounded, decaying downside.** Prediction: extensive-with-some-bad-parts should
beat minimal-clean for critic features — the opposite of the action-path setting, where one bad
stage can drive the robot into a wall (or an exploit). The break-even is where added variance
from dead columns outweighs marginal information; with 20–30 features and 256-wide critics,
that point is likely far away.

One genuine caveat: value error at the *wrong moment* matters more than average value error.
A confounded feature that misleads the critic right at the grasp→lift transition costs more
than its average ΔEV suggests, because advantages there gate the hardest credit assignment.

### Suggestion

**Feature-corruption ablation**, cheap and decisive (1h runs, existing config flags do most of
the work):

- Arms (same good program, seed-matched): (a) full features; (b) minimal-clean = success
  margins only (`--critic-stage-onehot 0 --critic-prior-action 0 --critic-prior-norms 0`
  equivalent); (c) full + K appended noise columns (feature-count control); (d) full with one
  margin replaced by an *inverted* margin and one gate deliberately mis-thresholded (confound
  control). Compare graded objective and critic explained-variance trajectories.
- The generation study's `simple` vs `complex` arms already bear on this from the authoring
  side; the corruption ablation isolates the consumption side.

### Feasibility

High. (c)/(d) need ~30 lines (a `--corrupt-features` debug flag in `critic_feats`); (a)/(b) are
config-only. 1h × 3 seeds × 4 arms = 12h sequential on the current box.

### Similar work

- Asymmetric actor-critic with privileged critic input: Pinto et al. 2017 (*Asymmetric Actor
  Critic for Image-Based Robot Learning*); OpenAI Dactyl (2018) and AlphaStar (Vinyals et al.
  2019) both gave the value function privileged/full-state inputs the policy lacked — the
  standing observation there is that extra critic-side inputs rarely hurt and often help.
- Contrast with reward shaping, where badness is *not* bounded unless potential-based
  (Ng, Harada & Russell 1999) — critic features inherit optimum-invariance "for free".
- Deep nets' robustness to nuisance input dimensions (standard result; the risk is small-sample
  spurious correlation, i.e., the confound case, not the noise case).

---

## 2. Best-of / intersection of priors, and priors + policies

### Reasoning

Because features are additive measurements (you can't execute two action priors at once, but you
can measure with several instruments at once), ensembling is natural here in a way it isn't for
the other methods. Three grades:

1. **Union**: concatenate feature blocks of K programs into one critic. The critic performs the
   "best-of" selection implicitly by weighting whichever blocks predict return. Cheapest, and
   doubles as an attribution experiment (per-block ablation ΔEV identifies the winning program).
2. **Intersection / consensus meta-features**: features computed *across* programs —
   mean/min of corresponding success margins, pairwise agreement of suggested prior actions
   (cosine or L2 between programs' actions at the current state), variance of suggested actions.
   Agreement is an epistemic-confidence proxy: states where independently-authored priors agree
   are states where the suggestion is probably right, and "the priors disagree here" is itself
   value-relevant information (frontier/uncertain regions). This gives the critic strictly more
   than the union does only if the critic couldn't synthesize it — plausible for the
   min/agreement nonlinearities under limited data.
3. **Priors + policies**: treat *trained artifacts* as instruments too. Candidate features from
   a previous run's checkpoint: (a) the frozen old policy's mean action and the current policy's
   deviation from it (the deviation feature already exists for the LLM prior — same code path);
   (b) the frozen old *critic's* value estimate as a feature (a cross-run value bootstrap).
   This is critic-side "kickstarting": the new run inherits measurement, not behavior, so a bad
   old policy again can't corrupt the optimum.

### Suggestion

Start with **union of 2–3 generation-study programs + one consensus block** (margin-min and
action-agreement), evaluated against the best single program at 1h × 3 seeds. If union wins,
the per-block ablation regression from question 4's instrumentation says which blocks earned it.
Defer priors+policies until a strong reference checkpoint exists (the 100%-success 6h baseline
qualifies).

### Feasibility

Union: moderate — `make_success_values_fn`/`prior_step` currently take one program; accepting a
list and concatenating blocks (plus one cursor per program) is a contained change in
`alt_methods_ppo.py`. Consensus block: small addition on top. Priors+policies: easy mechanically
(checkpoint load + frozen apply inside `critic_feats`) but adds a JAX param-passing wrinkle
(frozen params must ride through the jitted collector).

### Similar work

- Value-function ensembles and disagreement-as-uncertainty (e.g., bootstrapped value ensembles,
  Osband et al. 2016) — here the ensemble is over *authored instruments*, not learned nets.
- Kickstarting (Schmitt et al. 2018) and policy reuse (Fernández & Veloso 2006) are the
  actor-side analogues of priors+policies; the critic-side variant is safer for the optimum.
- Mixture-of-experts gating: the critic's first layer is doing soft expert selection over
  feature blocks.

---

## 3. Annealing features away as the critic outgrows the prior

### Reasoning

The classic annealing rationale does not apply: unlike `kl_prior` (which biases the objective
and must anneal to preserve the optimum), features are information, and information does not
need to be removed for correctness. The critic is discarded at deployment, so there's no
deployment reason either. Remaining honest reasons to anneal:

- **Crowding-out**: while features carry the value signal, the critic under-invests in the
  obs→value pathway; if features degrade in usefulness late (margins all saturated once the
  policy is competent), the critic is left with an under-trained obs pathway plus stale inputs.
  Scheduled feature dropout forces internalization — a distillation mechanism.
- **Hot-swap readiness**: if iterative prior improvement (revision mid-run) is coming, a critic
  that never became feature-dependent absorbs a program swap with less transient value error.

The counterargument is specific and strong: the **cursor one-hot is genuine non-Markov state**
(rollout history). It is not recoverable from obs by any amount of internalization by a
feedforward critic. Annealing it away removes real information, not scaffolding. So if annealing
is tried, anneal the *recoverable* blocks (prior action, norms, margins — all functions of
current obs) and keep the cursor.

Expected effect size: small. Priority: low, behind questions 1/4/5.

### Suggestion

If tested: per-block dropout probability ramped 0→1 over the second half of training,
cursor block exempt. Success criterion is not final score (likely a wash) but robustness of a
subsequent mid-run program swap.

### Feasibility

High — a per-block dropout mask in `critic_feats` keyed off a traced knob (the `knobs` vector
already exists to avoid recompilation under annealing).

### Similar work

- Privileged-information distillation and input-dropout curricula: Learning by Cheating
  (Chen et al. 2019), sensor-dropout for policy transfer; teacher-student with privileged
  teachers (Lee et al. 2020, quadruped locomotion — teacher sees privileged state, student
  internalizes). Those anneal on the *actor* side where deployment forces it; here it's optional.

---

## 4. How rapidly do critic features shape the policy?

### Reasoning

The influence path has a built-in lag with three stages: (i) the critic must fit the features
(fast — supervised, dense signal; expect v_loss/explained-variance divergence from baseline
within the first few hundred iterations); (ii) better advantages must accumulate into policy
updates (PPO-speed); (iii) behavioral divergence shows in eval metrics (delayed, and gated by
exploration).

The sharpest hypothesis worth testing: **critic features cannot help before the policy first
encounters reward-relevant states — they improve credit assignment, not exploration.** Predicted
signature: no change in time-to-first-lift; a steeper consolidation slope (success rate climb)
after first successes; largest gains mid-training when advantage noise, not exploration, is the
binding constraint. If instead the binding constraint in the lift task is exploration, critic
features will measure well (high ΔEV) and still barely move success — which would redirect
effort toward composition with an exploration-shaping method (question 5).

### Suggestion

**No new runs needed.** The `critic_vs_baseline_3seed` and 4h-comparison runs on disk already
contain per-eval histories. A plotting script should overlay, per seed: v_loss, success rate,
graded objective vs wall-clock/iteration, and report time-to-first-nonzero-success and
post-first-success slope for both arms. Add two cheap online metrics for future runs: critic
explained variance, and per-block ablation ΔEV every `eval_every` (train a small probe critic on
the same batches with each block zeroed). These are also the feedback signals any iterative
prior-improvement loop needs, so the instrumentation pays twice.

### Feasibility

Plotting: trivial (data exists). Online ΔEV probe: moderate (a second small critic per block is
K extra forward/backward passes on collected batches; acceptable at eval cadence).

### Similar work

- Variance-reduction theory for policy-gradient baselines: Greensmith et al. 2004; GAE
  (Schulman et al. 2015) — gains scale with how much advantage-estimation noise dominates.
- Auxiliary-task RL (UNREAL, Jaderberg et al. 2016): auxiliary signals speed representation
  learning but famously don't substitute for exploration — same expected split here.

---

## 5. Composition of the techniques

### Reasoning

Critic features are the most composable method in the family precisely because they occupy a
channel (critic input) no other method uses. Pairings, in order of expected complementarity:

- **critic_features + kl_prior**: the natural pair. KL supplies what features can't —
  exploration direction early (the policy samples near the instruction, so it *reaches*
  reward-relevant states) — and features supply what KL can't — accurate credit assignment,
  persisting after beta anneals to zero. Same program serves both roles. Predicted
  super-additive on tasks where the baseline stalls before first reward. Risk: while beta is
  high the on-policy distribution hugs the prior, so the critic initially fits values *of
  near-prior behavior* — the feature weights it learns there may need re-fitting as beta
  decays; expect a mid-training EV dip rather than a failure.
- **critic_features + curriculum/proposal**: prior-driven state visitation plus an informed
  critic. Compatible (curriculum masks warmup steps from the loss but the critic still values
  post-warmup states where margins are mid-range and informative). Weaker theoretical synergy
  than KL because visitation shaping already anneals away.
- **critic_features + value_shaping**: the same margins consumed twice — as potential-based
  shaping (reward) and as features (critic input). The potential-based term preserves the
  optimum, so safety composes; the non-potential `stage_reward` term does not, and should stay
  off in compositions. Redundancy risk: shaping already injects the margin signal into returns,
  which the critic then sees from both sides; expect sub-additive.
- **Iteration loop on top** (the separate iterative-improvement thread): composition matters
  here because hot-swapping the program mid-run is uniquely safe for critic_features but *not*
  for the kl arm of a combo (swapping the KL reference mid-run changes the objective). In a
  combo, iterate the feature program freely; freeze or only-anneal the KL program.

### Suggestion

A **2×2 factorial**: {kl_prior on/off} × {critic_features on/off}, one shared program, 3 seeds,
1–4h. This is the single most informative composition experiment: it measures each main effect
and the interaction with the same budget a two-arm test would need per pairing. Report
time-to-first-lift (expect KL owns this) and post-first-success slope (expect features own
this) separately, not just final success.

### Feasibility

Moderate. `METHODS` is currently a single string and `critic_feats`/the KL loss term are both
already gated independently (`cfg.method == "critic_features"` in `critic_feats`, `kl_coef` in
`loss_fn`), so the refactor is mechanical: replace the method string with per-mechanism flags
(or add a `combo` method that sets both). The loss already computes the KL term for all methods
and multiplies by a coefficient that is zero for non-KL methods — the plumbing is nearly there.

### Similar work

- Rainbow (Hessel et al. 2017) as the methodology template: factorial/ablative composition of
  individually-validated components, reporting interactions.
- Scheduled auxiliary control (SAC-X, Riedmiller et al. 2018) composes auxiliary rewards with
  main-task learning; the kl+features combo is the same philosophy split across channels
  (exploration via objective, credit via critic input).

---

## Priority order (cost-weighted)

1. **Q4 instrumentation + retrospective plots** — zero new training; produces the feedback
   signals every other question (and the iteration loop) needs.
2. **Q1 corruption ablation** — cheap, settles the minimal-vs-extensive authoring question and
   directly informs the generation-study prompt emphasis (simple vs complex).
3. **Q5 kl × features factorial** — the highest expected-value new capability; mechanical
   refactor.
4. **Q2 union + consensus features** — moderate code, natural follow-on once per-block ΔEV
   attribution exists.
5. **Q3 annealing** — low expected effect; only worth running as a rider on hot-swap experiments.

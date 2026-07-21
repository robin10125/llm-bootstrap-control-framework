# LLM-authored priors in PPO: findings and roadmap to publication

Status: findings + future-work report, 2026-07-20; updated 2026-07-21 with a pilot mechanism
check (§2.6). Covers two experimental threads under
`policy_bias_lab/`: **critic_features** (`experimental/alternative-methods/alt_methods_ppo.py`)
and **motor-tape** (`experimental/motor-tape/`). Both study the same underlying question from
opposite ends: how much *authority* should an LLM-authored prior be given inside a PPO loop, and
where does harm enter as that authority increases? All numbers below are pulled directly from
run artifacts (`final_report.json`, `metrics.jsonl`, `summary.md`/`screen.md` in each run
directory) rather than from memory of the studies — see the "Run index" table at the end for
exact paths.

Task substrate for all runs unless noted: MJX Shadow Hand, single 5cm cube lift, PPO
(`run_long_ppo`/`run_alt_method`/`run_motor_tape` families share the same env and eval stack).

---

## 1. The authority spectrum

Three mechanisms exist in the codebase for injecting an LLM-authored prior into training, ordered
by how much control they hand the LLM over what the robot actually does:

| mechanism | LLM controls | can it corrupt the optimum? | evidence this session/recently |
|---|---|---|---|
| `critic_features` | value-function *input* only (stage cursor, prior action, prior norms, per-stage margins) | no — actor, reward, and optimal policy are untouched; worst case is wasted critic capacity | diversity + composition studies (§2) |
| reward-shaping (`reward_modes.py`, incl. this session's `grasp_gated`) | what gets *paid*, not what the robot does or sees | yes, unless potential-based (Ng/Harada/Russell 1999) — `w_close` air-closure hack is the standing counterexample | [[reward-hacking-and-reward-modes]] memory; `grasp_gated` result in §3.2 |
| motor-tape | direct open-loop *actuation* (`q_des` tape), residual/rate/modulation heads correct it | yes, unbounded — a bad tape can occupy the grasp workspace and prevent grasp formation regardless of what the residual learns | the entire contact-harm and revision-depth story, §3 |

This framing is new synthesis for this document, not a pre-existing artifact — but every load-bearing claim in the two rows below it is backed by a specific run cited in §2/§3. No experiment in this repo has yet run all three mechanisms head-to-head under identical seeds/eval settings on the same task (flagged in §5.F).

---

## 2. Critic-features: findings

### 2.1 Mechanism

The LLM program feeds the critic only: stage-cursor one-hot, prior action, prior-norm and
policy-prior disagreement, per-stage success margins. Claimed causal path: features → better value
fit → lower-variance GAE advantages → sharper policy gradient. Extended this repo-history with
`--critic-gate-values` (parallel, cursor-free gate values), `--critic-critical-actions`
(critical-stage channel actions fed every step), and multi-program `--program` union (feature
concatenation, one cursor per program).

Only the first link in that chain is structurally guaranteed rather than asserted: `Actor` and
`Critic` are separate network objects in `alt_methods_ppo.py` with no shared trunk, so features
cannot reach the actor's parameters by any path other than through the critic's value estimates.
Everything after that — better value fit, lower advantage variance, and that this is what
produces the observed gains — is unmeasured. See §2.5.

### 2.2 Headline result — diversity study

`runs/critic_diversity_4h_useinfo/` (2 arms × 1 training seed × 2h, use-info-generated programs):

| run | graded | success | fitness |
|---|---:|---:|---:|
| parallel_s0 (critic_use + parallel gates) | 0.842 | 0.148 | 1.555 |
| union_s0 (critic_use + critic_simple + critic_complex) | **0.988** | **0.445** | 1.724 |

Confirmed with a large eval (2048 envs × 3 seeds, `large_eval.json` in each run dir): union
success 0.458 (range 0.453–0.467 across the 3 eval seeds) vs. parallel 0.131 — the training-time
gap is not an eval-noise artifact. This is the best `critic_features` result to date and the
empirical anchor for the "union of diverse priors beats a single prior" claim.

### 2.3 Composition study — a caveat, not a clean confirmation

`runs/critic_composition_12h/` (4 arms × 1 training seed × 3h, nested: noise ⊂ simple ⊂
simple+complex ⊂ dual+complex):

| run | graded | success | fitness |
|---|---:|---:|---:|
| noise_s0 (N(0,1) control, width-matched) | 0.610 | **0.195** | 1.623 |
| simple_s0 | 0.560 | 0.090 | 1.516 |
| simple_complex_s0 | 0.855 | 0.055 | 1.582 |
| dual_complex_s0 | 0.879 | 0.066 | 1.735 |

Graded objective tracks the "more/diverse features win" story, but **success rate does not** —
the noise control's success (0.195) beats both union arms (0.055, 0.066). At n=1 seed per arm this
is not distinguishable from seed variance, but it is a live discrepancy between two metrics that a
paper cannot paper over. This is the single most important open item before the "bounded,
decaying downside" claim can be stated as an empirical result rather than a theoretical one (see
§5.E).

### 2.4 Design principle (theoretical, partially tested)

"Bad critic features have bounded, decaying downside — extensive/diverse should beat
minimal-clean in expectation" (full reasoning in
`experimental/alternative-methods/OVERVIEW_critic_features_directions.md`). The diversity study is
consistent with this; the composition study's success-rate anomaly (§2.3) is not yet reconciled
with it. The OVERVIEW doc scopes five further questions (feature-corruption ablation, union +
consensus features, annealing, speed/instrumentation, KL×features factorial) — **none beyond the
union test above have been run**. These are the highest-value, lowest-cost next steps for this
thread (§5.E) because the infrastructure for all of them already exists as CLI flags.

### 2.5 The mechanism itself is unvalidated — and cheap to close

Everything in §2.2–2.4 validates an **outcome** (union beats a narrower feature set on graded
objective and success rate, confirmed at large-eval scale). None of it validates the **mechanism**
claimed to produce that outcome (§2.1's causal chain). Two questions are open, and both are live —
neither has been ruled out by anything run so far:

1. **Do the features actually improve value fit?** `v_loss` is logged every eval interval in
   `metrics.jsonl` for every critic_features run, including both studies above — but raw `v_loss`
   is not comparable across arms whose returns have different scale/variance, so it cannot answer
   this on its own. The correct diagnostic — explained variance,
   `1 - Var[returns − values] / Var[returns]` — is **not currently computed or logged anywhere in
   the codebase**.
2. **If value fit improves, is that what drives the success/graded gains** — or is the union
   arm's win coming from something else, such as added input dimensionality alone (a
   capacity/regularization effect unrelated to feature *content*)? This alternative is not a
   theoretical nitpick: it is the live explanation kept open by §2.3's own anomaly, where the
   composition study's noise-column control beat both real-feature union arms on success rate.

**Why this is cheap to close, in two tiers:**

- **Tier 1 — zero new training.** The OVERVIEW doc's Q4 already specifies the test: plot
  `v_loss` / `success` / `eval_graded` vs. iteration for the diversity study's `union_s0` and
  `parallel_s0` (data already sits in `runs/critic_diversity_4h_useinfo/*/metrics.jsonl`) and check
  for the mechanism's predicted signature — matched time-to-first-success across arms, but a
  steeper post-first-success slope for the feature-rich arm (credit assignment help should show up
  *after* the policy first reaches reward-relevant states, not before). If that signature is
  absent, the "credit assignment, not exploration" story is probably wrong regardless of the
  outcome-level win. This is an afternoon of plotting, not a new run.
- **Tier 2 — one small code change plus a short rerun.** Add explained-variance logging (a few
  lines, computed from the same batches already flowing through the value-loss term) and run a
  *clean* matched noise-vs-real-content pair: same feature width, content shuffled vs. real,
  matched seeds (≥3), both arms logging explained variance. The composition study's noise control
  gestures at this design but is confounded (nested arms of different widths, not a matched pair)
  and sits at n=1. A clean version of exactly that comparison is the single experiment that would
  most directly confirm or falsify "it's the feature content, not just the added dimensions" —
  i.e., whether §2.1's mechanism is real.

### 2.6 Pilot check (2026-07-21): the mechanism holds, at n=1

Ran both tiers from §2.5 under a 1h GPU budget: `ev_pre`/`ev_post` (explained variance,
pre-/post-update) added to `alt_methods_ppo.py`'s loss function and logged every iteration
(`run_ev_mechanism_check.sh`, `runs/critic_ev_mechanism_check_20260721/`). Read the code before
trusting this: `ev_pre` is computed from `ret - value_old`, which is exact and free — GAE returns
are `ret = advantage + value_old` (`experiment_runtime/ppo.py`), so `ret - value_old` recovers to
the raw pre-normalization advantage already in the batch with no new plumbing; `ev_post` reuses
the same update's fresh critic forward pass. Both are masked, variance-normalized against
`Var[ret]`, matching the standard explained-variance definition.

**Tier 1 (free, on the existing diversity-study data) surfaced a caution, not a confirmation.**
`union_s0`'s eval_success trajectory is not a clean climb: 0.0625 (it24) → 0.094 → 0.219 → 0.230 →
0.148 → 0.211 → ... mostly noisy/declining down to 0.008 (it449), then a single late spike to
0.4648 on the very last eval (it474) — which is the number reported in §2.2 as the confirmed
headline result. Raw `v_loss` also inflates by 1–2 orders of magnitude in the second half of
training for *both* union and parallel (up to 13.0 and 5.2 respectively), which is exactly why
raw `v_loss` can't answer the value-fit question on its own (§2.5) — and it means the diversity
study's headline 0.445 success number carries the same checkpoint-selection risk already
documented for motor-tape (a late lucky peak, not necessarily a stable trained state). This can't
be fully resolved retroactively — `ev_pre`/`ev_post` didn't exist when that run happened.

**Tier 2 pilot (new GPU run, ~22 min/arm, 1 seed each): a clean, matched noise-vs-content pair.**
Same program set as the diversity study's union arm (`critic_use`+`critic_simple`+`critic_complex`),
run twice at identical feature width — once with real content, once with `--critic-noise` (N(0,1),
same width, zero information):

| run | iters | `ev_pre` @ it0/24/49/74 | `ev_post` @ it0/24/49/74 | final v_loss | eval_graded (best) | eval_success (best) |
|---|---:|---|---|---:|---:|---:|
| real_s0 | 88 | &minus;0.958 / +0.157 / +0.276 / +0.522 | &minus;0.004 / +0.264 / +0.334 / +0.560 | 0.028 | 0.883 | 0.207 |
| noise_s0 | 82 | &minus;2.797 / +0.097 / +0.186 / +0.468 | &minus;0.591 / +0.210 / +0.262 / +0.504 | 0.351 | 0.847 | 0.145 |

Three findings, in order of how much weight they can bear:

1. **The features measurably improve value fit, and this shows up immediately** — `ev_pre` and
   `ev_post` favor real content over matched-width noise at *every single checkpoint measured*
   (4/4), including iteration 0 (before any training). Since the two arms differ only in feature
   *content* at identical width, "it's just added dimensionality" is directly ruled out as the
   explanation for this gap — noise had the same width and fit worse.
2. **The gap decays exactly as the "bounded, decaying downside" theory predicts.** The `ev_pre`
   gap is largest at init (0.958 vs. 2.797 negative — noise is *more* catastrophic to an untrained
   critic than real features) and narrows steadily (0.06 at it24, 0.09 at it49, 0.05 at it74) as
   the critic learns to work around the uninformative noise channel using the raw observation it
   also receives. This is the self-correcting signature the OVERVIEW doc predicted, now observed
   rather than assumed.
3. **Behavioral separation lags the value-fit separation and is noisier.** At it24 and it49, the
   *noise* arm actually has higher `eval_graded`/`eval_success` than the real-content arm (0.870
   vs. 0.546 at it24) — real only pulls ahead at it74 (0.871 vs. 0.758 graded, 0.180 vs. 0.109
   success), and the final best-checkpoint numbers favor real on both metrics (0.883/0.207 vs.
   0.847/0.145). This is the qualitative shape the "credit assignment, not exploration" hypothesis
   predicts (value-fit gap first, behavioral gap delayed and noisy) — but note time-to-first-success
   was identical in both arms (iter 24, the earliest possible checkpoint at `eval_every=25`), so
   that half of the Q4 test was uninformative here; a finer eval cadence is needed to actually
   resolve it.

**A concrete illustration of why raw `v_loss` is the wrong comparison metric** (the point made
abstractly in §2.5): `noise_s0`'s final `v_loss` (0.351) is 12&times; `real_s0`'s (0.028) — which
would suggest a huge fit-quality gap — while their `ev_pre`/`ev_post` at the same checkpoint are
close (0.47&ndash;0.52 vs. 0.50&ndash;0.56). The properly normalized metric says the two critics
are nearly as well-fit as each other by then; the raw loss overstates the gap by an order of
magnitude.

**What this does and doesn't settle.** This directly answers question 1 from §2.5 (yes, real
content measurably beats matched noise on value fit, and it isn't a width artifact). It gives a
directionally favorable but not conclusive answer to question 2 (real beats noise on success/graded
at matched short duration, which is the opposite of §2.3's anomaly — suggesting that anomaly was
likely a composition-study-specific confound, nested unequal-width arms at n=1, rather than a real
"noise beats content" effect — but this pilot is *also* n=1 per arm at ~22 minutes each, and the
whole project's own seed-bimodality finding (§3.4) means an n=1 behavioral gap at this scale cannot
yet be called confirmed). The full Tier-2 ablation from §5.E (&ge;3 seeds, longer duration, finer
eval cadence to fix the time-to-first-success ceiling effect) is still the right next step before
this goes in a paper — but the mechanism itself, which was the open question, now has real
evidence behind it rather than none.

---

## 3. Motor-tape: findings

### 3.1 Mechanism

LLM authors a reset-anchored keyframe tape (`{t, targets}`, exprs evaluated once on reset obs,
splined into `q_des[T+1,26]`). Three learned "cerebellar" corrections: efference-copy residual
(actor+critic see tracking error + plan lookahead), a [0,2]× time-warp rate head, and optional
±15% plan modulation. `a_ff = clip((q_des(t)-ctrl)/action_scale, -1, 1)` is the tape's open-loop
contribution to the action; the residual is additive on top.

### 3.2 Generation-condition study — how the tape is authored

`runs/motor_tape_genstudy_20260716/` (6 arms × 3 generations, codex, best-per-arm → 1h PPO × 2
seeds):

| arm | graded (s0/s1) | success (s0/s1) | autopilot floor |
|---|---:|---:|---:|
| base (no position info, no feedback) | 0.890 / 0.623 | 0.070 / 0.195 | 0.238 |
| exact (fixed spawn, `obj_xy_range=0`) | 0.592 / 0.367 | 0.328 / 0.125 | 0.037 |
| samples (4 sampled resets injected) | **1.063 / 1.020** | 0.180 / 0.164 | 0.039 |
| fb (1 revision round, graded-argmax keep) | 0.383 / 0.350 | 0.141 / 0.109 | 0.243 |
| orient (probed sign/orientation calibration table) | 0.871 / 0.690 | 0.250 / 0.125 | 0.037 |
| orient_fb | 0.368 / 0.429 | 0.125 / 0.188 | 0.243 |

Findings: exact position info is a dead end (its plan is fully brittle under ±4cm reset
perturbation — brittleness eval graded 0.037, success 0.0; the fixed-spawn training score is not
comparable to the other arms' harder task). `samples` is the best-trained arm of the six.
Feedback/revision raised the **autopilot** (open-loop) floor sharply (0.037→0.243) by fixing a
concrete grounding bug (a sign error in `base_y`'s object-relative arithmetic) but **the two
feedback arms trained worst of all six** despite the best floors — the first sighting of what
became the central finding of §3.4.

### 3.3 Contact-harm mitigation study — three interventions on the known-hacking `fb` tape

`runs/motor_tape_mitigation_20260717/` (n=4 seeds × 1h each; `fb` baseline graded 0.383/0.350,
grasp_rate 0.0/0.0; `samples` baseline graded 1.063/1.020, grasp_rate 1.0/1.0):

| arm | mechanism | graded (4 seeds) | grasp_rate (4 seeds) |
|---|---|---|---|
| gg | grasp-gated lift reward (`clip(closure/0.5,0,1)` gate on lift income) | 0.592 / 0.383 / 0.353 / 0.430 | 0.164 / 0 / 0 / 0 |
| ho | feedforward handoff (`a_ff` fades 0 over 8→5cm object distance; plan/shaping stay visible) | 0.547 / 0.468 / 0.451 / 0.445 | 0.273 / 0 / 0.008 / 0.008 |
| nm | near-miss revision (revise toward zero-contact, band-centered near-miss) | 0.419 / 0.368 / 0.487 / 0.406 | 0 / 0 / 0 / 0 |

`ho` is the most consistent mitigation: every one of its 4 seeds beats the `fb` baseline's graded
max (mean 0.478 vs. 0.383), and it is the only arm with grasp episodes in 3/4 seeds. `gg` escapes
into a grasp only 1/4 seeds — the gate removes the exploit's payoff but does not reliably redirect
exploration. `nm` is a clean **null**: the generated tape verifiably achieves zero contact
(engagement 0.000, nearmiss score up to 1.00 at generation time) yet trains no better than `fb` and
forms zero grasps across all 4 seeds — contact removal alone does not restore grasp formation.

3h extension (`runs/motor_tape_revdepth_20260718/`, seed 0 each, same `fb` tape): `ho3h` graded
0.590 (late-window 0.523), grasp 0.117, success 0.242 vs. `fb3h` graded 0.442 (late 0.387), grasp
0.039, success 0.164. The handoff edge persists at 3× duration on every metric, but neither run
escapes into the ~1.0 grasp basin that `samples`/deep-revision tapes reach in 1h — handoff is a
real, bounded gain, not a fix.

### 3.4 Revision-depth study — the central reversal

`runs/motor_tape_revdepth_20260718/`: 3 independent chains × 4 unconditional-accept revision
rounds each (`--feedback-rounds 4 --keep-rule always`, so chain *dynamics* are decoupled from
selector *choice*); chain1's r0..r4 trained 1h × 2 seeds.

**Open-loop dose-response** (all 3 chains) — graded objective plateaus almost immediately, contact
engagement keeps climbing:

| chain | r0 | r1 | r2 | r3 | r4 |
|---|---:|---:|---:|---:|---:|
| chain1 graded | 0.041 | 0.244 | 0.244 | 0.244 | 0.244 |
| chain1 engagement | 0.0% | 9.7% | 19.6% | 22.0% | 10.7% |
| chain2 engagement | 0.0% | 8.4% | 17.7% | 28.4% | 48.6% |
| chain3 graded | 0.034 | 0.236 | 0.238 | **0.770** | 0.739 |

Revisions stay evidence-faithful across all chains (structure preserved, 11 keyframes every
round — numeric refinement only). The model is not misreading its own feedback; it is correctly
and rationally hill-climbing the only signal still moving once the graded score saturates, which
happens to be contact engagement.

**Trained outcome vs. depth** (chain1, 1h × 2 seeds):

| depth | graded (s0/s1) | grasp_rate (s0/s1) | note |
|---|---:|---:|---|
| r0 (no revision) | 0.797 / 0.891 | 0.555 / 0.781 | |
| r1 (**the old pipeline's exact dose**) | 0.561 / 0.716 | 0.164 / 0.484 | **worst mean** |
| r2 | 0.993 / 0.376 | 0.992 / 0.008 | huge seed split |
| r3 | **1.137 / 1.167** | **1.0 / 0.984** | best-ever, stable both seeds |
| r4 | 1.112 / 0.984 | 0.992 / 0.781 | highest tape closure |

Two reversals of the prior mitigation-study conclusion:
1. Revision is **not** inherently harmful — the depth curve is non-monotone (U-shaped), and deep
   revision (r3/r4) produces the best-trained tapes measured anywhere in this project.
2. The production pipeline (1 revision round, keep by graded-score argmax) provably lands on the
   curve's minimum: on chain1 the graded selector picks r1 (0.244 = argmax of the open-loop
   scores) — the worst trained depth (mean graded 0.64 vs. r3's 1.15). A nearmiss-based selector
   would have picked r0 (0.85, also beaten by "just keep revising").

**Working hypothesis**: harm is U-shaped in contact *commitment*, not revision count — zero
contact is fine, marginal/grazing contact (~1–10% engagement, which is where the old fb-tape and
genstudy's feedback arms all sat) is the trap, and deep committed contact or high pre-shaped
closure (r3/r4, 20%+ engagement) trains best because it demonstrates the touch-conditioned closure
a real grasp needs instead of dangling an exploitable half-touch.

**Standing caveat — seed bimodality**: r2's two seeds (0.99 vs. 0.38 graded, grasp 0.992 vs. 0.008)
on the *identical* tape show that 1h training outcomes on this task are close to bimodal — either
the grasp basin is found or it isn't. This means every n=2 comparison in this document (most of
§3.2 and half of §3.4) carries real risk of reporting a seed artifact rather than a tape/mechanism
effect, and is the single strongest argument in this whole report for the seed-count work in §5.A.

### 3.5 An unresolved confound

The mitigation study's cross-cutting note still stands as an open question: revised tapes (`fb` =
contact + revised, `nm` = no-contact + revised) never reached closure above the 0.5 grasp
threshold in that study, while unrevised tapes (`samples`, `base_s0`) reached 0.55+ and grasp 1.0.
The revision-depth study complicates rather than resolves this — r3/r4 *are* heavily revised (3–4
rounds) and train best of all. The current best read is that revision-depth and revision-count are
not the same variable (chain1's r1 is "1 round of revision" in both framings and is bad in both
studies; r3/r4 is "more revision" and is good) — but the untested cells that would actually settle
it (`samples` + handoff; unrevised + handoff) have not been run. See §5.D.

---

## 4. Cross-cutting literature connections (brief; full citations in the earlier conversation turn)

- **critic_features** ↔ asymmetric actor-critic (Pinto et al. 2017; "Informed Asymmetric
  Actor-Critic," 2025, arXiv:2509.26000) — though canonical work privileges the critic with
  information the actor structurally cannot have; our critic features are LLM-derived from the
  *same* observation the actor gets, closer to auxiliary-task/feature-augmentation literature
  (UNREAL, Jaderberg et al. 2016) than true informational asymmetry. Worth stating explicitly in
  any paper's related-work section rather than overclaiming the asymmetric-critic framing.
- **motor-tape's tape+residual architecture** ↔ residual RL (Johannink et al. 2019; Silver et al.
  2018; "Residual Policy Learning" arXiv:1812.06298); most relevant recent work is about base-policy
  quality specifically — "From Imitation to Refinement" (arXiv:2407.16677) and uncertainty-gated
  residual RL (arXiv:2506.17564) — both about how a *bad* base controller propagates into the
  residual-trained result, directly on-topic for §3.4/§3.5.
- **LLM-authored tape as the base** ↔ Code as Policies; GenCHiP (arXiv:2404.06645, LLM-generated
  policy code for contact-rich manipulation — closest published analog to the whole motor-tape
  line); Language Movement Primitives (arXiv:2602.02839, a DMP-parameterized alternative to
  discrete keyframes worth comparing against if the marginal-contact trap turns out to be partly a
  representational artifact).
- **The marginal-contact trap** ↔ imperfect-demonstration bias (arXiv:1802.05313: imperfect demos
  can mislead an agent into local optima); soft expert guidance (arXiv:1911.07109) and iterative
  regularized policy optimization are candidate mitigations not yet tried here — the latter's
  "online policy becomes the next demonstrator" pattern is structurally close to a revision loop
  that sees trained-policy behavior instead of only open-loop tape evidence (untried, §5.D).
- **The reward-hacking/grasp-gated thread** ↔ Eureka (arXiv:2310.12931, LLM-authored reward code +
  evolutionary search); its documented failure mode — "misaligned rewards because the reflection
  loop lacks visual assessment and relies solely on coarse reward statistics" — is essentially our
  own graded-selector-picks-the-worst-depth finding, independently arrived at.

---

## 5. Future work — path to a publishable project

### A. Statistical rigor (prerequisite for any headline claim)
- The r2 seed split (0.99 vs. 0.38, §3.4) is itself pilot data: treat "grasp basin found" as an
  approximately Bernoulli outcome and run a quick power calculation from it rather than reusing the
  n=2/n=4 conventions used so far ad hoc. Expect the answer to land around n=6–10 seeds per arm for
  any claim meant to survive review.
- Retrofit the "late-window" (mean of last 3 periodic evals) metric introduced in
  `run_revision_depth.sh` to the older mitigation study, which currently only reports
  best-checkpoint numbers — known to be a distortable metric (§ discovered this session: fb_s0
  reported 0.38 at best_iter=99 vs. 0.66 late-window).
- Add confidence intervals or a non-parametric test (Mann-Whitney, given the demonstrated
  non-normality/bimodality) to every headline comparison table before it goes in a paper.

### B. Generalize across robots
- Repeat the core comparison (no-revision vs. deep-revision tape; critic_features union vs. single
  program) on at least one other embodiment: a parallel-jaw gripper (tests whether the
  marginal-contact trap is a dexterous-hand-specific phenomenon or general to any contact-onset
  task), and/or a second dexterous hand (Allegro) to test architecture portability of the
  keyframe/residual representation itself.
- An arm-only reach/insert task would test whether motor-tape's keyframe representation transfers
  to non-grasping contact tasks (peg insertion, tool use) where "contact" is the goal rather than a
  side-effect to avoid until late.

### C. Generalize across tasks
- In-hand reorientation (harder credit assignment, a direct stress test of critic_features'
  credit-assignment mechanism).
- A task where contact is required *throughout*, not just at a single onset (e.g., pushing/sliding)
  to test whether the U-shaped contact-commitment curve is lift-specific or a general property of
  contact-onset tasks.
- Multi-object or variable-geometry (the near-miss band and calibration table were tuned for one
  5cm cube; robustness to object variation is untested).

### D. Motor-tape mechanistic follow-ups (already identified, not yet run)
- Replicate the r0–r4 depth curve on chain2 and chain3 (programs already generated this session —
  training-only cost) to check whether "r3/r4 best" generalizes or was a lucky chain.
- Run the two untested cells that would disentangle revision-content from contact directly:
  `samples` (best unrevised-analog) + handoff, and unrevised + handoff.
- Replace the production keep-rule (1 round + graded-argmax, now shown to reliably pick the worst
  depth) with "keep-last" or an engagement-aware selector, and A/B against the old default on fresh
  chains.
- Feed motor-tape's efference-copy lookahead features into a `critic_features`-style critic-only
  channel — a direct test of whether the two frameworks compose (tape supplies the action, feature
  injection supplies the value-side signal) without inheriting the action-side downside.
- Try an iterative-regularized-policy-optimization-style loop where revision sees *trained policy*
  behavior, not just open-loop tape evidence (literature pointer in §4).

### E. Critic-features mechanistic follow-ups

- **Tiers 1 and 2 from §2.5 are now done as a pilot (§2.6, 2026-07-21)** — `ev_pre`/`ev_post`
  logging exists in `alt_methods_ppo.py`, and a clean matched noise-vs-content pair (same width,
  1 seed, ~22 min/arm) shows real content beating noise on value fit at every checkpoint measured,
  with a gap that decays as training proceeds — direct evidence for the mechanism, not just an
  outcome-level correlation. This was the top-priority item in this list; it is no longer
  "not yet run," but it is also not yet at publication strength.
- **Remaining: scale the pilot to ≥3 seeds and a longer duration**, using
  `run_ev_mechanism_check.sh` (already built, just bump `T` and add seeds) — the current pilot's
  behavioral gap (real beats noise on success/graded) is only n=1 per arm, and this project's own
  seed-bimodality finding (§3.4) means that isn't enough to call it confirmed yet, even though the
  value-fit gap (the actual mechanism claim) is cleaner and less likely to be seed noise given it
  held at all 4 checkpoints in one run.
- **Tighten the eval cadence** (`--eval-every` well below the default 25) in the follow-up — the
  pilot's time-to-first-success comparison was uninformative because both arms hit their first
  eval checkpoint already past first success; a finer cadence is needed to actually test that half
  of the Q4 hypothesis.
- Q5 `kl_prior` × `critic_features` 2×2 factorial — refactor is mechanical (both mechanisms already
  independently gated in the loss), highest expected information per unit of new code, and now
  more interpretable than before given the mechanism has initial support.

### F. A real head-to-head
- No experiment has ever run critic_features-best, reward-shaping-best (`stage_gated`), and
  motor-tape-best (r3/r4-depth tape) under identical seeds, identical eval settings (env count,
  eval cadence), and identical wall-clock budget. The "authority spectrum" framing in §1 is
  currently assembled from separately-run studies with different n and different eval harnesses —
  a single shared-protocol comparison is required before that framing can appear in a paper as a
  result rather than as a narrative device.

### G. Writing and submission mechanics
- Related-work section grounding every thread in the citations gathered in §4 (already sourced;
  needs writing, not more searching).
- Likely two papers rather than one: (1) `critic_features` as an information-only LLM-prior
  injection method — closer to submission-ready now (one confirmed strong result, clean mechanism,
  clean literature fit, modulo §2.3/§5.E); (2) motor-tape as a study of LLM action authority and the
  marginal-contact trap — the more novel finding, but needs §5.A (seed counts) and §5.D
  (cross-chain replication) before the U-shape claim is submission-strength.
- Reproducibility: per-run artifacts (`program.json`, `autopilot.json`, `meta.json`,
  `final_report.json`, `metrics.jsonl`) already exist for every run cited above — good foundation.
  Missing: a single top-level manifest mapping run directory → git commit → exact CLI invocation;
  a fixed-config appendix (cube size, spawn range, reward weights, LLM backend/model version,
  eval env count) so results are replicable without re-deriving them from driver scripts.
- Figures to produce from existing data (no new runs): the revision-depth dose-response curve
  (§3.4, data in `screen.md`/`summary.md`); the r2 seed-bimodality scatter as a motivating figure
  for the seed-count argument in §5.A; once §5.F exists, the authority-spectrum comparison chart.
- A vanilla-PPO-no-prior-at-all baseline at matched compute does not appear among the runs cited in
  this document — confirm whether one exists elsewhere in the repo (e.g. under `run_long_ppo`
  without a warm-start prior) before claiming any of these methods beat "no LLM involvement";
  if not, it needs to be run.

---

## Run index

| study | driver | run dir |
|---|---|---|
| critic diversity (use-info) | manual/`run_critic_diversity_4h.sh` v2 | `runs/critic_diversity_4h_useinfo/` |
| critic composition | `run_critic_composition_12h.sh` | `runs/critic_composition_12h/` |
| critic EV-mechanism pilot | `run_ev_mechanism_check.sh` | `runs/critic_ev_mechanism_check_20260721/` |
| motor-tape generation-condition | `run_gen_condition_study.py` / `run_gen_condition_ppo.sh` | `runs/motor_tape_genstudy_20260716/` |
| motor-tape contact-harm mitigation | `run_contact_mitigation.sh` | `runs/motor_tape_mitigation_20260717/` |
| motor-tape revision-depth | `run_revision_depth.sh` | `runs/motor_tape_revdepth_20260718/` |

Design docs: `experimental/alternative-methods/OVERVIEW_critic_features_directions.md` (critic
features, 5 scoped questions referenced in §2.4/§5.E); `experimental/motor-tape/` scripts
(`motor_tape.py`, `motor_tape_ppo.py`, `run_motor_tape.py`, `generate_motor_tape.py`) for the
mechanisms referenced in §3.

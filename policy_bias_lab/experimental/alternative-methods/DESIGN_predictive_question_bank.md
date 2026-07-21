# Predictive question bank for critic-only task knowledge

Status: experimental design proposal, 2026-07-15. This is not implemented. It proposes a new
LLM-authored input representation for the `critic_features` arm while keeping the actor, reward,
and deployed policy unchanged.

## Summary

The current critic-features method consumes an executable `freeform_staged` action prior:
stage-cursor one-hot, suggested action, prior norm, policy-prior disagreement, and per-stage
success margins. That representation was built to control the robot. The critic instead needs
compact measurements that distinguish situations with different expected futures.

This proposal asks the LLM to author a **predictive question bank**, not a controller. The bank
contains:

1. LLM-authored instantaneous signals over injected raw observables.
2. Independent LLM-authored event memories.
3. Questions about future events and future signal values.
4. Optional action advice, kept separate and omitted from the first experiment.

Small auxiliary heads learn answers to the questions from rollout transitions. Frozen answers and
event memories become critic-only features:

```text
authored questions -> learned forecasts -> better value fit
                  -> lower-noise GAE -> better PPO actor updates
```

The bank does not execute actions, alter reward, or enter the deployed actor. It also cannot
directly solve exploration; it can improve credit assignment after informative states are visited.

## Why replace the staged-controller representation?

A critic-aware generation paragraph still makes the LLM emit an executable staged controller. The
model spends effort authoring actuator laws, timing, a total behavior order, and gates that must
serve both execution and value abstraction. A single incorrect cursor transition can corrupt the
whole history feature.

A critic-native artifact should instead state:

- which current measurements may distinguish valuable situations;
- which past events matter;
- what is likely to happen at short and long horizons;
- whether a current condition is likely to persist, resolve, or regress;
- which independently authored events can coexist;
- which features are dead, redundant, unpredictable, or unrelated to value.

Suggested actions and a correct stage cursor may still contain information. They become optional
comparison blocks rather than mandatory structure.

## Proposed artifact

### Instantaneous signals

`signals` are expressions over raw observables mechanically enumerated from the environment and
robot. Every derived quantity and threshold is LLM-authored per candidate. The framework supplies
no derived signal vocabulary and attaches no physical interpretation to names.

Instantaneous signals also enter the critic directly. They provide useful information while the
prediction heads are immature and give dense supervision when authored events are rare.

### Independent event memory

A memory entry is a generic hysteretic latch:

```json
{
  "name": "event_a_seen",
  "set_when": "event_a_margin>0",
  "clear_when": "reset_a_margin>0"
}
```

The runtime only performs mechanical updates. It may expose the current bit, steps since set,
active duration, and firing count. These are structural transforms of authored conditions.

Independent latches improve on one monotone cursor for critic representation because several
events may coexist, happen in different orders, clear independently, or recur. One bad event does
not force all later features into the wrong state.

### Predictive questions

A question defines a future quantity to estimate under the current PPO policy. A minimal generic
schema can support:

- `event_probability`: probability an authored event occurs within a finite horizon;
- `time_to_event`: expected steps to an authored event, clipped at the horizon;
- `future_occupancy`: expected fraction of future steps satisfying a condition;
- `future_sum`: expected accumulated authored signal;
- `future_average`: expected average authored signal;
- `future_change`: expected final-minus-current change in an authored signal.

The LLM chooses horizons in control steps using injected control rate and episode length. It may
ask the same question at several horizons to distinguish imminent from eventual prospects.

These kinds are only convenience syntax. The compiler turns them into per-transition observed
quantities, continuation masks, and finite-horizon targets. Generic prompt prose may explain the
mechanism but must never recommend task-specific events or measurements.

### Optional action advice

An independent action-advice block may later restore policy-prior disagreement. It is excluded
from the first test so the experiment isolates predictive measurement from controller advice.

## Learning the answers

The LLM generates the bank once. It is not called during rollout.

At each step, a small predictor reads the current raw observation and stored event memory and emits
one answer per question. After the environment advances, the runtime evaluates what each question
observed on that transition. Each answer is trained toward the amount just observed plus the
predicted remaining amount from the next state if the question continues.

As training proceeds:

- event occurrence propagates backward to states that commonly precede it;
- time-to-event becomes smaller in states closer to the event;
- occupancy separates transient from persistent conditions;
- future-change learns whether an authored measurement tends to improve or regress.

Use a predictor separate from actor and main critic. Stop gradients from question answers into the
critic. Question losses therefore cannot reshape the actor, and critic regression cannot turn the
answers into arbitrary shortcuts.

### Stable PPO iteration

Question features must not change during repeated PPO epochs over one batch:

1. Freeze the predictor.
2. Collect a rollout and store answers plus event memory with every transition.
3. Run PPO actor and critic updates using those stored features.
4. Train the predictor from the collected transitions.
5. Freeze the updated predictor for the next rollout.

Stored state permits shuffled minibatches without reconstructing history. Episode reset clears all
memory. Current critic features must use only current observation and stored past state. Future
observations are training targets only; current-transition prediction error is diagnostic, not a
current-state feature.

## Critic input and predictive state

The critic receives:

```text
raw observation
+ bounded instantaneous signals
+ event latches and generic timing fields
+ frozen question answers
+ optional predictor uncertainty
```

The actor remains `raw observation -> action distribution`.

This feature vector is a **predictive state**: it describes the current situation through selected
expected futures. It is smaller than a world model because it predicts only quantities authored
for this run.

## Illustrative Shadow Hand lift bank

This is deliberately task-specific LLM output for the current injected Shadow Hand lift task. It
is not generic framework content, and none of its names, expressions, thresholds, or questions may
be copied into prompts, defaults, scoring logic, or later runs.

The environment uses a 0.025-second control step, so 20, 40, 80, and 160 steps correspond to 0.5,
1, 2, and 4 seconds. This proposed syntax is not accepted by the current compiler.

```json
{
  "mode": "predictive_question_bank",
  "signals": {
    "primary_finger_contact": "max(c_index,c_middle)",
    "opposed_contact": "min(c_thumb,primary_finger_contact)",
    "total_primary_contact": "c_thumb+primary_finger_contact",
    "max_environment_contact": "max(max(env_c_thumb,env_c_index),max(max(env_c_middle,env_c_ring),max(env_c_little,env_c_palm)))",
    "horizontal_error": "sqrt(obj_rel_x*obj_rel_x+obj_rel_y*obj_rel_y)",
    "object_speed": "sqrt(obj_vel_x*obj_vel_x+obj_vel_y*obj_vel_y+obj_vel_z*obj_vel_z)",
    "relative_contact_speed": "sqrt((v_base_x-obj_vel_x)*(v_base_x-obj_vel_x)+(v_base_y+obj_vel_y)*(v_base_y+obj_vel_y)+(v_base_z+obj_vel_z)*(v_base_z+obj_vel_z))"
  },
  "memory": [
    {
      "name": "opposed_contact_seen",
      "set_when": "min(opposed_contact>0.12,relative_contact_speed<0.04)",
      "clear_when": "opposed_contact<0.04"
    },
    {
      "name": "settled_contact_seen",
      "set_when": "min(min(opposed_contact>0.15,total_primary_contact>0.45),min(object_speed<0.02,max_environment_contact<0.05))",
      "clear_when": "max(opposed_contact<0.06,object_speed>0.20)"
    },
    {
      "name": "task_event_seen",
      "set_when": "task_success_signal",
      "clear_when": "episode_reset"
    },
    {
      "name": "task_failure_seen",
      "set_when": "task_failure_signal",
      "clear_when": "episode_reset"
    }
  ],
  "questions": [
    {
      "name": "opposed_contact_within_1s",
      "kind": "event_probability",
      "event": "opposed_contact_seen",
      "horizon_steps": 40
    },
    {
      "name": "steps_until_settled_contact",
      "kind": "time_to_event",
      "event": "settled_contact_seen",
      "horizon_steps": 80
    },
    {
      "name": "opposed_contact_persistence_1s",
      "kind": "future_occupancy",
      "measurement": "opposed_contact_seen",
      "horizon_steps": 40
    },
    {
      "name": "horizontal_error_change_500ms",
      "kind": "future_change",
      "measurement": "horizontal_error",
      "horizon_steps": 20
    },
    {
      "name": "environment_contact_exposure_500ms",
      "kind": "future_average",
      "measurement": "max_environment_contact",
      "horizon_steps": 20
    },
    {
      "name": "task_event_within_4s",
      "kind": "event_probability",
      "event": "task_event_seen",
      "horizon_steps": 160
    },
    {
      "name": "task_failure_within_2s",
      "kind": "event_probability",
      "event": "task_failure_seen",
      "horizon_steps": 80
    }
  ]
}
```

`task_success_signal` and `task_failure_signal` denote injected task data used for target
construction. If the implementation supports raw observables only, the LLM must author observable
expressions instead; the framework must not invent task-specific equivalents.

After training, illustrative answers might look like:

| Observed situation | contact within 1s | steps to settled contact | contact persistence | task event within 4s | failure within 2s |
|---|---:|---:|---:|---:|---:|
| episode start | 0.07 | 80 | 0.02 | 0.00 | 0.08 |
| authored relations improving | 0.68 | 18 | 0.10 | 0.03 | 0.04 |
| first opposed contact | 0.94 | 7 | 0.42 | 0.10 | 0.13 |
| settled contact | 0.99 | 0 | 0.86 | 0.48 | 0.05 |
| task progress underway | 0.99 | 0 | 0.91 | 0.82 | 0.03 |
| contact becoming transient | 0.72 | 14 | 0.24 | 0.17 | 0.31 |

These numbers are learned, not authored. Situation labels are explanatory prose in this document
only. Runtime diagnostics must report authored names and measurements without interpretation.

The useful case is two states with similar current contact but different histories and expected
futures. A settled-contact latch, predicted persistence, expected task event, and expected failure
allow the critic to value them differently without a global stage cursor.

## Task-agnosticism boundary

The implementation must preserve the hard task-agnosticism rule:

- Generic prompts document only signal, memory, and question mechanisms.
- `$task`, robot specification, environment field names, task definitions, and LLM-authored
  context supply all task content.
- Every derived signal and threshold belongs to the per-run artifact.
- The compiler validates structure and expressions but not physical meaning.
- Memory bookkeeping refers only to authored conditions.
- Targets are mechanically constructed from authored expressions or injected task/env fields.
- Diagnostics report values, coverage, trends, and errors under authored or injected names; they
  do not classify behavior.
- Generic empirical normalization may use mean, variance, clipping rate, and quantiles, but may
  not create derived signals or tune semantic thresholds.
- No bank or discovered threshold becomes a later-run default.
- Authored evals remain tie-breakers and cannot override real objective regression.

## Diagnostics

Report structural numeric evidence:

- signal variance, range, clipping, and saturation;
- event set/clear rate, occupancy, duration, and time to first set;
- question target variance, event rate, and horizon censoring;
- train and held-out prediction error;
- event-prediction calibration;
- answer correlation and effective rank;
- critic explained variance and held-out return error;
- feature-block ablation delta;
- GAE variance overall and near authored event transitions;
- actor/critic curves and the real task objective.

Accurate prediction is not sufficient: a constant or irrelevant quantity can be predicted
perfectly. Incremental held-out value fit is a proxy; the decisive measure is the real objective
across seeds.

## Risks and mitigations

- **Sparse events:** retain continuous signals and short-horizon continuous questions; reject dead
  questions from coverage diagnostics.
- **Changing policy:** use finite horizons and one frozen predictor snapshot per PPO iteration.
- **Auxiliary interference:** normalize targets mechanically and keep predictor, actor, and critic
  encoders separate initially.
- **Critic shortcuts:** retain the raw observation path and use dropout, corruption, and
  count-matched noise controls.
- **History aliasing:** include independent authored event memory; recurrence is a later option.
- **Future leakage:** unit-test that current features depend only on current observation and stored
  past state.
- **No exploration benefit:** separately report time to first task event and post-event
  consolidation; test exploration composition only as a later factorial.

## Experimental plan

### Phase 0: compiler and causality

Verify expression vocabulary, vectorized memory reset, hand-computed finite-horizon targets,
future-invariance of current features, shuffled-minibatch storage, and unchanged actor outputs and
rewards.

### Phase 1: fixed-replay proxy

On episode-split replay data, compare equal-capacity value predictors using:

1. raw observation only;
2. current staged critic features;
3. instantaneous authored signals;
4. signals plus independent memory;
5. full question bank;
6. count-matched random/noise features.

Report held-out return error, explained variance, effective rank, and block ablations. This cheaply
rejects dead representations but does not establish an RL improvement.

### Phase 2: matched on-policy PPO

Run at least three matched PPO seeds for:

1. no-prior control;
2. current staged `critic_features`;
3. signals plus memory;
4. full question bank;
5. full bank with count-matched permuted or noise questions.

Hold actor architecture, critic parameter count, optimizer, environment seeds, rollout budget, and
evaluation cadence fixed. If generation variance matters, cross multiple bank seeds with PPO seeds.

Primary outcome: injected-task `eval_graded_objective`. Secondary outcomes: task success, critic
explained variance, held-out return error, GAE variance, time to first task event, post-event slope,
and runtime overhead.

### Phase 3: attribution

If the bank wins, remove memory, forecasts, or long horizons separately; compare raw-observable
questions with task-field questions; add optional action advice; union independent banks; and
corrupt one question to test robustness.

## Acceptance criterion and priority

Adopt only if the real measured objective improves across matched seeds and the result is not
explained by critic size, feature count, runtime, or one generated artifact. Prediction metrics may
prune candidates but cannot rescue an objective regression. Do not loosen the authored-eval
noise-band guard.

Recommended order:

1. finish replicated validation of the existing critic-feature effect;
2. add explained-variance and block-ablation instrumentation;
3. implement signals plus event memory;
4. add a small finite-horizon question set;
5. run fixed-replay and matched-seed comparisons;
6. consider ensembles, recurrence, iterative revision, or exploration composition only after a
   positive component-level result.

This experiment has higher expected information value than further prompt tuning of
`freeform_staged` for critic use because it changes the represented object rather than merely
describing the same controller to the LLM differently.

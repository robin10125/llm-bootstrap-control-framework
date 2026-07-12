# Quiet Prior Framework: Intervention-Based Redesign

## Decision

Replace the additive action-prior architecture with an **intervention scaffold**.

The learned policy normally owns every actuator. An authored prior may temporarily replace the
policy proposal on an explicitly named actuator set, only while an authored eligibility condition
holds and only under a run-local intervention budget. The framework measures whether that
intervention causally improves an authored observable outcome. Useful interventions provide
training examples; redundant interventions retire. Final candidate selection is based primarily on
policy-only task performance.

This is intentionally not a new form of residual control. There is no continuously summed
`policy + prior` action and therefore no steady-state controller conflict for an actuator.

## Goals

The framework should:

1. use authored action knowledge to cross exploration barriers;
2. transfer the resulting capability into the neural policy;
3. measure intervention use and effect by authored intervention, actuator group, and rollout time;
4. stop intervening when policy-only behavior reaches the same observable outcome;
5. reject quietness improvements that cause a real task-objective regression;
6. contain no framework-authored task interpretation or cross-run behavioral knowledge.

## Non-goals

- Making an authored program solve the task open loop.
- Treating low action magnitude as evidence that the policy did the work.
- Inferring intervention value from learned scale alone.
- Keeping a prior active at deployment merely because training used it.
- Adding fixed derived signals, phase meanings, or task heuristics to framework code or prompts.

## Core Invariants

### Policy ownership

The policy action is the environment action unless an intervention owns a particular actuator at
that step. Ownership is exclusive per actuator:

```text
env_action[j] = intervention_action[j]  if authority[j]
                policy_action[j]        otherwise
```

Two authored interventions may not own the same actuator simultaneously. Deterministic structural
priority resolves eligible interventions before execution; priority must not be inferred from task
semantics.

### Bounded intervention

Every intervention has all of:

- an authored eligibility expression;
- an authored observable outcome expression;
- a finite outcome horizon;
- a finite consecutive-step authority limit;
- a cooldown;
- an explicit actuator set;
- a per-episode and per-training-round budget supplied by generic framework configuration.

An intervention cannot become an always-on controller by omission of these fields.

### Policy-only truth

The principal result of a run is the task-defined objective measured with all intervention authority
disabled. Assisted performance is diagnostic evidence, not the final score.

### Causal usefulness

Intervention usefulness is estimated from randomized eligible opportunities:

- **treatment:** execute the intervention on its actuator set;
- **control:** keep policy ownership;
- evaluate the authored outcome over the authored horizon;
- report treatment rate, control rate, uplift, uncertainty, and sample count.

The framework reports these measurements without interpreting what the outcome means.

### Retirement, not attenuation

The unit of quieting is a whole authored intervention on a named actuator set. An intervention is
retired when policy-only controls match its treatment outcome within a configured statistical
margin for a configured number of audits. Continuous learned scales are not the primary authority
mechanism.

Retirement state is run-local and is never copied into another run.

## Authored Representation

The LLM authors task structure. The framework supplies only raw observables, actuator information,
expression mechanics, and intervention mechanics.

```json
{
  "ir_version": 2,
  "name": "candidate",
  "rationale": "...",
  "signals": {
    "authored_quantity": "<expression over injected raw observables>"
  },
  "milestones": [
    {
      "name": "milestone_0",
      "reached": "<observable expression>"
    }
  ],
  "interventions": [
    {
      "name": "intervention_0",
      "eligibility": "<observable expression>",
      "outcome": "<observable expression>",
      "outcome_horizon_steps": 40,
      "max_authority_steps": 8,
      "cooldown_steps": 20,
      "channels": [
        {
          "actuators": ["<injected actuator name or group>"],
          "expr": "<bounded proposal expression>"
        }
      ]
    }
  ],
  "probes": [],
  "evals": [],
  "unused_dofs": []
}
```

`milestones` are measurements, not controller stages. They provide an authored progress ledger for
diagnostics and frontier localization. They do not automatically activate actions.

An intervention may reference authored signals and raw observables. Its `outcome` is the local
observable test used for causal audit and teaching eligibility. It does not replace the injected
task objective.

### Structural validation

Validation checks only mechanics:

- expression compilation;
- actuator existence and overlap;
- finite positive horizons and budgets;
- bounded channel expressions;
- unique names;
- milestone dependency ordering if one is declared;
- every intervention has eligibility and outcome measurements;
- all actuators are either mentioned or explicitly listed as unused.

Validation must not judge whether an intervention, outcome, or milestone is appropriate for a task.

## Runtime State Machine

Each environment carries generic run state:

```text
retired[I]                 bool
cooldown_remaining[I]      int
authority_remaining[I]     int
episode_uses[I]            int
pending_outcome[I]         ring-buffer entries
milestone_ledger[M]         bool
```

At every step:

1. Evaluate authored signals, milestones, and intervention eligibility.
2. Update the monotone milestone ledger from measured `reached` expressions.
3. Continue an already-authoritative intervention until its authority limit, eligibility loss, or
   explicit outcome, whichever comes first.
4. For free actuators, collect eligible non-retired interventions whose cooldown and budgets allow
   use.
5. Assign treatment or control according to the audit/scheduler decision.
6. Resolve actuator ownership structurally and construct the environment action.
7. Record the opportunity, assignment probability, policy proposal, intervention proposal,
   executed action, and subsequent authored outcome.

Authority does not imply milestone advancement. Only the authored measurement changes the ledger.

## Scheduler

The scheduler separates exploration from evaluation.

### Bootstrap rounds

Early rounds allocate a conservative intervention quota to every structurally valid intervention so
the framework can obtain initial treatment evidence. This is a generic experimental allocation, not
a claim that any intervention is useful.

### Evidence-guided rounds

Later rounds allocate the limited quota using run-local measurements:

- eligible opportunity count;
- uncertainty in treatment-control uplift;
- authored frontier at which opportunities occur;
- retirement status;
- intervention consumption.

The scheduler may prioritize unresolved estimates or the earliest measured frontier with poor
policy-only conversion. It may not use task nouns or framework-authored interpretations.

### Audit rounds

Regular audit rounds disable teaching updates and randomize treatment/control at eligible
opportunities. Audit data is kept separate from training data. Final audits include fully
policy-only rollouts.

## Learning

Overridden actions were not sampled from the policy and must not be treated as ordinary PPO actions.

### Autonomous data

Steps and actuator dimensions owned by the policy use the normal PPO objective.

### Overridden dimensions

For an intervention-owned actuator dimension:

- exclude that dimension from the PPO action log-probability ratio;
- continue training the value function on the observed return;
- retain the policy's unexecuted proposal for diagnostics;
- optionally apply an auxiliary imitation loss only after the intervention example receives
  generic evidence of usefulness.

The policy distribution therefore needs per-actuator or per-structural-group log-probability terms,
not only one action-vector sum.

### Evidence-weighted teaching

An intervention trace becomes a teaching example only when its authored outcome is observed and its
run-local treatment estimate is not worse than control. A conservative initial rule is:

```text
teacher_weight = positive_advantage * positive_uplift_confidence
```

The auxiliary loss applies only to the intervention-owned dimensions and is bounded relative to the
PPO loss. Failed or unresolved interventions remain exploration data but are not imitation targets.

This avoids blindly distilling an authored mistake into the policy.

### Capability transfer

Policy-only control outcomes are the transfer measurement. As their success rate approaches the
treatment outcome rate, intervention assignment probability decreases in discrete quota steps. Once
the retirement criterion holds across repeated audits, authority becomes zero.

The framework may archive the intervention and its measurements in the run output, but it does not
delete the authored artifact or carry the retirement decision to later runs.

## Diagnostics

All diagnostic names remain structural or authored.

### Per intervention

- eligible opportunities;
- assigned treatments and controls;
- treatment probability;
- authority steps and episode-use rate;
- outcome treatment/control rates;
- uplift and confidence interval;
- time to authored outcome;
- retirement state and audit history;
- authored milestone ledger at opportunity time.

### Per actuator group

- fraction of steps owned by policy vs intervention;
- mean absolute policy proposal;
- mean absolute intervention proposal;
- proposal agreement and opposition rates;
- executed-action clipping rate;
- policy proposal vs measured joint response;
- intervention proposal vs measured joint response.

Agreement and opposition are measurements, not proof of redundancy or conflict.

### Over rollout time

Report the same ownership and proposal statistics in fixed normalized rollout bins. Fixed bins are
generic measurement structure, not authored phase meanings.

### Per milestone

- reached fraction in assisted audits;
- reached fraction in policy-only audits;
- first-reach time distribution;
- conversion to the next authored milestone if an order is declared;
- intervention attribution at first reach as a factual ownership record.

## Candidate Evaluation and Selection

For each candidate, distinguish three quantities:

1. `policy_only_objective`: task-defined objective with no intervention authority;
2. `assisted_objective`: the same objective under the current intervention scheduler;
3. `sample_efficiency`: policy-only objective/frontier as a function of environment interactions.

Selection is lexicographic and guarded:

1. Reject candidates with a real policy-only objective regression outside the measured noise band.
2. Among statistically equivalent candidates, prefer deeper policy-only authored frontier and
   higher policy-only objective area under the training curve.
3. Then prefer fewer authority steps, fewer non-retired interventions, and lower actuator ownership.
4. Authored evals remain tie-break evidence only and cannot override an objective regression.

Assisted objective never outranks policy-only objective. A high assisted score with a low policy-only
score identifies dependence, not success.

### Necessary distinctions

`assisted - policy_only` after training measures immediate dependence. It does not by itself prove
that an intervention was necessary for discovery.

Discovery value requires a matched training comparison or randomized rollout evidence. The strongest
but most expensive test is a fixed-budget seed-matched run with an intervention removed. Routine runs
use local randomized uplift; final candidates may receive the expensive removal test.

## Revision Loop

The LLM receives:

- the authored candidate;
- injected task and robot data;
- policy-only and assisted objectives;
- authored milestone measurements;
- intervention opportunity, outcome, uplift, authority, and retirement tables;
- raw body, actuator, contact, and probe measurements already supported by the framework;
- rejected revision history.

Generic revision pressure is:

- remove or narrow interventions with measured use but no positive uplift;
- modify interventions with negative uplift;
- preserve interventions with positive uplift and poor policy-only control conversion;
- improve transfer when treatment succeeds but policy-only controls remain weak;
- do not increase authority to compensate for an objective regression without evidence.

These are statistical/mechanical instructions. The LLM remains responsible for interpreting the
authored signals and task behavior.

## Safety and Constraints

An action constraint is a distinct intervention kind, not an additive prior. A future constrained
mode may project only named actuator dimensions into an authored feasible interval and report every
projection. It must use the same eligibility, outcome, budget, causal-audit, and retirement
discipline unless an external safety requirement explicitly makes it permanent.

Permanent safety enforcement should not be counted as learned-policy work and should be evaluated as
a separate system contract.

## Repository Architecture

Implement the experiment beside the current path before replacing it:

```text
policy_bias_lab/experimental/quiet-prior/
  DESIGN.md

policy_bias_lab/
  intervention_ir.py          normalization and structural validation
  intervention_runtime.py     JAX runtime state and exclusive authority
  intervention_audit.py       opportunity matching, uplift, confidence, retirement
  intervention_diagnostics.py structural reports
  intervention_ppo.py         masked PPO and evidence-weighted teaching
```

Integration points:

- `bias.py`: compile authored intervention proposals but do not expose them as additive mean shifts.
- `ppo_bias.py`: factor action log probabilities, collect ownership traces, and dispatch to the new
  trainer behind an explicit mode until validated.
- `ppo_arbiter.py`: add policy-only, assisted, and randomized audit evaluation.
- `agentic_orchestrator.py`: use the guarded selection rule and feed intervention evidence to
  revisions.
- `prior_ir.py`: keep version 1 replay intact; route IR version 2 to `intervention_ir.py`.
- prompt templates: document only generic intervention mechanics; inject all authored behavioral
  content through the existing task/spec/context substitutions.

## Migration Plan

### Increment 1: observation-only shadow runtime

- Compile IR v2.
- Evaluate eligibility and proposals without executing them.
- Verify JAX behavior, accounting, milestone ledgers, and diagnostic shapes.
- Leave current training behavior unchanged.

### Increment 2: exclusive authority and audit

- Execute bounded replacement interventions.
- Add treatment/control randomization and fully policy-only evaluation.
- Do not add imitation yet.
- Establish whether authored interventions have measurable local uplift.

### Increment 3: correct learning credit

- Factor PPO log probabilities by actuator group/dimension.
- Mask overridden dimensions from the policy-gradient ratio.
- Train values on all transitions.
- Add bounded outcome- and advantage-weighted imitation.

### Increment 4: retirement scheduler

- Add repeated policy-only capability audits.
- Reduce quotas and retire statistically redundant interventions.
- Make policy-only objective the orchestrator's primary result.

### Increment 5: LLM generation and revision

- Add task-agnostic IR v2 representation documentation.
- Generate intervention candidates directly rather than converting staged controllers.
- Revise from causal intervention evidence.

### Increment 6: comparative validation

Use identical environment budgets and seeds to compare:

- policy without authored interventions;
- current additive staged prior;
- intervention scaffold without imitation;
- intervention scaffold with evidence-weighted imitation and retirement.

Compare final policy-only objective, policy-only learning curve, authored frontier, total authority
steps, non-retired interventions, and variance across seeds.

Only after this comparison should the intervention path replace the current default.

## Required Tests

### Unit tests

- all-policy authority exactly reproduces policy actions;
- intervention ownership replaces, never sums with, selected actuator actions;
- disjoint interventions compose and overlapping ownership resolves deterministically;
- authority limits, cooldowns, and budgets are exact across fragment boundaries;
- retired interventions never execute;
- milestones advance only from authored measurements;
- treatment assignment is reproducible from seeds;
- overridden dimensions contribute zero PPO action-ratio gradient;
- policy-owned dimensions retain the existing PPO gradient;
- failed interventions receive zero imitation weight;
- objective regressions cannot win through quietness or authored evals.

### Synthetic causal tests

- positive treatment effect produces positive uplift and eventual teaching;
- null treatment effect retires after policy-only equivalence audits;
- harmful treatment is not imitated;
- low sample counts remain unresolved rather than being labeled useful or redundant;
- retirement hysteresis prevents audit noise from repeatedly toggling authority.

### Integration tests

- fragment boundaries preserve intervention runtime state;
- checkpoint/resume preserves scheduler, audit, and retirement state;
- policy-only evaluation performs no intervention computation that can affect actions;
- IR v1 archived replay remains unchanged;
- diagnostics contain only raw, structural, injected, or authored names.

## Open Experimental Choices

These should be configuration sweeps, not hard-coded conclusions:

- randomization unit: eligible opportunity or whole episode;
- confidence method and minimum audit count;
- retirement equivalence margin and hysteresis length;
- intervention quota decay schedule;
- per-actuator versus semantic-group PPO masking;
- imitation coefficient and positive-advantage estimator;
- whether expensive seed-matched removal runs are used only for finalists or every revision.

The first implementation should choose conservative defaults while reporting every value needed to
re-evaluate those choices.

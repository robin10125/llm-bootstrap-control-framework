CONTEXT GENERATION (step 2 of 2). Given the comprehensive execution account below, turn it into
an exhaustive chronological procedure with observable phase exits. Reference context only; do NOT
design priors here.

TASK: $task

$spec_block

COMPREHENSIVE EXECUTION ACCOUNT (from step 1):
$body_account

Walk through the task moment by moment, starting from the INITIAL POSE in the spec. ORDER the
phases so that any required orientation or pre-shaping of a part is its OWN phase that completes
BEFORE the phase that transports that part toward contact -- never let a transport phase double as
the one that fixes orientation. Any orientation a phase establishes MUST be reachable within the
hinge degree limits in the spec; where the ideal orientation is unreachable, the orientation phase
drives the relevant joints to their limits and every later phase is built around that maximally-driven
pose (the exit_condition for such a phase tests the joint at/near its limit, not an unreachable
angle). For EVERY phase of the motion, state exhaustively:
 - moving: which robot parts move, in which observable direction, at what speed or effort budget;
 - still: which robot parts must stay still, and what observable quantities show stillness;
 - interactions: every physical interaction event, its order, its intended intensity, and how the
   robot can tell it happened from raw observables;
 - precision: which motions require tight error budgets, and the numeric observable budget;
 - must_not: what each moving part must not do, and the raw observable signature that would reveal
   the violation;
 - end_when: a prose statement of the observable condition that ends the phase;
 - exit_condition: that SAME end condition written as ONE OBSERVABLE PREDICATE the framework can
   evaluate directly. It must be an expression over the spec's OBSERVABLES and numeric constants
   that is > 0 exactly when the phase is complete. Use ONLY these operators/functions: + - * /,
   clip(x,lo,hi), sigmoid(x), min(a,b), max(a,b), abs(x), exp(x), sqrt(x), tanh(x), and a SINGLE
   comparison (a<b or a>b). Reference real observable names from the spec with explicit numeric
   thresholds and the correct direction. The exit_condition must be monotone in task order: it
   should stay true once the phase has done its job, and it must not already be true in the start
   state. Use a condition the phase actively brings about, and make sure no later phase's
   exit_condition can hold before this one's does.

State the things a competent operator would leave unsaid. Where a budget matters, make it numeric
and observable: maximum relative speed at first interaction, maximum interaction force, and minimum
force or displacement needed for each interaction.

Return JSON ONLY:
{"procedure": [{"phase": "<name>",
  "moving": "<parts, directions, speeds, efforts>",
  "still": "<parts that must not move, and observables that show stillness>",
  "interactions": "<physical interactions: order, intensity, detection>",
  "precision": "<numeric observable error budgets>",
  "must_not": "<forbidden motions and their observable signatures>",
  "end_when": "<prose observable condition that ends the phase>",
  "exit_condition": "<single observable predicate over the spec's observables, >0 exactly when the phase is complete, monotone, false at spawn>"}, ...],
 "global_invariants": ["<conditions that must hold for the whole task>", ...]}

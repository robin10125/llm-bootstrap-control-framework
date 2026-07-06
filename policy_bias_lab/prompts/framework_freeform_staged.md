FRAMEWORK: you design a SEQUENCE OF STAGES yourself -- there is NO fixed phase set -- and you
define your own SIGNAL VOCABULARY. The framework provides only the raw OBSERVABLES listed in the
spec; author `signals: {"<name>": "<expr over observables>"}` for every derived quantity your
stages need -- distances or offsets between observed positions, smooth activation gates over
them, aggregates of commanded or measured DOF positions, alignment or progress measures -- with
whatever definitions and thresholds fit the task. Later signals may reference earlier ones.
Gates, channels, success tests, and probes are expressions over the observables plus YOUR signal
names. Anything you will want measured and reported back must be expressible from your signals or
probes -- if you don't define a quantity, the diagnostics cannot track it.

DERIVE the stages from the upstream account of how the task is COMPLETED: find the natural
segments in its chain of actions -- each segment ends at an observable condition -- and make each
segment a stage. For EACH stage, author FIRST its EXIT MEASUREMENT: a signal `done_<stage>` whose
expression is > 0 exactly when that stage's job is complete (the segment's observable end
condition: a position entering a band, a measured force appearing, an error inside a budget).
Then BUILD THE GATES FROM THOSE SAME MEASUREMENTS as a PROGRESS LADDER:

  gate_0 = 1 - done_0
  gate_k = done_(k-1) * (1 - done_k)        [for k >= 1]

and give each stage its `done_<stage>` expression as its `success` field (they are the same
thing: the post-condition). The ladder makes sequencing come from the MEASUREMENTS: the active
stage is always the FIRST UNFINISHED one; a later stage cannot fire before its predecessor's exit
condition has actually been met; and if an earlier condition collapses (something drifts out of
its band), control returns to the stage that restores it -- statelessly, with no phase pointer.
You may add small shaping terms on top of a ladder gate, but the ladder must be its backbone:
free-form ADDITIVE activation sums, where every term contributes regardless of order, are how a
late stage ends up dominant in situations its predecessors never produced (e.g. terms like
(1 - moving) or (force ~ 0) are satisfied at spawn, so an end-of-task gate outbids the approach).

Stages are combined STATELESSLY from the CURRENT signals only -- no memory, latch, or phase
pointer -- so gates and channels must be pure functions of the present signals. Stage weights =
softmax of the gate values at a sharp temperature: the highest gate takes nearly all the weight,
with a brief smooth cross-fade when two gates are close. A shared constant added to every gate
cancels out -- only gate DIFFERENCES matter; UNEQUAL constants make the largest one a hidden
DEFAULT stage, so keep any offsets equal. During a gate hand-off the softmax briefly MIXES the
adjacent stages' channels, so a residual of one stage's motion leaks into the next; channels
whose exprs already go to 0 near their stage's boundary hand off cleanly, while channels still
commanding motion there carry it into the next stage. Define as many stages as the task needs;
the ladder covers every situation by construction (some stage is always the first unfinished one).

Your `success` expressions (the done_* measurements) are evaluated on the trained policy's
behavior and cross-checked against the observed stage hand-offs -- a stage whose hand-off fires
while its success test fails (or vice versa) is flagged back to you with the evidence, which
sharpens later revisions. Keep them simple and observable.

You may also author diagnostic PROBES (`probes: [{name, expr, stage?}]`, up to 8): named expressions
the framework measures on the trained policy's states and reports back with the diagnostics, so
you can request exactly the evidence your next revision needs.

You may also author EVALS (`evals: [{name, expr, when: 'ever'|'end'}]`, up to 8): named ACCEPTANCE
TESTS over the same vocabulary as probes, scored per episode as pass/fail -- 'ever': the expression
must exceed 0 for a sustained run somewhere in the episode; 'end': it must hold through the episode's
final steps -- and reported back as a pass fraction. Write each eval as the check you would use to
decide whether one specific problem is SOLVED. They carry selection weight: a revision whose
objective is unchanged within measurement noise is still adopted if it strictly improves the eval
battery, and a revision that flips a failing eval to passing is archived as a durable BRANCH of the
prior. Evals persist across revisions until you replace them; they can never rescue a revision whose
objective actually regressed.

One general principle: if you judge this to be a dexterous manipulation task, be gentle -- manage
velocity and force EXPLICITLY, the way a person does. Decelerate on approach so that any FIRST
CONTACT happens at near-zero relative speed; cap the force of contact, and apply only the MINIMUM
force that accomplishes each interaction (the spec's CONTROL law tells you how commanded-vs-measured
gaps map to applied force); make contact without displacing any items. Express these budgets as
signals and gate/channel conditions -- an approach-speed limit, a contact-force ceiling, a
minimal-grip condition -- so the diagnostics can measure whether they held.

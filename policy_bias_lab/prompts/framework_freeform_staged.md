FRAMEWORK: you design a SEQUENCE OF STAGES yourself -- there is NO fixed phase set -- and you
define your own SIGNAL VOCABULARY. The framework provides only the raw OBSERVABLES listed in the
spec; author `signals: {"<name>": "<expr over observables>"}` for every derived quantity your
stages need -- distances or offsets between observed positions, smooth activation gates over
them, aggregates of commanded or measured DOF positions, alignment or progress measures -- with
whatever definitions and thresholds fit the task. Later signals may reference earlier ones.
You may also author `parameters: {"<name>": {"init": number, "range": [lo, hi]}}`; parameter names
are scalar constants available inside signal, gate, success, channel, probe, and eval expressions,
so an empirical calibration pass can tune thresholds and gains without changing the structure.
Gates, channels, success tests, and probes are expressions over the observables plus YOUR signal
names. Anything you will want measured and reported back must be expressible from your signals or
probes -- if you don't define a quantity, the diagnostics cannot track it.

DERIVE the stages from the upstream account of how the task is COMPLETED: find the natural
segments in its chain of actions -- each segment ends at an observable condition -- and make each
segment a stage. ORIENT BEFORE MOVE: whenever a part must be in a particular pose/orientation to
do its job, put a stage that ESTABLISHES that pose (and whose `done_<stage>` measures the pose is
reached) BEFORE any stage that translates that part toward contact or carries it into the task --
never rely on the spawn pose being correct, and never let a transport stage double as the stage
that fixes orientation. For EACH stage, author FIRST its EXIT MEASUREMENT: a signal `done_<stage>` whose
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
pointer -- so gates and channels must be pure functions of the present signals. EXACTLY ONE stage
is active at any instant: the framework CLIPS every gate to [0, 1] and selects the single highest
one (ties break to the EARLIEST stage). There is NO cross-fade and NO mixing -- at a hand-off the
active stage switches cleanly from one to the next, so ONLY the active stage's channels ever drive
the robot. Because gates are clipped to [0, 1], a later stage cannot outbid a nearly-finished
current stage merely by having a large unbounded `1 - done_k`; keep each `done_<stage>` measurement
on a sane scale so its ladder gate crosses the others at the right moment. Since nothing leaks
across the boundary, a channel need NOT fade its expression to 0 near its stage's edge -- it stops
contributing the instant its stage deactivates. Define as many stages as the task needs; the
ladder covers every situation by construction (some stage is always the first unfinished one).

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
force that accomplishes each interaction (the spec exposes the measured contact force directly, and
its CONTROL law tells you how commanded-vs-measured gaps map to applied force); make contact without
displacing any items. AVOID BUMPING into other objects and surfaces unless the contact is
INTENTIONAL: every part should stay clear of anything it is not deliberately acting on, except where
that contact IS the intended action of the current stage. Use the body world-positions the spec
exposes to keep clearance -- gate a motion so a part that must stay off a surface keeps a margin of
distance from it -- and read the region-contact observables to catch a stray press early. Express
these budgets as signals and
gate/channel conditions -- an approach-speed limit, a contact-force ceiling, a minimal-grip condition,
a clearance floor -- so the diagnostics can measure whether they held.

GENTLENESS IS ONLY FOR THE DEXTEROUS PARTS -- everywhere else, be FAST. Gentleness has a cost: a
slow, asymptotic servo (a small proportional gain drifting toward a target with a tight tolerance)
can take many seconds to settle the last fraction of the way, and under the one-stage-at-a-time
ladder that stalls the ENTIRE prior behind it. A stage that only has to re-orient or coarsely
position a part in free space, away from any contact, carries no dexterous risk and should move at
FULL speed, then DECELERATE LATE -- drive hard until the part is near its target, brake sharply as it
arrives (a strong response with a velocity-damping term), and hand off as soon as it is within a
REAL margin of error (a few percent of travel, not a fraction of a degree). Do not gate such a stage
on an almost-exact match or on the velocity settling to near zero; that is what makes orientation
crawl. Reserve slow, near-zero-speed motion for the steps where speed would disturb an item it is
acting on or break a contact budget.

MOVE-FAST-THEN-HALT is a PRIMITIVE -- use it. A plain `gain*(target - q)` channel CRAWLS: because a
channel is a target-VELOCITY command (the CONTROL law moves the target by action*action_scale each
step), a proportional command shrinks itself as q nears target, so it approaches EXPONENTIALLY and
the last stretch takes as long as the first -- and if the gain never reaches the channel's clip it
never even cruises. You do NOT need to hand-tune this. For any coarse free-space positioning or
orientation, write the channel as `arrive(target - q_self, v_self, vmax)` -- it cruises at +-vmax,
decelerates late, and HALTS cleanly at the target (a trapezoidal move); pick vmax as the cruise speed
you want (~0.3-0.5 for a brisk reorient). Gate that stage's `done_<stage>` with `within(target -
q_<name>, tol)` so it hands off the instant it ARRIVES -- and do NOT also require the velocity to
settle to ~0 for a coarse stage; that is what makes a fast move sit around waiting to stop and eat the
rollout. Reserve gentle, hand-written, near-zero-speed channels (and any velocity/contact settling
condition) for the CONTACT and dexterous-closure stages, where slowness is correct.

FIT THE ROLLOUT. The whole prior runs in ONE rollout of the length given in the spec's ROLLOUT
BUDGET, one stage at a time in order, so every stage's duration is spent from that one budget. Give
each stage an `est_seconds` field: your estimate, in seconds, of how long that stage should take to
complete at the pace you designed. Keep the sum of all `est_seconds` comfortably below the budget so
the final stage has time to finish -- if it does not fit, make the coarse/positioning stages faster
rather than dropping stages. The diagnostics report both your estimate and the MEASURED seconds each
stage actually took, and flag when the rollout ends inside a stage (it ran out of time, i.e. it is
too slow, NOT that its hand-off is broken).

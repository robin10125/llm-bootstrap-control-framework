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
that fixes orientation. For EACH stage, author FIRST its EXIT MEASUREMENT: define the observable
condition that means the segment is complete (a position entering a band, a measured force
appearing, an error inside a budget). If the natural measurement is a signed margin, keep it as a
separate `<stage>_margin` signal if useful, but make `done_<stage>` a gate-ready COMPLETION
ACTIVATION: near 0 before the condition is met and near 1 once it is met within tolerance. When the
condition is met, this activation must fire strongly enough to hand control to the next stage.
Then BUILD THE GATES FROM THOSE COMPLETION ACTIVATIONS as a PROGRESS LADDER:

  gate_0 = 1 - done_0
  gate_k = done_(k-1) * (1 - done_k)        [for k >= 1]

and give each stage a `success` expression that tests the same post-condition. The ladder makes
sequencing come from the MEASUREMENTS: the active stage is always the FIRST UNFINISHED one; a later
stage cannot fire before its predecessor's exit condition has actually been met. The default runtime
uses a MONOTONE STAGE CURSOR: once a stage has handed off, ordinary earlier stages stay closed even if
their current raw gate later becomes large again. If recovery is needed, author it deliberately as a
later stage or as stage-local channels/constraints, with its own observable entry/exit conditions.
You may add small shaping terms on top of a ladder gate, but the ladder must be its backbone:
free-form ADDITIVE activation sums, where every term contributes regardless of order, are how a
late stage ends up dominant in situations its predecessors never produced (e.g. terms like
(1 - moving) or (force ~ 0) are satisfied at spawn, so an end-of-task gate outbids the approach).

EXACTLY ONE stage is active at any instant: the framework CLIPS every gate to [0, 1], takes hard
argmax, and uses that as an advancement request for the monotone cursor (ties break to the EARLIEST
stage). There is NO cross-fade and NO mixing -- at a hand-off the active stage switches cleanly from
one to the next, so ONLY the active stage's channels ever drive the robot. Because gates are clipped
to [0, 1], a later stage cannot outbid a nearly-finished current stage merely by having a large
unbounded `1 - done_k`; keep each `done_<stage>` activation scaled for competition under hard argmax.
A raw signed margin that only reaches 0.02 or 0.04 is fine as a success threshold, but it is NOT a
valid ladder activation by itself: `1 - done_k` would remain near 1 and self-lock the current stage.
Normalize, clip, compare, or combine margins so a completed stage's `done_<stage>` rises near 1,
while the next stage's gate can outbid the current one at the handoff. If more than one handoff
condition is valid, author an explicit fallback completion activation such as
`done_stage = max(done_primary, done_fallback)` and use it consistently in the stage gate and
success expression. Since nothing leaks across the boundary, a channel need NOT fade its expression
to 0 near its stage's edge -- it stops contributing the instant its stage deactivates. Define as many
stages as the task needs; the ladder covers every situation by construction.

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
distance from it -- and read both object-contact and environment-contact observables to distinguish
the interaction the stage intends from contact with other geometry. Express these budgets as signals
and gate/channel conditions -- movement-speed limits, approach-speed limits, object-contact force
ceilings, environment-contact force ceilings, minimal-interaction conditions, clearance floors -- so
the diagnostics can measure whether they held.

FOR DEXTEROUS TASKS, EVERY TRANSLATING BODY MOTION NEEDS A HARD SPEED LIMIT. Treat translating
parts differently from rotational/articulated joints: a carriage or body-position channel moves a
body through world space, so an excessive command can create violent approach, impact, or reversal.
For each stage that translates a part toward a target region, author a non-oscillating positioning
prior: a monotone approach direction, a clipped cruise speed, damping near the target, and a real
arrival margin that triggers hand-off. Do not use high-gain raw distance/error terms whose magnitude
grows when the part is far from the target; do not combine competing terms that can reverse the
translation direction across a narrow boundary; do not rely on overshoot-and-correction. Bound the
channel expression with an explicit speed ceiling and express the intended speed budget as a signal,
probe, or gate condition when it matters.

FINISH WITHIN THE ROLLOUT, BUT DO NOT SOLVE SLOWNESS BY MAKING MOTION VIOLENT. A slow, asymptotic
servo (a small proportional gain drifting toward a target with a tight tolerance) can take many
seconds to settle the last fraction of the way, and under the one-stage-at-a-time ladder that stalls
the ENTIRE prior behind it. The fix is a capped, damped, non-oscillating motion with a realistic
handoff tolerance: move at the chosen speed limit while far away, decelerate before the target, and
handoff as soon as the observable condition is within a REAL margin of error. Reserve near-zero-speed
motion for the steps where speed would disturb an item it is acting on or break a contact budget.

ARRIVE-THEN-HALT is a PRIMITIVE for single-DOF pose targets. A plain `gain*(target - q)` channel
CRAWLS: because a channel is a target-VELOCITY command (the CONTROL law moves the target by
action*action_scale each step), a proportional command shrinks itself as q nears target, so it
approaches EXPONENTIALLY and the last stretch takes as long as the first -- and if the gain never
reaches the channel's clip it never even cruises. You do NOT need to hand-tune this. For a
rotational/articulated free-space pose stage, `arrive(target - q_self, v_self, vmax)` cruises at the
chosen capped speed, decelerates, and HALTS cleanly at the target; pick `vmax` as the maximum speed
the stage may command, not as an uncapped aggression knob. For translating body stages, use the same
principle -- capped monotone approach plus damping -- only with the target, margin, and speed budget
defined from the relevant world-position signals in the spec. Gate that stage's `done_<stage>` with
an arrival margin so it hands off when it ARRIVES; do NOT require an almost-exact match. Add a
velocity/settling condition only where the dexterous budget actually requires settling before the
next interaction.

FIT THE ROLLOUT. The whole prior runs in ONE rollout of the length given in the spec's ROLLOUT
BUDGET, one stage at a time in order, so every stage's duration is spent from that one budget. Give
each stage an `est_seconds` field: your estimate, in seconds, of how long that stage should take to
complete at the pace you designed. Keep the sum of all `est_seconds` comfortably below the budget so
the final stage has time to finish -- if it does not fit, make the coarse/positioning stages faster
rather than dropping stages. The diagnostics report both your estimate and the MEASURED seconds each
stage actually took, and flag when the rollout ends inside a stage (it ran out of time, i.e. it is
too slow, NOT that its hand-off is broken).

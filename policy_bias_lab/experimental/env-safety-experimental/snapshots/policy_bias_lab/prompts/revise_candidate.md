Revise ONE action-prior candidate to score better on the objective. Keep the SAME framework and the
same STATELESS constraint as before, and keep the candidate's structure. Output exactly one improved
candidate -- a targeted edit of the one below, not a rewrite from scratch.

TASK: $task

$spec_block

$representation_doc

$dof_requirement

The upstream CONTEXT (comprehensive body-action account + embodied procedure):
$context_block

The context above includes a moment-by-moment account of what COMPLETING this task looks like (the
embodied procedure: its phases, interactions, budgets, forbidden motions/exploits, and per-phase
observable exit conditions). Before choosing an edit, locate the measured behavior on that account:
which phase does the policy actually reach, judged by the DIAGNOSTICS below (not by which gates
fire)? Your edit should be the one that carries the behavior into the NEXT phase of that account.

CURRENT CANDIDATE (program JSON):
$candidate

DIAGNOSTICS for it (real, observable; higher objective is better):
$diagnostics

The stage report's `body_motion` block gives measured kinematics of the observed bodies: speeds and
net displacement from each episode's start pose, overall and attributed per stage (statistics over
the steps where that stage is dominant). The `commanded_motion` block gives the target speed your
CHANNELS asked for, per stage and actuator group (ctrl-units/s) -- compare it with body_motion and
with the CONTROL law's units bridge to separate commanded aggression from passive drift. Read both
against your intent -- decide yourself which motion is progress and which is a side effect of your
channels, and revise accordingly.
The `contact_forces` block gives per-region object-contact and environment-contact force separately,
attributed by the stage that was dominant. Use it to calibrate action strength and to distinguish an
intended object interaction from contact with other geometry.
The `constraints` block gives each authored stage-local constraint's activation rate and strength,
including its activation while that stage is dominant. If an unwanted-contact or other violation is
present but no relevant constraint activates, add or rescale the constraint. If a constraint activates
often while progress stalls, revise it so it projects the stage goal to the nearest admissible boundary
plus a safety buffer and preserves still-admissible progress, instead of merely cancelling motion or
toggling against the nominal channels.
The stage report's `time_report` block is decisive for telling SLOW apart from STUCK. It gives, per
stage, your `authored_est_seconds` vs the `per_stage_measured_seconds` it actually took, the
`rollout_seconds` budget, and `ended_in_stage_frac` (how often the rollout ENDED inside each stage).
If a hand-off looks broken but the rollout keeps ending inside that stage (`stall_time_limited`), or
the measured seconds greatly exceed your estimate, the stage is TOO SLOW, not stuck: speed it up
(use a capped, damped, non-oscillating command and widen its handoff tolerance to a real margin) so
later stages get rollout time -- do NOT rewrite a gate that would fire fine given more time. If
`fits_rollout` is false, the whole chain is over-budget; shorten coarse stages by improving their
motion shape and tolerances, not by removing speed ceilings.
One general principle: if you judge this to be a dexterous manipulation task, be gentle -- manage
velocity and force EXPLICITLY: first contact at near-zero relative speed, a ceiling on contact
force, and only the MINIMUM force each interaction needs (the spec's CONTROL law maps
commanded-vs-measured gaps to applied force). Translating body motion is different from
rotational/articulated joint motion: in dexterous tasks, keep explicit hard speed ceilings for body
translation stages, use monotone approach commands with damping near the target, and avoid high-gain
distance/error terms or competing terms that can reverse direction and oscillate when far from the
target. A revision that gains objective by moving faster while the body_motion evidence shows excess
speed, reversal, or displacement is the wrong direction -- keep the budgets as explicit signal/gate
conditions. AVOID BUMPING into other objects and surfaces unless the contact is INTENTIONAL: keep
every part clear of anything it is not deliberately acting on -- use the body world-positions the spec
exposes for clearance (a part's position lets you keep it a margin clear of a surface it must not
touch), the object-contact observables to confirm intended interaction, and the environment-contact
observables to catch contact with other geometry. Add clearance floors and environment-contact force
ceilings as gate/channel conditions where a stage risks unintended contact. When such a condition is
a local admissibility boundary for the current stage, prefer a stage-local constraint that redefines
the affected movement target to the nearest feasible boundary plus an author-chosen safety buffer; do
not make a recovery that repeatedly undoes and redoes the same stage action.

WHERE THE POLICY STALLS (revise here):
$stage_focus

Diagnose what is limiting the objective from these diagnostics and the upstream procedure account
(including its forbidden-motion/exploit notes), and make a focused change to fix the stalling stage
named above. A stall usually means that stage's
channels never drive the signals into the NEXT stage's gate condition (so the hand-off never fires),
or the stage's own gate never activates -- fix whichever it is. For staged candidates, also check
the numeric gate competition under hard argmax: the framework clips gates to [0, 1] and runs the
single highest gate. A post-condition margin that barely becomes positive can pass `success > 0`
while still leaving `1 - done_<stage>` near 1, so the current stage self-locks and the next gate
never wins. If the condition is met within tolerance, make a separate completion activation fire
near 1 and use THAT in the ladder; keep raw signed margins separate when you need them for success
or diagnostics.

EDIT MENU -- pick the SMALLEST edit the evidence supports (a suggestion is given above):
 (a) reshape a channel's response (gentler / decelerating / sign-corrected) -- when a signal is
     trending the wrong way under the current channels, or commanded/body motion oscillates,
     reverses, or exceeds the intended speed budget;
 (b) nudge a gate threshold -- when the next gate's value is approaching but not yet firing;
 (c) REWRITE a gate condition -- when training has converged and the hand-off is not approaching:
     the gate reads the wrong signals or fires in the wrong region; do not just rescale it;
 (d) restructure the hand-off between two adjacent gates -- when they overlap or leave a dead zone
     (e.g. a hard-argmax self-lock): change BOTH sides of the boundary coherently and rescale the
     `done_<stage>` activations so the completed stage loses and the successor wins.
 (e) add or reshape a stage-local constraint -- when the nominal stage goal is valid in intent but
     violates an authored boundary in some states: use replace-mode channels to project the affected
     movement to the feasible boundary plus a safety buffer, and keep admissible components of the
     stage goal moving so the rollout does not deadlock.

PROBES -- request your own measurements. You may include `probes: [{name, expr, stage: '<stage
name>' (optional)}]` (up to 8) in the candidate. Each probe expression is evaluated by the framework
on the NEXT evaluation's visited states and its statistics (early/late episode means, min, max)
are reported back to you in the following revision's diagnostics (probe_report). Probe expressions
use the same grammar and vocabulary as gates (the raw OBSERVABLES plus the candidate's own
`signals` definitions -- you may also add or redefine `signals` in your revision if the evidence
needs a quantity that is not yet defined), plus two probe-only, episode-relative signals:
`obj_disp_xy` (object's horizontal distance from its own episode-start position, m) and
`obj_speed` (object's translational speed, m/s). With `stage` set, statistics are restricted to
steps where that stage is dominant. Use probes to TEST A HYPOTHESIS about why the policy fails --
ask for the measurement that discriminates between your candidate explanations, then act on the
numbers next iteration. If contact type matters, probe object-contact and environment-contact
signals separately; do not infer intended contact from a generic tracking gap. Probes you authored
earlier persist until you replace them.

EVALS -- author acceptance tests. You may include `evals: [{name, expr, when: 'ever'|'end'}]` (up
to 8): pass/fail checks scored per episode ('ever' = the expression exceeds 0 for a sustained run
somewhere in the episode; 'end' = it holds through the final steps), reported back as pass
fractions (eval_report in the diagnostics). Write each eval as the test that decides whether one
specific problem is SOLVED. They carry selection weight: a revision whose objective is unchanged
within measurement noise is still ADOPTED if it strictly improves the shared eval battery, and a
revision that flips a failing eval to passing is archived as a durable BRANCH of the prior. Evals
persist until you replace them; they can never rescue a revision whose objective regressed.

Use the signal trends above to pick
the DIRECTION of your change: if a signal the next gate needs is moving the wrong way under the
current channels, increasing their strength will move it further the wrong way -- change the
response shape (gentler, decelerating, or sign-corrected) or the gate hand-off instead. If the
diagnostics list `prior_failed_revisions`, those edits were already tried and scored WORSE than the
current candidate -- propose a genuinely different approach, not a variant of them. Keep
weights conservative (<=0.6).

Preserve the typed candidate structure: derived `signals` first, optional tunable `parameters`
second, then stage `success` exit measurements, progress-ladder gates, channels, optional
stage-local constraints, probes, and evals.
Parameter names are scalar constants available in expressions; use `{init, range:[lo,hi]}` for any
threshold or gain that should be calibrated empirically later.
Return JSON ONLY: {"candidate": {name, rationale, $output_item, unused_dofs:[{actuator, reason}]}}

Revise ONE action-prior candidate to score better on the objective. Keep the SAME framework and the
same STATELESS constraint as before, and keep the candidate's structure. Output exactly one improved
candidate -- a targeted edit of the one below, not a rewrite from scratch.

TASK: $task

$spec_block

$representation_doc

$dof_requirement

FAILURE MODES to steer away from and the upstream CONTEXT:
$context_block

CURRENT CANDIDATE (program JSON):
$candidate

DIAGNOSTICS for it (real, observable; higher objective is better):
$diagnostics

WHERE THE POLICY STALLS (revise here):
$stage_focus

Diagnose what is limiting the objective from these diagnostics and the upstream failure modes, and
make a focused change to fix the stalling stage named above. A stall usually means that stage's
channels never drive the signals into the NEXT stage's gate condition (so the hand-off never fires),
or the stage's own gate never activates -- fix whichever it is.

EDIT MENU -- pick the SMALLEST edit the evidence supports (a suggestion is given above):
 (a) reshape a channel's response (gentler / decelerating / sign-corrected) -- when a signal is
     trending the wrong way under the current channels;
 (b) nudge a gate threshold -- when the next gate's value is approaching but not yet firing;
 (c) REWRITE a gate condition -- when training has converged and the hand-off is not approaching:
     the gate reads the wrong signals or fires in the wrong region; do not just rescale it;
 (d) restructure the hand-off between two adjacent gates -- when they overlap or leave a dead zone
     (e.g. a hard-blend self-lock): change BOTH sides of the boundary coherently.

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
numbers next iteration. Probes you authored earlier persist until you replace them.

Use the signal trends above to pick
the DIRECTION of your change: if a signal the next gate needs is moving the wrong way under the
current channels, increasing their strength will move it further the wrong way -- change the
response shape (gentler, decelerating, or sign-corrected) or the gate hand-off instead. If the
diagnostics list `prior_failed_revisions`, those edits were already tried and scored WORSE than the
current candidate -- propose a genuinely different approach, not a variant of them. Keep
weights conservative (<=0.6).
Return JSON ONLY: {"candidate": {name, rationale, $output_item, unused_dofs:[{actuator, reason}]}}

Your motor plan below was compiled and PLAYED BACK in the real simulator, exactly as it will be
consumed. The measurements that follow are ground truth about what your plan actually did --
where it left the effector relative to the object, whether it made contact, and where its
commanded targets exceeded actuator ranges or slew limits.

YOUR CURRENT PLAN:
$candidate_json

MEASURED EVIDENCE:
$score_block

$grounding_block

$tape_report_block

REVISION OBJECTIVE -- ALIGNED NEAR-MISS. Your plan is the feedforward TRANSPORT-AND-POSTURE
layer of a hybrid controller: a learned closed-loop corrective policy runs on top of it with
full force/contact sensing, and FIRST CONTACT with the object is that policy's job, not yours.
An open-loop plan that touches the object cannot respond to where the object actually is or
what the contact forces say, and measurably traps the learner in a marginal-contact strategy.
Revise the plan so that, on every episode:
- the approach ends ALIGNED: the end-of-episode obj_rel x and y components are each as close to
  0 as the evidence lets you drive them (fix sign/arithmetic errors first);
- the closest approach stops SHORT of touch: minimum grasp-point-to-object distance inside
  [$nearmiss_lo, $nearmiss_hi] m, held steadily to the end of the episode;
- the object is NEVER contacted: measured contact steps must be 0 (a nonzero contact fraction
  is a failure of this objective, no matter what it does to lift or score);
- the hand arrives PRE-SHAPED for a grasp: fingers and thumb posed at a ready aperture around
  where the object sits, wrist attitude prepared -- posture the learner can close from, without
  the closing itself.
Do NOT add lift or grasp phases; delete or retarget any part of the current plan that produces
contact. Depth/closure ambition that overshoots into touch is worse than stopping early.

Rules:
- Diagnose FIRST: for each episode, compare spawn obj_rel to end obj_rel axis by axis. An axis
  whose magnitude GREW means your plan moved the grasp point the WRONG WAY on that axis --
  usually a sign error in the target arithmetic. Fix the arithmetic, do not just rescale.
- Keep what the evidence says is working; change what it says is broken. Structural rewrites
  are allowed but must be justified by the measurements, not taste.
- The representation, grammar, timing rules, and output format are UNCHANGED from the original
  instructions (same keyframe schema, absolute ctrl-unit targets, expressions over the reset
  observation only).
Return JSON ONLY, the complete corrected candidate in the same format:
{"candidates": [{"name": ..., "rationale": "<what the evidence showed and what you changed>",
"mode": "motor_tape", "signals": {...}, "parameters": {...},
"defaults": {"interp": "minjerk"}, "keyframes": [...], "unused_dofs": [...]}]}

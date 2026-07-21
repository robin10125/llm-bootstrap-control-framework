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

Revise the plan to fix what the evidence shows is wrong. Rules:
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

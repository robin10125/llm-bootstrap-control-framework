CONTEXT GENERATION (single job). Describe, in EXCRUCIATING physical detail, how this robot performs
this task -- an exhaustive, chronological execution score for every part of the robot. Reference
context only; do NOT design priors here.

TASK: $task

$spec_block

Walk through the task moment by moment. For EVERY phase of the motion, state exhaustively:
 - which parts move: direction, speed, and how much force, phrased against the actuators in the
   spec;
 - which parts must stay STILL at that moment, and what holds them still;
 - every contact event: what touches what, in what order, how firmly, and how the robot can tell
   the contact happened from its own observables;
 - the precision required: which motions must be slow and exact, which tolerate error, and what
   the error budget is;
 - what each moving part must NOT do at that moment, and the physical consequence of doing it;
 - the observable condition that ends the phase (a measurable state, not an intention).

Do not skip requirements because they are obvious -- state the things a competent operator would
leave unsaid. Where a budget matters, make it NUMERIC and observable: the maximum speed at which
any part may first touch anything, the maximum force of that first contact, and the minimum force
that accomplishes each interaction -- and say which observables would show a budget being violated.

Return JSON ONLY:
{"procedure": [{"phase": "<name>",
  "moving": "<parts, directions, speeds, forces>",
  "still": "<parts that must not move, and what keeps them still>",
  "contacts": "<contact events: order, firmness, how they are detected>",
  "precision": "<what must be exact vs what tolerates error>",
  "must_not": "<forbidden motions and their physical consequence>",
  "end_when": "<observable condition that ends the phase>"}, ...],
 "global_invariants": ["<conditions that must hold for the whole task>", ...]}

CONTEXT GENERATION (step 1 of 2). Produce a comprehensive account of completing the task
successfully with this robot. This is reference context only; do NOT design priors here.

TASK: $task

$spec_block

Describe the successful execution as a chronological physical account grounded in the robot spec.
Cover every listed actuator and semantic group, including parts that should remain still. For each
body part, state its role, when it moves, when it holds position, what observable quantities show
that it is doing the intended thing, and what observable quantities would show that it is interfering
with completion.

Begin from the INITIAL POSE in the spec. State explicitly whether the spawn configuration already
orients each part for its job or whether orientation must be actively established first -- do not
assume a neutral/home pose is interaction-ready. As a general rule, a part that must be oriented to
do its job is ORIENTED FIRST, before it is transported toward contact; name which actuators
establish that orientation and the observables that confirm it.

REASON WITHIN THE PHYSICAL LIMITS. Each hinge actuator's `motion` states its travel in DEGREES; a
segment can only reach orientations that are its home direction rotated within those degree limits,
and no further. Before you assume a part will be in the "textbook" orientation for its job, CHECK
that orientation against the joint limits: compute the extreme reachable orientation (each relevant
joint driven to its limit) and state it in degrees. If the ideal orientation is OUTSIDE the reachable
envelope, say so explicitly and design the execution around the closest reachable extreme instead --
treat that maximally-driven pose as the working orientation and describe how the job is accomplished
from it, rather than describing a motion that silently assumes an orientation the robot cannot reach.
Never require a pose the degree limits forbid.

Keep the account generic and mechanical: use the task text and injected robot/environment data as
the only task content. Make the implicit requirements explicit, including contact order, speed
budgets, force budgets, stillness requirements, and recovery conditions when they matter for the
task. Do not emit gates, channels, prior stages, or code.

Return JSON ONLY:
{"execution_account": "<chronological account of the successful execution>",
 "body_parts": [{"part": "<actuator name or semantic group>",
   "role": "<what this part contributes during success>",
   "moves_when": "<when it moves and in which observable direction>",
   "holds_when": "<when it must hold position or stop changing>",
   "observable_evidence": "<raw observables that show correct behavior>",
   "observable_interference": "<raw observables that show this part is preventing success>"}],
 "global_requirements": ["<requirements that apply across the whole execution>", ...]}

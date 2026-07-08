CONTEXT GENERATION (step 1 of 2). Produce a BASIC PLAN for how this robot performs the task -- a
short, ordered sketch of the strategy that a detailed procedure step will expand next. Reference
context only; do NOT design priors here.

TASK: $task

$spec_block

First think about how a capable agent -- e.g. a person, with their actions mapped onto THIS robot's
actual DOF -- would accomplish the task. Then lay the strategy out as an ORDERED list of high-level
steps. For each step, name the primary actuators / semantic groups it recruits and the OBSERVABLE
milestone that marks it complete. Keep to the natural high-level segments (not moment-by-moment
detail -- that comes in step 2). Deliberately surface any DOF a purely mechanical reading would leave
idle, and say what moving it would buy.

Return JSON ONLY:
{"strategy": "<one-paragraph account of the overall approach, in this robot's terms>",
 "plan": [{"step": "<short name>",
   "actuators": "<primary DOF / semantic groups this step drives>",
   "milestone": "<observable condition that marks this step complete>"}, ...]}

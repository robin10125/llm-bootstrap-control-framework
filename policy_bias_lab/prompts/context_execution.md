CONTEXT GENERATION (single job). Produce an itemized, per-actuator execution description of ONE
successful run of the task -- excruciating detail, phase by phase. Reference context for a later
prior-design step; do NOT design priors here.

TASK: $task

$spec_block

For EACH phase in [$phases], describe -- per actuator or semantic group -- what it does and WHY
(direction in world/ctrl terms, rough magnitude, what observable signal would trigger it). Cover
EVERY actuator listed; do not skip any, and do not default an actuator to "held/stable" without
first considering whether actively moving it would improve the outcome. If an actuator genuinely
stays at rest in a phase, say so and justify why moving it would not help.

Return JSON ONLY:
{"phases": [{"phase": "<phase name>", "per_actuator": [{"actuator": "<name or group>",
  "motion": "<what it does, world/ctrl direction>", "trigger": "<observable signal>",
  "why": "<one line>"}, ...]}, ...]}

CONTEXT GENERATION (single job, OPTIONAL). Describe how a HUMAN would perform this task using their
own body, then map each human action onto this robot's actual DOF. Reference context only; do NOT
design priors here. The point is to surface DOF that a purely mechanical analysis might leave idle.

TASK: $task

$spec_block

Return JSON ONLY:
{"human_account": "<how a person does it, step by step>",
 "mapping": [{"human_action": "<one human action>", "robot_actuator": "<name or group>",
  "note": "<one line>"}, ...]}

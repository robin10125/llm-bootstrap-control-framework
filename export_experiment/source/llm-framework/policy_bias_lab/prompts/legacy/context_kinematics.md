CONTEXT GENERATION (single job). Analyze the kinematics/affordances of this robot and its
environment and derive the REQUIRED interaction configuration and the DOF it implicates. Reference
context only; do NOT design priors here.

TASK: $task

$spec_block

Reason from: the reachable workspace of the actuators, the relevant object/environment geometry, and
the mechanics of achieving the task's success predicate. Conclude which actuators are load-bearing
for success and which provide fine adjustment -- and, for EACH actuator, whether its ORIENTATION or
configuration matters for aligning the effector with the task geometry (do not assume any actuator is
purely a stabilizer; consider whether actively changing it would help).

Return JSON ONLY:
{"interaction_config": "<how the actuators should interact with the environment to succeed>",
 "implicated_dof": [{"actuator": "<name or group>", "role": "load-bearing|adjustment|rest",
  "orientation_matters": "<does actively changing this actuator help, and how>",
  "why": "<one line>"}, ...]}

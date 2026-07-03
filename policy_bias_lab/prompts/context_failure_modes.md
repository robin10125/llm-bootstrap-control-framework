CONTEXT GENERATION (single job). Enumerate how this task is commonly FAILED or reward-HACKED, so a
later prior-design step can steer away from them. Do NOT design priors here.

TASK: $task

$spec_block

Cover both genuine failures (ways the task is attempted but not achieved) and non-transferable
exploits that must be avoided (reward hacks, or sim-only tricks that would not survive on real
hardware). Derive these from the task's success predicate and constraints and the available signals
-- do not assume a particular task. For each, give the observable signature and a one-line
mitigation expressed in terms of the available signals.

Return JSON ONLY:
{"failure_modes": [{"name": "<short>", "kind": "failure|exploit",
  "signature": "<observable signal pattern>", "mitigation": "<one line in terms of signals>"}, ...]}

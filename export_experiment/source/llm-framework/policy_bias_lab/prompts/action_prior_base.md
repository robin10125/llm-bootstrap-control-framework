Design action priors: weak per-step mean-shifts added to a PPO policy's action to bias early
exploration. They are stateless functions of the observation. FRAMEWORK (fixed): a STACKED phase
gate blends the sub-priors for the phases [$phases] automatically by observable signals (earlier
phases when far from the goal, later phases as the task progresses). You design one sub-prior per
phase.

TASK: $task. Design for the task's success predicate and honor its constraints; prefer real,
transferable behavior and avoid non-transferable or sim-only exploits.

ACTUATORS (every one must be ACTIVELY MOVABLE by your prior -- holding at 0 is NOT enough):
$actuators
SEMANTIC GROUPS: $semantic_groups
WORLD-SIGN CONVENTIONS: $base_world_sign
CONTROL: $control
SIGNALS available to gates/expressions: $signals

$representation_doc

$dof_requirement

Return JSON ONLY: {"candidates": [ {name, rationale, $output_item, unused_dofs:[{actuator, reason}]}, ... ]}. Produce 3 DIVERSE candidates (different strategies). Keep weights conservative (<=0.6).

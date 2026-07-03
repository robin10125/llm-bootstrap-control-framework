Design action priors: weak per-step mean-shifts added to a PPO policy's action to bias early
exploration. They are STATELESS functions of the observation (recomputed on shuffled minibatches at
update time -- no hidden state, phase pointer, or integrator).
$framework

TASK: $task. Design for the task's success predicate and honor its constraints (see the spec below);
prefer real, transferable behavior and avoid non-transferable or sim-only exploits.

$spec_block

$representation_doc

$dof_requirement

CONTEXT (generated upstream -- a per-actuator execution account, the failure modes to avoid, the
kinematic/affordance analysis, and an optional human analogy). USE it to ground your design and to
recruit every relevant DOF:
$context_block

Produce $n_seeds GENUINELY DIVERSE candidates -- different strategies / DOF recruitment / gating
emphasis, not minor variations. Keep weights conservative (<=0.6). Return JSON ONLY:
{"candidates": [{name, rationale, $output_item, unused_dofs:[{actuator, reason}]}, ...]}

Design action priors: weak per-step mean-shifts added to a PPO policy's action to bias early
exploration. Channels, gates, and success tests are pure expressions over the current observation and
your authored signals; staged priors use the framework's monotone rollout cursor for stage progress.
$framework

TASK: $task. Design for the task's success predicate and honor its constraints (see the spec below);
prefer real, transferable behavior and avoid non-transferable or sim-only exploits.

$spec_block

$representation_doc

$dof_requirement

CONTEXT (generated upstream -- a comprehensive body-action account, then a moment-by-moment
embodied procedure: its phases, interactions, budgets, forbidden motions/exploits, and a per-phase
OBSERVABLE exit condition). USE it to ground your design, derive your stages from its phases, and
recruit every relevant DOF:
$context_block

Build each candidate in this order:
 1. Define `signals` for every derived quantity the stages, channels, probes, or evals need.
 2. Optionally define `parameters` for numeric thresholds/gains that should be calibrated later;
    each parameter is a scalar expression name with `{init, range:[lo,hi]}`.
 3. Define each stage's exit measurement as `success`.
 4. Build gates as a progress ladder from those exit measurements. Make every
    `done_<stage>` used by the ladder a completion activation scaled for hard-argmax competition:
    near 0 before the condition is met and near 1 when it is met within tolerance. Do not put a
    tiny signed margin directly into `1 - done`; define a separate margin signal if needed. If a
    stage can safely hand off by either a primary or fallback condition, define that explicitly with
    `max(done_primary, done_fallback)` or an equivalent completion activation.
 5. Add channels only after the stage exits and gates are defined. Make coarse free-space
    positioning/orientation stages finish promptly with capped, damped, non-oscillating motion,
    handing off within a real margin; for dexterous tasks, put hard speed ceilings on translating
    body stages and reserve near-zero-speed motion for contact/dexterous steps (see the framework
    note).
 6. Give each stage an `est_seconds` (how long it should take at your designed pace) and check the
    sum fits well inside the spec's ROLLOUT BUDGET, with time to spare for the final stage.
 7. Add probes/evals for assumptions that empirical diagnostics should check.

Produce $n_seeds GENUINELY DIVERSE candidates -- different strategies / DOF recruitment / gating
emphasis, not minor variations. Keep weights conservative (<=0.6). Return JSON ONLY:
{"candidates": [{name, rationale, $output_item, unused_dofs:[{actuator, reason}]}, ...]}

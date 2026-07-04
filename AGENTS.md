# Agent instructions for llm-framework

## The task-agnosticism rule (hard constraint)

`policy_bias_lab/` is a DIAGNOSTIC of whether an LLM can author useful task knowledge itself. The
framework (orchestrator, arbiter, diagnostics, prompt templates) must therefore contain **no
task-specific information, interpretation, or instructions**. Every piece of task knowledge in the
loop must come from exactly one of two sources:

1. **Injected environment/robot/task data** — the robot spec (`robot_spec(env)`: actuators, ranges,
   groups, world-sign conventions), the environment's own eval fields and their names, and the task
   definitions in `policy_bias_lab/tasks.py` (NL description, success predicate, graded objective,
   progress-metric list). These are *data handed to the framework*, keyed by task, never baked into
   framework logic or prompt prose.
2. **LLM-authored artifacts produced during the run** — context calls (execution account, failure
   modes, kinematics), candidate programs (stages, gates, channels, per-stage `success`
   expressions), and diagnostic `probes`.

Concretely, when working on this code:

- **Never add task nouns or task heuristics to prompt templates** (`policy_bias_lab/prompts/*.md`).
  Templates hold structure and generic mechanism docs only; task content arrives via `$task`,
  `$spec_block`, `$context_block` substitution at runtime.
- **Never add framework-authored interpretation of behavior.** Diagnostics report *measurements*
  (rates, signal means/trends, gate values, hand-off statistics, intent-vs-executed and
  command-vs-measured gaps, probe results) under the environment's own field names. Labeling a
  pattern ("air closure", "flinging", "knocked the object") is the LLM's job in the revision loop.
  If a run exposes a new failure mode, the fix is to surface *more generic evidence* (a new generic
  measurement or a wider probe vocabulary), not to hardcode the diagnosis.
- **Do not carry lessons across runs through the framework.** A discovery made in one run (e.g. a
  useful gate structure, a wrist posture, an exploit) must not be transplanted into prompts,
  defaults, or scoring logic so later runs inherit it. The system has to be able to re-derive it
  organically. Cross-run knowledge lives only in `tasks.py` (task definitions) or the env.
- **New diagnostics must be structural.** They may reference stages, gates, signals, actuators,
  transitions — the model's own authored objects and the injected vocabulary — never what those
  objects are "supposed" to do for a particular task.
- **The signal vocabulary is not framework property.** The framework exposes only RAW observables,
  mechanically enumerated from the env/robot (`raw_signal_fn`: world positions/velocities in obs,
  commanded actuator targets, measured joint positions). Every DERIVED quantity — distances,
  proximity/grip gates, closure fractions, alignment/orientation measures, and their thresholds —
  is task structure and must be LLM-authored per candidate (`signals: {name: expr}`). Do not add
  derived signals to the framework, advertise them in prompts, or tune their thresholds in code:
  that both leaks task assumptions and silently restricts what candidates can express or measure
  (the fixed 8-signal vocabulary made palm orientation literally unrepresentable). The legacy
  derived names (`near`, `gripped`, `closure`, `lift`, `palm_obj_dist`) survive only for
  archived-program replay and must not appear in prompt templates. Probe-only episode-relative
  extras (`obj_disp_xy`, `obj_speed`) are the narrow exception: raw rigid-body quantities that
  need a per-episode baseline the stateless obs cannot carry.

Known legacy exceptions (quarantined, not in the default path): `policy_bias_lab/legacy/*` and the
open-loop prefilter `prior_eval.score_program` (its `fling_fraction` etc. predate this rule; it is
prefilter-grade only — do not route its labeled fields into prompts).

## Practical notes

- Python: use `.venv/bin/python`. Env init takes ~4 min (JAX/XLA); run env-touching scripts in the
  background. One GPU (8 GB) — run PPO jobs sequentially.
- `../bootstrapping/mjx_env.py` is shared and read-only from this project's perspective; env-side
  reward/eval changes belong there only with explicit approval.
- Long PPO runs: `run_long_ppo.py` (resumable); selection runs: `run_agentic_selection.py`
  (checkpointed, resumable). Both handle SIGINT/SIGTERM by finishing the in-flight step and saving.

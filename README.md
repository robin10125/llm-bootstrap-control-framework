# llm-framework

Experimental framework for comparing task-agnostic LLM control interfaces on a dexterous
robot hand. The first target is the adjacent Shadow Hand MuJoCo/MJX environment in
`../bootstrapping`.

The action interfaces intentionally avoid task primitives such as `grasp`, `throw`,
`fold`, or `pick_up`. They expose general control concepts: base/frame targets, hand
shape interpolation, per-joint targets, appendage-local subprograms, contacts, monitors,
generic controllers, and latent decoder blocks.

The hybrid DSL supports appendage-local reasoning through `call_appendage_agent` blocks.
The top-level LLM can delegate a finger/thumb/wrist subprogram inline:

```json
{
  "op": "call_appendage_agent",
  "appendage": "index",
  "program": {
    "blocks": [
      {"op": "set_joint_target", "joint": "rh_A_FFJ4", "target": 0.4}
    ]
  },
  "duration_s": 0.2
}
```

This still compiles to bounded actuator targets, so the runtime can validate appendage
programs before execution.

The `recursive_units` interface goes further: it makes a top-level LLM call to decompose
the task into commands for every fundamental unit, then one subagent call per unit, then a
final compiler call that combines the unit policies into lowest-level actuator/base
commands. Use it explicitly:

```bash
python -m llm_framework.experiments.compare_interfaces \
  --env shadow \
  --backend codex \
  --interfaces recursive_units \
  --tasks lift \
  --seeds 0
```

## Quick checks

```bash
python -m pytest
python -m llm_framework.experiments.compare_interfaces \
  --backend mock \
  --env shadow \
  --tasks lift \
  --seeds 0 \
  --budget-per-interface 1
```

Use `--backend codex` to call the adjacent `bootstrapping/llm_backend.py` Codex backend.

# LLM Primitive-Policy Prompt Template

Use this with `robot_primitives.json` and a concrete current state.

```text
You control a robot only by emitting a JSON primitive policy. Do not write Python.

Your output must be exactly one JSON object with:
- "name": short snake_case name
- "goal": restatement of the goal
- "setup": copied from the provided setup, if any
- "steps": ordered primitive calls
- "success": final checks

Each step must use one primitive from the provided primitive vocabulary:
- "primitive": primitive name
- "params": object matching that primitive's parameter schema
- "duration_s": how long to execute that primitive
- optional "assert": checks that should hold after that step

You may tune only:
- primitive order,
- primitive durations,
- primitive input parameters,
- assertions and final success checks.

Prefer conservative, robust schedules:
- open before approaching the object,
- approach above the object before descending,
- close only when centred and at grasp height,
- lift only after an explicit grasp assertion.

Return JSON only. No prose, markdown, code fences, or comments.

Primitive vocabulary:
<paste robot_primitives.json here>

Current state:
<paste state JSON here>

Goal:
<paste goal here>
```

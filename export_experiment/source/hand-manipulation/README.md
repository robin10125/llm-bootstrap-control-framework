# hand-manipulation

The **code-as-action** experiment ported to **robotic hand manipulation in MuJoCo**. A
coding model is given a concrete observation + goal and writes a **bespoke short Python
control script** that, when executed against the simulation, achieves the goal. The same
env-agnostic machinery as the Minecraft env (`../game-agent`) drives it — recursive
`achieve()` (script-or-plan), a `SuccessCheck`, the `--gen-mode` knob, and the repair loop.
Only three things are robot-specific: the **bridge**, the **observation schema**, and the
**primitive vocabulary**. That boundary is the whole point — porting domains swaps those,
not the loop.

## Stepping through manipulation

The robot is a floating parallel-jaw gripper over a table with a free cube. A generated
skill is the **body of a Python function** that drives the robot by the *set-target → step*
pattern — it closes its own control loop:

```python
set_ctrl('slide_z', 0.12)   # command an actuator toward a target
step(500)                   # advance physics so it gets there
if grasped('cube'): ...
```

`step` draws from a per-execution budget, so a runaway script fails with structured
feedback instead of hanging — the robot analog of the Minecraft bridge's exec timeout.

## Architecture

```
 tasks/*.yaml          goal + success check (expr over obs, or llm_judge)
        │
        ▼
 harness/agent_hand/  (Python)
   controller.py   ── env-agnostic loop: generate → execute → check → repair → save
   prompts.py      ── robot primitive vocabulary  (env-specific surface)
   tasks.py        ── SuccessCheck = {expr | llm_judge}
   llm.py          ── AnthropicLLM + offline MockLLM (canned, verified lift skill)
   skills.py       ── skill library (keyword retrieval; embeddings later)
   bridge.py       ── in-process bridge: runs the skill body in the primitive scope,
        │              returns structured feedback (error / world-diff / result)
        ▼
   sim.py          ── MuJoCo wrapper + the primitive surface  (env-specific surface)
        │
        ▼
 env/models/gripper_cube.xml   ── the scene (swap for a dexterous hand later)
```

The **structured feedback channel is the core instrument**: every attempt returns *why* it
failed (exception + traceback, or budget exhaustion) and *what changed* in the world (obs
before/after + a compact diff), which the repair loop consumes.

## Setup & run

```bash
cd harness
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

# offline: canned skill, no API key — validates the whole pipeline
python -m agent_hand.run --task ../tasks/lift_cube.yaml --mock

# real controller (needs ANTHROPIC_API_KEY)
python -m agent_hand.run --task ../tasks/lift_cube.yaml --gen-mode scratch
```

`--gen-mode {scratch | edit-nearest | skeleton}` is the comparable experimental knob
(starting point for generation). `--max-depth` caps subgoal decomposition. Runs and their
per-iteration logs land in `runs/`; successful leaf skills are saved to `skills/`.

## Injecting a hand-written control script

The project-constitution loop, done by hand: the sim dumps its state to a text file, an
author (you, or a coding model) reads it and writes a **bespoke control script** for that
specific state + goal, then the script is injected and run. Each script must drive toward a
**clear end state**, verify it, and carry its **own failsafe** so it returns a failure
instead of running forever (the bridge also enforces a hard step-budget backstop).

```bash
# 1. start the sim and write all relevant state to a text file
python -m agent_hand.inject dump                 # -> ../inject/state.txt

# 2. read inject/state.txt, author a script (see scripts/grab_and_lift.py), then run it
python -m agent_hand.inject run --script ../scripts/grab_and_lift.py
python -m agent_hand.stepper --latest            # step through what the script did
```

`run` exits 0 only if the script ran cleanly AND the end state holds (`grasped` and
`cube_z >= 0.12`); otherwise 1. It records a trajectory, the script, and the feedback under
`runs/<ts>-inject_<name>/`. `scripts/grab_and_lift.py` is a worked example: a closed loop
that opens → centres → descends → squeezes → lifts → verifies, retrying within an
8000-step self-imposed budget and returning `FAILURE: ...` if the budget is spent.

## Live step-through viewer

Every run records a `trajectory.npz` — the initial state + the agent's piecewise-constant
**control timeline** (disable with `--no-record`). The stepper does **not** replay stored
positions: it holds a live MuJoCo sim and **re-integrates real physics on demand**, so the
sim is paused between clicks and each click actually advances the simulation. A tiny local
server owns the sim + renderer; the browser page is buttons + an image:

```bash
python -m agent_hand.stepper --latest          # newest run under runs/
python -m agent_hand.stepper --run runs/2026... # a specific run
python -m agent_hand.stepper --latest --port 8008 --no-open
```

Controls: **Step / Back / Play / Reset** buttons, a **steps-per-click** field (1 = single
physics step), and a **slider** that seeks anywhere (seeking back re-integrates from the
start, since physics is deterministic). Arrow keys step, spacebar plays. The status line
shows the step / sim time / cube height / grasped flag and the live actuator targets — so
you can pause on any step and advance the real simulation one step at a time. Rendering
needs a GL backend (`MUJOCO_GL=glfw` on a desktop, `egl` on a headless GPU box); Ctrl-C stops
the server.

## Status

Offline-validated end-to-end (2026-06-08): MuJoCo gripper grasps and lifts the cube; the
loop, structured feedback (errors / budget / world-diff), the `expr` check, plan-envelope
parsing, and skill saving all pass with the MockLLM. The grasp generalizes to off-centre
cube positions. **Not yet:** a real-LLM run (needs `ANTHROPIC_API_KEY`); swapping in a
dexterous hand from `mujoco_menagerie` (the next milestone — a model-file change);
`llm_judge`-checked tasks (e.g. "place the cube on the left").

## Why a simple gripper first

This mirrors how the Minecraft env validated offline before touching a live server: prove
the pipeline with zero external assets, then raise difficulty. `env/models/gripper_cube.xml`
is authored so the primitive/observation names are the only coupling — the immediate next
step is dropping in the 24-DOF Shadow Dexterous Hand and adjusting `sim.py`'s names, with
the harness untouched.

# Agent instructions — inject a control script to manipulate the MuJoCo hand

You are the **script author** in a code-as-action loop. The simulation writes its state to a
text file; you read it, write a **bespoke Python control script** for that specific state and
goal, inject it, and confirm it reached the goal. You do **not** emit primitive actions one at
a time — you write one short program that closes its own control loop.

Default goal (unless told otherwise): **make the gripper grab the red cube and lift it at
least 10 cm off the table, keeping it gripped.**

## Environment

- Project dir: `/home/robin/Documents/agent-mini-script-control/hand-manipulation`
- Always work from the harness dir with the venv active:
  ```bash
  cd /home/robin/Documents/agent-mini-script-control/hand-manipulation/harness
  source .venv/bin/activate
  ```
  First time only (if `.venv` is missing): `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt pillow`
- Rendering (the stepper) needs a GL backend and a display; `MUJOCO_GL=glfw` is set
  automatically. Run from a desktop session, not a bare SSH shell. The `dump`/`run` steps do
  not need a display.

## Procedure

1. **Dump the sim state** to a text file you can read:
   ```bash
   python -m agent_hand.inject dump            # optional: --cube-pos X Y
   ```
   This writes `../inject/state.txt`.

2. **Read `inject/state.txt`** with the Read tool. It contains the GOAL, the reset
   (`cube_pos`), the current OBSERVATION (JSON), MODEL FACTS (actuator names + ranges, cube
   size, grasp-height hints), and the full CONTROL API. Treat it as the ground truth for this
   attempt — do not assume positions from memory.

3. **Author the script** at `../scripts/<name>.py` (e.g. `grab_and_lift.py`). See the script
   contract below. Write the function BODY only — no `def` line, no imports.

4. **Inject and run it:**
   ```bash
   python -m agent_hand.inject run --script ../scripts/<name>.py   # match dump's --cube-pos
   ```
   Exit code is **0 only if** the script ran without error **and** the end state holds
   (`grasped` and `cube_z >= 0.12`); otherwise **1**. Artifacts land in
   `runs/<ts>-inject_<name>/` (`trajectory.npz`, `script.py`, `feedback.json`).

5. **If it failed**, read the printed logs / `runs/<ts>-inject_<name>/feedback.json` (it
   reports the error or world-state diff and why the end state was unmet), revise the script,
   and re-run. Repeat until exit 0.

6. **Inspect the behaviour** (optional, needs a display):
   ```bash
   python -m agent_hand.stepper --latest
   ```
   Opens a browser; Step/Back/Play/Reset advance the live re-simulated physics.

## Script contract (REQUIRED)

The script is the **body of a Python function**; the primitives below are already in scope and
`return` ends the script. Your script MUST:

- **Define a clear end state and verify it before stopping.** For the default goal:
  `grasped('cube') and obj_pos('cube')[2] >= 0.12`. Check it (a) up front (return early if
  already satisfied) and (b) after acting.
- **Carry its own failsafe.** Track a self-imposed physics-step budget (e.g. `MAX_STEPS`). If
  the end state is not reached within it, **return a `FAILURE: ...` string** rather than
  looping forever. (The bridge also hard-stops at ~20000 steps as a backstop, but your script
  must fail gracefully before that.)
- **Return a short summary string**, prefixed `SUCCESS:` or `FAILURE:`, including the final
  `cube_z`, `grasped`, and steps used.
- Use only the primitives (no `import`, no fs/network). Remember: **set a target, then step** —
  a bare `set_ctrl` does nothing until you `step`.

### Control API (also in state.txt)

```
set_ctrl(name, value)   # name: slide_x/slide_y (±0.30), slide_z (0.06–0.45, palm height),
                        #       left_finger/right_finger (0.05 open .. 0.0 closed/squeezing)
step(n=1)               # advance physics n steps (2 ms each); nothing moves without this
get_ctrl(name)
obj_pos('cube') -> [x,y,z]      palm_pos() -> [x,y,z]      obj_vel('cube') -> [vx,vy,vz]
joint_qpos(name)   finger_opening()   grasped('cube') -> bool   contact_pairs()   obs() -> dict
log(...)   np
```

### Grasp recipe that works on this model

Open fingers (0.05) → centre `slide_x/slide_y` over `obj_pos('cube')` → descend `slide_z` to
~**0.115** (lower jams the fingers into the table and they get friction-pinned) → command both
fingers toward **0.0** so the cube blocks them and creates grip force → check `grasped('cube')`
→ raise `slide_z` to ~0.32 → verify the end state. Step a few hundred steps after each move so
it settles.

## Worked example

`scripts/grab_and_lift.py` is a complete, working reference implementing exactly this contract
(closed loop with retry + an 8000-step failsafe). Read it before writing a new one; for the
default goal you can run it as-is. For a different goal, copy it and change the end-state check
and the motions, keeping the failsafe.

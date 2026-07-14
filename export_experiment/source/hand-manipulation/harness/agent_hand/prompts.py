"""Prompt construction for the code-as-action controller (MuJoCo hand domain).

The model's whole job: read a concrete observation + goal and emit the BODY of a Python
control function (the "skill"). The body runs inside the bridge with the primitives
documented below already in scope. This file is the env-specific surface that replaced the
Minecraft primitives — the controller, envelope, checks, and gen-mode knob are unchanged.
"""
from __future__ import annotations

import json

# Mirror of agent_hand/sim.py's primitive scope — keep in sync. This is the model's API.
PRIMITIVES_DOC = """\
You write the BODY of a Python `def` (do NOT include the signature). The robot is a
floating parallel-jaw gripper over a table. You control it by commanding actuator TARGETS
and then stepping physics so it moves toward them — nothing happens until you `step`.

The following are already in scope:

  // actuation (set a target, then step to let it take effect)
  set_ctrl(name, value)        # name in: 'slide_x','slide_y','slide_z',
                               #          'left_finger','right_finger'
        # slide_x/y in [-0.30,0.30] (palm horizontal position, metres)
        # slide_z   in [ 0.06,0.45] (palm height; ~0.12 puts fingers around a table cube)
        # *_finger  in [ 0.00,0.05] (0.05 = open, 0.0 = fully closed/squeezing)
  step(n=1) -> steps_used      # advance physics n steps (timestep 0.002s). Use a few
                               # hundred steps after a move so it settles. Budget-limited.
  get_ctrl(name) -> float

  // sensing (all positions are [x,y,z] in metres)
  obj_pos('cube') -> [x,y,z]   # the red cube's centre
  palm_pos() -> [x,y,z]
  obj_vel('cube') -> [vx,vy,vz]
  joint_qpos(name) -> float    # e.g. 'slide_z', 'left_finger'
  finger_opening() -> float    # mean finger position (0 closed .. 0.05 open)
  grasped('cube') -> bool      # True when BOTH fingers contact the cube
  contact_pairs() -> [(g1,g2)] # current geom-geom contacts by name
  obs() -> dict                # the full observation dict (same shape you were given)

  // misc
  log(...)   # record a debug line returned in feedback; use it to explain your steps
  np         # numpy

Rules:
- Use only the names above; no import (np is provided), no network, no fs.
- Command a target THEN step — e.g. set_ctrl('slide_z',0.12); step(500). A bare set_ctrl
  with no following step does nothing.
- To grasp: open fingers, centre over the object, descend to graspable height, command
  fingers CLOSED (toward 0.0 — the object blocks them and that creates grip force), then
  lift. Don't drive fingers into the table.
- Make it robust: read obj_pos/grasped before & after, and return when the goal is reached.
- Keep it focused on THIS goal. Return a short string summary at the end.
"""

# A domain-agnostic scaffold for the skeleton gen-mode: the same five beats apply whether
# the body uses Minecraft primitives or robot primitives.
SKELETON = """\
# 1. GUARD: if the goal is already satisfied (check obs()/grasped), return early.
# 2. SENSE: read obj_pos('cube') / palm_pos(); log what you found.
# 3. ACT: set_ctrl targets then step() to move; approach, grasp, then act on the goal.
# 4. VERIFY: re-check the goal condition (grasped, obj_pos height, ...).
# 5. RETURN a short string summary of the outcome."""

RESPONSE_MODES = """\
A goal is either small enough for one short robust script, or too large — in which case you
break it into ordered subgoals (each itself handled by this same loop). Respond with exactly
ONE of these two forms, and nothing else:

(A) LEAF SCRIPT — a ```python block holding the function BODY (as documented above).
    Use this when a single short script can robustly achieve the goal from here.

(B) PLAN — a ```plan block holding JSON of the form:
      {"subgoals": [
        {"goal": "<imperative subgoal>", "check": {<one check clause>}},
        ...
      ]}
    Use this when the goal is too large/uncertain for one script. Order subgoals so each is
    reachable once the earlier ones are done. Each `check` decides when its subgoal is done
    and is ONE of:
      {"expr": "<python bool over `obs`>"}   // e.g. "obs['grasped']" or "obs['cube']['z']>0.1"
      {"llm_judge": "<natural-language success criterion>"}  // when not a clean state expr
    Prefer a precise `expr` over the observation; use llm_judge only when the subgoal isn't a
    clean world-state condition. Keep plans short (a handful of subgoals); don't decompose
    trivially."""

SYSTEM = f"""\
You control a robotic gripper in a MuJoCo simulation by writing bespoke short Python skills
— one per attempt. You do not take primitive actions; you emit either a program that
achieves the goal or a plan that breaks it into subgoals, then read structured feedback
(errors, world-state diff, unmet checks) and revise.

{PRIMITIVES_DOC}

{RESPONSE_MODES}

Respond with ONLY the single fenced block (```python or ```plan). No prose outside it."""


# --- Subgoal judge -----------------------------------------------------------
# Used when a check carries an `llm_judge` clause (a goal not expressible as a clean
# world-state predicate). The judge reads the before/after observation and the NL criterion
# and returns a strict pass/fail. Env-agnostic: the same call works for any domain whose
# observation is a JSON state summary.
JUDGE_SYSTEM = """\
You are a strict evaluator for an embodied agent. Given a success CRITERION and the agent's
observation BEFORE and AFTER an attempt, decide whether the criterion is now satisfied.
Judge only what the observations support; do not assume unobserved success.

Respond with ONLY a JSON object: {"ok": true|false, "reason": "<one short sentence>"}."""


def build_judge_prompt(criterion: str, obs_before: dict, obs_after: dict) -> str:
    return (
        f"# Criterion\n{criterion}\n\n"
        "# Observation BEFORE\n```json\n" + json.dumps(obs_before, indent=2) + "\n```\n\n"
        "# Observation AFTER\n```json\n" + json.dumps(obs_after, indent=2) + "\n```\n\n"
        "Is the criterion satisfied in the AFTER state?"
    )


def build_user_prompt(goal: str, observation: dict, *, gen_mode: str = "scratch",
                      starting_skill=None, can_decompose: bool = True,
                      last_attempt: dict | None = None) -> str:
    """Assemble the per-node user prompt.

    gen_mode controls the *starting point* offered to the model (a comparable knob):
      - "scratch":      blank page, write from the goal + observation.
      - "edit-nearest": start by editing the most similar library skill (falls back to the
                        skeleton when the library has no match).
      - "skeleton":     fill in a fixed five-beat scaffold.
    can_decompose is False at the depth limit: the model must return a leaf script.
    """
    parts = [f"# Goal\n{goal}\n"]
    parts.append("# Current observation\n```json\n" + json.dumps(observation, indent=2) + "\n```\n")

    if gen_mode == "edit-nearest" and starting_skill is not None:
        parts.append("# Starting point — EDIT this existing skill to fit the goal and observation")
        parts.append(f"## {starting_skill.name} — {starting_skill.description}\n"
                     f"```python\n{starting_skill.code}\n```\n")
    elif gen_mode == "skeleton" or (gen_mode == "edit-nearest" and starting_skill is None):
        note = "" if gen_mode == "skeleton" else " (no similar skill in library — use this scaffold)"
        parts.append(f"# Starting point — FILL IN this skeleton{note}")
        parts.append(f"```python\n{SKELETON}\n```\n")
    # gen_mode == "scratch": no starting code offered.

    if not can_decompose:
        parts.append("# Constraint\nYou are at the decomposition depth limit: respond with a "
                     "```python LEAF SCRIPT only (no ```plan).\n")

    if last_attempt:
        fb = last_attempt
        status = "succeeded" if fb.get("success") else "FAILED"
        parts.append(f"# Your previous attempt {status}")
        if fb.get("error"):
            parts.append("Error:\n```\n" + json.dumps(fb["error"], indent=2) + "\n```")
        if fb.get("logs"):
            parts.append("Logs:\n```\n" + "\n".join(fb["logs"]) + "\n```")
        if fb.get("stateDiff"):
            parts.append("World change: " + json.dumps(fb["stateDiff"]))
        if fb.get("unmet"):
            parts.append("Goal still unmet because: " + "; ".join(fb["unmet"]))
        parts.append("\nFix the issue and write an improved skill.")
    else:
        parts.append("Write a skill to achieve the goal from the current state.")

    return "\n".join(parts)

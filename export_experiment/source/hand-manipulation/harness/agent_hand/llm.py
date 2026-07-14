"""LLM clients that turn (system, user) prompts into a Python control skill.

- AnthropicLLM: real controller. Caches the large static system prompt so repeated repair
  iterations within/across episodes hit the cache.
- MockLLM: returns a canned skill so the full pipeline can be exercised offline (no API
  key). The canned lift_cube skill is the open-loop sequence verified against the model.
"""
from __future__ import annotations

import json
import os
import re

from . import prompts

DEFAULT_MODEL = os.environ.get("AGENT_HAND_MODEL", "claude-opus-4-8")
# A smaller/cheaper model is plenty for the pass/fail judge; override via env.
JUDGE_MODEL = os.environ.get("AGENT_HAND_JUDGE_MODEL", "claude-haiku-4-5-20251001")

_CODE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)
_PLAN_RE = re.compile(r"```(?:plan|json)\s*\n(.*?)```", re.DOTALL)
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def extract_code(text: str) -> str:
    """Pull the Python function body out of a fenced code block (or take the whole text)."""
    m = _CODE_RE.search(text)
    return (m.group(1) if m else text).strip()


def extract_envelope(text: str) -> dict:
    """Classify a node response as a plan or a leaf script.

    Returns {"type": "plan", "subgoals": [...]} when a ```plan/```json block holds a
    well-formed {"subgoals": [...]}; otherwise {"type": "script", "code": <python body>}.
    """
    m = _PLAN_RE.search(text)
    if m:
        try:
            data = json.loads(m.group(1))
        except json.JSONDecodeError:
            data = None
        if isinstance(data, dict) and isinstance(data.get("subgoals"), list):
            subgoals = [s for s in data["subgoals"] if isinstance(s, dict) and s.get("goal")]
            if subgoals:
                return {"type": "plan", "subgoals": subgoals}
    return {"type": "script", "code": extract_code(text)}


class AnthropicLLM:
    def __init__(self, model: str | None = None, max_tokens: int = 2000):
        import anthropic  # imported lazily so MockLLM works without the dep
        self.client = anthropic.Anthropic()
        self.model = model or DEFAULT_MODEL
        self.max_tokens = max_tokens

    def generate(self, system: str, user: str) -> str:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in resp.content if b.type == "text")

    def judge(self, criterion: str, obs_before: dict, obs_after: dict) -> tuple[bool, str]:
        resp = self.client.messages.create(
            model=JUDGE_MODEL,
            max_tokens=200,
            system=[{"type": "text", "text": prompts.JUDGE_SYSTEM,
                     "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user",
                       "content": prompts.build_judge_prompt(criterion, obs_before, obs_after)}],
        )
        text = "".join(b.text for b in resp.content if b.type == "text")
        return _parse_judge(text)


def _parse_judge(text: str) -> tuple[bool, str]:
    """Parse a {"ok":..., "reason":...} judge reply; fail closed on garbage."""
    m = _JSON_RE.search(text)
    if not m:
        return (False, f"unparseable judge reply: {text[:120]}")
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return (False, f"invalid judge JSON: {text[:120]}")
    return (bool(data.get("ok")), str(data.get("reason", "")))


class MockLLM:
    """Deterministic stand-in. Returns a skill keyed off the goal text."""

    def __init__(self, skill: str | None = None):
        self._skill = skill

    def generate(self, system: str, user: str) -> str:
        if self._skill is not None:
            return f"```python\n{self._skill}\n```"
        goal = user.lower()
        if "lift" in goal or "pick up" in goal:
            return "```python\n" + _CANNED_LIFT_CUBE + "\n```"
        return "```python\nlog('mock: no canned skill for this goal')\nreturn 'noop'\n```"

    def judge(self, criterion: str, obs_before: dict, obs_after: dict) -> tuple[bool, str]:
        # Offline stand-in: assume the attempt met the criterion so --mock pipelines flow.
        return (True, "mock judge: assumed pass")


# Open-loop grasp-and-lift verified against gripper_cube.xml; the repair loop can improve it.
_CANNED_LIFT_CUBE = """\
# Grasp the red cube and lift it clear of the table.
if grasped('cube') and obj_pos('cube')[2] > 0.12:
    return 'already lifted'
cube = obj_pos('cube')
# 1. open the gripper and centre the palm over the cube, up high.
set_ctrl('left_finger', 0.05); set_ctrl('right_finger', 0.05)
set_ctrl('slide_x', cube[0]); set_ctrl('slide_y', cube[1]); set_ctrl('slide_z', 0.30)
step(300)
# 2. descend so the fingers straddle the cube (not so low they jam into the table).
set_ctrl('slide_z', 0.12)
step(500)
# 3. squeeze: command fingers fully closed — the cube blocks them, creating grip force.
set_ctrl('left_finger', 0.0); set_ctrl('right_finger', 0.0)
step(400)
log('grasped after squeeze:', grasped('cube'))
# 4. lift straight up and verify.
set_ctrl('slide_z', 0.30)
step(800)
log('cube z:', obj_pos('cube')[2])
return 'cube_z=%.3f grasped=%s' % (obj_pos('cube')[2], grasped('cube'))
"""

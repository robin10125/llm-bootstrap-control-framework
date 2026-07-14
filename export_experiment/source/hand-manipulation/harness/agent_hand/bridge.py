"""In-process bridge: runs a generated Python skill against the MuJoCo sim.

Implements the same surface the controller expects from the Minecraft bridge
(`wait_until_spawned` / `reset` / `observe` / `execute`), so controller.py is identical
across the two domains. Unlike Minecraft (a separate Node process over HTTP), the robot
sim is Python, so the bridge runs in-process — but the *contract* (a skill is a function
body executed in a fixed primitive scope, returning structured feedback) is the same.

The structured feedback channel is the core research instrument: it reports WHY a skill
failed (exception + traceback, or budget exhaustion) and WHAT changed in the world
(observation before/after + a compact diff), which the repair loop consumes.
"""
from __future__ import annotations

import textwrap
import traceback
from typing import Any

from .sim import HandSim, StepBudgetExceeded


class Bridge:
    def __init__(self, sim: HandSim | None = None):
        self.sim = sim or HandSim()

    # -- controller-facing API (mirrors the Minecraft BridgeClient) -----------
    def wait_until_spawned(self) -> None:
        return  # the in-process sim is ready as soon as it is constructed

    def reset(self, **setup: Any) -> dict:
        self.sim.reset(**setup)
        return {"ok": True}

    def observe(self) -> dict:
        return self.sim.observe()

    def enable_recording(self) -> None:
        self.sim.enable_recording()

    def save_trajectory(self, path) -> int:
        return self.sim.save_trajectory(path)

    def execute(self, code: str, timeout_ms: int = 0) -> dict:
        """Run a generated skill (a function body) and return structured feedback.

        `timeout_ms`, if >0, is converted to a step budget (timestep 0.002s); otherwise the
        sim's default budget applies. The skill's `return` value becomes feedback["result"].
        """
        budget = int(timeout_ms / 2) if timeout_ms else None  # 2 ms per physics step
        self.sim.begin_execution(budget)
        obs_before = self.sim.observe()

        fn = self._compile(code)
        feedback: dict[str, Any] = {
            "success": False, "result": None, "logs": None, "error": None,
            "observationBefore": obs_before, "observationAfter": obs_before,
            "stateDiff": {}, "stepsUsed": 0,
        }
        if fn is None:
            feedback["error"] = {"type": "SyntaxError",
                                 "message": "generated skill did not compile"}
            return feedback

        try:
            result = fn()
            feedback["success"] = True
            feedback["result"] = None if result is None else str(result)
        except StepBudgetExceeded as e:
            feedback["error"] = {"type": "StepBudgetExceeded", "message": str(e)}
        except Exception as e:  # noqa: BLE001 — surface any skill error as feedback
            feedback["error"] = {
                "type": type(e).__name__, "message": str(e),
                "traceback": traceback.format_exc(limit=4),
            }

        obs_after = self.sim.observe()
        feedback["logs"] = list(self.sim.logs)
        feedback["observationAfter"] = obs_after
        feedback["stateDiff"] = _diff(obs_before, obs_after)
        feedback["stepsUsed"] = obs_after.get("t", 0)
        return feedback

    # -- internals ------------------------------------------------------------
    def _compile(self, code: str):
        """Wrap a function BODY into a callable with the primitives in scope."""
        body = textwrap.indent(code.strip() or "pass", "    ")
        src = "def _skill():\n" + body + "\n"
        ns: dict[str, Any] = dict(self._scope())
        try:
            exec(compile(src, "<skill>", "exec"), ns)  # noqa: S102 — trusted experiment code
        except SyntaxError:
            return None
        return ns["_skill"]

    def _scope(self) -> dict[str, Any]:
        s = self.sim
        import numpy as np
        return {
            # actuation + time
            "set_ctrl": s.set_ctrl, "get_ctrl": s.get_ctrl, "step": s.step,
            "actuator_names": s.actuator_names,
            # sensing
            "obs": s.observe, "palm_pos": s.palm_pos, "obj_pos": s.obj_pos,
            "obj_vel": s.obj_vel, "site_pos": s.site_pos, "body_pos": s.body_pos,
            "joint_qpos": s.joint_qpos, "finger_opening": s.finger_opening,
            "grasped": s.grasped, "contact_pairs": s.contact_pairs,
            # misc
            "log": s.log, "np": np,
        }


def _diff(before: dict, after: dict) -> dict:
    """Compact, human-readable change summary for the repair prompt."""
    out: dict[str, Any] = {}
    bz, az = before["cube"]["z"], after["cube"]["z"]
    if abs(az - bz) > 1e-3:
        out["cube_z"] = {"from": bz, "to": az, "delta": round(az - bz, 4)}
    if before["grasped"] != after["grasped"]:
        out["grasped"] = {"from": before["grasped"], "to": after["grasped"]}
    bo, ao = before["fingers"]["opening"], after["fingers"]["opening"]
    if abs(ao - bo) > 1e-3:
        out["finger_opening"] = {"from": bo, "to": ao}
    bp, ap = before["palm"]["pos"], after["palm"]["pos"]
    if any(abs(a - b) > 1e-3 for a, b in zip(ap, bp)):
        out["palm_pos"] = {"from": bp, "to": ap}
    return out

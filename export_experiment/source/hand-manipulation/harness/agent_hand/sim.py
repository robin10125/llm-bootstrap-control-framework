"""MuJoCo simulation wrapper + the primitive surface a generated script controls.

This is the env-specific half of the experiment (the analog of the Minecraft bridge's
primitive vocabulary). The harness loop in controller.py never imports this file; it only
talks to a Bridge that talks to a HandSim. Swapping the gripper for a dexterous hand from
mujoco_menagerie means a new model XML + adjusting the names below, not touching the loop.

A generated "skill" is the BODY of a Python function. It runs with the primitives in this
module in scope and drives the robot by the set-target-then-step pattern:

    set_ctrl("slide_z", 0.12)   # command an actuator toward a target
    step(400)                   # advance physics so it gets there
    if grasped("cube"): ...

`step` draws from a per-execution budget so a runaway script fails with structured
feedback instead of hanging (the budget is the robot analog of the bridge's exec timeout).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import mujoco
import numpy as np

MODEL_PATH = Path(__file__).resolve().parents[2] / "env" / "models" / "gripper_cube.xml"

# Default ceiling on physics steps per execute() call (timestep 0.002s -> 40 sim-seconds).
DEFAULT_STEP_BUDGET = 20000


class StepBudgetExceeded(RuntimeError):
    pass


class HandSim:
    def __init__(self, model_path: str | Path = MODEL_PATH):
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        self._home_key = self._maybe_key("home")
        self._steps_used = 0
        self._total_steps = 0          # monotonic across executes, for the recorded timeline
        self._budget = DEFAULT_STEP_BUDGET
        self._logs: list[str] = []
        # control-timeline recording for the live step-through viewer (off until enabled).
        # The episode is reproducible from (initial state + piecewise-constant ctrl), so the
        # viewer re-integrates real physics on demand rather than replaying stored positions.
        self._recording = False
        self._ctrl_steps: list[int] = []   # total-step index at each set_ctrl
        self._ctrl_vals: list = []         # full ctrl vector at that point
        self._init_qpos = None
        self._init_qvel = None
        self.reset()

    # -- lifecycle ------------------------------------------------------------
    def _maybe_key(self, name: str) -> int | None:
        try:
            return self.model.key(name).id
        except KeyError:
            return None

    def reset(self, cube_pos: list[float] | None = None, settle: int = 100,
              **_ignored: Any) -> None:
        """Reset to the home keyframe, optionally repositioning the cube, then settle.

        `cube_pos` is [x, y] (z is forced to rest on the table). Unknown setup keys are
        ignored so task YAML can carry env-specific fields without breaking the reset.
        """
        if self._home_key is not None:
            mujoco.mj_resetDataKeyframe(self.model, self.data, self._home_key)
        else:
            mujoco.mj_resetData(self.model, self.data)
        if cube_pos is not None:
            adr = self.model.joint("cube").qposadr[0]
            self.data.qpos[adr] = float(cube_pos[0])
            self.data.qpos[adr + 1] = float(cube_pos[1])
        mujoco.mj_forward(self.model, self.data)
        for _ in range(settle):
            mujoco.mj_step(self.model, self.data)

    # -- execution budget -----------------------------------------------------
    def begin_execution(self, budget: int | None = None) -> None:
        self._steps_used = 0
        self._budget = budget or DEFAULT_STEP_BUDGET
        self._logs = []

    @property
    def logs(self) -> list[str]:
        return self._logs

    # -- control-timeline recording (for the live step-through viewer) --------
    def enable_recording(self) -> None:
        """Pin the current state as the timeline origin and start logging ctrl changes."""
        self._recording = True
        self._total_steps = 0
        self._init_qpos = self.data.qpos.copy()
        self._init_qvel = self.data.qvel.copy()
        self._ctrl_steps = [0]
        self._ctrl_vals = [self.data.ctrl.copy()]

    def save_trajectory(self, path: str | Path) -> int:
        """Write the initial state + ctrl timeline to an .npz; returns total physics steps."""
        if not self._recording:
            return 0
        np.savez(
            str(path),
            init_qpos=self._init_qpos, init_qvel=self._init_qvel,
            ctrl_steps=np.array(self._ctrl_steps),
            ctrl_vals=np.array(self._ctrl_vals),
            total_steps=np.array(self._total_steps),
        )
        return int(self._total_steps)

    # -- primitives exposed to generated scripts ------------------------------
    def step(self, n: int = 1) -> int:
        n = int(n)
        if self._steps_used + n > self._budget:
            raise StepBudgetExceeded(
                f"step budget exhausted ({self._budget} steps); the script is taking too "
                "long — act in fewer/larger steps or check why a target is never reached")
        for _ in range(n):
            mujoco.mj_step(self.model, self.data)
            self._total_steps += 1
        self._steps_used += n
        return self._steps_used

    def set_ctrl(self, name: str, value: float) -> None:
        act = self.model.actuator(name)  # raises KeyError on a bad name
        lo, hi = act.ctrlrange
        self.data.ctrl[act.id] = float(np.clip(value, lo, hi))
        if self._recording:
            self._ctrl_steps.append(self._total_steps)
            self._ctrl_vals.append(self.data.ctrl.copy())

    def get_ctrl(self, name: str) -> float:
        return float(self.data.ctrl[self.model.actuator(name).id])

    def actuator_names(self) -> list[str]:
        return [self.model.actuator(i).name for i in range(self.model.nu)]

    def joint_qpos(self, name: str) -> float:
        return float(self.data.joint(name).qpos[0])

    def site_pos(self, name: str) -> list[float]:
        return self.data.site(name).xpos.round(4).tolist()

    def body_pos(self, name: str) -> list[float]:
        return self.data.body(name).xpos.round(4).tolist()

    def palm_pos(self) -> list[float]:
        return self.body_pos("palm")

    def obj_pos(self, name: str = "cube") -> list[float]:
        return self.body_pos(name)

    def obj_vel(self, name: str = "cube") -> list[float]:
        adr = self.model.joint(name).dofadr[0]
        return self.data.qvel[adr:adr + 3].round(4).tolist()

    def finger_opening(self) -> float:
        return round((self.joint_qpos("left_finger") + self.joint_qpos("right_finger")) / 2, 4)

    def _geom_name(self, gid: int) -> str:
        return self.model.geom(gid).name or f"geom{gid}"

    def contact_pairs(self) -> list[tuple[str, str]]:
        seen: set[tuple[str, str]] = set()
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            seen.add(tuple(sorted((self._geom_name(c.geom1), self._geom_name(c.geom2)))))
        return sorted(seen)

    def grasped(self, name: str = "cube") -> bool:
        """True when BOTH fingers are touching `name` (a stable two-sided grip)."""
        touching = {a for pair in self.contact_pairs() if name in pair for a in pair}
        return "left_finger" in touching and "right_finger" in touching

    def log(self, *args: Any) -> None:
        self._logs.append(" ".join(str(a) for a in args))

    # -- observation ----------------------------------------------------------
    def observe(self) -> dict:
        cube = self.obj_pos("cube")
        return {
            "t": self._steps_used,
            "palm": {"pos": self.palm_pos(), "height": round(self.joint_qpos("slide_z"), 4)},
            "fingers": {
                "left": round(self.joint_qpos("left_finger"), 4),
                "right": round(self.joint_qpos("right_finger"), 4),
                "opening": self.finger_opening(),
            },
            "cube": {"pos": cube, "z": cube[2], "vel": self.obj_vel("cube")},
            "grasped": self.grasped("cube"),
            "actuators": {n: round(self.get_ctrl(n), 4) for n in self.actuator_names()},
            "contacts": [list(p) for p in self.contact_pairs()],
        }

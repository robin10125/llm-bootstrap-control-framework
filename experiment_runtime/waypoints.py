#!/usr/bin/env python3
"""LLM control interface: waypoint plans compiled to a per-step action stream.

The LLM does not emit hundreds of per-step joint targets. It emits a short **waypoint
plan** — a few via-poses with timings — which this module compiles into the *same*
per-step action stream the RL policy outputs, then rolls out in MJX. The rollout returns
both a task score (to decide whether the LLM beat the current policy) and the
`(obs, action)` pairs used as behavior-cloning demonstrations.

The compiler is behind a small interface (`Compiler`) so other LLM control representations
(raw per-step targets, the old primitive language) can be swapped in later without touching
the trainer.

Gripper plan schema (JSON the LLM produces):

    {"waypoints": [
        {"t": 0.0, "pos": [x, y, z], "open": 1.0},   # open hand, above object
        {"t": 1.2, "pos": [x, y, z], "open": 1.0},   # descend
        {"t": 1.8, "pos": [x, y, z], "open": 0.0},   # close
        {"t": 3.0, "pos": [x, y, z], "open": 0.0}    # lift
    ]}

`pos` are base actuator targets (base_x, base_y, base_z within their ctrlrange); `open` in
[0, 1] is finger openness (1 = fully open, 0 = closed). Times are clamped to the episode.
"""
from __future__ import annotations

from typing import Any, Protocol

import jax
import jax.numpy as jp
import numpy as np

from experiment_runtime import eval_metrics as EM
from experiment_runtime.environment import MjxEnv


class PlanError(ValueError):
    pass


class Compiler(Protocol):
    def compile(self, plan: dict[str, Any], env: MjxEnv) -> np.ndarray:
        """Return per-control-step ctrl targets, shape [horizon, nu]."""
        ...


class WaypointCompiler:
    """Compile a gripper waypoint plan into per-step ctrl targets (host-side interp)."""

    def __init__(self, env: MjxEnv):
        m = env.model
        self.env = env
        self.nu = m.nu
        names = [m.actuator(i).name for i in range(m.nu)]
        self.idx = {n: i for i, n in enumerate(names)}
        if not {"base_x", "base_y", "base_z"} <= self.idx.keys():
            raise PlanError(f"scene lacks base actuators; have {names}")
        self.lo = m.actuator_ctrlrange[:, 0].copy()
        self.hi = m.actuator_ctrlrange[:, 1].copy()
        # Open/closed control targets come from the env (gripper finger range or, for an
        # articulated hand, the keyframe-derived grasp poses). The single `open` scalar
        # interpolates the whole hand between them; the LLM only commands base pose + grasp
        # fraction, the same abstraction for the gripper and the Shadow Hand.
        self.ctrl_open = np.asarray(jax.device_get(env.ctrl_open))
        self.ctrl_close = np.asarray(jax.device_get(env.ctrl_close))
        self.hand_ids = list(env.hand_act_ids)

    def _target_vec(self, pos, opening: float) -> np.ndarray:
        t = self.ctrl_open.copy()
        frac = float(np.clip(opening, 0.0, 1.0))  # 1 = open, 0 = closed
        for i in self.hand_ids:
            t[i] = frac * self.ctrl_open[i] + (1.0 - frac) * self.ctrl_close[i]
        t[self.idx["base_x"]] = pos[0]
        t[self.idx["base_y"]] = pos[1]
        t[self.idx["base_z"]] = pos[2]
        return np.clip(t, self.lo, self.hi)

    def compile(self, plan: dict[str, Any], env: MjxEnv) -> np.ndarray:
        wps = plan.get("waypoints")
        if not isinstance(wps, list) or len(wps) < 1:
            raise PlanError("plan must have a non-empty 'waypoints' list")
        ep_T = env.horizon
        dt = env.cfg.control_dt
        ts, vecs = [], []
        for w in wps:
            if not {"t", "pos", "open"} <= set(w):
                raise PlanError(f"waypoint needs t/pos/open: {w}")
            pos = [float(v) for v in w["pos"]]
            if len(pos) != 3:
                raise PlanError(f"pos must be [x,y,z]: {w}")
            ts.append(float(w["t"]))
            vecs.append(self._target_vec(pos, float(w["open"])))
        order = np.argsort(ts)
        ts = np.asarray(ts)[order]
        vecs = np.asarray(vecs)[order]

        times = np.arange(ep_T) * dt
        targets = np.empty((ep_T, self.nu), dtype=np.float32)
        for d in range(self.nu):
            targets[:, d] = np.interp(times, ts, vecs[:, d])
        return targets


def make_plan_rollout(env: MjxEnv):
    """Return a jitted single-env rollout driven by precomputed ctrl targets.

    The action that reaches each target is the rate-limited delta the policy would also
    output, so the recorded (obs, action) pairs are valid BC demonstrations.
    """
    scale = env.cfg.action_scale

    def rollout(state, targets, _key):
        def body(state, target):
            cur = state.data.ctrl
            action = jp.clip((target - cur) / scale, -1.0, 1.0)
            nstate = env.step(state, action)
            return nstate, (state.obs, action, nstate.reward, nstate.metrics["success"],
                            nstate.metrics["lift"], nstate.metrics["eval"])

        state, (obs, act, rew, succ, lift, ev) = jax.lax.scan(body, state, targets)
        return {
            "obs": obs, "action": act,
            "return": rew.sum(), "success": (succ.max() > 0.5).astype(jp.float32),
            "lift_max": lift.max(), "eval_summary": EM.reduce_summary(ev),
        }

    return jax.jit(rollout)


def scripted_lift_plan(env: MjxEnv, obj_xy=(0.0, 0.0)) -> dict[str, Any]:
    """A hand-written lift plan; doubles as the scripted-expert baseline and an env sanity
    check (the env is solvable if this lifts)."""
    x, y = float(obj_xy[0]), float(obj_xy[1])
    z_lo, z_hi = env.model.actuator("base_z").ctrlrange  # noqa: F841
    descend = min(0.16, float(z_hi))  # positive base_z lowers the palm onto the object
    return {"waypoints": [
        {"t": 0.0, "pos": [x, y, 0.0], "open": 1.0},
        {"t": 0.9, "pos": [x, y, descend], "open": 1.0},
        {"t": 1.5, "pos": [x, y, descend], "open": 0.0},
        {"t": 2.2, "pos": [x, y, 0.0], "open": 0.0},
        {"t": 3.0, "pos": [x, y, 0.0], "open": 0.0},
    ]}


def _selftest(n: int = 8) -> None:
    import numpy as np
    env = MjxEnv()
    comp = WaypointCompiler(env)
    roll = make_plan_rollout(env)
    reset = jax.jit(env.reset)
    succdraws, lifts = [], []
    for s in range(n):
        key = jax.random.PRNGKey(s)
        state = reset(key)
        obj = np.asarray(jax.device_get(state.data.xpos[env.object_bid]))
        plan = scripted_lift_plan(env, (obj[0], obj[1]))
        targets = jp.asarray(comp.compile(plan, env))
        out = roll(state, targets, key)
        succdraws.append(float(out["success"]))
        lifts.append(float(out["lift_max"]))
        if s == 0:
            print(f"obs_seq={out['obs'].shape} act_seq={out['action'].shape} "
                  f"frame_skip={env.frame_skip} horizon={env.horizon} dt={env.model.opt.timestep}")
    print(f"scripted lift over {n} setups: success={np.mean(succdraws):.2f} "
          f"lift_max_mean={np.mean(lifts):.3f} (per-setup lifts: "
          f"{', '.join(f'{l:.3f}' for l in lifts)})")


if __name__ == "__main__":
    _selftest()

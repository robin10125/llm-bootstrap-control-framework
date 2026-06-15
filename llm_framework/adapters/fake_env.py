from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np


class FakeActuator:
    def __init__(self, name: str, idx: int, ctrlrange: tuple[float, float]):
        self.name = name
        self.id = idx
        self.ctrlrange = ctrlrange


class FakeBody:
    def __init__(self, idx: int):
        self.id = idx


class FakeModel:
    def __init__(self):
        self.names = ["base_x", "base_y", "base_z", "rh_A_FFJ4", "rh_A_THJ5"]
        self.actuator_ctrlrange = np.array([
            [-0.30, 0.30],
            [-0.30, 0.30],
            [-0.15, 0.20],
            [0.00, 1.00],
            [0.00, 1.00],
        ], dtype=np.float32)

    def actuator(self, key: int | str) -> FakeActuator:
        if isinstance(key, int):
            return FakeActuator(self.names[key], key, tuple(self.actuator_ctrlrange[key]))
        idx = self.names.index(key)
        return FakeActuator(key, idx, tuple(self.actuator_ctrlrange[idx]))

    def body(self, name: str) -> FakeBody:
        if name != "object":
            raise KeyError(name)
        return FakeBody(0)


@dataclass
class FakeData:
    xpos: np.ndarray
    cvel: np.ndarray
    qpos: np.ndarray
    ctrl: np.ndarray


@dataclass
class FakeState:
    data: FakeData
    reward: float
    step: int


class FakeEnv:
    """Small deterministic env for framework integration tests.

    It is not a physics model. It only verifies that interfaces produce bounded action
    streams and that the experiment runner can score/report them end to end.
    """

    is_fake_env = True

    def __init__(self, *, episode_seconds: float = 3.0, control_dt: float = 0.1):
        self.model = FakeModel()
        self.nu = len(self.model.names)
        self.cfg = SimpleNamespace(action_scale=0.05, control_dt=control_dt, episode_seconds=episode_seconds)
        self.horizon = max(1, round(episode_seconds / control_dt))
        self.ctrl_open = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.ctrl_close = np.array([0.0, 0.0, 0.0, 1.0, 1.0], dtype=np.float32)
        self.hand_act_ids = [3, 4]
        self.object_bid = 0
        self.base_qadr = [0, 1, 2]
        self.hand_qadr: list[int] = []

    def reset(self, seed: int = 0) -> FakeState:
        rng = np.random.default_rng(int(seed))
        obj = np.array([rng.uniform(-0.03, 0.03), rng.uniform(-0.03, 0.03), 0.025], dtype=np.float32)
        return FakeState(
            data=FakeData(
                xpos=obj[None, :].copy(),
                cvel=np.zeros((1, 6), dtype=np.float32),
                qpos=np.zeros(3, dtype=np.float32),
                ctrl=self.ctrl_open.copy(),
            ),
            reward=0.0,
            step=0,
        )

    def step(self, state: FakeState, action: np.ndarray) -> FakeState:
        lo = self.model.actuator_ctrlrange[:, 0]
        hi = self.model.actuator_ctrlrange[:, 1]
        ctrl = np.clip(state.data.ctrl + np.clip(action, -1.0, 1.0) * self.cfg.action_scale, lo, hi)
        obj = state.data.xpos[0].copy()
        base_xy = ctrl[:2]
        closed = float(ctrl[3:].mean()) > 0.55
        near = float(np.linalg.norm(base_xy - obj[:2])) < 0.07
        if closed and near:
            obj[:2] += 0.25 * (base_xy - obj[:2])
            obj[2] = max(obj[2], 0.025 + max(0.0, 0.16 - float(ctrl[2])) * 0.5)
        reward = float(obj[2] - 0.025)
        return FakeState(
            data=FakeData(
                xpos=obj[None, :],
                cvel=np.zeros((1, 6), dtype=np.float32),
                qpos=ctrl[:3].copy(),
                ctrl=ctrl.astype(np.float32),
            ),
            reward=reward,
            step=state.step + 1,
        )


def make_fake_env(**overrides: Any) -> FakeEnv:
    return FakeEnv(**overrides)

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from llm_framework.core.state import SafetyLimits, WorldState
from llm_framework.core.tasks import build_task_context
from llm_framework.interfaces.hybrid import HybridInterface
from llm_framework.interfaces.latent import LatentStubInterface
from llm_framework.interfaces.recursive_units import RecursiveUnitsInterface
from llm_framework.interfaces.script_dsl import ScriptDSLInterface
from llm_framework.llm.worker import complete_for_interface
from llm_framework.runtime.dsl_interpreter import compile_dsl


class FakeActuator:
    def __init__(self, name: str, idx: int, ctrlrange: tuple[float, float]):
        self.name = name
        self.id = idx
        self.ctrlrange = ctrlrange


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

    def actuator(self, key):
        if isinstance(key, int):
            return FakeActuator(self.names[key], key, tuple(self.actuator_ctrlrange[key]))
        idx = self.names.index(key)
        return FakeActuator(key, idx, tuple(self.actuator_ctrlrange[idx]))


class FakeEnv:
    def __init__(self):
        self.model = FakeModel()
        self.nu = 5
        self.horizon = 12
        self.cfg = SimpleNamespace(action_scale=0.05, control_dt=0.1)
        self.ctrl_open = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.ctrl_close = np.array([0.0, 0.0, 0.0, 1.0, 1.0], dtype=np.float32)
        self.hand_act_ids = [3, 4]


def fake_world() -> WorldState:
    return WorldState(
        time_s=0.0,
        object_pos=np.array([0.02, -0.01, 0.025], dtype=np.float32),
        object_vel=np.zeros(3, dtype=np.float32),
        base_q=np.zeros(3, dtype=np.float32),
        hand_q=np.zeros(2, dtype=np.float32),
        ctrl=np.zeros(5, dtype=np.float32),
        ctrl_lo=np.array([-0.3, -0.3, -0.15, 0.0, 0.0], dtype=np.float32),
        ctrl_hi=np.array([0.3, 0.3, 0.2, 1.0, 1.0], dtype=np.float32),
        actuator_names=["base_x", "base_y", "base_z", "rh_A_FFJ4", "rh_A_THJ5"],
        appendages={
            "base": ["base_x", "base_y", "base_z"],
            "index": ["rh_A_FFJ4"],
            "thumb": ["rh_A_THJ5"],
            "hand": ["rh_A_FFJ4", "rh_A_THJ5"],
        },
        joint_schema={},
        derived={},
    )


def test_mock_interfaces_compile_to_action_stream() -> None:
    env = FakeEnv()
    world = fake_world()
    ctx = build_task_context("lift", 0, world, episode_seconds=1.2)

    for interface in (ScriptDSLInterface(), HybridInterface(), LatentStubInterface()):
        program = interface.parse(interface.mock_response(ctx, world))
        validation = interface.validate(program, SafetyLimits(max_episode_seconds=1.2))
        assert validation.ok, validation.errors

        compiled = interface.compile(program, ctx, world, env)

        assert compiled.action_stream.shape == (env.horizon, env.nu)
        assert compiled.action_stream.min() >= -1.0
        assert compiled.action_stream.max() <= 1.0


def test_appendage_agent_targets_only_its_joints() -> None:
    env = FakeEnv()
    world = fake_world()
    ctx = build_task_context("lift", 0, world, episode_seconds=1.2)
    interface = HybridInterface()
    program = interface.parse("""
    {"blocks": [{
      "op": "call_appendage_agent",
      "appendage": "index",
      "duration_s": 0.2,
      "program": {"blocks": [
        {"op": "set_joint_target", "joint": "rh_A_FFJ4", "target": 0.7},
        {"op": "set_joint_target", "joint": "rh_A_THJ5", "target": 1.0}
      ]}
    }]}
    """)

    validation = interface.validate(program, SafetyLimits(max_episode_seconds=1.2))
    assert validation.ok, validation.errors
    compiled = compile_dsl(program, env, world)

    assert compiled.action_stream[:, 3].max() > 0.0
    assert compiled.action_stream[:, 4].max() == 0.0


def test_recursive_units_mock_returns_trace_and_compiles() -> None:
    env = FakeEnv()
    world = fake_world()
    ctx = build_task_context("lift", 0, world, episode_seconds=1.2)
    interface = RecursiveUnitsInterface()

    completion = complete_for_interface(interface, ctx, world, backend="mock")

    assert completion.ok
    program = interface.parse(completion.text)
    assert "recursive_trace" in program.source
    validation = interface.validate(program, SafetyLimits(max_episode_seconds=1.2))
    assert validation.ok, validation.errors
    compiled = interface.compile(program, ctx, world, env)
    assert compiled.action_stream.shape == (env.horizon, env.nu)
    assert compiled.metadata["recursive_units"] is True


def test_recursive_units_sorts_phase_blocks_before_compile() -> None:
    env = FakeEnv()
    world = fake_world()
    ctx = build_task_context("lift", 0, world, episode_seconds=1.2)
    interface = RecursiveUnitsInterface()
    program = interface.parse("""
    {"blocks": [
      {"phase": "lift_or_transport", "op": "set_joint_target", "joint": "base_z", "target": -0.1, "duration_s": 0.2},
      {"phase": "close_until_touch_or_settle", "op": "set_joint_target", "joint": "rh_A_FFJ4", "target": 0.7, "duration_s": 0.2},
      {"phase": "approach", "op": "set_joint_target", "joint": "base_x", "target": 0.02, "duration_s": 0.2},
      {"op": "return"}
    ]}
    """)

    assert [block.get("phase") for block in program.source["blocks"][:3]] == [
        "approach",
        "close_until_touch_or_settle",
        "lift_or_transport",
    ]
    validation = interface.validate(program, SafetyLimits(max_episode_seconds=1.2))
    assert validation.ok, validation.errors
    compiled = interface.compile(program, ctx, world, env)
    assert compiled.metadata["trace"][0]["op"] == "set_joint_target"


def test_recursive_units_expands_generated_primitive() -> None:
    env = FakeEnv()
    world = fake_world()
    ctx = build_task_context("lift", 0, world, episode_seconds=1.2)
    interface = RecursiveUnitsInterface()
    program = interface.parse("""
    {
      "generated_primitives": {
        "close_until_touching": {
          "blocks": [
            {"phase": "close_until_touch_or_settle", "op": "set_joint_target", "joint": "$joint", "target": "$target", "duration_s": 0.2},
            {"phase": "verify_contact_or_settle", "op": "monitor", "duration_s": 0.1}
          ]
        }
      },
      "blocks": [
        {"op": "close_until_touching", "params": {"joint": "rh_A_FFJ4", "target": 0.7}},
        {"op": "return"}
      ]
    }
    """)

    assert program.source["blocks"][0]["op"] == "set_joint_target"
    validation = interface.validate(program, SafetyLimits(max_episode_seconds=1.2))
    assert validation.ok, validation.errors
    compiled = interface.compile(program, ctx, world, env)
    assert compiled.action_stream[:, 3].max() > 0.0

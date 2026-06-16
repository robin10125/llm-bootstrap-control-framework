from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from llm_framework.core.state import RolloutResult, SafetyLimits, WorldState
from llm_framework.core.tasks import build_task_context
from llm_framework.interfaces.hybrid import HybridInterface
from llm_framework.interfaces.latent import LatentStubInterface
from llm_framework.interfaces.recursive_units import RecursiveUnitsInterface, _repair_prompt
from llm_framework.interfaces.script_dsl import ScriptDSLInterface
from llm_framework.llm.worker import complete_for_interface
from llm_framework.runtime.dsl_interpreter import compile_dsl


class FakeActuator:
    def __init__(self, name: str, idx: int, ctrlrange: tuple[float, float]):
        self.name = name
        self.id = idx
        self.ctrlrange = ctrlrange


class FakeBody:
    def __init__(self, idx: int):
        self.id = idx


SHADOW_ACTUATOR_NAMES = [
    "base_x",
    "base_y",
    "base_z",
    "rh_A_WRJ2",
    "rh_A_WRJ1",
    "rh_A_THJ5",
    "rh_A_THJ4",
    "rh_A_THJ3",
    "rh_A_THJ2",
    "rh_A_THJ1",
    "rh_A_FFJ4",
    "rh_A_FFJ3",
    "rh_A_FFJ2",
    "rh_A_FFJ1",
    "rh_A_MFJ4",
    "rh_A_MFJ3",
    "rh_A_MFJ2",
    "rh_A_MFJ1",
    "rh_A_RFJ4",
    "rh_A_RFJ3",
    "rh_A_RFJ2",
    "rh_A_RFJ1",
    "rh_A_LFJ5",
    "rh_A_LFJ4",
    "rh_A_LFJ3",
    "rh_A_LFJ2",
    "rh_A_LFJ1",
]

BODY_IDS = {
    "object": 0,
    "rh_ffdistal": 1,
    "rh_mfdistal": 2,
    "rh_rfdistal": 3,
    "rh_lfdistal": 4,
    "rh_thdistal": 5,
}


class FakeModel:
    def __init__(self):
        self.names = SHADOW_ACTUATOR_NAMES
        ranges = [[-0.30, 0.30], [-0.30, 0.30], [-0.15, 0.20]]
        ranges += [[0.00, 1.00] for _ in self.names[3:]]
        self.actuator_ctrlrange = np.asarray(ranges, dtype=np.float32)

    def actuator(self, key):
        if isinstance(key, int):
            return FakeActuator(self.names[key], key, tuple(self.actuator_ctrlrange[key]))
        idx = self.names.index(key)
        return FakeActuator(key, idx, tuple(self.actuator_ctrlrange[idx]))

    def body(self, name: str) -> FakeBody:
        if name not in BODY_IDS:
            raise KeyError(name)
        return FakeBody(BODY_IDS[name])


class FakeEnv:
    jit_step = False

    def __init__(self, *, episode_seconds: float = 1.2, control_dt: float = 0.1):
        self.model = FakeModel()
        self.nu = len(self.model.names)
        self.horizon = max(1, round(episode_seconds / control_dt))
        self.cfg = SimpleNamespace(action_scale=0.05, control_dt=control_dt, episode_seconds=episode_seconds, contact_eps=0.03)
        self.ctrl_open = np.zeros(self.nu, dtype=np.float32)
        self.ctrl_close = np.zeros(self.nu, dtype=np.float32)
        self.ctrl_close[3:] = 1.0
        self.hand_act_ids = list(range(3, self.nu))
        self.object_bid = BODY_IDS["object"]
        self.fingertip_bids = np.asarray([BODY_IDS[name] for name in (
            "rh_ffdistal",
            "rh_mfdistal",
            "rh_rfdistal",
            "rh_lfdistal",
            "rh_thdistal",
        )], dtype=np.int32)
        self.base_qadr = [0, 1, 2]
        self.hand_qadr = list(range(3, self.nu))

    def reset(self, seed: int = 0):
        rng = np.random.default_rng(seed)
        obj = np.array([rng.uniform(-0.03, 0.03), rng.uniform(-0.03, 0.03), 0.025], dtype=np.float32)
        ctrl = self.ctrl_open.copy()
        return SimpleNamespace(
            data=SimpleNamespace(
                xpos=self._xpos(obj, ctrl),
                cvel=np.zeros((len(BODY_IDS), 6), dtype=np.float32),
                qpos=ctrl.copy(),
                ctrl=ctrl,
            ),
            reward=0.0,
            step=0,
        )

    def step(self, state, action: np.ndarray):
        lo = self.model.actuator_ctrlrange[:, 0]
        hi = self.model.actuator_ctrlrange[:, 1]
        ctrl = np.clip(state.data.ctrl + np.clip(action, -1.0, 1.0) * self.cfg.action_scale, lo, hi).astype(np.float32)
        obj = state.data.xpos[self.object_bid].copy()
        base_xy = ctrl[:2]
        near = float(np.linalg.norm(base_xy - obj[:2])) < 0.07
        closed = self._closure(ctrl) > 0.55
        if closed and near:
            obj[:2] += 0.25 * (base_xy - obj[:2])
            obj[2] = max(obj[2], 0.025 + max(0.0, -float(ctrl[2])) * 0.5)
        reward = float(obj[2] - 0.025)
        return SimpleNamespace(
            data=SimpleNamespace(
                xpos=self._xpos(obj, ctrl),
                cvel=np.zeros((len(BODY_IDS), 6), dtype=np.float32),
                qpos=ctrl.copy(),
                ctrl=ctrl,
            ),
            reward=reward,
            step=state.step + 1,
        )

    def _closure(self, ctrl: np.ndarray) -> float:
        ids = [self.model.names.index(name) for name in ("rh_A_FFJ4", "rh_A_THJ5")]
        return float(ctrl[ids].mean())

    def _xpos(self, obj: np.ndarray, ctrl: np.ndarray) -> np.ndarray:
        xpos = np.zeros((len(BODY_IDS), 3), dtype=np.float32)
        xpos[self.object_bid] = obj
        closed = self._closure(ctrl) > 0.55
        near = float(np.linalg.norm(ctrl[:2] - obj[:2])) < 0.07
        for bid in self.fingertip_bids:
            xpos[int(bid)] = obj if closed and near else obj + np.array([0.18, 0.0, 0.0], dtype=np.float32)
        return xpos


def fake_world() -> WorldState:
    env = FakeEnv()
    ctrl = env.ctrl_open.copy()
    return WorldState(
        time_s=0.0,
        object_pos=np.array([0.02, -0.01, 0.025], dtype=np.float32),
        object_vel=np.zeros(3, dtype=np.float32),
        base_q=np.zeros(3, dtype=np.float32),
        hand_q=np.zeros(env.nu - 3, dtype=np.float32),
        ctrl=ctrl,
        ctrl_lo=env.model.actuator_ctrlrange[:, 0],
        ctrl_hi=env.model.actuator_ctrlrange[:, 1],
        actuator_names=env.model.names,
        appendages={
            "base": ["base_x", "base_y", "base_z"],
            "wrist": ["rh_A_WRJ2", "rh_A_WRJ1"],
            "index": ["rh_A_FFJ4"],
            "thumb": ["rh_A_THJ5"],
            "middle": ["rh_A_MFJ4"],
            "ring": ["rh_A_RFJ4"],
            "little": ["rh_A_LFJ5"],
            "hand": env.model.names[3:],
        },
        joint_schema={},
        derived={
            "palm_pos": [0.01, 0.0, 0.30],
            "grasp_site_pos": [0.05, -0.01, 0.125],
            "object_to_palm_xy_distance": 0.02,
            "object_to_grasp_site_xy_distance": 0.03,
        },
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

    assert compiled.action_stream[:, env.model.names.index("rh_A_FFJ4")].max() > 0.0
    assert compiled.action_stream[:, env.model.names.index("rh_A_THJ5")].max() == 0.0


def test_set_frame_target_uses_actual_grasp_site_coordinates() -> None:
    env = FakeEnv()
    world = fake_world()
    program = RecursiveUnitsInterface().parse("""
    {"blocks": [
      {"op": "set_frame_target", "frame": "grasp_site",
       "target": {"object": "object", "offset": [0.0, 0.0, 0.03]},
       "duration_s": 0.1},
      {"op": "return"}
    ]}
    """)
    validation = RecursiveUnitsInterface().validate(program, SafetyLimits(max_episode_seconds=1.2))
    assert validation.ok, validation.errors

    compiled = compile_dsl(program, env, world)
    base_x = env.model.names.index("base_x")
    base_y = env.model.names.index("base_y")
    base_z = env.model.names.index("base_z")

    assert compiled.action_stream[0, base_x] < 0.0
    assert abs(float(compiled.action_stream[0, base_y])) < 1e-6
    assert compiled.action_stream[0, base_z] < 0.0


def test_set_frame_target_accepts_world_coordinate_list() -> None:
    env = FakeEnv()
    world = fake_world()
    program = RecursiveUnitsInterface().parse("""
    {"blocks": [
      {"op": "set_frame_target", "frame": "grasp_site",
       "target": [0.02, -0.01, 0.055],
       "duration_s": 0.1},
      {"op": "return"}
    ]}
    """)

    compiled = compile_dsl(program, env, world)
    base_x = env.model.names.index("base_x")
    base_y = env.model.names.index("base_y")
    base_z = env.model.names.index("base_z")

    assert compiled.action_stream[0, base_x] < 0.0
    assert abs(float(compiled.action_stream[0, base_y])) < 1e-6
    assert compiled.action_stream[0, base_z] < 0.0


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
    assert compiled.action_stream[:, env.model.names.index("rh_A_FFJ4")].max() > 0.0


def test_recursive_units_prompt_requests_timed_schedule_without_loops() -> None:
    world = fake_world()
    ctx = build_task_context("lift", 0, world, episode_seconds=1.2)
    prompt = RecursiveUnitsInterface().build_prompt(ctx, world)

    assert "explicit timed schedule" in prompt
    assert "monitor or wait blocks" in prompt
    assert "runtime conditionals" in prompt
    assert "stop_condition" not in prompt


def test_unknown_control_op_is_rejected() -> None:
    world = fake_world()
    ctx = build_task_context("lift", 0, world, episode_seconds=1.2)
    interface = RecursiveUnitsInterface()
    program = interface.parse("""
    {
      "blocks": [
        {
          "phase": "close_until_touch_or_settle",
          "op": "unsupported_control_op",
          "duration_s": 0.1
        },
        {"op": "return"}
      ]
    }
    """)

    validation = interface.validate(program, SafetyLimits(max_episode_seconds=1.2, max_loop_iterations=4))
    assert not validation.ok
    assert any("unknown op 'unsupported_control_op'" in error for error in validation.errors)


def test_repair_prompt_requests_scheduled_error_correction() -> None:
    world = fake_world()
    ctx = build_task_context("lift", 0, world, episode_seconds=1.2)
    previous_result = RolloutResult(
        interface="recursive_units",
        task="lift",
        seed=0,
        success=False,
        score=0.0,
        total_return=0.0,
        final_object_pos=ctx.object_start,
        max_object_z=ctx.object_start[2],
        metadata={"task_metrics": {"n_contacts": 0, "base_xy_error": 0.05}},
    )

    prompt = _repair_prompt(ctx, world, {"blocks": []}, previous_result)

    assert "There is no runtime conditional loop" in prompt
    assert "Tune fixed phase durations" in prompt
    assert "Previous rollout result" in prompt

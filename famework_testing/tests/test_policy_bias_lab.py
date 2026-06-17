from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jp
import numpy as np

from policy_bias_lab.bias import compile_bias
from policy_bias_lab.es import BIAS_ARMS, ESConfig, rollout_policy, train_arm
from policy_bias_lab.policy import init_params
from policy_bias_lab.schema import default_bias_spec, validate_bias_spec


class FakeActuator:
    def __init__(self, name: str):
        self.name = name


class FakeModel:
    names = ("base_x", "base_y", "base_z", "rh_A_FFJ4", "rh_A_THJ5")

    def actuator(self, i: int):
        return FakeActuator(self.names[i])


class FakeState(NamedTuple):
    obs: jp.ndarray
    reward: jp.ndarray
    step: jp.ndarray
    metrics: dict[str, jp.ndarray]


class FakeEnv:
    def __init__(self):
        self.model = FakeModel()
        self.nu = len(self.model.names)
        self.action_size = self.nu
        self.obs_size = 4 + 3 + self.action_size
        self.horizon = 3
        self.base_act_ids = [0, 1, 2]
        self.hand_act_ids = [3, 4]
        self.ctrl_open = jp.asarray([0.0, 0.0, 0.0, 0.0, 0.0])
        self.ctrl_close = jp.asarray([0.0, 0.0, 0.0, 1.0, 1.0])

    def reset(self, key):
        rel = jp.asarray([0.05, -0.03, 0.02])
        ctrl = jp.zeros((self.action_size,))
        obs = jp.concatenate([jp.zeros((4,)), rel, ctrl])
        eval_vec = jp.asarray([0.2, 0.2, 0.0, 0.0, 0.0, 0.0])
        return FakeState(obs=obs, reward=jp.float32(0.0), step=jp.int32(0), metrics={"eval": eval_vec})

    def step(self, state, action):
        ctrl = jp.clip(state.obs[-self.action_size:] + 0.1 * action, -1.0, 1.0)
        rel = state.obs[4:7] - jp.asarray([0.01 * action[0], 0.01 * action[1], 0.0])
        lift = jp.maximum(0.0, -ctrl[2]) * 0.05 + jp.maximum(0.0, ctrl[3] + ctrl[4]) * 0.01
        closure = jp.clip((ctrl[3] + ctrl[4]) * 0.5, 0.0, 1.0)
        eval_vec = jp.asarray([
            jp.linalg.norm(rel),
            jp.maximum(0.0, 0.2 - closure * 0.1),
            jp.where(closure > 0.1, 2.0, 0.0),
            closure,
            lift,
            jp.linalg.norm(ctrl[:2]) * 0.02,
        ])
        obs = jp.concatenate([jp.zeros((4,)), rel, ctrl])
        return FakeState(obs=obs, reward=lift, step=state.step + 1, metrics={"eval": eval_vec})


def test_bias_spec_validation_rejects_unknown_observable() -> None:
    spec = default_bias_spec("lift")
    spec["reward_terms"][0]["observable"] = "magic_sensor"

    result = validate_bias_spec(spec)

    assert not result.ok
    assert any("unknown observable" in error for error in result.errors)


def test_compile_bias_generates_action_prior_and_targets() -> None:
    env = FakeEnv()
    bias = compile_bias(default_bias_spec("lift"), env)
    state = env.reset(jax.random.PRNGKey(0))

    prior = bias.action_prior(state.obs, "lift")
    target = bias.supervised_target(state.obs, "lift")

    assert prior.shape == (env.action_size,)
    assert target.shape == (env.action_size,)
    assert float(prior[0]) > 0.0
    assert float(prior[1]) < 0.0
    assert float(prior[2]) < 0.0
    assert float(prior[3]) > 0.0


def test_rollout_policy_uses_reward_bias_and_action_prior() -> None:
    env = FakeEnv()
    bias = compile_bias(default_bias_spec("lift"), env)
    params = init_params(jax.random.PRNGKey(1), env.obs_size, env.action_size, hidden=8)

    baseline = rollout_policy(
        env=env,
        params=params,
        bias=bias,
        task="lift",
        key=jax.random.PRNGKey(2),
        n_envs=2,
        use_reward_bias=False,
        use_action_prior=False,
    )
    biased = rollout_policy(
        env=env,
        params=params,
        bias=bias,
        task="lift",
        key=jax.random.PRNGKey(2),
        n_envs=2,
        use_reward_bias=True,
        use_action_prior=True,
    )

    assert biased.fitness != baseline.fitness
    assert "lift" in biased.eval_summary


def test_train_arm_runs_one_generation_on_fake_env() -> None:
    env = FakeEnv()
    bias = compile_bias(default_bias_spec("lift"), env)
    params = init_params(jax.random.PRNGKey(3), env.obs_size, env.action_size, hidden=8)
    cfg = ESConfig(generations=1, population=2, envs=2, supervised_steps=1, supervised_batch=2)

    trained, metrics = train_arm(
        env=env,
        init_params=params,
        bias=bias,
        task="lift",
        arm="full",
        seed=4,
        cfg=cfg,
    )

    assert set(trained) == set(params)
    assert len(metrics) == 1
    assert metrics[0]["arm"] == "full"


def test_isolated_arms_enable_one_bias_against_baseline() -> None:
    assert BIAS_ARMS["baseline"] == (False, False, False, False)
    isolated = ("reward", "action_prior", "exploration", "supervised_init")

    for arm in isolated:
        assert sum(bool(flag) for flag in BIAS_ARMS[arm]) == 1

    assert BIAS_ARMS["reward"] == (True, False, False, False)
    assert BIAS_ARMS["action_prior"] == (False, True, False, False)
    assert BIAS_ARMS["exploration"] == (False, False, True, False)
    assert BIAS_ARMS["supervised_init"] == (False, False, False, True)

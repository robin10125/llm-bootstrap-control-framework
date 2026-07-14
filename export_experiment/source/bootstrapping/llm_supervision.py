#!/usr/bin/env python3
"""Glue between the async LLM worker and the PPO trainer (main-thread / JAX side).

Owns the worker, the demo buffer, and the per-iteration cycle:
  feed fresh samples -> drain finished plans -> score each in MJX -> clone the winners.
All JAX runs here on the main thread; the worker thread only does LLM I/O. The trainer
calls `Supervisor.step(...)` once per PPO iteration and gets back a BC minibatch (or None)
plus stats to log. See `rl_redesign.md` and `llm_worker.py`.
"""
from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jp
import numpy as np

import eval_metrics as EM
import reward_spec as RS
from curriculum import progress_score
from llm_backend import call_llm
from llm_worker import (DemoBuffer, LLMWorker, build_reward_prompt, parse_reward_spec)
from mjx_env import MjxEnv
from waypoints import PlanError, WaypointCompiler, make_plan_rollout


def author_reward_spec(backend: str, model: str | None, *, embodiment: str, log_dir: Path,
                       prior_spec=None, reflection: str | None = None,
                       timeout_s: float = 360.0):
    """One LLM call to author/revise a reward-curriculum spec. Returns a *validated* spec dict
    or None (caller falls back to the hand-written default). The LLM never runs code; the spec
    is checked against the audited operator library before use."""
    ctx = {"embodiment": embodiment, "fields_doc": EM.fields_doc()}
    if reflection is not None:
        ctx.update(reflection=reflection, prior_spec=prior_spec)
    resp = call_llm(backend, build_reward_prompt(ctx), model=model, timeout_s=timeout_s,
                    log_dir=log_dir, tag="reward_author")
    if not resp.ok:
        return None
    spec = parse_reward_spec(resp.text)
    if spec is None:
        return None
    try:
        RS.validate_spec(spec)
    except RS.SpecError:
        return None
    return spec


def make_policy_rollout(env: MjxEnv, net):
    """Deterministic (mean-action) single-env rollout, for LLM-vs-policy scoring + traces."""
    def rollout(params, state):
        def body(state, _):
            mean, _ls, _v = net.apply(params, state.obs[None])
            nstate = env.step(state, mean[0])
            return nstate, (state.obs, nstate.reward, nstate.metrics["success"],
                            nstate.metrics["eval"])
        state, (obs, rew, succ, ev) = jax.lax.scan(body, state, None, length=env.horizon)
        return {"return": rew.sum(), "success": (succ.max() > 0.5).astype(jp.float32),
                "obs": obs, "eval_summary": EM.reduce_summary(ev)}
    return jax.jit(rollout)


class Supervisor:
    def __init__(self, env: MjxEnv, net, *, backend: str, model: str | None,
                 budget: int, bc_batch: int, log_dir: Path,
                 residual_after: float = 0.3, samples_per_iter: int = 1,
                 max_pending: int = 2, demo_capacity: int = 200_000,
                 switch_rate: float = 0.15, switch_min_attempts: int = 8,
                 give_up_rate: float = 0.05, give_up_min_attempts: int = 12,
                 window: int = 12, task: str = "gripper"):
        self.env = env
        self.net = net
        self.bc_batch = bc_batch
        self.embodiment = "hand" if task == "shadow" else "gripper"
        # When a curriculum/reward spec is attached, judge demos by task progress (partial
        # credit) instead of the binary success+return gate. self.stage tracks the trainer's
        # curriculum stage so the policy is scored at the same difficulty.
        self.compiled = getattr(env, "compiled_reward", None)
        self.stage = 0
        self.residual_after = residual_after
        self.samples_per_iter = samples_per_iter
        self.max_pending = max_pending

        # Lifecycle of the LLM helper:
        #   scratch  -> author plans from scratch (early acquisition)
        #   residual -> correct the policy's own rollouts, once RL beats from-scratch plans
        #   off      -> disabled, because residual corrections are consistently not helping
        # `scratch->residual` fires when the policy outperforms the LLM's scratch demos
        # (the recent scratch accept-rate collapses) or, as a fallback, once training success
        # passes `residual_after`. `residual->off` fires when, after enough tries, the recent
        # residual accept-rate is ~0 (the LLM can no longer beat the policy). Both transitions
        # latch. Accepts are gated on actually beating the policy (see `_drain`), so a low
        # accept-rate is exactly "the LLM is no longer helping".
        self.mode = "scratch"
        self.switch_rate = switch_rate
        self.switch_min_attempts = switch_min_attempts
        self.give_up_rate = give_up_rate
        self.give_up_min_attempts = give_up_min_attempts
        self._scratch_outcomes: deque[int] = deque(maxlen=window)
        self._residual_outcomes: deque[int] = deque(maxlen=window)

        self.compiler = WaypointCompiler(env)
        self.reset_single = jax.jit(env.reset)
        self.plan_rollout = make_plan_rollout(env)
        self.policy_rollout = make_policy_rollout(env, net)
        self.buffer = DemoBuffer(demo_capacity, env.obs_size, env.action_size)
        self.worker = LLMWorker(backend, model, budget, log_dir=log_dir)
        self.worker.start()

        # observation field offsets (must match mjx_env._observe layout)
        self.obj_off = 3 + 3 + len(env.hand_qadr) + len(env.hand_vadr)
        self.ctrl_off = env.obs_size - env.nu
        self.finger_ctrl_ids = [i for i in range(env.nu)
                                if env.model.actuator(i).name in ("g_left", "g_right")]

        self.stats = {"submitted": 0, "returned": 0, "parse_fail": 0,
                      "compile_fail": 0, "accepted": 0, "rejected": 0,
                      "scratch_accepted": 0, "residual_accepted": 0,
                      "residual_rejected": 0}

    # --- helpers --------------------------------------------------------------

    def _object_xy(self, state) -> tuple[float, float, float]:
        obj = np.asarray(jax.device_get(state.data.xpos[self.env.object_bid]))
        return float(obj[0]), float(obj[1]), float(obj[2])

    def _trace(self, obs_seq) -> str:
        obs = np.asarray(jax.device_get(obs_seq))
        every = max(1, round(0.3 / self.env.cfg.control_dt))
        lines = []
        for t in range(0, obs.shape[0], every):
            o = obs[t]
            base = o[0:3]
            objp = o[self.obj_off:self.obj_off + 3]
            opening = float(o[self.ctrl_off + self.finger_ctrl_ids[0]]) / 0.06 if self.finger_ctrl_ids else 0.0
            lines.append(f"  t={t*self.env.cfg.control_dt:.1f}s base=({base[0]:+.2f},{base[1]:+.2f},"
                         f"{base[2]:+.2f}) obj_z={objp[2]:.3f} open={opening:.2f}")
        return "\n".join(lines)

    # --- per-iteration cycle --------------------------------------------------

    @staticmethod
    def _rate(window: deque[int]) -> float:
        return (sum(window) / len(window)) if window else 1.0

    def _update_mode(self, training_success: float) -> None:
        """Advance the latching scratch -> residual -> off lifecycle (see __init__)."""
        if self.mode == "scratch":
            scratch_rate = self._rate(self._scratch_outcomes)
            beats_demos = (len(self._scratch_outcomes) >= self.switch_min_attempts
                           and scratch_rate <= self.switch_rate)
            if beats_demos or training_success >= self.residual_after:
                self.mode = "residual"
        if self.mode == "residual":
            if (len(self._residual_outcomes) >= self.give_up_min_attempts
                    and self._rate(self._residual_outcomes) <= self.give_up_rate):
                self.mode = "off"  # residuals consistently unhelpful -> stop spending calls

    def step(self, params, key, training_success: float, stage: int = 0):
        self.stage = stage
        self._update_mode(training_success)
        if self.mode != "off":
            self._feed(params, key, self.mode)
        self._drain(params)
        bc = self.buffer.sample(self.bc_batch) if self.buffer.size > 0 else None
        return bc, dict(self.stats, demos=self.buffer.size, accepted_demos=self.buffer.accepted_demos,
                        budget_left=self.worker.budget_left(), mode=self.mode,
                        scratch_rate=round(self._rate(self._scratch_outcomes), 3),
                        residual_rate=round(self._rate(self._residual_outcomes), 3))

    def _feed(self, params, key, mode: str) -> None:
        fed = 0
        while fed < self.samples_per_iter and self.worker.budget_left() > 0:
            key, sk = jax.random.split(key)
            state = self.reset_single(sk)
            ox, oy, oz = self._object_xy(state)
            ctx: dict[str, Any] = {
                "key": [int(v) for v in np.asarray(jax.device_get(sk))],
                "object_xy": (ox, oy), "object_top_z": oz,
                "episode_seconds": self.env.cfg.episode_seconds, "mode": mode,
                "embodiment": self.embodiment,
            }
            if mode == "residual":
                pout = self.policy_rollout(params, state)
                ctx["policy_trace"] = self._trace(pout["obs"])
                ctx["policy_lifted"] = bool(pout["success"] > 0.5)
                obs = np.asarray(jax.device_get(pout["obs"]))
                ctx["policy_peak_obj_z"] = float(obs[:, self.obj_off + 2].max())
            if not self.worker.submit(ctx):
                break
            self.stats["submitted"] += 1
            fed += 1

    def _drain(self, params) -> None:
        for cand in self.worker.poll():
            self.stats["returned"] += 1
            if cand.plan is None:
                self.stats["parse_fail"] += 1
                continue
            key = jp.asarray(np.asarray(cand.ctx["key"], dtype=np.uint32))
            state = self.reset_single(key)
            try:
                targets = jp.asarray(self.compiler.compile(cand.plan, self.env))
            except PlanError:
                self.stats["compile_fail"] += 1
                continue
            plan_out = self.plan_rollout(state, targets, key)
            pol_out = self.policy_rollout(params, state)
            cand_mode = cand.ctx.get("mode", "scratch")
            # Decide whether the LLM plan is MORE HELPFUL than the current policy on the same
            # reset. With a curriculum/reward spec, "more helpful" = further along the task
            # (progress-dominance: partial credit for reaching, getting fingers on, a small
            # lift), so helpful-but-incomplete plans are accepted. Without one, fall back to
            # the binary success+return gate. Either way, a plan that doesn't beat the policy
            # is thrown out (and in residual mode counts toward the give-up signal).
            if self.compiled is not None:
                ps = progress_score(np.asarray(jax.device_get(plan_out["eval_summary"])), self.compiled)
                qs = progress_score(np.asarray(jax.device_get(pol_out["eval_summary"])), self.compiled)
                margin = ps - qs
                accepted = margin > 0.0
            else:
                plan_succ = bool(plan_out["success"] > 0.5)
                pol_succ = bool(pol_out["success"] > 0.5)
                margin = float(plan_out["return"]) - float(pol_out["return"])
                accepted = plan_succ and (not pol_succ or margin > 0.0)
            if accepted:
                # weight the demo by how much it beat the policy (advantage-weighted BC)
                self.buffer.add(plan_out["obs"], plan_out["action"], weight=margin)
                self.stats["accepted"] += 1
            else:
                self.stats["rejected"] += 1
            if cand_mode == "residual":
                self._residual_outcomes.append(int(accepted))
                self.stats["residual_accepted" if accepted else "residual_rejected"] += 1
            else:
                self._scratch_outcomes.append(int(accepted))
                if accepted:
                    self.stats["scratch_accepted"] += 1

    def close(self):
        self.worker.stop()
        return self.worker.usage.summary()

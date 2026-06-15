from __future__ import annotations

from typing import Any

import numpy as np

from llm_framework.core.state import CompiledPolicy, RolloutResult
from llm_framework.core.tasks import TaskContext, score_task, task_metrics


def rollout_compiled(env: Any, state: Any, compiled: CompiledPolicy, ctx: TaskContext, *, seed: int) -> RolloutResult:
    if getattr(env, "is_fake_env", False):
        return _rollout_fake(env, state, compiled, ctx, seed)
    return _rollout_jax(env, state, compiled, ctx, seed)


def _rollout_jax(env: Any, state: Any, compiled: CompiledPolicy, ctx: TaskContext, seed: int) -> RolloutResult:
    try:
        import jax
        import jax.numpy as jp

        @jax.jit
        def run(initial_state, actions):
            def body(cur, action):
                nxt = env.step(cur, action)
                obj = nxt.data.xpos[env.object_bid]
                return nxt, (nxt.reward, obj)

            final_state, (rewards, objects) = jax.lax.scan(body, initial_state, actions)
            return final_state, rewards.sum(), objects[:, 2].max()

        cur, total, max_z = run(state, jp.asarray(compiled.action_stream))
        final = tuple(float(v) for v in np.asarray(jax.device_get(cur.data.xpos[env.object_bid]), dtype=float))
        draft = RolloutResult(
            interface=compiled.interface,
            task=ctx.name,
            seed=seed,
            success=False,
            score=0.0,
            total_return=round(float(jax.device_get(total)), 4),
            final_object_pos=final,
            max_object_z=round(float(jax.device_get(max_z)), 4),
            metadata=compiled.metadata,
        )
        success, score = score_task(ctx, draft)
        metadata = compiled.metadata | {"task_metrics": task_metrics(ctx, draft)}
        return RolloutResult(
            interface=compiled.interface,
            task=ctx.name,
            seed=seed,
            success=success,
            score=score,
            total_return=draft.total_return,
            final_object_pos=final,
            max_object_z=draft.max_object_z,
            metadata=metadata,
        )
    except Exception as exc:
        return _error_result(compiled, ctx, seed, exc)


def _rollout_fake(env: Any, state: Any, compiled: CompiledPolicy, ctx: TaskContext, seed: int) -> RolloutResult:
    try:
        total = 0.0
        max_z = float(state.data.xpos[env.object_bid, 2])
        cur = state
        for action in compiled.action_stream:
            cur = env.step(cur, action)
            total += float(cur.reward)
            max_z = max(max_z, float(cur.data.xpos[env.object_bid, 2]))
        final = tuple(float(v) for v in cur.data.xpos[env.object_bid])
        draft = RolloutResult(
            interface=compiled.interface,
            task=ctx.name,
            seed=seed,
            success=False,
            score=0.0,
            total_return=round(total, 4),
            final_object_pos=final,
            max_object_z=round(max_z, 4),
            metadata=compiled.metadata,
        )
        success, score = score_task(ctx, draft)
        metadata = compiled.metadata | {"task_metrics": task_metrics(ctx, draft)}
        return RolloutResult(
            interface=compiled.interface,
            task=ctx.name,
            seed=seed,
            success=success,
            score=score,
            total_return=draft.total_return,
            final_object_pos=final,
            max_object_z=draft.max_object_z,
            metadata=metadata,
        )
    except Exception as exc:
        return _error_result(compiled, ctx, seed, exc)


def _error_result(compiled: CompiledPolicy, ctx: TaskContext, seed: int, exc: Exception) -> RolloutResult:
    return RolloutResult(
        interface=compiled.interface,
        task=ctx.name,
        seed=seed,
        success=False,
        score=0.0,
        total_return=0.0,
        final_object_pos=ctx.object_start,
        max_object_z=ctx.object_start[2],
        errors=(f"{type(exc).__name__}: {exc}",),
        metadata=compiled.metadata,
    )

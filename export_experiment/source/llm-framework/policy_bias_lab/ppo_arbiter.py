"""PPO arbiter for action-prior candidates.

The open-loop rollout scorer (prior_eval.score_program) is known to INVERT the true PPO ranking
(prelim: DSL open-loop -0.10 "beat" free-form -0.57, yet free-form trained better). So the agentic
orchestrator's real arbiter trains a PPO run per candidate and ranks on the TASK-DEFINED
graded objective over the trained policy's eval -- the actual objective. It also returns the
trained policy's eval measurements (uninterpreted) to drive the revision feedback loop, so the LLM
critiques what the policy actually DID, not an open-loop proxy.

TASK-AGNOSTICISM: this module carries NO task knowledge. The objective definition and the list of
progress metrics come from the injected task spec (tasks.py); the diagnostics are raw measurements
with the environment's own field names, with NO framework-authored interpretation -- diagnosing WHY
a policy fails is the LLM's job in the revision loop (see AGENTS.md).

One fragmented-stage PPO evaluation == one budget ITERATION (the scarce, real-robot-relevant resource).
"""
from __future__ import annotations

import gc
from dataclasses import fields, replace
from pathlib import Path
from typing import Any

import jax

import ppo  # bootstrapping module; on sys.path before this imports (see run_* CLIs)

from policy_bias_lab.bias import compile_bias
from policy_bias_lab.ppo_bias import (
    PPOBiasConfig,
    _policy_dim,
    evaluate_fragmented_policy,
    train_fragmented_stage_ppo,
)
from policy_bias_lab.schema import FIELD_INDEX
from policy_bias_lab.tasks import task_graded_objective, task_progress_metrics

# Historical label kept for checkpoints and reports that recorded the old arbiter arm name.
ARBITER_ARM = "freeform_encourage"


def _default_fragment_steps(env: Any) -> int:
    for steps in (100, 80, 64, 50, 40, 25, 20, 10, 5, 1):
        if int(env.horizon) % steps == 0:
            return steps
    return 1


def _ppo_cfg(env: Any, *, train_envs: int, eval_envs: int, train_seconds: float | None,
             cfg_overrides: dict | None = None) -> PPOBiasConfig:
    allowed = {f.name for f in fields(PPOBiasConfig)}
    kwargs = {
        "envs": int(train_envs),
        "eval_envs": int(eval_envs),
        "target_train_seconds": train_seconds,
        "fragment_steps": _default_fragment_steps(env),
    }
    for k, v in (cfg_overrides or {}).items():
        if v is not None and k in allowed:
            kwargs[k] = v
    cfg = PPOBiasConfig(**kwargs)
    if int(env.horizon) % int(cfg.fragment_steps) != 0:
        cfg = replace(cfg, fragment_steps=_default_fragment_steps(env))
    return cfg


def behavioral_diagnostics(ev: dict) -> dict:
    """The trained policy's eval measurements, uninterpreted, for the revision prompt.

    Numbers only, under the environment/task's own field names: rates, per-signal episode means
    (eval_summary named via the env schema), and action statistics. No failure-mode labels -- any
    framework-authored interpretation is task knowledge leaking into the loop, and it preempts the
    diagnosis the model is supposed to make organically from the evidence.
    """
    out: dict[str, Any] = {}
    for k, v in ev.items():
        if isinstance(v, (int, float)) and k.startswith("eval_"):
            out[k.removeprefix("eval_")] = round(float(v), 4)
    summ = ev.get("eval_summary")
    if summ is not None:
        out["episode_signal_means"] = {name: round(float(summ[i]), 4)
                                       for name, i in FIELD_INDEX.items() if i < len(summ)}
        out.pop("summary", None)
    return out


def training_convergence(rows: list[dict], metrics: tuple[str, ...]) -> dict:
    """Task-agnostic under-training probe from the fragmented-stage PPO training telemetry.

    A candidate can score low for two very different reasons: the prior is wrong, or the prior is
    fine and the policy simply ran out of training budget. The revision loop must not "fix" the
    latter. We compare each training metric's mean over the MIDDLE third vs the LAST third of
    iterations (the first third is warmup/compile noise): any metric still clearly rising at budget
    end means more training would likely help -- verdict "undertrained". All metrics flat or
    falling -> "converged": the prior, not the budget, is the limiter. The metric list is injected
    (task_progress_metrics); the trend logic itself carries no task knowledge.
    """
    n = len(rows)
    if n < 6:
        return {"verdict": "insufficient_data", "n_iters": n}
    mid, late = rows[n // 3: 2 * n // 3], rows[2 * n // 3:]
    trend, improving = {}, []
    for m in metrics:
        a = sum(float(r.get(m, 0.0)) for r in mid) / len(mid)
        b = sum(float(r.get(m, 0.0)) for r in late) / len(late)
        trend[m] = [round(a, 4), round(b, 4)]
        if b - a > max(0.02 * max(abs(a), abs(b)), 1e-4):
            improving.append(m)
    return {"n_iters": n, "mid_vs_late": trend, "still_improving": improving,
            "verdict": "undertrained" if improving else "converged"}



def _eval_policy(env: Any, bias: Any, params: Any, program: dict, *, task: str, seed: int,
                 cfg: PPOBiasConfig, apw) -> tuple[float, dict, dict]:
    """Deterministic eval of one fragmented-stage PPO policy."""
    want_stages = program.get("mode") == "freeform_staged"
    ev = evaluate_fragmented_policy(
        env=env, params=params, bias=bias, program=program, task=task, seed=seed,
        cfg=cfg, action_prior_weights=apw, return_obs=want_stages)
    obs = ev.pop("eval_obs", None)
    obj = float(ev.get("eval_graded_objective", task_graded_objective(task, ev)))
    diag = behavioral_diagnostics(ev)
    if want_stages and obs is not None:
        from policy_bias_lab.freeform_priors import stage_occupancy
        diag["stage_report"] = stage_occupancy(env, program, obs)
    return obj, diag, ev


PRETRAIN_NOTE = ("behavior at PPO iteration 0: the prior acting through an UNTRAINED policy "
                 "(network output ~0, so this is essentially the prior alone). Compare with the "
                 "trained metrics -- a defect present here is a PRIOR problem; one that only "
                 "appears after training points at the reward/training side.")



def evaluate_candidate_prior_only(
    env: Any, program: dict, *, task: str = "lift", seed: int = 0, eval_envs: int = 256,
    cfg_overrides: dict | None = None, n_batches: int = 1,
) -> dict:
    """No-training arbiter: evaluate the prior at PPO iteration 0, one rollout batch per
    candidate. Isolates PRIOR quality from training/reward effects and is cheaper than a trained
    PPO evaluation."""
    del n_batches  # Fragmented eval is already full-episode; batch-spread is prior-only legacy.
    spec = {"name": "arbiter", "action_priors": [], "prior_program": program}
    bias = compile_bias(spec, env)
    cfg = _ppo_cfg(env, train_envs=eval_envs, eval_envs=eval_envs, train_seconds=None,
                   cfg_overrides=cfg_overrides)
    apw = bias.default_action_prior_weights()
    _net, params0 = ppo.init_params(jax.random.PRNGKey(seed), env.obs_size,
                                    _policy_dim(env, cfg), cfg.hidden)
    obj, diag, ev = _eval_policy(env, bias, params0, program, task=task, seed=seed + 20_000,
                                 cfg=cfg, apw=apw)
    diag = {"note": PRETRAIN_NOTE, **diag}
    jax.clear_caches()
    gc.collect()
    return {"objective_score": obj, "diagnostics": diag, "eval": ev,
            "best_train_success": 0.0, "best_checkpoint_iter": -1, "best_params": None,
            "telemetry": None}



def evaluate_candidate_ppo(
    env: Any, program: dict, *, task: str = "lift", seed: int = 0,
    train_seconds: float = 180.0, train_envs: int = 256, eval_envs: int = 256,
    checkpoint_dir: Path | None = None, cfg_overrides: dict | None = None,
) -> dict:
    """Train fragmented-stage PPO with `program` injected; return objective and diagnostics.

    Returns: {objective_score, diagnostics, eval, best_train_success, best_checkpoint_iter,
    best_params}. `best_params` lets the orchestrator keep the trained weights of the winner.
    """
    spec = {"name": "arbiter", "action_priors": [], "prior_program": program}
    bias = compile_bias(spec, env)
    cfg = _ppo_cfg(env, train_envs=train_envs, eval_envs=eval_envs, train_seconds=train_seconds,
                   cfg_overrides=cfg_overrides)
    apw = bias.default_action_prior_weights()

    # PRE-TRAIN eval: the same candidate at PPO iteration 0, so the revision loop can separate
    # prior defects (present before training) from reward/training effects (appear only after).
    _net0, params0 = ppo.init_params(jax.random.PRNGKey(seed), env.obs_size,
                                     _policy_dim(env, cfg), cfg.hidden)
    pre_obj, pre_diag, _pre_ev = _eval_policy(
        env, bias, params0, program, task=task, seed=seed + 20_000, cfg=cfg, apw=apw)

    _params, rows, best_params, best_score, best_iter = train_fragmented_stage_ppo(
        env=env, bias=bias, program=program, task=task, seed=seed, cfg=cfg,
        out_dir=checkpoint_dir, action_prior_weights=apw,
    )
    obj, diag, ev = _eval_policy(
        env, bias, best_params, program, task=task, seed=seed + 10_000, cfg=cfg, apw=apw)
    diag["training_report"] = training_convergence(rows, task_progress_metrics(task))
    diag["pretrained_prior"] = {"note": PRETRAIN_NOTE, "objective_score": round(pre_obj, 4),
                                **pre_diag}
    jax.clear_caches()
    gc.collect()
    stride = max(1, len(rows) // 150)
    telemetry = [{k: (round(float(v), 5) if isinstance(v, (int, float)) else v)
                  for k, v in r.items()} for r in rows[::stride]]
    return {
        "objective_score": obj, "diagnostics": diag, "eval": ev,
        "best_train_success": round(float(ev.get("eval_success_rate", 0.0)), 6),
        "best_checkpoint_iter": int(best_iter), "best_params": best_params,
        "best_train_objective": round(float(best_score), 6),
        "telemetry": telemetry,
    }

"""Short-PPO arbiter for action-prior candidates.

The open-loop rollout scorer (prior_eval.score_program) is known to INVERT the true PPO ranking
(prelim: DSL open-loop -0.10 "beat" free-form -0.57, yet free-form trained better). So the agentic
orchestrator's real arbiter trains a SHORT PPO run per candidate and ranks on TRAINED contact-gated
success -- the actual objective. It also returns rich behavioral diagnostics from the trained
policy (reach/grasp/lift/closure/saturation + a failure-mode read) to drive the revision feedback
loop, so the LLM critiques what the policy actually DID, not an open-loop proxy.

One short-PPO evaluation == one budget ITERATION (the scarce, real-robot-relevant resource).
"""
from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

import jax

from policy_bias_lab.bias import compile_bias, default_reward_template_weights
from policy_bias_lab.ppo_bias import PPOBiasConfig, evaluate_ppo_policy, train_ppo_arm

# An arm whose BIAS_ARMS flags are (use_reward=True, use_action_prior=True): the prior is injected
# and trained on the base contact-gated reward + the fixed lift template. The arbiter only needs
# those two flags on; the arm name is otherwise irrelevant to a file-injected prior program.
ARBITER_ARM = "freeform_encourage"


SUCCESS_LIFT_THRESHOLD = 0.05  # m; matches PPOBiasConfig.success_lift_threshold default


def graded_objective(ev: dict) -> float:
    """DENSE, contact-gated trained-policy objective along reach -> grasp -> lift -> sustain.

    Ranking on sustained success ALONE is blind here: no arm in this project has ever achieved
    sustained lift even after hours, so that term is ~0 for every candidate at any feasible
    per-candidate budget (see the marginal-value run that flatlined at the 1e-4 tie-breaker floor).
    Instead we reward HOW FAR down the manipulation chain the trained policy gets, with the later
    (harder) stages weighted more, and we GATE lift by grasp so flinging the object up without a
    secure contact grip is NOT credited (anti-transfer-exploit, consistent with the contact-gated
    reward). Sustained success still dominates whenever it is nonzero.
    """
    reach = float(ev.get("eval_reach_rate", 0.0))            # contact made (gated)
    grasp = float(ev.get("eval_grasp_rate", 0.0))            # contact + closure: secure grip (gated)
    lift_reached = float(ev.get("eval_lift_reached_rate", 0.0))  # raw lift>thresh (NOT gated)
    lift_max_norm = min(float(ev.get("eval_lift_max", 0.0)) / SUCCESS_LIFT_THRESHOLD, 1.0)
    succ = float(ev.get("eval_success_rate", 0.0))           # sustained contact-gated lift (the goal)
    # Credit lift only as far as there is a grasp -> a flinging policy (lift high, grasp low) scores
    # via the min() at ~0 on the lift terms.
    gated_lift_reached = min(lift_reached, grasp)
    gated_lift_max = min(lift_max_norm, grasp)
    return (1.0 * succ + 0.5 * grasp + 0.2 * reach
            + 0.3 * gated_lift_reached + 0.1 * gated_lift_max)


def behavioral_diagnostics(ev: dict) -> dict:
    """Condense the trained-policy eval into a failure-mode read for the revision prompt."""
    reach = float(ev.get("eval_reach_rate", 0.0))
    grasp = float(ev.get("eval_grasp_rate", 0.0))
    lift_reached = float(ev.get("eval_lift_reached_rate", 0.0))
    lift_max = float(ev.get("eval_lift_max", 0.0))
    succ = float(ev.get("eval_success_rate", 0.0))
    instant = float(ev.get("eval_instant_success_rate", 0.0))
    sat = float(ev.get("eval_saturation_frac", 0.0))
    # eval_summary layout: [palm_obj_dist, ?, contacts, closure, lift, xy_disp]
    summ = ev.get("eval_summary") or [0.0] * 6
    closure = float(summ[3]) if len(summ) > 3 else 0.0
    contacts = float(summ[2]) if len(summ) > 2 else 0.0
    xy = float(summ[5]) if len(summ) > 5 else 0.0
    flags = []
    if reach < 0.3:
        flags.append("never_reaches (palm not getting to object -> approach too weak/misaimed)")
    if closure > 0.5 and contacts < 0.3:
        flags.append("air_closure (fingers curling before contact, locking out the grasp)")
    if grasp > 0.4 and lift_reached < 0.1:
        flags.append("grasp_no_lift (grips but never raises -> lift phase too weak / grip slips)")
    if lift_max > 0.05 and contacts < 0.3:
        flags.append("flinging (object up without sustained contact -> non-prehensile)")
    if instant > succ + 0.1:
        flags.append("transient_only (brief success not sustained -> grip unstable)")
    if sat > 0.5:
        flags.append("saturating (actions hitting limits -> weights too high)")
    return {
        "trained_success": round(succ, 4), "instant_success": round(instant, 4),
        "reach_rate": round(reach, 4), "grasp_rate": round(grasp, 4),
        "lift_reached_rate": round(lift_reached, 4), "lift_max": round(lift_max, 4),
        "mean_contacts": round(contacts, 4), "mean_closure": round(closure, 4),
        "obj_xy_drift": round(xy, 4), "saturation_frac": round(sat, 4),
        "likely_failure_modes": flags or ["none obvious (low-signal run)"],
    }


def training_convergence(rows: list[dict]) -> dict:
    """Task-agnostic under-training probe from the short-PPO training telemetry.

    A candidate can score low for two very different reasons: the prior is wrong, or the prior is
    fine and the policy simply ran out of training budget. The revision loop must not "fix" the
    latter. We compare each training metric's mean over the MIDDLE third vs the LAST third of
    iterations (the first third is warmup/compile noise): any metric still clearly rising at budget
    end means more training would likely help -- verdict "undertrained". All metrics flat or
    falling -> "converged": the prior, not the budget, is the limiter. Uses only the generic
    training rows (env base return + graded progress rates); no task nouns.
    """
    metrics = ("base_return", "reach_rate", "grasp_rate", "lift_reached_rate", "success")
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


def evaluate_candidate_ppo(
    env: Any, program: dict, *, task: str = "lift", seed: int = 0,
    train_seconds: float = 180.0, train_envs: int = 256, eval_envs: int = 256,
    checkpoint_dir: Path | None = None, cfg_overrides: dict | None = None,
) -> dict:
    """Train a short PPO with `program` injected; return graded objective + behavioral diagnostics.

    Returns: {objective_score, diagnostics, eval, best_train_success, best_checkpoint_iter,
    best_params}. `best_params` lets the orchestrator keep the trained weights of the winner.
    """
    spec = {"name": "arbiter", "action_priors": [], "prior_program": program}
    bias = compile_bias(spec, env)
    cfg = PPOBiasConfig(envs=train_envs, target_train_seconds=train_seconds,
                        **(cfg_overrides or {}))
    reward_weights = default_reward_template_weights(task)
    apw = bias.default_action_prior_weights()

    _params, rows, best_params, best_success, best_iter = train_ppo_arm(
        env=env, bias=bias, task=task, arm=ARBITER_ARM, seed=seed, cfg=cfg,
        checkpoint_dir=checkpoint_dir, reward_weights=reward_weights, base_reward_weight=1.0,
        action_prior_weights=apw,
    )
    want_stages = program.get("mode") == "freeform_staged"
    ev = evaluate_ppo_policy(
        env=env, params=best_params, bias=bias, task=task, arm=ARBITER_ARM,
        seed=seed + 10_000, n_envs=eval_envs, cfg=cfg, reward_weights=reward_weights,
        base_reward_weight=1.0, action_prior_weights=apw, return_obs=want_stages,
    )
    obs = ev.pop("eval_obs", None)  # keep the (large) obs array out of the persisted eval dict
    obj = graded_objective(ev)
    diag = behavioral_diagnostics(ev)
    diag["training_report"] = training_convergence(rows)
    if want_stages and obs is not None:
        # Task-agnostic stage localization: where in the model's OWN stage sequence does the
        # trained policy stall? Drives single-stage-focused revision.
        from policy_bias_lab.freeform_priors import stage_occupancy
        diag["stage_report"] = stage_occupancy(env, program, obs)
    jax.clear_caches()
    gc.collect()
    return {
        "objective_score": obj, "diagnostics": diag, "eval": ev,
        "best_train_success": round(float(best_success), 6),
        "best_checkpoint_iter": int(best_iter), "best_params": best_params,
    }

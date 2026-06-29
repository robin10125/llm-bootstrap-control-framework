from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import pickle
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_triton_gemm=false")
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", str(Path(".xla_cache").resolve()))

import jax  # noqa: E402  (must follow the XLA env-var configuration above)

from policy_bias_lab.action_priors import (
    ActionPriorCoach,
    ActionPriorConfig,
    load_action_prior_rules,
    load_pareto_action_prior_rules,
    load_pareto_supervised_target_rules,
)
from policy_bias_lab.bias import REWARD_TEMPLATE_NAMES, compile_bias, default_reward_template_weights, reward_template_metadata
from policy_bias_lab.composed_priors import default_library, prior_program_for_arm
from policy_bias_lab.dynamic_rewards import DynamicRewardCoach, DynamicRewardConfig, load_pre_run_reward_analysis
from policy_bias_lab.es import BIAS_ARMS
from policy_bias_lab.llm_bias import load_bias_spec
from policy_bias_lab.phase_controller import load_phase_teacher
from policy_bias_lab.ppo_bias import PPOBiasConfig, evaluate_ppo_policy, train_ppo_arm
from policy_bias_lab.run_ppo_experiment import summarize, write_csv
from policy_bias_lab.tasks import task_metadata

ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAPPING = ROOT.parent / "bootstrapping"


def main() -> int:
    args = parse_args()
    # --no-coach: the DynamicRewardCoach (LLM reward authoring + mid-run rewrite/checkups) is
    # deactivated, so "shaped reward" arms are a deterministic, hand-written null-control (base
    # contact-gated reward + the fixed reward TEMPLATES) with no LLM in the reward loop at all.
    # Implemented by forcing the existing gates off: no initial program generation, no rewrite,
    # no checkups. (Analysis showed the coach added no measurable benefit and a small net drag.)
    if args.no_coach:
        args.initial_reward_program = False   # no LLM-generated program; fixed templates only
        args.allow_reward_rewrite = False     # no mid-run recompile / reward rewrite
        args.freeze_reward_shaping = True
        args.cheap_checkup_steps = 1_000_000_000  # checkups never fire (no mid-run LLM calls)
    if str(BOOTSTRAPPING) not in sys.path:
        sys.path.insert(0, str(BOOTSTRAPPING))
    from mjx_env import make_env

    if args.out is None:
        args.out = Path("runs") / f"dynamic_reward_combo_{time.strftime('%Y%m%d-%H%M%S')}"
    args.out.mkdir(parents=True, exist_ok=True)

    tasks = [item.strip() for item in args.tasks.split(",") if item.strip()]
    arms = [item.strip() for item in args.arms.split(",") if item.strip()]
    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    for arm in arms:
        if arm not in BIAS_ARMS:
            raise KeyError(f"unknown arm {arm!r}; choose from {sorted(BIAS_ARMS)}")

    env = make_env(
        "shadow",
        control_dt=args.control_dt,
        episode_seconds=args.episode_seconds,
        physics_dt=args.physics_dt,
        obj_xy_range=args.obj_xy_range,
    )
    env_summary = {
        "obs_size": env.obs_size,
        "action_size": env.action_size,
        "horizon": env.horizon,
        "frame_skip": env.frame_skip,
        "actuators": [env.model.actuator(i).name for i in range(env.nu)],
        "tasks": {name: task_metadata(name) for name in tasks},
    }
    previous_run_context = load_previous_run_context(args.previous_run_dir)

    if args.bias_spec is not None:
        bias_spec = json.loads(args.bias_spec.read_text())
    else:
        bias_spec = load_bias_spec(
            backend=args.llm_backend,
            model=args.llm_model,
            task="+".join(tasks),
            tasks=tasks,
            env_summary=env_summary,
            log_dir=args.out / "llm_initial_bias",
        )
    pre_run_reward_analysis = {}
    if args.pre_run_reward_analysis:
        pre_run_reward_analysis = load_pre_run_reward_analysis(
            llm_backend=args.llm_backend,
            llm_model=args.llm_model,
            task="+".join(tasks),
            tasks=tasks,
            env_summary=env_summary,
            bias_spec=bias_spec,
            previous_run_context=previous_run_context,
            log_dir=args.out / "pre_run_reward_analysis",
        )
        (args.out / "pre_run_reward_analysis.json").write_text(json.dumps(pre_run_reward_analysis, indent=2) + "\n")
    if args.pareto_action_prior:
        action_prior_rules = load_pareto_action_prior_rules(
            ActionPriorConfig(
                llm_backend=args.llm_backend,
                llm_model=args.llm_model,
                task="+".join(tasks),
                tasks=tasks,
                arm="global",
                env_summary=env_summary,
                log_dir=args.out / "pareto_action_prior",
                max_weight=args.max_action_prior_weight,
                max_checkups=args.action_prior_max_checkups,
                previous_run_context=previous_run_context,
                pre_run_reward_analysis=pre_run_reward_analysis,
                candidate_count=args.action_prior_candidates,
                selection_envs=args.selection_envs,
                selection_steps=args.selection_steps,
                selection_seed=args.selection_seed,
                success_hold_seconds=args.success_hold_seconds,
                success_lift_threshold=args.success_lift_threshold,
                min_contacts=args.min_contacts,
                max_xy_disp=args.max_xy_disp,
            ),
            bias_spec,
            env,
        )
        bias_spec = dict(bias_spec)
        bias_spec["action_priors"] = action_prior_rules
    elif args.llm_action_prior:
        action_prior_rules = load_action_prior_rules(
            ActionPriorConfig(
                llm_backend=args.llm_backend,
                llm_model=args.llm_model,
                task="+".join(tasks),
                tasks=tasks,
                arm="global",
                env_summary=env_summary,
                log_dir=args.out / "llm_action_prior",
                max_weight=args.max_action_prior_weight,
                max_checkups=args.action_prior_max_checkups,
                previous_run_context=previous_run_context,
                pre_run_reward_analysis=pre_run_reward_analysis,
            ),
            bias_spec,
        )
        bias_spec = dict(bias_spec)
        bias_spec["action_priors"] = action_prior_rules
    if args.pareto_supervised_init:
        supervised_target_rules = load_pareto_supervised_target_rules(
            ActionPriorConfig(
                llm_backend=args.llm_backend,
                llm_model=args.llm_model,
                task="+".join(tasks),
                tasks=tasks,
                arm="global",
                env_summary=env_summary,
                log_dir=args.out / "pareto_supervised_init",
                max_weight=args.max_supervised_target_weight,
                previous_run_context=previous_run_context,
                pre_run_reward_analysis=pre_run_reward_analysis,
                candidate_count=args.supervised_candidates,
                selection_envs=args.selection_envs,
                selection_steps=args.selection_steps,
                selection_seed=args.selection_seed + 50_000,
                success_hold_seconds=args.success_hold_seconds,
                success_lift_threshold=args.success_lift_threshold,
                min_contacts=args.min_contacts,
                max_xy_disp=args.max_xy_disp,
            ),
            bias_spec,
            env,
        )
        bias_spec = dict(bias_spec)
        bias_spec["supervised_targets"] = supervised_target_rules
    (args.out / "bias_spec.json").write_text(json.dumps(bias_spec, indent=2) + "\n")
    compiled_bias = compile_bias(bias_spec, env)

    # Shared legacy sub-prior library for the situation-dependent prior arms (constant across a
    # bake-off so only the gating discipline varies). None -> composed_priors.default_library().
    prior_library = None
    if args.prior_library_json is not None:
        prior_library = json.loads(Path(args.prior_library_json).read_text())

    # Build the closed-loop curriculum teacher ONCE (shared across supervised_init arms/seeds),
    # validate it with the staged contact-gated metric, and only commit it as the warm-start
    # teacher if it actually reaches contact-grasp-lift -- otherwise fall back to no teacher
    # (legacy static targets). Phase A uses the hand-written default program; an LLM-authored
    # program (Phase B) can be supplied via --phase-program-json.
    phase_teacher = None
    if args.phase_teacher and any(bool(BIAS_ARMS[arm][3]) for arm in arms):
        program = None
        if args.phase_program_json is not None:
            program = json.loads(Path(args.phase_program_json).read_text())
        teacher, teacher_record = load_phase_teacher(
            env,
            task="+".join(tasks) if len(tasks) == 1 else tasks[0],
            program=program,
            log_dir=args.out / "phase_teacher",
            validate_envs=args.selection_envs,
            seed=args.selection_seed,
            min_contacts=args.min_contacts,
            max_xy_disp=args.max_xy_disp,
            lift_threshold=args.success_lift_threshold,
            hold_seconds=args.success_hold_seconds,
        )
        gate = float(teacher_record.get("final_phase_frac", 0.0))
        if gate >= args.phase_teacher_min_progress:
            phase_teacher = teacher
            print(f"[phase_teacher] committed: staged_score={teacher_record['staged_score']} "
                  f"final_phase_frac={gate} contact_gated_success={teacher_record['contact_gated_success']}")
        else:
            print(f"[phase_teacher] REJECTED (final_phase_frac={gate} < "
                  f"{args.phase_teacher_min_progress}); falling back to no teacher")

    # Generate the frozen reward program ONCE, globally, so every arm and seed trains on the
    # IDENTICAL shaping. This is the fairness fix: the only thing that differs between the
    # `reward` and `reward_action_prior` arms is the action prior, not the reward program.
    shared_program = None
    if args.initial_reward_program:
        shared_coach = DynamicRewardCoach(DynamicRewardConfig(
            llm_backend=args.llm_backend,
            llm_model=args.llm_model,
            arm="shared",
            task="+".join(tasks),
            log_dir=args.out / "shared_reward_program",
            cheap_checkup_steps=args.cheap_checkup_steps,
            deep_checkup_seconds=args.deep_checkup_seconds,
            max_template_weight=args.max_template_weight,
            min_base_reward_weight=args.min_base_reward_weight,
            allow_reward_rewrite=True,
            previous_run_context=previous_run_context,
            pre_run_reward_analysis=pre_run_reward_analysis,
        ))
        shared_spec, shared_weights, shared_record = shared_coach.initial_reward_program(bias_spec)
        shared_program = {
            "spec": shared_spec,
            "weights": shared_weights,
            "base_reward_weight": shared_coach.base_reward_weight,
            "adaptive_terms": shared_spec.get("adaptive_reward_terms", []),
            "record": shared_record,
        }
        (args.out / "shared_initial_reward_program.json").write_text(json.dumps({
            "base_reward_weight": shared_program["base_reward_weight"],
            "weights": shared_weights,
            "adaptive_reward_terms": shared_program["adaptive_terms"],
        }, indent=2) + "\n")

    unit_count = max(1, len(tasks) * len(seeds) * len(arms))
    target_train_seconds = args.target_arm_seconds
    if target_train_seconds is None and args.target_total_seconds is not None:
        target_train_seconds = args.target_total_seconds / unit_count
    cfg = PPOBiasConfig(
        iters=args.iters,
        envs=args.envs,
        lr=args.lr,
        gamma=args.gamma,
        lam=args.lam,
        hidden=tuple(args.hidden),
        ent_coef=args.ent_coef,
        supervised_steps=args.supervised_steps,
        supervised_batch=args.supervised_batch,
        supervised_lr=args.supervised_lr,
        bc_critic_pretrain=not args.no_bc_critic_pretrain,
        bc_rollout_states=not args.no_bc_rollout_states,
        bc_kl_coef=args.bc_kl_coef,
        bc_kl_anneal_iters=args.bc_kl_anneal_iters,
        checkpoint_count=args.checkpoint_count,
        target_train_seconds=target_train_seconds,
        max_env_steps=args.max_env_steps,
        action_transform=args.action_transform,
        saturation_penalty=args.saturation_penalty,
        saturation_threshold=args.saturation_threshold,
        prior_logit_clip=args.prior_logit_clip,
        action_target_reward_weight=args.action_target_reward_weight,
        success_hold_seconds=args.success_hold_seconds,
        success_lift_threshold=args.success_lift_threshold,
        warmup_compile=args.warmup_compile,
    )
    config = {
        "learner": "ppo_dynamic_reward",
        "tasks": tasks,
        "arms": arms,
        "arm_mechanisms": {
            arm: {
                "reward": bool(BIAS_ARMS[arm][0]),
                "action_prior": bool(BIAS_ARMS[arm][1]),
                "exploration": bool(BIAS_ARMS[arm][2]),
                "supervised_init": bool(BIAS_ARMS[arm][3]),
            }
            for arm in arms
        },
        "seeds": seeds,
        "prior_program_arms": {arm: prior_program_for_arm(arm, library=prior_library)
                               for arm in arms if prior_program_for_arm(arm, library=prior_library) is not None},
        "prior_library": prior_library or default_library(),
        "env": env_summary,
        "ppo": cfg.__dict__,
        "dynamic_reward": {
            "cheap_checkup_steps": args.cheap_checkup_steps,
            "deep_checkup_seconds": args.deep_checkup_seconds,
            "max_template_weight": args.max_template_weight,
            "min_base_reward_weight": args.min_base_reward_weight,
            "post_new_reward_checkup_steps": args.post_new_reward_checkup_steps,
            "post_new_reward_fast_checkups": args.post_new_reward_fast_checkups,
            "allow_reward_rewrite": args.allow_reward_rewrite,
            "pre_run_reward_analysis": args.pre_run_reward_analysis,
            "initial_reward_program": args.initial_reward_program,
            "freeze_reward_shaping": args.freeze_reward_shaping,
            "anneal_shaping": args.anneal_shaping,
            "shaping_anneal_start_fraction": args.shaping_anneal_start_fraction,
            "max_env_steps": args.max_env_steps,
            "efficiency_success_threshold": args.efficiency_success_threshold,
            "rewrite_fraction": args.rewrite_fraction,
            "previous_run_dir": str(args.previous_run_dir) if args.previous_run_dir is not None else None,
            "xla_cache": str(Path(".xla_cache").resolve()),
        },
        "dynamic_action_prior": {
            "llm_action_prior": args.llm_action_prior,
            "pareto_action_prior": args.pareto_action_prior,
            "action_prior_candidates": args.action_prior_candidates,
            "pareto_supervised_init": args.pareto_supervised_init,
            "supervised_candidates": args.supervised_candidates,
            "selection_envs": args.selection_envs,
            "selection_steps": args.selection_steps,
            "selection_seed": args.selection_seed,
            "checkup_steps": args.action_prior_checkup_steps,
            "max_checkups": args.action_prior_max_checkups,
            "max_weight": args.max_action_prior_weight,
        },
        "llm_backend": args.llm_backend,
        "llm_model": args.llm_model,
    }
    (args.out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (args.out / "reward_templates.json").write_text(json.dumps(reward_template_metadata(), indent=2) + "\n")
    if previous_run_context:
        (args.out / "previous_run_context.json").write_text(json.dumps(previous_run_context, indent=2) + "\n")

    metrics_path = args.out / "metrics.jsonl"
    eval_rows: list[dict[str, Any]] = []
    with metrics_path.open("w") as metrics_file:
        for task in tasks:
            for seed in seeds:
                for arm in arms:
                    run_dir = args.out / f"{task}_s{seed}_{arm}"
                    run_dir.mkdir(parents=True, exist_ok=True)
                    coach = DynamicRewardCoach(DynamicRewardConfig(
                        llm_backend=args.llm_backend,
                        llm_model=args.llm_model,
                        arm=arm,
                        task=task,
                        log_dir=run_dir / "reward_checkups",
                        cheap_checkup_steps=args.cheap_checkup_steps,
                        deep_checkup_seconds=args.deep_checkup_seconds,
                        max_template_weight=args.max_template_weight,
                        min_base_reward_weight=args.min_base_reward_weight,
                        post_new_reward_checkup_steps=args.post_new_reward_checkup_steps,
                        post_new_reward_fast_checkups=args.post_new_reward_fast_checkups,
                        allow_reward_rewrite=args.allow_reward_rewrite and bool(BIAS_ARMS[arm][0]),
                        freeze_reward_shaping=args.freeze_reward_shaping,
                        previous_run_context=previous_run_context,
                        pre_run_reward_analysis=pre_run_reward_analysis,
                        anneal_shaping=args.anneal_shaping and bool(BIAS_ARMS[arm][0]),
                        target_train_seconds=target_train_seconds,
                        shaping_anneal_start_fraction=args.shaping_anneal_start_fraction,
                    ))
                    active_bias = compiled_bias
                    # "No shaped reward" arms (reward bias off) train on the base contact-gated
                    # reward ONLY -- zero shaping-template weights -- so they are a true base-reward
                    # baseline. Reward arms get the front-loaded LLM shaping program below. (Without
                    # this, default_reward_template_weights leaks lift_basin_curriculum=1.0 into the
                    # "no shaped reward" arms, which would invalidate the shaped-vs-unshaped contrast.)
                    active_reward_weights = default_reward_template_weights(task)
                    if not bool(BIAS_ARMS[arm][0]):
                        active_reward_weights = active_reward_weights * 0.0
                    active_base_reward_weight = coach.base_reward_weight
                    active_action_prior_weights = active_bias.default_action_prior_weights()
                    # Front-load shaping: compile the pre-run analysis into a concrete reward
                    # program before training so shaping guides exploration from iteration 0.
                    initial_program_record = None
                    rewrite_record = None
                    if shared_program is not None and bool(BIAS_ARMS[arm][0]):
                        # Adopt the single shared program (no per-arm regeneration).
                        active_reward_weights = list(shared_program["weights"])
                        coach.apply_program(shared_program["weights"], shared_program["base_reward_weight"], shared_program["adaptive_terms"])
                        active_base_reward_weight = coach.base_reward_weight
                        initial_program_record = shared_program["record"]
                        (run_dir / "initial_bias_spec.json").write_text(json.dumps(shared_program["spec"], indent=2) + "\n")
                        active_bias = compile_bias(shared_program["spec"], env)
                    # Situation-dependent prior arms: inject the arm's prior program into the
                    # (possibly reward-augmented) spec and recompile. The program is a stateless
                    # fn(obs, weights) and replaces the legacy rule-sum prior. See
                    # EXPERIMENT_situation_dependent_priors.md.
                    arm_prior_program = prior_program_for_arm(arm, library=prior_library)
                    if arm_prior_program is not None:
                        arm_spec = dict(active_bias.spec)
                        arm_spec["prior_program"] = arm_prior_program
                        active_bias = compile_bias(arm_spec, env)
                        active_action_prior_weights = active_bias.default_action_prior_weights()
                        (run_dir / "prior_program.json").write_text(json.dumps(arm_prior_program, indent=2) + "\n")
                    prior_coach = None
                    if bool(BIAS_ARMS[arm][1]) and args.action_prior_checkup_steps > 0 and arm_prior_program is None:
                        prior_coach = ActionPriorCoach(
                            ActionPriorConfig(
                                llm_backend=args.llm_backend,
                                llm_model=args.llm_model,
                                task=task,
                                tasks=tasks,
                                arm=arm,
                                env_summary=env_summary,
                                log_dir=run_dir / "action_prior_checkups",
                                max_weight=args.max_action_prior_weight,
                                max_checkups=args.action_prior_max_checkups,
                                previous_run_context=previous_run_context,
                                pre_run_reward_analysis=pre_run_reward_analysis,
                            ),
                            list(active_bias.spec.get("action_priors", [])),
                        )
                    arm_phase_teacher = phase_teacher if bool(BIAS_ARMS[arm][3]) else None
                    should_rewrite = args.allow_reward_rewrite and bool(BIAS_ARMS[arm][0]) and not args.freeze_reward_shaping
                    if should_rewrite:
                        first_cfg, second_cfg = split_cfg_for_rewrite(cfg, args.rewrite_fraction)
                        params, first_rows, best_params, best_success, best_iter = train_ppo_arm(
                            env=env,
                            bias=active_bias,
                            task=task,
                            arm=arm,
                            seed=seed,
                            cfg=first_cfg,
                            checkpoint_dir=run_dir / "checkpoints",
                            reward_weights=active_reward_weights,
                            base_reward_weight=active_base_reward_weight,
                            checkup_interval=args.cheap_checkup_steps,
                            checkup_fn=coach,
                            action_prior_weights=active_action_prior_weights,
                            action_prior_checkup_interval=args.action_prior_checkup_steps,
                            action_prior_checkup_fn=prior_coach,
                            phase_teacher=arm_phase_teacher,
                        )
                        if prior_coach is not None:
                            active_action_prior_weights = prior_coach.current_weights
                        rewrite_ctx = {
                            "rows": first_rows,
                            "elapsed_seconds": first_rows[-1]["elapsed_seconds"] if first_rows else 0.0,
                            "iter": first_rows[-1]["iter"] if first_rows else -1,
                            "arm": arm,
                            "task": task,
                        }
                        adapted_spec, active_reward_weights, rewrite_record = coach.rewrite_reward_code(rewrite_ctx, bias_spec)
                        active_base_reward_weight = coach.base_reward_weight
                        if arm_prior_program is not None:
                            adapted_spec = {**adapted_spec, "prior_program": arm_prior_program}
                        (run_dir / "adapted_bias_spec.json").write_text(json.dumps(adapted_spec, indent=2) + "\n")
                        active_bias = compile_bias(adapted_spec, env)
                        params, second_rows, best2_params, best2_success, best2_iter = train_ppo_arm(
                            env=env,
                            bias=active_bias,
                            task=task,
                            arm=arm,
                            seed=seed + 1_000_000,
                            cfg=second_cfg,
                            checkpoint_dir=run_dir / "checkpoints",
                            reward_weights=active_reward_weights,
                            base_reward_weight=active_base_reward_weight,
                            checkup_interval=args.cheap_checkup_steps,
                            checkup_fn=coach,
                            action_prior_weights=active_action_prior_weights,
                            action_prior_checkup_interval=args.action_prior_checkup_steps,
                            action_prior_checkup_fn=prior_coach,
                            initial_params=params,
                            iter_offset=len(first_rows),
                        )
                        # Best across both segments (held-out best-checkpoint, not collapsed final).
                        if best2_success >= best_success:
                            best_params, best_success, best_iter = best2_params, best2_success, best2_iter
                        active_reward_weights = coach.applied_weights()
                        active_base_reward_weight = coach.applied_base_reward_weight()
                        if prior_coach is not None:
                            active_action_prior_weights = prior_coach.current_weights
                        rows = [
                            row | {"segment": "pre_rewrite"} for row in first_rows
                        ] + [
                            row | {"segment": "post_rewrite"} for row in second_rows
                        ]
                        (run_dir / "reward_rewrite_summary.json").write_text(json.dumps(rewrite_record, indent=2) + "\n")
                    else:
                        params, rows, best_params, best_success, best_iter = train_ppo_arm(
                            env=env,
                            bias=active_bias,
                            task=task,
                            arm=arm,
                            seed=seed,
                            cfg=cfg,
                            checkpoint_dir=run_dir / "checkpoints",
                            reward_weights=active_reward_weights,
                            base_reward_weight=active_base_reward_weight,
                            checkup_interval=args.cheap_checkup_steps,
                            checkup_fn=coach,
                            action_prior_weights=active_action_prior_weights,
                            action_prior_checkup_interval=args.action_prior_checkup_steps,
                            action_prior_checkup_fn=prior_coach,
                            phase_teacher=arm_phase_teacher,
                        )
                        active_reward_weights = coach.applied_weights()
                        active_base_reward_weight = coach.applied_base_reward_weight()
                        if prior_coach is not None:
                            active_action_prior_weights = prior_coach.current_weights
                    for row in rows:
                        row = row | {"seed": seed}
                        metrics_file.write(json.dumps(row) + "\n")
                        metrics_file.flush()
                    with (run_dir / "params.pkl").open("wb") as f:
                        pickle.dump(params, f)
                    (run_dir / "final_reward_weights.json").write_text(
                        json.dumps({name: float(value) for name, value in zip(REWARD_TEMPLATE_NAMES, active_reward_weights)}, indent=2) + "\n"
                    )
                    (run_dir / "final_base_reward_weight.json").write_text(
                        json.dumps({"base_reward_weight": float(active_base_reward_weight)}, indent=2) + "\n"
                    )
                    (run_dir / "final_action_prior_weights.json").write_text(
                        json.dumps(_action_prior_weight_dict(active_bias, active_action_prior_weights), indent=2) + "\n"
                    )
                    # Consolidated, easy-to-inspect record of every shaped reward this arm
                    # generated: the pre-training program, the mid-run feedback rewrite, and the
                    # final (post-anneal) weights actually used at evaluation.
                    def _terms(record: Any) -> Any:
                        if isinstance(record, dict) and isinstance(record.get("decision"), dict):
                            return record["decision"].get("adaptive_reward_terms")
                        return None
                    (run_dir / "generated_shaped_rewards.json").write_text(
                        json.dumps({
                            "arm": arm,
                            "task": task,
                            "initial_reward_program": {
                                "status": (initial_program_record or {}).get("status", "not_generated"),
                                "base_reward_weight": (initial_program_record or {}).get("base_reward_weight"),
                                "adaptive_reward_terms": _terms(initial_program_record),
                            },
                            "mid_run_rewrite": {
                                "status": (rewrite_record or {}).get("status", "not_generated"),
                                "base_reward_weight": (rewrite_record or {}).get("base_reward_weight"),
                                "adaptive_reward_terms": _terms(rewrite_record),
                            },
                            "final_reward_weights": {
                                name: float(value)
                                for name, value in zip(REWARD_TEMPLATE_NAMES, active_reward_weights)
                            },
                            "final_base_reward_weight": float(active_base_reward_weight),
                            "anneal_factor_at_end": round(float(coach.last_anneal_factor), 6),
                        }, indent=2) + "\n"
                    )
                    # Evaluate the best-by-training-success checkpoint, not the (possibly
                    # coach-collapsed) final policy, so a late bad intervention doesn't tank the
                    # reported ceiling. Final policy is still saved to params.pkl.
                    eval_row = {
                        "task": task,
                        "seed": seed,
                        "arm": arm,
                        "best_checkpoint_iter": int(best_iter),
                        "best_train_success": round(float(best_success), 6),
                        **evaluate_ppo_policy(
                            env=env,
                            params=best_params,
                            bias=active_bias,
                            task=task,
                            arm=arm,
                            seed=seed + 10_000,
                            n_envs=args.eval_envs,
                            cfg=cfg,
                            reward_weights=active_reward_weights,
                            base_reward_weight=active_base_reward_weight,
                            action_prior_weights=active_action_prior_weights,
                        ),
                        **learning_efficiency(rows, success_threshold=args.efficiency_success_threshold),
                    }
                    (run_dir / "eval.json").write_text(json.dumps(eval_row, indent=2) + "\n")
                    eval_rows.append(eval_row)
                    # Free this arm's compiled executables before the next arm. Each arm builds its
                    # own jitted reset/step/update/collect; sequential arms otherwise accumulate the
                    # XLA compilation cache and OOM the GPU around the 4th arm on an 8 GB card
                    # (observed: a 6-arm run killed mid-arm-4 with EXIT 137 / SIGKILL).
                    jax.clear_caches()
                    gc.collect()

    summary = summarize(eval_rows)
    # Fold in sample-efficiency metrics (steps-to-threshold, step-normalized AUC) so the
    # headline comparison is about data, not wall-clock.
    for arm in summary:
        arm_rows = [r for r in eval_rows if r["arm"] == arm]
        reached = [r for r in arm_rows if r.get("reached_success_threshold")]
        steps_vals = [r["steps_to_success_threshold"] for r in reached if r.get("steps_to_success_threshold") is not None]
        auc_vals = [float(r.get("success_step_auc", 0.0)) for r in arm_rows]
        total_vals = [int(r.get("total_env_steps", 0)) for r in arm_rows]
        summary[arm].update({
            "efficiency_success_threshold": args.efficiency_success_threshold,
            "reached_success_threshold_frac": round(len(reached) / len(arm_rows), 6) if arm_rows else 0.0,
            "mean_steps_to_success_threshold": round(sum(steps_vals) / len(steps_vals), 1) if steps_vals else None,
            "mean_success_step_auc": round(sum(auc_vals) / len(auc_vals), 6) if auc_vals else 0.0,
            "mean_total_env_steps": round(sum(total_vals) / len(total_vals), 1) if total_vals else 0.0,
        })
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_csv(args.out / "eval.csv", eval_rows)
    write_report(args.out, config, summary)
    print(f"Wrote {args.out}")
    print(json.dumps(summary, indent=2))
    return 0


def split_cfg_for_rewrite(cfg: PPOBiasConfig, rewrite_fraction: float) -> tuple[PPOBiasConfig, PPOBiasConfig]:
    fraction = min(max(float(rewrite_fraction), 0.05), 0.95)
    first_seconds = None
    second_seconds = None
    if cfg.target_train_seconds is not None:
        first_seconds = cfg.target_train_seconds * fraction
        second_seconds = cfg.target_train_seconds - first_seconds
    # The env-step break in train_ppo_arm counts *global* cumulative steps (iter_offset based),
    # whereas the time budget resets per segment. So the first segment stops at the rewrite
    # fraction of the budget, and the second segment carries the full (cumulative) budget.
    first_steps = None
    second_steps = None
    if cfg.max_env_steps is not None:
        first_steps = max(1, round(cfg.max_env_steps * fraction))
        second_steps = cfg.max_env_steps
    first_iters = max(1, round(cfg.iters * fraction))
    second_iters = max(1, cfg.iters - first_iters)
    first_ckpts = max(0, round(cfg.checkpoint_count * fraction))
    second_ckpts = max(0, cfg.checkpoint_count - first_ckpts)
    return (
        replace(cfg, iters=first_iters, target_train_seconds=first_seconds, max_env_steps=first_steps, checkpoint_count=first_ckpts),
        replace(cfg, iters=second_iters, target_train_seconds=second_seconds, max_env_steps=second_steps, checkpoint_count=second_ckpts),
    )


def learning_efficiency(rows: list[dict[str, Any]], *, success_threshold: float) -> dict[str, Any]:
    """Sample-efficiency metrics from a single arm's training curve, keyed on environment
    steps (the resource that is scarce on real robots). Reports the env-step count at which
    sustained success first crosses ``success_threshold`` and the step-normalized area under
    the success curve (mean success per collected step)."""
    pts = sorted(
        (int(r["env_steps"]), float(r["success"]))
        for r in rows
        if r.get("env_steps") is not None and r.get("success") is not None
    )
    if not pts:
        return {
            "efficiency_success_threshold": success_threshold,
            "steps_to_success_threshold": None,
            "reached_success_threshold": False,
            "success_step_auc": 0.0,
            "total_env_steps": 0,
        }
    steps_to = next((s for s, v in pts if v >= success_threshold), None)
    auc = sum(0.5 * (v0 + v1) * (s1 - s0) for (s0, v0), (s1, v1) in zip(pts, pts[1:]))
    span = pts[-1][0] - pts[0][0]
    mean_success = auc / span if span > 0 else pts[-1][1]
    return {
        "efficiency_success_threshold": success_threshold,
        "steps_to_success_threshold": steps_to,
        "reached_success_threshold": steps_to is not None,
        "success_step_auc": round(mean_success, 6),
        "total_env_steps": pts[-1][0],
    }


def load_previous_run_context(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    context: dict[str, Any] = {"run_dir": str(path)}
    summary_path = path / "summary.json"
    if summary_path.exists():
        context["summary"] = _read_json(summary_path)
    evals: dict[str, Any] = {}
    for eval_path in sorted(path.glob("*_s*_*/eval.json")):
        data = _read_json(eval_path)
        if isinstance(data, dict):
            evals[str(data.get("arm") or eval_path.parent.name)] = data
    if evals:
        context["evals"] = evals
    deep_notes: dict[str, str] = {}
    for completion in sorted(path.glob("*_s*_*/reward_checkups/llm/deep_*_completion.txt")):
        text = completion.read_text(errors="replace")
        deep_notes[completion.parts[-4]] = text[:3000]
    if deep_notes:
        context["deep_checkup_notes"] = deep_notes
    return context


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _action_prior_weight_dict(bias: Any, weights: Any) -> dict[str, float]:
    rules = list(bias.spec.get("action_priors", []))
    return {
        str(rule.get("name", f"prior_{idx}")): round(float(value), 6)
        for idx, (rule, value) in enumerate(zip(rules, weights))
    }


def write_report(out: Path, config: dict[str, Any], summary: dict[str, Any]) -> None:
    lines = [
        "# Dynamic Reward Combination Experiment",
        "",
        f"Created: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Configuration",
        "",
        f"- Arms: `{','.join(config['arms'])}`",
        f"- Tasks: `{','.join(config['tasks'])}`",
        f"- Target seconds per arm: `{config['ppo']['target_train_seconds']}`",
        f"- Cheap checkup steps: `{config['dynamic_reward']['cheap_checkup_steps']}`",
        f"- Deep checkup seconds: `{config['dynamic_reward']['deep_checkup_seconds']}`",
        f"- Post-new-reward checkup steps: `{config['dynamic_reward']['post_new_reward_checkup_steps']}`",
        f"- Post-new-reward fast checkups: `{config['dynamic_reward']['post_new_reward_fast_checkups']}`",
        f"- Reward rewrite: `{config['dynamic_reward']['allow_reward_rewrite']}`",
        f"- Rewrite fraction: `{config['dynamic_reward']['rewrite_fraction']}`",
        f"- LLM action prior: `{config['dynamic_action_prior']['llm_action_prior']}`",
        f"- Action prior checkup steps: `{config['dynamic_action_prior']['checkup_steps']}`",
        f"- Action transform: `{config['ppo']['action_transform']}`",
        "",
        "## Summary",
        "",
        "```json",
        json.dumps(summary, indent=2),
        "```",
        "",
        "## Notes",
        "",
        "Reward coach prompts intentionally omit env reward, train return, base return, and PPO losses.",
        "Rejected and deep-checkup proposals are stored under each arm's `reward_checkups/` directory.",
    ]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm-backend", required=True)
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--bias-spec", type=Path, default=None)
    parser.add_argument("--tasks", default="lift")
    parser.add_argument("--arms", default="baseline,reward_action_prior,reward_supervised_init")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--iters", type=int, default=100000)
    parser.add_argument("--envs", type=int, default=1024)
    parser.add_argument("--eval-envs", type=int, default=1024)
    parser.add_argument("--hidden", type=int, nargs="+", default=[128, 128])
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--supervised-steps", type=int, default=80)
    parser.add_argument("--supervised-batch", type=int, default=128)
    parser.add_argument("--supervised-lr", type=float, default=1e-3)
    parser.add_argument("--no-bc-critic-pretrain", action="store_true",
                        help="Disable critic pretraining in the warm-start (revert toward legacy BC).")
    parser.add_argument("--no-bc-rollout-states", action="store_true",
                        help="BC on reset states only instead of controller-rollout states (legacy).")
    parser.add_argument("--bc-kl-coef", type=float, default=0.5,
                        help="Initial KL-anchor coefficient pulling the warm-started actor toward the "
                        "frozen BC policy; decays to 0 over --bc-kl-anneal-iters. 0 disables the anchor.")
    parser.add_argument("--bc-kl-anneal-iters", type=int, default=200)
    parser.add_argument("--episode-seconds", type=float, default=2.5)
    parser.add_argument("--control-dt", type=float, default=0.025)
    parser.add_argument("--physics-dt", type=float, default=0.01)
    parser.add_argument("--obj-xy-range", type=float, default=0.04)
    parser.add_argument("--checkpoint-count", type=int, default=5)
    parser.add_argument("--target-arm-seconds", type=float, default=None)
    parser.add_argument("--target-total-seconds", type=float, default=43200.0)
    parser.add_argument(
        "--max-env-steps",
        type=int,
        default=None,
        help="Per-arm environment-step budget (envs*horizon per iteration). Compares arms on "
        "equal data instead of equal wall-clock; the thesis-relevant scarce resource.",
    )
    parser.add_argument(
        "--efficiency-success-threshold",
        type=float,
        default=0.2,
        help="Sustained-success level used to report steps-to-threshold sample efficiency.",
    )
    parser.add_argument("--cheap-checkup-steps", type=int, default=50)
    parser.add_argument("--deep-checkup-seconds", type=float, default=7200.0)
    parser.add_argument("--max-template-weight", type=float, default=1.25)
    parser.add_argument("--min-base-reward-weight", type=float, default=0.5,
                        help="Floor on base_reward_weight so the coach cannot starve the true "
                        "objective while fighting reward hacks (a fair-baseline guard).")
    parser.add_argument("--post-new-reward-checkup-steps", type=int, default=10)
    parser.add_argument("--post-new-reward-fast-checkups", type=int, default=3)
    parser.add_argument("--allow-reward-rewrite", action="store_true")
    parser.add_argument("--pre-run-reward-analysis", action="store_true")
    parser.add_argument("--rewrite-fraction", type=float, default=0.5)
    parser.add_argument(
        "--initial-reward-program",
        action="store_true",
        help="Compile the pre-run analysis into shaped rewards before training so shaping is "
        "active from iteration 0 (the mid-run rewrite then only adds feedback-driven terms).",
    )
    parser.add_argument(
        "--anneal-shaping",
        action="store_true",
        help="Dense-to-sparse: linearly anneal adaptive shaping toward 0 (and restore base "
        "reward weight toward 1.0) over the final fraction of each arm's wall-clock budget.",
    )
    parser.add_argument("--shaping-anneal-start-fraction", type=float, default=0.7)
    parser.add_argument(
        "--freeze-reward-shaping",
        action="store_true",
        help="Fixed-baseline mode: compile the initial reward program once, then hold reward "
        "weights and base_reward_weight constant (no deep checkups, no mid-run rewrite). Gives a "
        "reproducible, maximally-shaped baseline to isolate the action prior's marginal value.",
    )
    parser.add_argument("--previous-run-dir", type=Path, default=None)
    parser.add_argument("--llm-action-prior", action="store_true")
    parser.add_argument("--pareto-action-prior", action="store_true")
    parser.add_argument("--action-prior-candidates", type=int, default=5)
    parser.add_argument("--pareto-supervised-init", action="store_true")
    parser.add_argument("--supervised-candidates", type=int, default=5)
    parser.add_argument("--selection-envs", type=int, default=128)
    parser.add_argument("--selection-steps", type=int, default=None)
    parser.add_argument("--selection-seed", type=int, default=0)
    parser.add_argument("--action-prior-checkup-steps", type=int, default=0)
    parser.add_argument("--action-prior-max-checkups", type=int, default=3)
    parser.add_argument("--max-action-prior-weight", type=float, default=0.6)
    parser.add_argument("--max-supervised-target-weight", type=float, default=0.9)
    parser.add_argument("--action-transform", choices=["raw", "tanh"], default="tanh")
    parser.add_argument("--saturation-penalty", type=float, default=0.0)
    parser.add_argument("--saturation-threshold", type=float, default=0.98)
    parser.add_argument("--prior-logit-clip", type=float, default=0.95)
    parser.add_argument("--action-target-reward-weight", type=float, default=0.0)
    parser.add_argument("--success-hold-seconds", type=float, default=0.5)
    parser.add_argument("--success-lift-threshold", type=float, default=0.05)
    # Contact-gated thresholds shared by candidate selection and the phase teacher validation.
    parser.add_argument("--min-contacts", type=float, default=1.0,
                        help="Fingertip contacts required for a 'lift' to count (anti-fling gate).")
    parser.add_argument("--max-xy-disp", type=float, default=0.08,
                        help="Max object lateral drift for a lift to count as a grasp, not a fling.")
    # Closed-loop curriculum phase teacher (warm-start). See PHASE_B_curriculum_aware_selection.md.
    parser.add_argument("--phase-teacher", dest="phase_teacher", action="store_true", default=True,
                        help="Use the closed-loop contact->close->lift phase controller as the BC "
                        "warm-start teacher for supervised_init arms.")
    parser.add_argument("--no-phase-teacher", dest="phase_teacher", action="store_false")
    parser.add_argument("--phase-program-json", type=Path, default=None,
                        help="Optional JSON phase program (Phase B / LLM-authored). Defaults to the "
                        "hand-written curriculum for the task.")
    parser.add_argument("--prior-library-json", type=Path, default=None,
                        help="Optional JSON list of legacy sub-priors [{name, rules:[...]}] shared by "
                        "the situation-dependent prior arms (prior_gate_* / prior_monolithic). "
                        "Defaults to composed_priors.default_library(). Keep it constant across a "
                        "bake-off so only the gating discipline varies.")
    parser.add_argument("--phase-teacher-min-progress", type=float, default=0.15,
                        help="Reject the teacher (fall back to no warm-start) if its validated "
                        "final_phase_frac is below this; keeps a useless teacher from poisoning BC. "
                        "A teacher that reliably APPROACHES (reach phase 1) is already a strong "
                        "warm-start basin even if it can't autonomously complete the lift.")
    parser.add_argument("--warmup-compile", dest="warmup_compile", action="store_true", default=True)
    parser.add_argument("--no-warmup-compile", dest="warmup_compile", action="store_false")
    parser.add_argument("--no-coach", action="store_true",
                        help="Deactivate the DynamicRewardCoach entirely: no LLM reward authoring, "
                        "no mid-run rewrite/recompile, no checkups. 'Shaped reward' arms become a "
                        "fixed hand-written shaping control (base reward + reward templates). Use for "
                        "clean null-hypothesis testing of whether shaping helps.")
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())

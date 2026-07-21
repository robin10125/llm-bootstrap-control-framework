# Experimental runs log

This directory is a normalized historical log of every substantive top-level experimental run found under `runs/` and the older `famework_testing/runs/` archive. Each run has a separate Markdown report with date evidence, an abstract, experimental structure, quantitative results, interpretation, limitations, and links to the original artifacts.

- **Substantive run entries:** 111
- **Excluded smoke/support entries:** 40
- **Inventory date:** 2026-07-14

## Inclusion policy

A directory is included when it represents a substantive training, comparison, generation, selection, diagnostic, or failed/interrupted experimental attempt. A completed success is not required: preserving negative and partial evidence prevents survivorship bias. Directories explicitly named `smoke` are excluded, as are render-only collections and empty debug placeholders. Short runs not named as smoke are retained when their artifacts show they were used as a scientific diagnostic or comparison.

Dates come first from run-directory timestamps, then from contemporaneous reports/logs, and finally from the earliest surviving artifact modification time. The per-run report states which source was used.

## Chronological index

| Date | Run | Status |
|---|---|---|
| 2026-06-12 22:54:45 | [exports/export_experiment/source/bootstrapping/runs_rl_s0.log](exports_bootstrapping__runs_rl_s0.md) | failed/aborted attempt; no training rows or terminal result survive |
| 2026-06-12 23:51:15 | [exports/export_experiment/source/bootstrapping/runs_rl_baseline.log](exports_bootstrapping__runs_rl_baseline.md) | completed |
| 2026-06-13 01:01:50 | [exports/export_experiment/source/bootstrapping/runs_rl_codex.log](exports_bootstrapping__runs_rl_codex.md) | completed |
| 2026-06-15 | [famework_testing/runs/conditional_vs_flat_video](famework_testing__conditional_vs_flat_video.md) | completed or completed-with-caveats |
| 2026-06-15 13:26:15 | [famework_testing/runs/real_shadow_codex_lift_s0](famework_testing__real_shadow_codex_lift_s0.md) | completed or completed-with-caveats |
| 2026-06-15 23:08:26 | [famework_testing/runs/shadowhand_recursive_codex_replay](famework_testing__shadowhand_recursive_codex_replay.md) | completed or completed-with-caveats |
| 2026-06-16 | [famework_testing/runs/shadowhand_actual_coords_codex_recompiled](famework_testing__shadowhand_actual_coords_codex_recompiled.md) | artifact-only or generation-only; no terminal evaluation recorded |
| 2026-06-16 00:15:36 | [famework_testing/runs/shadowhand_actual_coords_codex](famework_testing__shadowhand_actual_coords_codex.md) | completed or completed-with-caveats |
| 2026-06-17 | [famework_testing/runs/policy_bias_lab_shadow_isolation_chunk4_s0](famework_testing__policy_bias_lab_shadow_isolation_chunk4_s0.md) | artifact-only or generation-only; no terminal evaluation recorded |
| 2026-06-17 | [famework_testing/runs/policy_bias_lab_shadow_isolation_env16_s0](famework_testing__policy_bias_lab_shadow_isolation_env16_s0.md) | artifact-only or generation-only; no terminal evaluation recorded |
| 2026-06-17 | [famework_testing/runs/policy_bias_lab_shadow_isolation_pilot_s0](famework_testing__policy_bias_lab_shadow_isolation_pilot_s0.md) | completed or completed-with-caveats |
| 2026-06-17 | [famework_testing/runs/policy_bias_lab_shadow_isolation_s0](famework_testing__policy_bias_lab_shadow_isolation_s0.md) | artifact-only or generation-only; no terminal evaluation recorded |
| 2026-06-17 | [famework_testing/runs/policy_bias_lab_shadow_isolation_vec_s0](famework_testing__policy_bias_lab_shadow_isolation_vec_s0.md) | artifact-only or generation-only; no terminal evaluation recorded |
| 2026-06-17 16:56:13 | [runs/policy_bias_lab_30min_lift_isolation_20260617-165613](policy_bias_lab_30min_lift_isolation_20260617-165613.md) | artifact-only or generation-only; no terminal evaluation recorded |
| 2026-06-17 17:17:54 | [runs/policy_bias_lab_30min_lift_isolation_short_horizon_20260617-171754](policy_bias_lab_30min_lift_isolation_short_horizon_20260617-171754.md) | artifact-only or generation-only; no terminal evaluation recorded |
| 2026-06-17 17:29:09 | [runs/policy_bias_lab_30min_lift_isolation_env16_20260617-172909](policy_bias_lab_30min_lift_isolation_env16_20260617-172909.md) | artifact-only or generation-only; no terminal evaluation recorded |
| 2026-06-17 17:40:31 | [runs/policy_bias_lab_30min_lift_isolation_runnable_20260617-174031](policy_bias_lab_30min_lift_isolation_runnable_20260617-174031.md) | artifact-only or generation-only; no terminal evaluation recorded |
| 2026-06-17 17:51:44 | [runs/policy_bias_lab_30min_lift_isolation_env1_20260617-175144](policy_bias_lab_30min_lift_isolation_env1_20260617-175144.md) | completed or completed-with-caveats |
| 2026-06-18 00:46:40 | [runs/policy_bias_ppo_shadow_lift_1h_isolation_20260618-004640](policy_bias_ppo_shadow_lift_1h_isolation_20260618-004640.md) | completed or completed-with-caveats |
| 2026-06-18 12:39:03 | [runs/policy_bias_ppo_shadow_lift_200iter_20260618-123903](policy_bias_ppo_shadow_lift_200iter_20260618-123903.md) | completed or completed-with-caveats |
| 2026-06-19 03:09:00 | [runs/policy_bias_ppo_basin_reward_200iter_20260619-030900](policy_bias_ppo_basin_reward_200iter_20260619-030900.md) | completed or completed-with-caveats |
| 2026-06-19 13:46:50 | [runs/saturation_strategy_compare_20260619-134650](saturation_strategy_compare_20260619-134650.md) | completed or completed-with-caveats |
| 2026-06-19 17:49:24 | [runs/dynamic_reward_combo_20260619-174924](dynamic_reward_combo_20260619-174924.md) | completed or completed-with-caveats |
| 2026-06-20 08:48:46 | [runs/dynamic_reward_rewrite_combo_20260620-084846](dynamic_reward_rewrite_combo_20260620-084846.md) | artifact-only or generation-only; no terminal evaluation recorded |
| 2026-06-20 08:49:50 | [runs/dynamic_reward_rewrite_combo_20260620-084950](dynamic_reward_rewrite_combo_20260620-084950.md) | completed or completed-with-caveats |
| 2026-06-21 20:23:52 | [runs/pre_run_failure_action_prior_codex_20260621-202352](pre_run_failure_action_prior_codex_20260621-202352.md) | artifact-only or generation-only; no terminal evaluation recorded |
| 2026-06-22 12:57:50 | [runs/action_prior_general_prompt_2h_20260622-125750](action_prior_general_prompt_2h_20260622-125750.md) | completed or completed-with-caveats |
| 2026-06-22 18:17:01 | [runs/pareto_fixed_prior_shaped_reward_12h_20260622-181701](pareto_fixed_prior_shaped_reward_12h_20260622-181701.md) | artifact-only or generation-only; no terminal evaluation recorded |
| 2026-06-22 19:06:03 | [runs/pareto_fixed_prior_shaped_reward_12h_512env_20260622-190603](pareto_fixed_prior_shaped_reward_12h_512env_20260622-190603.md) | artifact-only or generation-only; no terminal evaluation recorded |
| 2026-06-22 19:35:56 | [runs/pareto_fixed_prior_shaped_reward_12h_128env_20260622-193556](pareto_fixed_prior_shaped_reward_12h_128env_20260622-193556.md) | artifact-only or generation-only; no terminal evaluation recorded |
| 2026-06-22 20:07:26 | [runs/pareto_fixed_prior_shaped_reward_12h_32env_20260622-200726](pareto_fixed_prior_shaped_reward_12h_32env_20260622-200726.md) | artifact-only or generation-only; no terminal evaluation recorded |
| 2026-06-22 20:29:13 | [runs/diagnostic_no_warmup_32env_20260622-202913](diagnostic_no_warmup_32env_20260622-202913.md) | completed or completed-with-caveats |
| 2026-06-22 20:32:37 | [runs/pareto_fixed_prior_shaped_reward_12h_nowarmup_20260622-203237](pareto_fixed_prior_shaped_reward_12h_nowarmup_20260622-203237.md) | completed or completed-with-caveats |
| 2026-06-23 15:19:04 | [runs/shaped_vs_shaped_prior_2h_20260623-151904](shaped_vs_shaped_prior_2h_20260623-151904.md) | completed or completed-with-caveats |
| 2026-06-24 11:06:26 | [runs/frozen_baseline_20260624-110626](frozen_baseline_20260624-110626.md) | completed or completed-with-caveats |
| 2026-06-25 20:18:23 | [runs/test1h_20260625-201823](test1h_20260625-201823.md) | completed or completed-with-caveats |
| 2026-06-26 01:09:10 | [runs/eight_hour_6arm_20260626-010910](eight_hour_6arm_20260626-010910.md) | partial/interrupted; training metrics survive but no complete run-level report |
| 2026-06-26 12:33:19 | [runs/reward_fix_test_20260626-123319](reward_fix_test_20260626-123319.md) | completed or completed-with-caveats |
| 2026-06-26 13:57:31 | [runs/pbrs_test_20260626-135731](pbrs_test_20260626-135731.md) | completed or completed-with-caveats |
| 2026-06-26 16:50:30 | [runs/twelve_hour_6arm_20260626-165030](twelve_hour_6arm_20260626-165030.md) | completed or completed-with-caveats |
| 2026-06-27 17:16:33 | [runs/grip_test_20260627-171633](grip_test_20260627-171633.md) | completed or completed-with-caveats |
| 2026-06-27 19:24:47 | [runs/shaped3_liftfix_20260627-192447](shaped3_liftfix_20260627-192447.md) | completed or completed-with-caveats |
| 2026-06-29 00:39:16 | [runs/priorfix_quicktest_20260629-003916](priorfix_quicktest_20260629-003916.md) | completed or completed-with-caveats |
| 2026-06-29 10:42:17 | [runs/situational_priors_7arm_20260629-104217](situational_priors_7arm_20260629-104217.md) | completed or completed-with-caveats |
| 2026-06-30 02:27:53 | [runs/dsl_vs_freeform_20260630-022753](dsl_vs_freeform_20260630-022753.md) | artifact-only or generation-only; no terminal evaluation recorded |
| 2026-06-30 02:32:09 | [runs/dsl_vs_freeform_20260630-023209](dsl_vs_freeform_20260630-023209.md) | artifact-only or generation-only; no terminal evaluation recorded |
| 2026-06-30 02:43:21 | [runs/dsl_vs_freeform_ppo_20260630-024321](dsl_vs_freeform_ppo_20260630-024321.md) | completed or completed-with-caveats |
| 2026-06-30 12:08:22 | [runs/dsl_vs_freeform_debias_20260630-120822](dsl_vs_freeform_debias_20260630-120822.md) | artifact-only or generation-only; no terminal evaluation recorded |
| 2026-06-30 12:24:38 | [runs/dofmode_consider_20260630-122438](dofmode_consider_20260630-122438.md) | artifact-only or generation-only; no terminal evaluation recorded |
| 2026-06-30 12:35:26 | [runs/dofmode_encourage_20260630-123526](dofmode_encourage_20260630-123526.md) | artifact-only or generation-only; no terminal evaluation recorded |
| 2026-06-30 12:48:17 | [runs/dofmode_ppo_20260630-124817](dofmode_ppo_20260630-124817.md) | completed or completed-with-caveats |
| 2026-06-30 17:33:49 | [runs/mv_20260630-173349](mv_20260630-173349.md) | artifact-only or generation-only; no terminal evaluation recorded |
| 2026-06-30 18:39:54 | [runs/mv_20260630-183954](mv_20260630-183954.md) | artifact-only or generation-only; no terminal evaluation recorded |
| 2026-07-01 00:42:26 | [runs/twelve_hour_4arm_20260701-004226](twelve_hour_4arm_20260701-004226.md) | completed or completed-with-caveats |
| 2026-07-01 16:51:23 | [runs/staged_20260701-165123](staged_20260701-165123.md) | completed or completed-with-caveats |
| 2026-07-01 21:28:51 | [runs/staged_20260701-212851](staged_20260701-212851.md) | completed or completed-with-caveats |
| 2026-07-02 | [runs/stalldir_frontier](stalldir_frontier.md) | artifact-only or generation-only; no terminal evaluation recorded |
| 2026-07-02 | [runs/stalldir_replay](stalldir_replay.md) | artifact-only or generation-only; no terminal evaluation recorded |
| 2026-07-02 00:03:02 | [runs/staged_stagefocus_20260702-000302](staged_stagefocus_20260702-000302.md) | completed or completed-with-caveats |
| 2026-07-02 12:58:23 | [runs/staged_stalldir_20260702-125823](staged_stalldir_20260702-125823.md) | completed or completed-with-caveats |
| 2026-07-02 19:10:14 | [runs/staged_handoff_val_20260702-191014](staged_handoff_val_20260702-191014.md) | completed or completed-with-caveats |
| 2026-07-02 19:43:33 | [runs/staged_handoff_val2_20260702-194333](staged_handoff_val2_20260702-194333.md) | completed or completed-with-caveats |
| 2026-07-02 23:45:28 | [runs/longppo_20260702-234528](longppo_20260702-234528.md) | partial/interrupted; training metrics survive but no complete run-level report |
| 2026-07-03 | [runs/agentic_v3_20260703](agentic_v3_20260703.md) | completed or completed-with-caveats |
| 2026-07-03 | [runs/arm_adjusted](arm_adjusted.md) | completed or completed-with-caveats |
| 2026-07-03 | [runs/arm_lift_only](arm_lift_only.md) | completed or completed-with-caveats |
| 2026-07-03 | [runs/arm_stage_gated](arm_stage_gated.md) | partial/interrupted; training metrics survive but no complete run-level report |
| 2026-07-04 | [runs/agentic_v4_20260704](agentic_v4_20260704.md) | completed or completed-with-caveats |
| 2026-07-04 | [runs/lift_sustain_4seed_4refine_shortppo_20260704](lift_sustain_4seed_4refine_shortppo_20260704.md) | completed or completed-with-caveats |
| 2026-07-04 | [runs/prior_fable_20260704](prior_fable_20260704.md) | completed or completed-with-caveats |
| 2026-07-04 | [runs/prior_refine_20260704](prior_refine_20260704.md) | completed or completed-with-caveats |
| 2026-07-04 | [runs/prior_refine_mb_20260704](prior_refine_mb_20260704.md) | completed or completed-with-caveats |
| 2026-07-04 14:52:38 | [runs/prior_only_5s_earlyterm_20260704-145238](prior_only_5s_earlyterm_20260704-145238.md) | completed or completed-with-caveats |
| 2026-07-05 | [runs/prior_v5_20260705](prior_v5_20260705.md) | completed or completed-with-caveats |
| 2026-07-05 | [runs/prior_v6_20260705](prior_v6_20260705.md) | completed or completed-with-caveats |
| 2026-07-06 | [runs/prior_v10_20260706](prior_v10_20260706.md) | completed or completed-with-caveats |
| 2026-07-06 | [runs/prior_v7_20260706](prior_v7_20260706.md) | completed or completed-with-caveats |
| 2026-07-06 | [runs/prior_v8_20260706](prior_v8_20260706.md) | completed or completed-with-caveats |
| 2026-07-06 | [runs/prior_v9_20260706](prior_v9_20260706.md) | completed or completed-with-caveats |
| 2026-07-06 16:25:26 | [runs/priortest_20260706-162526](priortest_20260706-162526.md) | completed or completed-with-caveats |
| 2026-07-06 19:12:59 | [runs/wristtest_20260706-191259](wristtest_20260706-191259.md) | completed or completed-with-caveats |
| 2026-07-06 19:42:33 | [runs/wristtest2_20260706-194233](wristtest2_20260706-194233.md) | completed or completed-with-caveats |
| 2026-07-07 | [runs/ab_ladder_20260707](ab_ladder_20260707.md) | completed or completed-with-caveats |
| 2026-07-07 | [runs/ab_oldstyle_20260707](ab_oldstyle_20260707.md) | completed or completed-with-caveats |
| 2026-07-07 01:17:18 | [runs/armtest_20260707-011718](armtest_20260707-011718.md) | completed or completed-with-caveats |
| 2026-07-07 14:21:43 | [runs/spawnattitude_gen_20260707-142143](spawnattitude_gen_20260707-142143.md) | completed or completed-with-caveats |
| 2026-07-07 15:05:55 | [runs/wristfix_gen_20260707-150555](wristfix_gen_20260707-150555.md) | completed or completed-with-caveats |
| 2026-07-07 15:51:32 | [runs/limits_gen_20260707-155132](limits_gen_20260707-155132.md) | completed or completed-with-caveats |
| 2026-07-07 16:27:10 | [runs/persist_gen_20260707-162710](persist_gen_20260707-162710.md) | completed or completed-with-caveats |
| 2026-07-07 17:57:51 | [runs/timing_gen_20260707-175751](timing_gen_20260707-175751.md) | completed or completed-with-caveats |
| 2026-07-07 18:27:22 | [runs/timing_gen2_20260707-182722](timing_gen2_20260707-182722.md) | completed or completed-with-caveats |
| 2026-07-07 19:15:37 | [runs/pace_gen_20260707-191537](pace_gen_20260707-191537.md) | completed or completed-with-caveats |
| 2026-07-07 23:50:47 | [runs/primitives_gen_20260707-235047](primitives_gen_20260707-235047.md) | completed or completed-with-caveats |
| 2026-07-08 12:28:46 | [runs/staged_generic_20260708-122846](staged_generic_20260708-122846.md) | artifact-only or generation-only; no terminal evaluation recorded |
| 2026-07-08 12:42:19 | [runs/prior_only_1seed_4rev_20260708-124219](prior_only_1seed_4rev_20260708-124219.md) | completed or completed-with-caveats |
| 2026-07-08 13:40:57 | [runs/prior_only_gatefix_1seed_4rev_20260708-134057](prior_only_gatefix_1seed_4rev_20260708-134057.md) | completed or completed-with-caveats |
| 2026-07-08 14:47:56 | [runs/prior_only_speedlimit_1seed_4rev_20260708-144756](prior_only_speedlimit_1seed_4rev_20260708-144756.md) | completed or completed-with-caveats |
| 2026-07-08 15:29:00 | [runs/generation_only_envcontact_1seed_20260708-1529](generation_only_envcontact_1seed_20260708-1529.md) | artifact-only or generation-only; no terminal evaluation recorded |
| 2026-07-08 17:03:57 | [runs/prior_only_constraints_1seed_1rev_20260708-170357](prior_only_constraints_1seed_1rev_20260708-170357.md) | completed or completed-with-caveats |
| 2026-07-08 17:53:39 | [runs/prior_only_reset_1seed_4rev_20260708-175339](prior_only_reset_1seed_4rev_20260708-175339.md) | completed or completed-with-caveats |
| 2026-07-08 18:55:27 | [runs/full_long_prior_vs_baseline_20260708-185527](full_long_prior_vs_baseline_20260708-185527.md) | completed or completed-with-caveats |
| 2026-07-09 12:45:17 | [runs/env_contact_prioronly_20260709-124517](env_contact_prioronly_20260709-124517.md) | completed or completed-with-caveats |
| 2026-07-09 14:46:12 | [runs/monotone_prioronly_20260709-144612](monotone_prioronly_20260709-144612.md) | completed or completed-with-caveats |
| 2026-07-09 15:29:35 | [runs/monotone_prior_6h_ppo_20260709-152935](monotone_prior_6h_ppo_20260709-152935.md) | artifact-only or generation-only; no terminal evaluation recorded |
| 2026-07-09 15:30:33 | [runs/monotone_prior_6h_ppo_20260709-153033](monotone_prior_6h_ppo_20260709-153033.md) | partial/interrupted; training metrics survive but no complete run-level report |
| 2026-07-09 15:30:33 | [runs/monotone_prior_6h_ppo_20260709-153033_restart](monotone_prior_6h_ppo_20260709-153033_restart.md) | completed or completed-with-caveats |
| 2026-07-11 | [runs/framework_cmp_4h_20260711](framework_cmp_4h_20260711.md) | completed or completed-with-caveats |
| 2026-07-11 | [runs/prior_gen_study_20260711](prior_gen_study_20260711.md) | completed or completed-with-caveats |
| 2026-07-13 | [runs/critic_diversity_4h](critic_diversity_4h.md) | completed or completed-with-caveats |
| 2026-07-13 | [runs/critic_useinfo_1h_20260713](critic_useinfo_1h_20260713.md) | completed or completed-with-caveats |
| 2026-07-14 | [runs/critic_diversity_4h_useinfo](critic_diversity_4h_useinfo.md) | completed or completed-with-caveats |

## Excluded smoke tests and non-run support directories

| Directory | Reason |
|---|---|
| `famework_testing/runs/check_fake_reactive_executor` | functional check directory, treated as a smoke/integration test |
| `famework_testing/runs/check_fake_reactive_timeout_repair` | functional check directory, treated as a smoke/integration test |
| `famework_testing/runs/check_fake_timeout_continues` | functional check directory, treated as a smoke/integration test |
| `famework_testing/runs/check_gripper_reactive_timeout_repair` | functional check directory, treated as a smoke/integration test |
| `famework_testing/runs/check_gripper_recursive_codex` | functional check directory, treated as a smoke/integration test |
| `famework_testing/runs/check_gripper_recursive_codex_macros_repair1` | functional check directory, treated as a smoke/integration test |
| `famework_testing/runs/check_gripper_recursive_codex_repair1` | functional check directory, treated as a smoke/integration test |
| `famework_testing/runs/check_gripper_recursive_codex_repair1_recompiled` | functional check directory, treated as a smoke/integration test |
| `famework_testing/runs/check_gripper_recursive_codex_scheduled` | functional check directory, treated as a smoke/integration test |
| `famework_testing/runs/check_gripper_recursive_codex_schemafix` | functional check directory, treated as a smoke/integration test |
| `famework_testing/runs/check_gripper_recursive_mock` | functional check directory, treated as a smoke/integration test |
| `famework_testing/runs/check_gripper_recursive_mock_scheduled` | functional check directory, treated as a smoke/integration test |
| `famework_testing/runs/check_gripper_recursive_reactive_loop` | functional check directory, treated as a smoke/integration test |
| `famework_testing/runs/check_gripper_recursive_reactive_loop_bound` | functional check directory, treated as a smoke/integration test |
| `famework_testing/runs/check_reporting_generated_primitives` | functional check directory, treated as a smoke/integration test |
| `famework_testing/runs/optimal_bias_mock_smoke` | explicit smoke test by directory name |
| `famework_testing/runs/policy_bias_lab_smoke` | explicit smoke test by directory name |
| `famework_testing/runs/policy_bias_lab_smoke2` | explicit smoke test by directory name |
| `famework_testing/runs/smoke_fake` | explicit smoke test by directory name |
| `famework_testing/runs/smoke_recursive` | explicit smoke test by directory name |
| `famework_testing/runs/smoke_shadow_only_mock` | explicit smoke test by directory name |
| `runs/action_prior_runtime_smoke_20260621-182835` | explicit smoke test by directory name |
| `runs/composed_prior_smoke_20260629-023005` | explicit smoke test by directory name |
| `runs/dynamic_reward_rewrite_combo_foreground_debug` | empty debug directory; no run artifacts |
| `runs/dynamic_reward_rewrite_smoke_20260620-084301` | explicit smoke test by directory name |
| `runs/dynamic_reward_smoke_20260619-173653` | explicit smoke test by directory name |
| `runs/dynamic_reward_smoke_20260619-174050` | explicit smoke test by directory name |
| `runs/policy_bias_lab_memory_smoke_env64` | explicit smoke test by directory name |
| `runs/policy_bias_lab_timed_smoke` | explicit smoke test by directory name |
| `runs/policy_bias_ppo_basin_reward_smoke_20260619-030416` | explicit smoke test by directory name |
| `runs/policy_bias_ppo_reward_fix_smoke` | explicit smoke test by directory name |
| `runs/policy_bias_ppo_smoke` | explicit smoke test by directory name |
| `runs/policy_bias_ppo_timed_smoke` | explicit smoke test by directory name |
| `runs/prior_only_videos` | render-only collection, not an experimental run |
| `runs/saturation_tanh_smoke_20260619-133447` | explicit smoke test by directory name |
| `runs/saturation_tanh_smoke_20260619-134157` | explicit smoke test by directory name |
| `runs/slim_smoke_20260630-154430` | explicit smoke test by directory name |
| `runs/smoke8` | explicit smoke test by directory name |
| `runs/smoke_pareto_selection_20260622-180124` | explicit smoke test by directory name |
| `runs/smoke_pareto_selection_20260622-180919` | explicit smoke test by directory name |

## Cross-run cautions

The history spans several runner generations. Episode horizons, environment counts, reward definitions, success predicates, action transforms, selection objectives, and report schemas changed. Cross-run comparisons should therefore be made only when the reports establish compatible conditions. Within-run controlled arm comparisons are generally stronger evidence than comparing headline numbers from unrelated runs.

This log is documentary only. It introduces no task knowledge into `policy_bias_lab`, changes no environment/reward code, and does not transplant discoveries into framework prompts or defaults.

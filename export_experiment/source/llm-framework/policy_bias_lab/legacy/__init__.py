"""Quarantined legacy subsystems (kept for reference / paper ablations, OUT of the active path):
- dynamic_rewards: the DynamicRewardCoach (LLM-in-reward-loop; proven null, always run --no-coach).
- phase_controller: the curriculum/warm-start teacher (supervised-init; deprioritized).
- action_priors: the old pareto open-loop action-prior selection (superseded by the agentic
  generation in run_dsl_vs_freeform; reused helpers live in policy_bias_lab.llm_util).
- run_ppo_experiment: old PPO runner (eval/CSV utils extracted to policy_bias_lab.report_utils).
- short_rollout_ppo / run_short_prior_ppo / run_long_short_ppo: the previous
  full-horizon short-rollout PPO implementation, superseded by the fragmented-stage default in
  policy_bias_lab.ppo_bias, policy_bias_lab.run_prior_ppo, and policy_bias_lab.run_long_ppo.
- run_dynamic_reward_experiment: the old orchestrator (coach/phase/pareto plumbing).
The active prior path (run_dsl_vs_freeform + run_prior_ppo + composed_priors/freeform_priors/
bias/ppo_bias) should not import anything here; compatibility-only legacy commands may.
"""

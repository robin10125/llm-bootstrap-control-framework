"""Quarantined legacy subsystems (kept for reference / paper ablations, OUT of the active path):
- dynamic_rewards: the DynamicRewardCoach (LLM-in-reward-loop; proven null, always run --no-coach).
- phase_controller: the curriculum/warm-start teacher (supervised-init; deprioritized).
- action_priors: the old pareto open-loop action-prior selection (superseded by the agentic
  generation in run_dsl_vs_freeform; reused helpers live in policy_bias_lab.llm_util).
- run_ppo_experiment: old PPO runner (eval/CSV utils extracted to policy_bias_lab.report_utils).
- run_dynamic_reward_experiment: the old orchestrator (coach/phase/pareto plumbing). Superseded by
  policy_bias_lab.run_prior_ppo (the slim PPO runner: injected prior programs + fixed reward, no
  coach/phase/pareto).
The active prior path (run_dsl_vs_freeform + run_prior_ppo + composed_priors/freeform_priors/
bias/ppo_bias) does NOT import anything here.
"""

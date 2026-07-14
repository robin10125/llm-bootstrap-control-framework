# Legacy policy-bias implementations

These files are preserved for replay, paper ablations, and provenance. Active modules must not
import them.

## Moved during the 2026-07-14 cleanup

- `llm_bias.py`: the original fixed `BiasSpec` prompt/compiler entry point, referenced only by
  legacy runners.
- `es.py` and `policy.py`: the retired evolution-strategy trainer and its small MLP. Active code had
  used only `BIAS_ARMS`; that registry was extracted to `policy_bias_lab/arms.py`.
- `reward_modes.py`: experimental shaping modes referenced only by the legacy short-rollout runner.
- `prior_calibration.py`: generic bounded parameter search that had no callers in the active or
  experimental code.
- `agentic_orchestrator_fixed_phase.py`: the pre-cleanup orchestrator preserved before removing
  its fixed-phase representation and hand-authored fallback from the active path.
- `run_dsl_vs_freeform.py`: the preliminary fixed-phase comparison harness, including its
  hand-authored fallback and fixed phase/operator vocabulary. Its result report is now
  `policy_bias_lab/reports/dsl_vs_freeform.md`.

## Previously quarantined

- `action_priors.py`: old Pareto/open-loop prior selection.
- `dynamic_rewards.py` and `run_dynamic_reward_experiment.py`: LLM-in-the-reward-loop coach.
- `phase_controller.py`: curriculum/warm-start teacher.
- `short_rollout_ppo.py`, `run_short_prior_ppo.py`, and `run_long_short_ppo.py`: previous
  full-horizon short-rollout PPO path.
- `run_ppo_experiment.py`: old combined PPO experiment runner.

All legacy files now use the repository-local `experiment_runtime` package; replay does not require
sibling source trees.

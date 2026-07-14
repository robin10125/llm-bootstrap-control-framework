# Experiment runtime

This package contains the runtime pieces used by both `llm_framework` and
`policy_bias_lab`. Keeping them here makes this repository self-contained: commands use the
repository's `.venv` and do not add sibling checkouts to `sys.path`.

The environment, PPO primitives, eval-vector implementation, LLM backend, and waypoint compiler
were centralized from the sibling `bootstrapping` experiment on 2026-07-14. The Shadow Hand model
under `assets/shadow_hand/` retains its upstream `LICENSE`, and the original source remains in git
history and the sibling project.

Runtime code is environment/mechanism infrastructure. Task definitions used by the policy-bias
selection loop remain in `policy_bias_lab/tasks.py`, as required by the task-agnosticism rule.

The copied runtime sources are maintained here going forward; sibling projects are no longer
loaded at runtime.

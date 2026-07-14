"""Experiment-arm registry shared by the active PPO runners."""

BIAS_ARMS = {
    "baseline": (False, False, False, False),
    "reward": (True, False, False, False),
    "action_prior": (False, True, False, False),
    "exploration": (False, False, True, False),
    "supervised_init": (False, False, False, True),
    "reward_action_prior": (True, True, False, False),
    "reward_supervised_init": (True, False, False, True),
    "full": (True, True, True, True),
    "prior_monolithic": (True, True, False, False),
    "prior_gate_soft": (True, True, False, False),
    "prior_gate_subgoal": (True, True, False, False),
    "prior_gate_options": (True, True, False, False),
    "prior_gate_stacked": (True, True, False, False),
    "prior_reactive_law": (True, True, False, False),
    "prior_dmp": (True, True, False, False),
    "dsl_stacked": (True, True, False, False),
    "freeform_stacked": (True, True, False, False),
    "freeform_consider": (True, True, False, False),
    "freeform_encourage": (True, True, False, False),
}


def arm_features(name: str) -> tuple[bool, bool, bool, bool]:
    """Return reward, action-prior, exploration, and supervised-init flags."""
    try:
        return BIAS_ARMS[name]
    except KeyError as exc:
        raise KeyError(f"unknown arm {name!r}; choose from {sorted(BIAS_ARMS)}") from exc


__all__ = ["BIAS_ARMS", "arm_features"]

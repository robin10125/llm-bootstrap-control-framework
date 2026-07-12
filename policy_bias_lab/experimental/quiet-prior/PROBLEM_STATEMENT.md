# Quiet Action Priors: Problem Statement

The current action-prior foundation is too active. A strong prior can keep injecting actions after
the neural policy has become capable of solving the same part of the behavior, which makes the prior
and policy overlap in function. When that overlap is large, PPO is not simply learning the task; it
is also learning around, against, or through the prior. This can create oppositional dynamics where
the residual policy and authored prior fight over the same actuators, timing, or state transition.

The goal of this experiment is to generate an action-prior system that is quieter: the prior should
guide the neural policy only where useful behavior is hard to discover from reward alone, and it
should leave as much of the final behavior as possible to the learned policy. The prior is not meant
to be a full controller. It should act above the policy as a scaffold, hint, unlock, or constraint
that helps the policy reach useful regions of behavior, then becomes weak, local, or inactive once
the policy can act competently.

The optimization target is therefore not "make the prior solve the task." The target is:

- maximize the work done by the neural policy;
- minimize functional overlap between neural-policy capabilities and prior behavior;
- preserve only the parts of the prior that unlock behaviors the neural policy would otherwise have
  difficulty reaching;
- make prior influence legible and measurable by actuator group, stage, and rollout time.

In this framing, the action prior should do what the neural policy cannot yet do, not what the neural
policy can already learn. A successful quiet-prior system should improve exploration, stage reach,
or initial behavioral structure while still allowing the trained policy to become the primary source
of control authority.

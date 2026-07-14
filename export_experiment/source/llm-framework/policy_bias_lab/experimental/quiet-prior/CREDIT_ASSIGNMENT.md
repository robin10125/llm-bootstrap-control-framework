# Credit Assignment Amendment

This amendment supersedes the mixed-rollout PPO treatment described in `DESIGN.md`.

Overridden actions and the states reached through them were not sampled from the autonomous policy.
Removing intervention-owned actuator dimensions from a summed log probability does not make the
remaining hybrid trajectory on-policy.

The first implementation must use two separate collection streams:

1. **Autonomous stream:** interventions are disabled for the complete rollout. These transitions
   alone supply PPO policy-gradient and value-function targets.
2. **Scaffold stream:** bounded interventions and randomized controls may execute. These transitions
   supply causal-audit records and possible supervised examples, but no PPO ratio or value target.

For each scaffold transition, retain the policy proposal, intervention proposal, authority mask,
assignment probability, executed action, and subsequent authored outcome. A scaffold trace may enter
a separate imitation minibatch only when the intervention has positive run-local causal evidence and
the particular trace reaches its authored outcome. Failed or unresolved interventions remain audit
data and are not imitation targets.

This separation costs environment interactions but makes policy-only training and evaluation
literal. Admitting policy-owned suffixes or dimensions from hybrid trajectories is a later research
option requiring an explicit behavior-policy derivation and importance correction.

Consequent implementation changes:

- preserve the existing PPO loss for the autonomous stream;
- add a separate scaffold collector and demonstration buffer;
- optimize a bounded auxiliary imitation loss outside the PPO surrogate;
- account for both streams in the environment-interaction budget;
- test that scaffold batches contribute no PPO policy or value gradient.

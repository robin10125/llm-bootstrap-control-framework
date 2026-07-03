REPRESENTATION = DSL. Each phase sub-prior is a list of `rules`. Two rule kinds:
  - operator: {kind:'operator', group:<one of $groups + ['all']>, direction:<one of $operators>, weight:0..0.6}. Operators are pre-calibrated primitives; each name indicates its effect and its exact semantics are defined by the spec.
  - basis: {kind:'basis', group:<semantic group>, sign:+1 or -1, weight:0..0.6}. Drives that group's actuators toward +/- ctrl. Use basis rules to actively move actuators the operators do not cover.

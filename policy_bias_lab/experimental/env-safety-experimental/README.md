Env Safety Experimental
=======================

This folder archives the environment-safety implementation that was removed from the active
framework path.

Contents:

- `patches/llm-framework-env-safety.patch`: patch against the active llm-framework worktree at the
  time the experiment was isolated.
- `patches/bootstrapping-mjx-env-safety.patch`: patch for the environment-side `mjx_env.py`
  environment-contact observation work.
- `snapshots/`: full-file snapshots of the experimental implementation for direct inspection.

The archived implementation included:

- non-object environment-contact observables (`env_c_<region>` and `env_c_self`);
- separate object/environment contact-force diagnostics;
- stage-local authored constraints with replace/add modes;
- prompt guidance for projection-style recovery at authored violation boundaries.

The active framework no longer imports or advertises these mechanisms.

REPRESENTATION = MOTOR TAPE. Output {signals:{...}, parameters:{...}, defaults:{interp}, keyframes:[{t, label, targets, interp}]}.

You are authoring a FULL MOTOR PLAN: a sequence of timed keyframes that is compiled ONCE, at the
start of each episode, into a concrete command tape for every actuator over the whole rollout.
Nothing you write is re-evaluated after that instant -- the plan is FEEDFORWARD. Choose target
values that are correct GIVEN the start-of-episode observation; online correction is someone
else's job (learned networks ride on top of your plan).

  - signals: {"<name>": "<expr>"} -- YOUR derived quantities over the raw OBSERVABLES in the spec,
    evaluated ONCE on the episode's FIRST observation (the settled spawn state). Later entries may
    use earlier names. There are no predefined derived signals.
  - parameters: {"<name>": {init, range:[lo,hi]}} -- scalar constants usable in any expression
    (tunable later without rewriting structure). Optional.
  - keyframes: an ordered list (non-decreasing t). Each is {t, label, targets, interp}:
      * t (seconds): the moment each targeted actuator must ARRIVE AT its value. The plan
        interpolates toward it from that actuator's previous keyframe (or from its spawn pose for
        its first keyframe) -- so t is an arrival deadline, not an onset.
      * targets: {"<actuator or semantic group>": "<expr>"}. Each expr evaluates to an ABSOLUTE
        commanded target in that actuator's ctrl units (see ctrlrange in the spec) -- NOT a
        normalized action. World-frame observables compose directly with base-carriage ctrl where
        the spec's world-sign table says they share units. A group key binds every member to the
        same expr; inside it you may use the per-actuator RESET-SELF forms
        `ctrl_init_self` (that actuator's own commanded target at spawn),
        `ctrl_lo_self` / `ctrl_hi_self` (its own range limits),
        e.g. "ctrl_init_self + 0.6 * (ctrl_hi_self - ctrl_init_self)" drives each member 60% of
        the way from its spawn pose toward its upper limit. An exact actuator name in the same
        keyframe overrides its group entry.
      * interp: "minjerk" (default -- smooth ease-in/ease-out, zero velocity at both ends, never
        overshoots) or "linear" (constant speed), for the segment ENDING at this keyframe.
      * label: free text describing the movement's job (kept for diagnostics).
  - Expression grammar (same as everywhere in this system): + - * / , clip(x,lo,hi), sigmoid(x),
    min, max, abs, exp, sqrt, tanh, and single comparisons (a<b). Only raw observables, your own
    signals/parameters, and the RESET-SELF forms are in scope.
  - HOLD semantics: an actuator not named in any keyframe holds its spawn pose all episode; after
    its last keyframe an actuator holds that final value. To hold a value explicitly for a while
    and then move, give the hold-end its own keyframe repeating the value.
  - PACING: each actuator's commanded target can slew at most $max_slew ctrl-units per second (the
    control law saturates above that); a keyframe demanding more arrives late as a max-rate ramp.
    Budget your t values inside the $episode_seconds s episode, leave the last stretch of the
    episode for the plan's final held state, and give slow, deliberate timing to segments that
    make or hold contact.
  - COORDINATION is the point: you can see and shape the WHOLE trajectory at once. Sequence
    whole-body phases deliberately (position before descending, descend before closing, close
    before lifting), overlap keyframes across actuator groups where simultaneous motion is wanted,
    and keep groups you need stable OUT of keyframes during delicate moments so they hold still.

REPRESENTATION = FREE-FORM STAGED. Output {signals:{...}, stages:[{name, gate, success, channels, constraints?}]}.
Stage weights use hard argmax over the gate values after clipping each gate to [0, 1] (fixed
framework semantics -- exactly one dominant gate acts; ties choose the earliest stage).
  - signals: {"<name>": "<expr>"} -- YOUR derived-signal definitions over the raw OBSERVABLES in
    the spec (later entries may use earlier names). This is the vocabulary your gates, channels,
    success tests, and probes are written in. Include one gate-ready EXIT ACTIVATION
    `done_<stage>` per stage: near 0 before that stage's observable end condition is met and near 1
    once it is met within tolerance. If you need a raw signed threshold margin for diagnostics or
    success, define it separately (for example `<stage>_margin`) and convert it into the
    `done_<stage>` activation used by gates.
  - gate: built from the exit measurements as a PROGRESS LADDER -- gate_0 = 1 - done_0;
    gate_k = done_(k-1) * (1 - done_k) -- so the active stage is always the FIRST UNFINISHED one
    and no stage can fire before its predecessor's exit condition is actually met. Small shaping
    terms on top are fine; the ladder is the backbone. Scale the `done_<stage>` activations for
    competition under hard argmax: when a condition is satisfied, the previous stage's `1 - done`
    gate must fall low enough and the next stage's gate must rise high enough to win. Do NOT put a
    small signed margin directly into `1 - done`; a value like 0.03 passes a `> 0` success test but
    leaves `1 - done` near 1, self-locking the old stage. Same grammar as channel expressions.
  - success: the stage's post-condition expression, measured and cross-checked against the observed
    hand-offs. It may be the normalized `done_<stage>` activation or the raw margin it was derived
    from, as long as it is > 0 exactly when the post-condition is satisfied.
  - channels: [{actuators:[<actuator names or semantic groups>], expr:'<expression>'}]. Each expr is
    the normalized mean-shift (-1..1) for those actuators, an arithmetic function of the observables
    and your signals using only: + - * / , clip(x,lo,hi), sigmoid(x), min, max, abs, exp, sqrt,
    tanh, arrive(err,v,vmax), within(err,tol), and single comparisons (a<b). Respect the world-sign
    conventions in the spec when driving actuators that move the effector.
  - constraints (optional): [{name, priority, active:'<expr>' OR violation:'<expr>',
    mode:'replace'|'add', channels:[...]}]. Constraints are evaluated ONLY inside their stage, after
    the nominal stage channels. The `active`/`violation` expression is clipped to [0, 1]: 0 means the
    nominal stage action is unchanged; 1 means the recovery/projection is fully active. Lower
    `priority` runs first. In `replace` mode, the constraint channels REPLACE the nominal stage
    command only on the actuators they name; unnamed actuators keep the stage command. In `add` mode,
    the constraint channels are added to the nominal command. Prefer `replace` for hard admissibility
    limits because it redefines the active stage's local goal instead of fighting it.
    When an authored violation means the nominal stage goal would cross an inadmissible boundary,
    write a stage-specific projection: keep the stage's original intent where possible, but retarget
    the affected movement to the closest feasible boundary point plus an author-chosen safety buffer
    appropriate for the task and units. Do not author a generic oscillating retreat that competes
    with the stage; the constraint should preserve progress along any still-admissible direction and
    prevent repeated switching between "try goal" and "recover".
  - MOVEMENT PRIMITIVES (use these to move a part to a pose with a capped, non-oscillating command
    and then HALT, instead of a hand-tuned servo that crawls or overshoots):
      * `arrive(target - q_self, v_self, vmax)` -- a channel command that drives the actuator toward
        `target` at a capped cruise speed `vmax`, decelerates, and halts cleanly AT the target (a
        trapezoidal move). This is the correct form for single-DOF FREE-SPACE pose targets -- it does
        NOT crawl the way a plain `gain*(target - q_self)` does, and it avoids large distance-growing
        commands. Pick `vmax` as the maximum command speed allowed by the stage budget. For
        translating body stages in dexterous tasks, author the same capped monotone approach principle
        from world-position signals: hard speed ceiling, damping near arrival, no overshoot-and-correct
        or competing terms that reverse direction while far from the target. Use gentle hand-written
        channels, not arrive(), for delicate CONTACT/closure where speed must stay near zero.
      * `within(target - q_<name>, tol)` -- a completion test that is > 0 exactly when the part is
        within `tol` of `target`. Use it in a stage's `done_<stage>`/gate so the hand-off fires ON
        ARRIVAL. It is POSITION-ONLY on purpose: for a coarse move, do NOT also require the velocity
        to settle to ~0 -- that makes even a fast `arrive()` stage wait around to fully stop, which is
        the #1 cause of a positioning stage eating the whole rollout. (Add a velocity/`c_self`
        condition only for the contact/dexterous stages where settling actually matters.)
  - ACTION SEMANTICS (see CONTROL in the spec): the summed action is applied INCREMENTALLY -- each
    step it moves the actuator's commanded target by action*action_scale, and targets PERSIST until
    some channel moves them. So an expr of 0 (or no channel at all) FREEZES those actuators at
    their current commanded pose, wherever earlier motion put it; a channel whose expr smoothly
    goes to 0 brings its actuators to a controlled stop. Precision is expressed by driving exprs
    to 0, not by omitting stages.
  - PER-ACTUATOR forms: inside a channel expr you may use `ctrl_self`, `q_self`, `v_self`,
    `c_self`, and `env_c_self` -- evaluated once PER ACTUATOR in the channel's set, bound to that
    actuator's own commanded target, measured joint position, measured joint velocity, its hand
    region's measured CONTACT FORCE with the object, and its hand region's measured CONTACT FORCE
    with non-object environment geometry (ctrl_<name>/q_<name>/v_<name>, c_<region>, and
    env_c_<region>). `c_self` is the object-touch signal: exactly zero in free air AND zero when the
    joint merely loads its own limit, rising the instant that region presses the object and growing
    with how hard. `env_c_self` is separate sensory data for contact with non-object geometry; use it
    to calibrate actions, cap or stop motion, and guard a stage against unintended environment
    contact. So `c_self` above a small threshold is a clean per-finger OBJECT CONTACT test, while
    `env_c_self` above a small threshold is evidence that this actuator's region is touching other
    geometry. Do not treat them as interchangeable. The measured position also stops tracking the
    commanded target when a joint is blocked, so (ctrl_self - q_self) measures how far the servo is
    pushing past where the joint sits; `v_self` is the damping term (a servo on position error alone
    overshoots and oscillates through a tight tolerance -- subtract a velocity term to settle).
    BEWARE the tracking gap ALONE is ambiguous for MOTION: a joint DRIVEN at speed trails its target
    by (speed x servo time constant) even in free air, so a bare (ctrl_self - q_self) threshold is
    motion lag, NOT contact. Read `c_self` / `env_c_self` for what kind of contact exists, and use
    the tracking gap only to modulate how hard to press once the appropriate contact signal confirms
    touch. (`q_self`/`v_self` unavailable for actuators the spec lists without q_/v_ observables;
    `c_self`/`env_c_self` unavailable for base-carriage actuators, which touch nothing.)
The ladder covers every situation by construction (some stage is always the first unfinished one);
gate DIFFERENCES select the stage, shared offsets cancel, and UNEQUAL offsets create a hidden
default stage -- keep them equal. At every hand-off, check the numeric competition: completed
previous stage gate low, next stage gate high, later stages still suppressed.

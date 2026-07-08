REPRESENTATION = FREE-FORM STAGED. Output {signals:{...}, stages:[{name, gate, success, channels}]}.
Stage weights are a sharp softmax over the gate values (fixed framework semantics -- the dominant
gate acts, with a brief smooth hand-off when two gates are close).
  - signals: {"<name>": "<expr>"} -- YOUR derived-signal definitions over the raw OBSERVABLES in
    the spec (later entries may use earlier names). This is the vocabulary your gates, channels,
    success tests, and probes are written in. Include one EXIT MEASUREMENT `done_<stage>` per
    stage: > 0 exactly when that stage's job is complete (its observable end condition).
  - gate: built from the exit measurements as a PROGRESS LADDER -- gate_0 = 1 - done_0;
    gate_k = done_(k-1) * (1 - done_k) -- so the active stage is always the FIRST UNFINISHED one
    and no stage can fire before its predecessor's exit condition is actually met. Small shaping
    terms on top are fine; the ladder is the backbone. Same grammar as channel expressions.
  - success: the stage's `done_<stage>` expression (its post-condition; measured and cross-checked
    against the observed hand-offs).
  - channels: [{actuators:[<actuator names or semantic groups>], expr:'<expression>'}]. Each expr is
    the normalized mean-shift (-1..1) for those actuators, an arithmetic function of the observables
    and your signals using only: + - * / , clip(x,lo,hi), sigmoid(x), min, max, abs, exp, sqrt,
    tanh, arrive(err,v,vmax), within(err,tol), and single comparisons (a<b). Respect the world-sign
    conventions in the spec when driving actuators that move the effector.
  - MOVEMENT PRIMITIVES (use these to move a part to a pose QUICKLY and then HALT, instead of a
    hand-tuned servo that crawls):
      * `arrive(target - q_self, v_self, vmax)` -- a channel command that drives the actuator toward
        `target` at cruise speed `vmax`, decelerates late, and halts cleanly AT the target (a
        trapezoidal move: it cruises at +-vmax across most of the travel, then brakes in the last
        little bit). This is the correct form for FREE-SPACE positioning/orientation -- it does NOT
        crawl the way a plain `gain*(target - q_self)` does. Pick `vmax` as the fraction of full
        command speed you want (e.g. 0.3-0.5 for a brisk reorient). Use gentle hand-written channels,
        not arrive(), for delicate CONTACT/closure where speed must stay near zero.
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
  - PER-ACTUATOR forms: inside a channel expr you may use `ctrl_self`, `q_self`, `v_self`, and
    `c_self` -- evaluated once PER ACTUATOR in the channel's set, bound to that actuator's own
    commanded target, measured joint position, measured joint velocity, and its hand region's
    measured CONTACT FORCE with the object (ctrl_<name>/q_<name>/v_<name>, and c_<region>).
    `c_self` is the ground-truth touch signal: exactly zero in free air AND zero when the joint
    merely loads its own limit, rising the instant that region presses the object and growing with
    how hard. So `c_self` above a small threshold is a clean per-finger CONTACT test, and driving
    c_self toward a small positive target is a minimum-force grip -- advance each finger until IT
    makes contact, then hold a gentle force, independently per actuator. The measured position also
    stops tracking the commanded target when a joint is blocked, so (ctrl_self - q_self) measures
    how far the servo is pushing past where the joint sits; `v_self` is the damping term (a servo
    on position error alone overshoots and oscillates through a tight tolerance -- subtract a
    velocity term to settle). BEWARE the tracking gap ALONE is ambiguous for MOTION: a joint DRIVEN
    at speed trails its target by (speed x servo time constant) even in free air, so a bare
    (ctrl_self - q_self) threshold is motion lag, NOT contact. Read `c_self` for whether contact
    exists, and use the tracking gap only to modulate how hard to press once c_self confirms touch.
    (`q_self`/`v_self` unavailable for actuators the spec lists without q_/v_ observables; `c_self`
    unavailable for base-carriage actuators, which touch nothing.)
The ladder covers every situation by construction (some stage is always the first unfinished one);
gate DIFFERENCES select the stage, shared offsets cancel, and UNEQUAL offsets create a hidden
default stage -- keep them equal.

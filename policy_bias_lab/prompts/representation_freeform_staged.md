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
    tanh, and single comparisons (a<b). Respect the world-sign conventions in the spec when driving actuators that
    move the effector.
  - ACTION SEMANTICS (see CONTROL in the spec): the summed action is applied INCREMENTALLY -- each
    step it moves the actuator's commanded target by action*action_scale, and targets PERSIST until
    some channel moves them. So an expr of 0 (or no channel at all) FREEZES those actuators at
    their current commanded pose, wherever earlier motion put it; a channel whose expr smoothly
    goes to 0 brings its actuators to a controlled stop. Precision is expressed by driving exprs
    to 0, not by omitting stages.
  - PER-ACTUATOR forms: inside a channel expr you may use `ctrl_self`, `q_self`, `v_self`, and
    `f_self` -- evaluated once PER ACTUATOR in the channel's set, bound to that actuator's own
    commanded target, measured joint position, measured joint velocity, and measured constraint
    force (the same quantities as ctrl_<name>/q_<name>/v_<name>/f_<name>). `f_self` is the
    ground-truth contact signal: exactly zero in free air no matter how the joint moves, nonzero
    only under real load. The measured position stops tracking the commanded target
    when a joint is physically blocked or loaded, so expressions over (ctrl_self - q_self) give
    each actuator in one channel its own reactive response: advance only while the joint still
    follows its target and stop when it no longer does, hold a commanded tension constant, or
    move the target back toward the measured position to release load. `v_self` is the damping
    term: a servo built on a position error alone will overshoot and oscillate through a tight
    tolerance; subtracting a velocity term lets it settle. BEWARE: the tracking gap ALONE is
    ambiguous -- a joint DRIVEN at speed trails its target by (speed x servo time constant) even
    in free air, so a bare (ctrl_self - q_self) threshold fires from motion lag with nothing
    touching anything. The discriminating signature: BLOCKED = gap with near-zero v_self;
    LAG = gap while v_self still tracks the commanded rate. (`q_self`/`v_self` are unavailable
    for actuators the spec lists without q_/v_ observables.)
The ladder covers every situation by construction (some stage is always the first unfinished one);
gate DIFFERENCES select the stage, shared offsets cancel, and UNEQUAL offsets create a hidden
default stage -- keep them equal.

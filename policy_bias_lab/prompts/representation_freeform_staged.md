REPRESENTATION = FREE-FORM STAGED. Output {signals:{...}, stages:[{name, gate, channels}]}. Stage
weights are a sharp softmax over the gate values (fixed framework semantics -- the dominant gate
acts, with a brief smooth hand-off when two gates are close).
  - signals: {"<name>": "<expr>"} -- YOUR derived-signal definitions over the raw OBSERVABLES in
    the spec (later entries may use earlier names). This is the vocabulary your gates, channels,
    success tests, and probes are written in.
  - gate: a symbolic expression over the observables and your signals giving this stage's
    activation (higher = more active). Same grammar as channel expressions.
  - channels: [{actuators:[<actuator names or semantic groups>], expr:'<expression>'}]. Each expr is
    the normalized mean-shift (-1..1) for those actuators, an arithmetic function of the observables
    and your signals using only: + - * / , clip(x,lo,hi), sigmoid(x), min, max, abs, exp, and single
    comparisons (a<b). Respect the world-sign conventions in the spec when driving actuators that
    move the effector.
Design the gates so the stages together cover every situation, and so that in each situation the
intended stage's gate clearly dominates (gate DIFFERENCES select the stage; shared offsets cancel).

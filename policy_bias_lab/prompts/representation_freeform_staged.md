REPRESENTATION = FREE-FORM STAGED. Output {blend:'soft'|'hard', stages:[{name, gate, channels}]}.
  - gate: a symbolic expression over the SIGNALS above giving this stage's activation (>=0; higher =
    more active). Same grammar as channel expressions.
  - channels: [{actuators:[<actuator names or semantic groups>], expr:'<expression>'}]. Each expr is
    the normalized mean-shift (-1..1) for those actuators, an arithmetic function of the SIGNALS using
    only: + - * / , clip(x,lo,hi), sigmoid(x), min, max, abs, exp, and single comparisons (a<b).
    Respect the world-sign conventions in the spec when driving actuators that move the effector.
Design the gates so the stages together cover every situation, and (for soft) so adjacent stages
overlap smoothly rather than switching abruptly.

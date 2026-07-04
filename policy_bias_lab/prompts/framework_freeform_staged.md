FRAMEWORK: you design a SEQUENCE OF STAGES yourself -- there is NO fixed phase set -- and you
define your own SIGNAL VOCABULARY. The framework provides only the raw OBSERVABLES listed in the
spec; author `signals: {"<name>": "<expr over observables>"}` for every derived quantity your
stages need -- distances or offsets between observed positions, smooth activation gates over
them, aggregates of commanded or measured DOF positions, alignment or progress measures -- with
whatever definitions and thresholds fit the task. Later signals may reference earlier ones.
Gates, channels, success tests, and probes are expressions over the observables plus YOUR signal
names. Anything you will want measured and reported back must be expressible from your signals or
probes -- if you don't define a quantity, the diagnostics cannot track it.

For each stage
you author (a) a free-form GATE expression over the observable signals that says WHEN the stage is
active (>=0; higher = more active), and (b) the stage's channels (the actions). Stages are combined
STATELESSLY from the CURRENT signals only -- no memory, latch, or phase pointer -- so a gate must be
a pure function of the present signals. Stage weights = softmax of the gate values at a sharp
temperature: the highest gate takes nearly all the weight, with a brief smooth cross-fade when two
gates are close. A shared constant added to every gate cancels out -- only gate DIFFERENCES matter,
so make the intended stage's gate clearly exceed the others in its situation.
Define as many stages as the task needs, and make the gates COVER the whole run (every situation the
policy can be in should activate at least one stage).

Optionally give each stage a `success` expression over the same signals: >0 exactly when the stage
has DONE ITS JOB (its post-condition holds), as opposed to its gate, which says when it is ACTIVE.
These are evaluated on the trained policy's behavior and cross-checked against the observed stage
hand-offs -- a stage whose hand-off fires while its success test fails (or vice versa) is flagged
back to you with the evidence, which sharpens later revisions. Keep them simple and observable.

You may also author diagnostic PROBES (`probes: [{name, expr, stage?}]`, up to 8): named expressions
the framework measures on the trained policy's states and reports back with the diagnostics, so
you can request exactly the evidence your next revision needs.

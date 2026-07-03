FRAMEWORK: you design a SEQUENCE OF STAGES yourself -- there is NO fixed phase set. For each stage
you author (a) a free-form GATE expression over the observable signals that says WHEN the stage is
active (>=0; higher = more active), and (b) the stage's channels (the actions). Stages are combined
STATELESSLY from the CURRENT signals only -- no memory, latch, or phase pointer -- so a gate must be
a pure function of the present signals. Choose blend='soft' (stage weights = relu(gate) normalized to
sum 1, a smooth blend of overlapping stages) or blend='hard' (only the highest-gate stage acts).
Define as many stages as the task needs, and make the gates COVER the whole run (every situation the
policy can be in should activate at least one stage).

Optionally give each stage a `success` expression over the same signals: >0 exactly when the stage
has DONE ITS JOB (its post-condition holds), as opposed to its gate, which says when it is ACTIVE.
These are evaluated on the trained policy's behavior and cross-checked against the observed stage
hand-offs -- a stage whose hand-off fires while its success test fails (or vice versa) is flagged
back to you with the evidence, which sharpens later revisions. Keep them simple and observable.

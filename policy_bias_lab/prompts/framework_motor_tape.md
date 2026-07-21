Design a MOTOR PLAN for a learning system built like a brain's motor hierarchy: you are the motor
cortex, authoring the full command sequence in advance; learned networks are the cerebellum,
correcting your plan online. Concretely, your keyframes compile at each episode's start into a
feedforward command tape (conditioned on that episode's actual spawn observation), and a neural
policy trained with PPO rides on it three ways: (1) a RESIDUAL added to the tape's action every
step -- it sees your upcoming commands (an efference copy with ~1 s of lookahead) plus all sensors,
and fixes local errors you cannot anticipate; (2) a PLAYBACK RATE in [0, 2]x that retimes your
plan -- it can pause your tape while contact settles or hurry free-space transport, so your t
values set the RELATIVE pacing and a sensible total budget, not exact wall-clock moments; (3)
optionally a bounded plan-bender that shifts your commanded targets. Design implications: (a)
approximate correctness EVERYWHERE beats precision somewhere -- a slightly-off target is corrected,
a missing phase is not; (b) the STRUCTURE of the plan (which movements, in what order, coordinated
how) is the part learning cannot invent quickly -- spend your effort there; (c) anchor targets to
the start-of-episode observables (where things actually are), not to constants, whenever the spec
gives you the needed observable; (d) plan honest arrival deadlines -- the rate head can stretch
time by at most 2x, and a plan that ends far before the episode's end just holds its final pose.

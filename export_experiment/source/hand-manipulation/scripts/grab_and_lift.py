# Bespoke control script — authored from inject/state.txt.
# Goal (clear END STATE it must satisfy before stopping):
#     grasped('cube') is True AND obj_pos('cube')[2] >= 0.12
# Strategy: open -> centre over cube -> descend to straddle height -> squeeze closed
#           (the cube blocks the fingers, creating grip force) -> lift -> verify.
# FAILSAFE: a self-imposed physics-step budget. If the end state is not reached within it,
#           the script returns a FAILURE summary instead of looping forever. (The bridge
#           also hard-stops at its global step budget as a backstop.)
#
# This is the BODY of a function: the primitives (set_ctrl/step/obj_pos/grasped/...) are
# already in scope, and `return` ends the script.

GOAL_Z = 0.12
MAX_STEPS = 8000      # ~16 sim-seconds of compute before giving up
GRASP_H = 0.115       # palm height that puts the fingers around a cube on the table
LIFT_H = 0.32         # clearly above the 0.12 m success threshold

used = 0

def adv(n):
    # advance physics while charging the self-imposed compute budget
    nonlocal used
    step(n)
    used += n

def goal_met():
    return grasped('cube') and obj_pos('cube')[2] >= GOAL_Z

if goal_met():
    return 'SUCCESS: goal already satisfied'

attempt = 0
# Closed loop: keep attempting the grasp-and-lift until the end state holds or the
# failsafe budget is spent. Each full attempt costs ~1800 steps.
while used < MAX_STEPS:
    attempt += 1
    log('attempt %d (steps used so far: %d)' % (attempt, used))

    # 1. open the fingers and centre the palm over the cube, up high to clear it.
    c = obj_pos('cube')
    set_ctrl('left_finger', 0.05); set_ctrl('right_finger', 0.05)
    set_ctrl('slide_x', c[0]); set_ctrl('slide_y', c[1]); set_ctrl('slide_z', LIFT_H)
    adv(300)

    # 2. descend to grasp height (re-read x/y in case the cube moved on a prior attempt).
    c = obj_pos('cube')
    set_ctrl('slide_x', c[0]); set_ctrl('slide_y', c[1]); set_ctrl('slide_z', GRASP_H)
    adv(500)

    # 3. squeeze: command both fingers fully closed; the cube blocks them -> grip force.
    set_ctrl('left_finger', 0.0); set_ctrl('right_finger', 0.0)
    adv(400)
    if not grasped('cube'):
        log('no grip after squeeze (cube_z=%.3f); retrying' % obj_pos('cube')[2])
        continue

    # 4. lift straight up, then verify the end state.
    set_ctrl('slide_z', LIFT_H)
    adv(600)
    if goal_met():
        return ('SUCCESS: cube_z=%.3f grasped=%s after %d attempt(s), %d steps'
                % (obj_pos('cube')[2], grasped('cube'), attempt, used))
    log('grip slipped during lift (cube_z=%.3f); retrying' % obj_pos('cube')[2])

# Failsafe reached: stop with an explicit failure rather than running forever.
return ('FAILURE: end state not reached within %d-step budget '
        '(cube_z=%.3f grasped=%s after %d attempt(s))'
        % (MAX_STEPS, obj_pos('cube')[2], grasped('cube'), attempt))

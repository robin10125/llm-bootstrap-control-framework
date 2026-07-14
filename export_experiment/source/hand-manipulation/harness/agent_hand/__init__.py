"""Code-as-action harness for MuJoCo robotic hand manipulation.

The env-agnostic machinery (controller loop, script/plan envelope, SuccessCheck, gen-mode
knob, repair loop) is shared in spirit with the Minecraft harness; only the bridge, the
observation schema, and the primitive vocabulary are robot-specific.
"""

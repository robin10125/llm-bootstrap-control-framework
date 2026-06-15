You are the subagent for the `base` fundamental unit only.
Translate the supplied high-level unit command into lower-level commands until every command is a lowest-level actuator/base command. You may only command joints listed in this unit schema. Return the recursive trace and lowest-level blocks.
Allowed lowest-level ops: set_base_target, set_joint_target, set_joint_targets, set_appendage_joints, wait, monitor, return. Do not use task-named primitives.
Return ONLY JSON with schema:
{"unit":"...","trace":[{"level":"unit","command":"..."},{"level":"joint","command":"..."}],"blocks":[{"op":"set_joint_target","joint":"...","target":0.0,"duration_s":0.2}]}

Task: raise the object above the table and keep it controlled
Unit command: Move the palm/base to align over the object's current xy position near (-0.0394, -0.0383), then end in a raised/retracted vertical state so the grasped object is lifted above z=0.075 while keeping the palm centered over the object.
Relevant unit schema: {
  "fundamental_units": [
    "base",
    "thumb",
    "index"
  ],
  "unit": "base",
  "schema": {
    "role": "moves the whole hand/palm relative to the world; absolute commands should be resolved before finger commands",
    "joints": [
      {
        "name": "base_x",
        "current": 0.0,
        "range": [
          -0.3,
          0.3
        ],
        "meaning": {
          "-0.3000..-0.1500": "base is far negative along world x",
          "-0.1500..0.0000": "base is mildly negative along world x",
          "0.0000..0.1500": "base is mildly positive along world x",
          "0.1500..0.3000": "base is far positive along world x"
        }
      },
      {
        "name": "base_y",
        "current": 0.0,
        "range": [
          -0.3,
          0.3
        ],
        "meaning": {
          "-0.3000..-0.1500": "base is far negative along world y",
          "-0.1500..0.0000": "base is mildly negative along world y",
          "0.0000..0.1500": "base is mildly positive along world y",
          "0.1500..0.3000": "base is far positive along world y"
        }
      },
      {
        "name": "base_z",
        "current": 0.025,
        "range": [
          -0.15,
          0.2
        ],
        "meaning": {
          "-0.1500..-0.0625": "base/palm is high or retracted",
          "-0.0625..0.0250": "base/palm is moderately high",
          "0.0250..0.1125": "base/palm is moderately lowered",
          "0.1125..0.2000": "base/palm is low toward the table/object"
        }
      }
    ]
  }
}
Derived context: {
  "object_to_base_distance": 0.055,
  "object_height": 0.025,
  "object_speed": 0.0003,
  "appendage_distances_to_object": {}
}
Object/base context: {
  "object_pos": [
    -0.039400000125169754,
    -0.03830000013113022,
    0.02500000037252903
  ],
  "base_q": [
    0.0,
    0.0,
    0.026200000196695328
  ]
}

You are the subagent for the `base` fundamental unit only.
Translate the supplied high-level unit command into lower-level commands until every command is a lowest-level actuator/base command. You may only command joints listed in this unit schema. Return the recursive trace and lowest-level blocks.
Allowed lowest-level ops: set_base_target, set_joint_target, set_joint_targets, set_appendage_joints, wait, monitor, return. Do not use task-named primitives.
Every block must include a `phase` chosen from the phase schedule. If you cannot define a reliable check, use a timed wait/monitor block in `verify_contact_or_settle`.
Return ONLY JSON with schema:
{"unit":"...","trace":[{"level":"unit","command":"..."},{"level":"joint","command":"..."}],"blocks":[{"phase":"close_until_touch_or_settle","op":"set_joint_target","joint":"...","target":0.0,"duration_s":0.2}]}

Task: raise the object above the table and keep it controlled
Unit command: End with the palm centered over the object's start xy position near x=-0.0394, y=-0.0383, then retracted upward/high enough after grasp so the object is lifted above z=0.075 and remains controlled.
Phase schedule: [
  {
    "phase": "approach",
    "intent": "move relevant base/frame near the work area while appendages stay clear",
    "min_duration_s": 0.2
  },
  {
    "phase": "descend_or_precontact",
    "intent": "move the base/palm or appendages toward the object/contact region",
    "min_duration_s": 0.2
  },
  {
    "phase": "close_until_touch_or_settle",
    "intent": "move the affected appendages toward contact; use touch checks if obvious",
    "min_duration_s": 0.2
  },
  {
    "phase": "verify_contact_or_settle",
    "intent": "confirm contact or wait long enough for contact/force to settle",
    "min_duration_s": 0.2
  },
  {
    "phase": "lift_or_transport",
    "intent": "move the base or object only after the preceding phases",
    "min_duration_s": 0.2
  },
  {
    "phase": "stabilize_or_release",
    "intent": "hold, stabilize, or release according to the task",
    "min_duration_s": 0.2
  }
]
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

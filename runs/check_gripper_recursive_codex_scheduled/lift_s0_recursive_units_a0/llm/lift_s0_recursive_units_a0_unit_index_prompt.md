You are the subagent for the `index` fundamental unit only.
Translate the supplied high-level unit command into lower-level commands until every command is a lowest-level actuator/base command. You may only command joints listed in this unit schema. Return the recursive trace and lowest-level blocks.
Allowed lowest-level ops: set_base_target, set_joint_target, set_joint_targets, set_appendage_joints, wait, monitor, return. Do not use task-named primitives.
Every block must include a `phase` chosen from the phase schedule. If you cannot define a reliable check, use a timed wait/monitor block in `verify_contact_or_settle`.
Return ONLY JSON with schema:
{"unit":"...","trace":[{"level":"unit","command":"..."},{"level":"joint","command":"..."}],"blocks":[{"phase":"close_until_touch_or_settle","op":"set_joint_target","joint":"...","target":0.0,"duration_s":0.2}]}

Task: raise the object above the table and keep it controlled
Unit command: Relative to the final palm/base pose, keep the index flexed inward from open to a firm near-closed opposing pinch against the object, matching the thumb to stabilize and control the object.
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
  "unit": "index",
  "schema": {
    "role": "controls the index finger relative to the palm",
    "joints": [
      {
        "name": "g_left",
        "current": 0.06,
        "range": [
          0.0,
          0.06
        ],
        "meaning": {
          "0.0000..0.0150": "joint/actuator is near extended/open",
          "0.0150..0.0300": "joint/actuator is flexed a small amount",
          "0.0300..0.0450": "joint/actuator is flexed a medium amount",
          "0.0450..0.0600": "joint/actuator is flexed a large amount or near closed"
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

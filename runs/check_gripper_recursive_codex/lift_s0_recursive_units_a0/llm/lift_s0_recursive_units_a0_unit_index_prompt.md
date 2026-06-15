You are the subagent for the `index` fundamental unit only.
Translate the supplied high-level unit command into lower-level commands until every command is a lowest-level actuator/base command. You may only command joints listed in this unit schema. Return the recursive trace and lowest-level blocks.
Allowed lowest-level ops: set_base_target, set_joint_target, set_joint_targets, set_appendage_joints, wait, monitor, return. Do not use task-named primitives.
Return ONLY JSON with schema:
{"unit":"...","trace":[{"level":"unit","command":"..."},{"level":"joint","command":"..."}],"blocks":[{"op":"set_joint_target","joint":"...","target":0.0,"duration_s":0.2}]}

Task: raise the object above the table and keep it controlled
Unit command: Relative to the lifted palm, keep the index finger flexed near closed to maintain opposing left-side contact and stabilize the object against the thumb.
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
          "0.0000..0.0150": "joint is near extended/open",
          "0.0150..0.0300": "joint is flexed a small amount",
          "0.0300..0.0450": "joint is flexed a medium amount",
          "0.0450..0.0600": "joint is flexed a large amount or near closed"
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

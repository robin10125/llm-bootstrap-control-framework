You are the final compiler agent. Combine all unit policies into one executable JSON policy for the robot. Preserve recursive_trace. Use only lowest-level blocks. Because base targets are absolute, order base/wrist stabilization before finger closure unless a unit policy explicitly requires waiting. Do not invent unsupported ops.
Allowed ops: set_base_target, set_joint_target, set_joint_targets, set_appendage_joints, wait, monitor, return.
Return ONLY JSON: {"recursive_trace": {...}, "blocks": [...]}

Task: raise the object above the table and keep it controlled
Actuator schema: {
  "fundamental_units": [
    "base",
    "thumb",
    "index"
  ],
  "units": {
    "thumb": {
      "role": "controls the thumb finger relative to the palm",
      "joints": [
        {
          "name": "g_right",
          "current": 0.06,
          "range": [
            0.0,
            0.06
          ],
          "meaning": {
            "0.0000..0.0150": "joint/actuator is flexed a large amount or near closed",
            "0.0150..0.0300": "joint/actuator is flexed a medium amount",
            "0.0300..0.0450": "joint/actuator is flexed a small amount",
            "0.0450..0.0600": "joint/actuator is near extended/open",
            "open_target": "0.0600",
            "closed_target": "0.0000"
          }
        }
      ]
    },
    "index": {
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
            "0.0000..0.0150": "joint/actuator is flexed a large amount or near closed",
            "0.0150..0.0300": "joint/actuator is flexed a medium amount",
            "0.0300..0.0450": "joint/actuator is flexed a small amount",
            "0.0450..0.0600": "joint/actuator is near extended/open",
            "open_target": "0.0600",
            "closed_target": "0.0000"
          }
        }
      ]
    },
    "hand": {
      "role": "appendage-local actuator group",
      "joints": [
        {
          "name": "g_left",
          "current": 0.06,
          "range": [
            0.0,
            0.06
          ],
          "meaning": {
            "0.0000..0.0150": "joint/actuator is flexed a large amount or near closed",
            "0.0150..0.0300": "joint/actuator is flexed a medium amount",
            "0.0300..0.0450": "joint/actuator is flexed a small amount",
            "0.0450..0.0600": "joint/actuator is near extended/open",
            "open_target": "0.0600",
            "closed_target": "0.0000"
          }
        },
        {
          "name": "g_right",
          "current": 0.06,
          "range": [
            0.0,
            0.06
          ],
          "meaning": {
            "0.0000..0.0150": "joint/actuator is flexed a large amount or near closed",
            "0.0150..0.0300": "joint/actuator is flexed a medium amount",
            "0.0300..0.0450": "joint/actuator is flexed a small amount",
            "0.0450..0.0600": "joint/actuator is near extended/open",
            "open_target": "0.0600",
            "closed_target": "0.0000"
          }
        }
      ]
    },
    "base": {
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
}
Input traces and unit policies: {
  "top_trace": [
    {
      "level": "task",
      "command": "Lift the object above the table height threshold while maintaining a stable controlled grasp."
    }
  ],
  "unit_commands": [
    {
      "unit": "base",
      "command": "Move the palm/base to align over the object's current xy position near (-0.0394, -0.0383), then end in a raised/retracted vertical state so the grasped object is lifted above z=0.075 while keeping the palm centered over the object."
    },
    {
      "unit": "thumb",
      "command": "Relative to the raised, centered palm, flex the thumb inward from open to form and maintain a firm opposing contact on the object without over-squeezing."
    },
    {
      "unit": "index",
      "command": "Relative to the raised, centered palm, flex the index inward from open to oppose the thumb and maintain a firm stabilizing pinch contact on the object during the lift."
    }
  ],
  "unit_policies": [
    {
      "unit": "base",
      "trace": [
        {
          "level": "unit",
          "command": "Move palm/base to object xy near (-0.0394, -0.0383), then retract vertically while maintaining xy centering so the grasped object is lifted above z=0.075."
        },
        {
          "level": "joint",
          "command": "Set base_x to -0.0394 to center over object x."
        },
        {
          "level": "joint",
          "command": "Set base_y to -0.0383 to center over object y."
        },
        {
          "level": "joint",
          "command": "Set base_z to -0.0750 for a raised/retracted vertical state."
        }
      ],
      "blocks": [
        {
          "op": "set_joint_target",
          "joint": "base_x",
          "target": -0.0394,
          "duration_s": 0.4
        },
        {
          "op": "set_joint_target",
          "joint": "base_y",
          "target": -0.0383,
          "duration_s": 0.4
        },
        {
          "op": "set_joint_target",
          "joint": "base_z",
          "target": -0.075,
          "duration_s": 0.6
        },
        {
          "op": "monitor",
          "joint": "base_x",
          "target": -0.0394,
          "duration_s": 0.2
        },
        {
          "op": "monitor",
          "joint": "base_y",
          "target": -0.0383,
          "duration_s": 0.2
        },
        {
          "op": "return"
        }
      ]
    },
    {
      "unit": "thumb",
      "trace": [
        {
          "level": "unit",
          "command": "Relative to the raised, centered palm, flex the thumb inward from open to form and maintain a firm opposing contact on the object without over-squeezing."
        },
        {
          "level": "joint",
          "command": "Set g_right to a large but not maximum flexion for firm opposing contact, then hold."
        }
      ],
      "blocks": [
        {
          "op": "set_joint_target",
          "joint": "g_right",
          "target": 0.052,
          "duration_s": 0.25
        },
        {
          "op": "wait",
          "duration_s": 0.5
        },
        {
          "op": "return"
        }
      ]
    },
    {
      "unit": "index",
      "trace": [
        {
          "level": "unit",
          "command": "Relative to the raised, centered palm, flex the index inward from open to oppose the thumb and maintain a firm stabilizing pinch contact on the object during the lift."
        },
        {
          "level": "joint",
          "command": "Set g_left to large flexion/near closed and hold to maintain inward opposing contact."
        }
      ],
      "blocks": [
        {
          "op": "set_joint_target",
          "joint": "g_left",
          "target": 0.06,
          "duration_s": 0.2
        },
        {
          "op": "monitor",
          "joint": "g_left",
          "target": 0.06,
          "duration_s": 1.0
        },
        {
          "op": "return"
        }
      ]
    }
  ]
}

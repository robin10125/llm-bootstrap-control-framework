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
            "0.0000..0.0150": "joint is near extended/open",
            "0.0150..0.0300": "joint is flexed a small amount",
            "0.0300..0.0450": "joint is flexed a medium amount",
            "0.0450..0.0600": "joint is flexed a large amount or near closed"
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
            "0.0000..0.0150": "joint is near extended/open",
            "0.0150..0.0300": "joint is flexed a small amount",
            "0.0300..0.0450": "joint is flexed a medium amount",
            "0.0450..0.0600": "joint is flexed a large amount or near closed"
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
            "0.0000..0.0150": "joint is near extended/open",
            "0.0150..0.0300": "joint is flexed a small amount",
            "0.0300..0.0450": "joint is flexed a medium amount",
            "0.0450..0.0600": "joint is flexed a large amount or near closed"
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
            "0.0000..0.0150": "joint is near extended/open",
            "0.0150..0.0300": "joint is flexed a small amount",
            "0.0300..0.0450": "joint is flexed a medium amount",
            "0.0450..0.0600": "joint is flexed a large amount or near closed"
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
      "command": "Lift the object above the table height threshold while maintaining a stable opposing grasp."
    }
  ],
  "unit_commands": [
    {
      "unit": "base",
      "command": "Move the palm/base to stay centered over the object at x=-0.0394, y=-0.0383 and raise upward enough for the object to exceed z=0.075 while keeping the palm steady and aligned with the object."
    },
    {
      "unit": "thumb",
      "command": "Relative to the lifted palm, keep the thumb flexed near closed to maintain firm right-side contact and prevent the object from slipping during the upward motion."
    },
    {
      "unit": "index",
      "command": "Relative to the lifted palm, keep the index finger flexed near closed to maintain opposing left-side contact and stabilize the object against the thumb."
    }
  ],
  "unit_policies": [
    {
      "unit": "base",
      "trace": [
        {
          "level": "unit",
          "command": "Move the palm/base to stay centered over the object at x=-0.0394, y=-0.0383 and raise upward enough for the object to exceed z=0.075 while keeping the palm steady and aligned with the object."
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
          "command": "Move base_z upward from 0.0262 to -0.0350 so the controlled object rises above z=0.075."
        },
        {
          "level": "joint",
          "command": "Hold base_x/base_y/base_z steady and monitor object height and alignment."
        }
      ],
      "blocks": [
        {
          "op": "set_joint_targets",
          "targets": {
            "base_x": -0.0394,
            "base_y": -0.0383,
            "base_z": -0.035
          },
          "duration_s": 0.8
        },
        {
          "op": "monitor",
          "conditions": {
            "object_z_min": 0.075,
            "base_xy_target": [
              -0.0394,
              -0.0383
            ],
            "max_xy_error": 0.01
          },
          "duration_s": 0.4
        },
        {
          "op": "set_joint_targets",
          "targets": {
            "base_x": -0.0394,
            "base_y": -0.0383,
            "base_z": -0.035
          },
          "duration_s": 0.3
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
          "command": "Relative to the lifted palm, keep the thumb flexed near closed to maintain firm right-side contact and prevent the object from slipping during the upward motion."
        },
        {
          "level": "joint",
          "command": "Hold thumb joint g_right at near-closed flexion to maintain firm contact."
        }
      ],
      "blocks": [
        {
          "op": "set_joint_target",
          "joint": "g_right",
          "target": 0.06,
          "duration_s": 0.2
        },
        {
          "op": "monitor",
          "joint": "g_right",
          "target": 0.06,
          "duration_s": 1.0
        }
      ]
    },
    {
      "unit": "index",
      "trace": [
        {
          "level": "unit",
          "command": "Relative to the lifted palm, keep the index finger flexed near closed to maintain opposing left-side contact and stabilize the object against the thumb."
        },
        {
          "level": "joint",
          "command": "Hold g_left near closed at 0.0600 for opposing left-side contact."
        }
      ],
      "blocks": [
        {
          "op": "set_joint_target",
          "joint": "g_left",
          "target": 0.06,
          "duration_s": 0.2
        }
      ]
    }
  ]
}

You are the final compiler agent. Combine all unit policies into one executable JSON policy for the robot. Preserve recursive_trace. Use only lowest-level blocks. Respect phase order strictly: approach -> descend_or_precontact -> close_until_touch_or_settle -> verify_contact_or_settle -> lift_or_transport -> stabilize_or_release. This is more important than grouping by unit. Do not raise or transport the base before the close/touch and verify/settle phases have occurred. Do not invent unsupported ops.
Allowed ops: set_base_target, set_joint_target, set_joint_targets, set_appendage_joints, wait, monitor, return.
Return ONLY JSON: {"recursive_trace": {...}, "blocks": [...]}

Task: raise the object above the table and keep it controlled
Required phase schedule: [
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
      "command": "Lift the object from its start position above the table and maintain a stable opposed thumb-index grasp until the object remains controlled above the height threshold."
    }
  ],
  "phase_schedule": [
    {
      "phase": "approach",
      "intent": "Move the palm/base laterally over the object while keeping thumb and index open for clearance.",
      "min_duration_s": 0.35
    },
    {
      "phase": "descend_precontact",
      "intent": "Lower the palm toward the object to establish a centered pre-grasp without pushing it sideways.",
      "min_duration_s": 0.35
    },
    {
      "phase": "close_to_contact",
      "intent": "Flex thumb and index toward each other around the object until contact or near-closed timed grasp is reached.",
      "min_duration_s": 0.45
    },
    {
      "phase": "settle_verify",
      "intent": "Hold the closed grasp briefly so opposing contacts stabilize before lifting.",
      "min_duration_s": 0.25
    },
    {
      "phase": "lift_transport",
      "intent": "Retract/raise the palm vertically while maintaining the closed opposed grasp so the object rises above 0.075 m.",
      "min_duration_s": 0.8
    },
    {
      "phase": "hold_control",
      "intent": "Keep palm steady above the table and maintain finger closure to prevent slip.",
      "min_duration_s": 0.3
    }
  ],
  "unit_commands": [
    {
      "unit": "base",
      "command": "End with the palm centered over the object start xy at approximately base_x=-0.0394, base_y=-0.0383, then retracted upward after grasp to a moderately high final base_z around -0.04 to -0.06 so the grasped object is lifted above the table height threshold; keep xy steady during lift and hold."
    },
    {
      "unit": "thumb",
      "command": "Relative to the final centered palm pose, move from open g_right=0.06 toward strong flexion/near closed, about g_right=0.005 to 0.015, maintaining inward opposing pressure on the object throughout lift and hold."
    },
    {
      "unit": "index",
      "command": "Relative to the final centered palm pose, move from open g_left=0.06 toward strong flexion/near closed, about g_left=0.005 to 0.015, matching thumb closure to form a stable pinch grip and maintain contact during lift and hold."
    }
  ],
  "unit_policies": [
    {
      "unit": "base",
      "trace": [
        {
          "level": "unit",
          "command": "End with palm centered over object start xy, then retract upward after grasp to moderately high base_z while keeping xy steady and holding controlled."
        },
        {
          "level": "joint",
          "command": "Move base_x to -0.0394 and base_y to -0.0383 at current/precontact height."
        },
        {
          "level": "joint",
          "command": "Hold base position while appendage units close and contact settles."
        },
        {
          "level": "joint",
          "command": "Lift by changing only base_z to -0.05 while holding base_x=-0.0394 and base_y=-0.0383."
        },
        {
          "level": "joint",
          "command": "Hold final base pose to keep object controlled."
        }
      ],
      "blocks": [
        {
          "phase": "approach",
          "op": "set_base_target",
          "target": {
            "base_x": -0.0394,
            "base_y": -0.0383,
            "base_z": 0.0262
          },
          "duration_s": 0.4
        },
        {
          "phase": "descend_or_precontact",
          "op": "set_base_target",
          "target": {
            "base_x": -0.0394,
            "base_y": -0.0383,
            "base_z": 0.04
          },
          "duration_s": 0.25
        },
        {
          "phase": "close_until_touch_or_settle",
          "op": "wait",
          "duration_s": 0.3
        },
        {
          "phase": "verify_contact_or_settle",
          "op": "monitor",
          "duration_s": 0.3
        },
        {
          "phase": "lift_or_transport",
          "op": "set_base_target",
          "target": {
            "base_x": -0.0394,
            "base_y": -0.0383,
            "base_z": -0.05
          },
          "duration_s": 0.55
        },
        {
          "phase": "stabilize_or_release",
          "op": "set_base_target",
          "target": {
            "base_x": -0.0394,
            "base_y": -0.0383,
            "base_z": -0.05
          },
          "duration_s": 0.3
        },
        {
          "phase": "stabilize_or_release",
          "op": "return",
          "status": "complete"
        }
      ]
    },
    {
      "unit": "thumb",
      "trace": [
        {
          "level": "unit",
          "command": "Relative to final centered palm pose, move thumb actuator from g_right=0.06 toward strong opposing closure around g_right=0.005..0.015 and maintain inward pressure during lift/hold."
        },
        {
          "level": "joint",
          "command": "Use only g_right: keep clear during approach/precontact, close progressively to 0.010, wait for settle, then hold 0.010 through lift and stabilization."
        }
      ],
      "blocks": [
        {
          "phase": "approach",
          "op": "wait",
          "duration_s": 0.2
        },
        {
          "phase": "descend_or_precontact",
          "op": "wait",
          "duration_s": 0.2
        },
        {
          "phase": "close_until_touch_or_settle",
          "op": "set_joint_target",
          "joint": "g_right",
          "target": 0.03,
          "duration_s": 0.25
        },
        {
          "phase": "close_until_touch_or_settle",
          "op": "set_joint_target",
          "joint": "g_right",
          "target": 0.015,
          "duration_s": 0.25
        },
        {
          "phase": "close_until_touch_or_settle",
          "op": "set_joint_target",
          "joint": "g_right",
          "target": 0.01,
          "duration_s": 0.25
        },
        {
          "phase": "verify_contact_or_settle",
          "op": "wait",
          "duration_s": 0.25
        },
        {
          "phase": "lift_or_transport",
          "op": "set_joint_target",
          "joint": "g_right",
          "target": 0.01,
          "duration_s": 0.4
        },
        {
          "phase": "stabilize_or_release",
          "op": "set_joint_target",
          "joint": "g_right",
          "target": 0.01,
          "duration_s": 0.5
        },
        {
          "phase": "stabilize_or_release",
          "op": "return"
        }
      ]
    },
    {
      "unit": "index",
      "trace": [
        {
          "level": "unit",
          "command": "Relative to the final centered palm pose, move from open g_left=0.06 toward strong flexion/near closed, about g_left=0.005 to 0.015, and maintain contact during lift and hold."
        },
        {
          "level": "joint",
          "command": "Command only index joint g_left: close from 0.06 to 0.010, wait for contact/settle, then keep g_left at 0.010 during lift/hold."
        }
      ],
      "blocks": [
        {
          "phase": "close_until_touch_or_settle",
          "op": "set_joint_target",
          "joint": "g_left",
          "target": 0.01,
          "duration_s": 0.35
        },
        {
          "phase": "verify_contact_or_settle",
          "op": "wait",
          "duration_s": 0.25
        },
        {
          "phase": "lift_or_transport",
          "op": "set_joint_target",
          "joint": "g_left",
          "target": 0.01,
          "duration_s": 0.3
        },
        {
          "phase": "stabilize_or_release",
          "op": "set_joint_target",
          "joint": "g_left",
          "target": 0.01,
          "duration_s": 0.5
        },
        {
          "phase": "stabilize_or_release",
          "op": "return",
          "status": "index_hold_target_set"
        }
      ]
    }
  ]
}

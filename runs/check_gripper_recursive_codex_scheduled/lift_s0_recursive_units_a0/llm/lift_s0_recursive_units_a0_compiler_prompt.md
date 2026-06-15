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
      "command": "Lift the object from its start position above the table to above the height threshold while maintaining a stable opposing pinch grip."
    }
  ],
  "phase_schedule": [
    {
      "phase": "approach",
      "intent": "Move the palm/base laterally over the object at its current xy position while keeping thumb and index open.",
      "min_duration_s": 0.3
    },
    {
      "phase": "descend_pre_contact",
      "intent": "Lower the palm/base toward the object until the open thumb and index straddle it without pushing it away.",
      "min_duration_s": 0.3
    },
    {
      "phase": "close_settle",
      "intent": "Flex thumb and index inward symmetrically to establish opposing contact and allow the object to settle in the grasp.",
      "min_duration_s": 0.5
    },
    {
      "phase": "verify_or_settle",
      "intent": "Maintain the pinch briefly to confirm contact by stability or timed settling before lifting.",
      "min_duration_s": 0.3
    },
    {
      "phase": "lift_hold",
      "intent": "Raise/retract the palm/base while keeping thumb and index flexed enough to hold the object controlled above the table threshold.",
      "min_duration_s": 0.8
    }
  ],
  "unit_commands": [
    {
      "unit": "base",
      "command": "End with the palm centered over the object's start xy position near x=-0.0394, y=-0.0383, then retracted upward/high enough after grasp so the object is lifted above z=0.075 and remains controlled."
    },
    {
      "unit": "thumb",
      "command": "Relative to the final palm/base pose, keep the thumb flexed inward from open to a firm near-closed opposing pinch against the object, maintaining grip during lift."
    },
    {
      "unit": "index",
      "command": "Relative to the final palm/base pose, keep the index flexed inward from open to a firm near-closed opposing pinch against the object, matching the thumb to stabilize and control the object."
    }
  ],
  "unit_policies": [
    {
      "unit": "base",
      "trace": [
        {
          "level": "unit",
          "command": "End with the palm centered over the object's start xy position near x=-0.0394, y=-0.0383, then retracted upward/high enough after grasp so the object is lifted above z=0.075 and remains controlled."
        },
        {
          "level": "joint",
          "command": "Set base_x to -0.0394 and base_y to -0.0383 to center the palm over the object start xy."
        },
        {
          "level": "joint",
          "command": "Lower base_z toward the object/contact region before grasp."
        },
        {
          "level": "joint",
          "command": "Hold base position while non-base appendages grasp and contact settles."
        },
        {
          "level": "joint",
          "command": "Retract base_z upward to a high negative target while maintaining base_x/base_y so the grasped object lifts above z=0.075."
        },
        {
          "level": "joint",
          "command": "Hold the lifted base pose to keep the object controlled."
        }
      ],
      "blocks": [
        {
          "phase": "approach",
          "op": "set_joint_targets",
          "targets": {
            "base_x": -0.0394,
            "base_y": -0.0383,
            "base_z": 0.025
          },
          "duration_s": 0.4
        },
        {
          "phase": "descend_or_precontact",
          "op": "set_joint_target",
          "joint": "base_z",
          "target": 0.075,
          "duration_s": 0.3
        },
        {
          "phase": "close_until_touch_or_settle",
          "op": "wait",
          "duration_s": 0.2
        },
        {
          "phase": "verify_contact_or_settle",
          "op": "monitor",
          "duration_s": 0.3
        },
        {
          "phase": "lift_or_transport",
          "op": "set_joint_targets",
          "targets": {
            "base_x": -0.0394,
            "base_y": -0.0383,
            "base_z": -0.075
          },
          "duration_s": 0.5
        },
        {
          "phase": "stabilize_or_release",
          "op": "monitor",
          "duration_s": 0.3
        },
        {
          "phase": "stabilize_or_release",
          "op": "return"
        }
      ]
    },
    {
      "unit": "thumb",
      "trace": [
        {
          "level": "unit",
          "command": "Relative to the final palm/base pose, keep the thumb flexed inward from open to a firm near-closed opposing pinch against the object, maintaining grip during lift."
        },
        {
          "level": "joint",
          "command": "Use thumb joint g_right only: start open/clear, flex inward to near-closed pinch, wait for contact/settle, then hold the near-closed target through lift and stabilization."
        }
      ],
      "blocks": [
        {
          "phase": "approach",
          "op": "set_joint_target",
          "joint": "g_right",
          "target": 0.0,
          "duration_s": 0.2
        },
        {
          "phase": "descend_or_precontact",
          "op": "set_joint_target",
          "joint": "g_right",
          "target": 0.025,
          "duration_s": 0.2
        },
        {
          "phase": "close_until_touch_or_settle",
          "op": "set_joint_target",
          "joint": "g_right",
          "target": 0.058,
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
          "joint": "g_right",
          "target": 0.058,
          "duration_s": 0.2
        },
        {
          "phase": "stabilize_or_release",
          "op": "set_joint_target",
          "joint": "g_right",
          "target": 0.058,
          "duration_s": 0.3
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
          "command": "Relative to the final palm/base pose, keep the index flexed inward from open to a firm near-closed opposing pinch against the object, matching the thumb to stabilize and control the object."
        },
        {
          "level": "joint",
          "command": "Drive index joint g_left inward to near-closed flexion and hold it for stable opposing contact."
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
          "joint": "g_left",
          "target": 0.055,
          "duration_s": 0.25
        },
        {
          "phase": "verify_contact_or_settle",
          "op": "monitor",
          "duration_s": 0.25,
          "condition": "timed_settle_no_index_touch_sensor"
        },
        {
          "phase": "stabilize_or_release",
          "op": "set_joint_target",
          "joint": "g_left",
          "target": 0.055,
          "duration_s": 0.3
        },
        {
          "phase": "stabilize_or_release",
          "op": "return"
        }
      ]
    }
  ]
}

You are a reflection and error-correction agent for a failed robot policy.
Analyze why the previous scheduled policy failed, then output a corrected executable JSON policy. Preserve or update recursive_trace, include a short `reflection`, and use only supported lowest-level ops.

Important execution constraints:
- Keep strict phase order: approach -> descend_or_precontact -> close_until_touch_or_settle -> verify_contact_or_settle -> lift_or_transport -> stabilize_or_release.
- If max_object_z did not increase, assume the object was not contacted or not held. Do not lift/transport before a sufficiently deep descend, closure, and settle period.
- Use the joint schema for actuator direction. For base_z, high values lower the palm toward the table/object and low/negative values raise/retract it.
- For gripper-like fingers, use closed_target/open_target from the schema; do not guess that larger means more closed.
- If no reliable condition is available, use timed wait/monitor in verify_contact_or_settle.

Allowed ops: set_base_target, set_joint_target, set_joint_targets, set_appendage_joints, wait, monitor, return.
Return ONLY JSON: {"reflection": {...}, "recursive_trace": {...}, "blocks": [...]}

Task: raise the object above the table and keep it controlled
Task context: {
  "name": "lift",
  "goal": "raise the object above the table and keep it controlled",
  "seed": 0,
  "episode_seconds": 2.5,
  "object_start": [
    -0.0394,
    -0.0383,
    0.025
  ],
  "target_xy": null,
  "metadata": {
    "height_threshold": 0.07498264610767365
  }
}
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
Derived context: {
  "object_to_base_distance": 0.055,
  "object_height": 0.025,
  "object_speed": 0.0003,
  "appendage_distances_to_object": {}
}
Previous rollout result: {
  "interface": "recursive_units",
  "task": "lift",
  "seed": 0,
  "success": false,
  "score": 0.0002,
  "total_return": -3.183,
  "final_object_x": -0.03941647708415985,
  "final_object_y": -0.03833186626434326,
  "final_object_z": 0.024992680177092552,
  "max_object_z": 0.025,
  "errors": ""
}
Previous program: {
  "recursive_trace": {
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
    "unit_traces": {
      "base": [
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
      "thumb": [
        {
          "level": "unit",
          "command": "Relative to final centered palm pose, move thumb actuator from g_right=0.06 toward strong opposing closure around g_right=0.005..0.015 and maintain inward pressure during lift/hold."
        },
        {
          "level": "joint",
          "command": "Use only g_right: keep clear during approach/precontact, close progressively to 0.010, wait for settle, then hold 0.010 through lift and stabilization."
        }
      ],
      "index": [
        {
          "level": "unit",
          "command": "Relative to the final centered palm pose, move from open g_left=0.06 toward strong flexion/near closed, about g_left=0.005 to 0.015, and maintain contact during lift and hold."
        },
        {
          "level": "joint",
          "command": "Command only index joint g_left: close from 0.06 to 0.010, wait for contact/settle, then keep g_left at 0.010 during lift/hold."
        }
      ]
    }
  },
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
      "phase": "approach",
      "op": "wait",
      "duration_s": 0.2
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
      "joint": "g_left",
      "target": 0.01,
      "duration_s": 0.35
    },
    {
      "phase": "close_until_touch_or_settle",
      "op": "set_joint_target",
      "joint": "g_right",
      "target": 0.01,
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
      "phase": "verify_contact_or_settle",
      "op": "wait",
      "duration_s": 0.25
    },
    {
      "phase": "verify_contact_or_settle",
      "op": "wait",
      "duration_s": 0.25
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
      "phase": "lift_or_transport",
      "op": "set_joint_target",
      "joint": "g_right",
      "target": 0.01,
      "duration_s": 0.4
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
      "op": "set_joint_target",
      "joint": "g_right",
      "target": 0.01,
      "duration_s": 0.5
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
      "status": "complete"
    }
  ]
}

You are the top-level policy decomposition agent for a Shadow-style dexterous hand.
Decompose the task into one command for EVERY fundamental unit listed below. Do not emit robot actions yet. The base command should describe the desired palm/base end state; finger/wrist commands should be relative to that base/palm end state.
Also produce a sequential phase schedule. For contact manipulation, prefer this general order: approach over/near object, descend or pre-contact, close/flex until touch or timed settle, verify contact or timed settle, then lift/transport. This prevents units from moving away prematurely when exact checks are not obvious.
Return ONLY JSON with this schema:
{"trace":[{"level":"task","command":"..."}],"phase_schedule":[{"phase":"approach","intent":"...","min_duration_s":0.2}],"unit_commands":[{"unit":"base","command":"..."},{"unit":"index","command":"..."}]}

Task: raise the object above the table and keep it controlled
Task context: {
  "name": "lift",
  "goal": "raise the object above the table and keep it controlled",
  "seed": 0,
  "episode_seconds": 3.0,
  "object_start": [
    0.0082,
    -0.0138,
    0.025
  ],
  "target_xy": null,
  "metadata": {
    "height_threshold": 0.07500000037252903
  }
}
Fundamental units: ["base", "thumb", "index"]
World and derived context: {
  "time_s": 0.0,
  "object": {
    "pos": [
      0.008200000040233135,
      -0.013799999840557575,
      0.02500000037252903
    ],
    "vel": [
      0.0,
      0.0,
      0.0
    ]
  },
  "base_q": [
    0.0,
    0.0,
    0.0
  ],
  "hand_q": [],
  "ctrl": {
    "base_x": 0.0,
    "base_y": 0.0,
    "base_z": 0.0,
    "rh_A_FFJ4": 0.0,
    "rh_A_THJ5": 0.0
  },
  "appendages": {
    "thumb": [
      "rh_A_THJ5"
    ],
    "index": [
      "rh_A_FFJ4"
    ],
    "hand": [
      "rh_A_FFJ4",
      "rh_A_THJ5"
    ],
    "base": [
      "base_x",
      "base_y",
      "base_z"
    ]
  },
  "joint_schema": {
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
            "name": "rh_A_THJ5",
            "current": 0.0,
            "range": [
              0.0,
              1.0
            ],
            "meaning": {
              "0.0000..0.2500": "joint/actuator is near extended/open",
              "0.2500..0.5000": "joint/actuator is flexed a small amount",
              "0.5000..0.7500": "joint/actuator is flexed a medium amount",
              "0.7500..1.0000": "joint/actuator is flexed a large amount or near closed",
              "open_target": "0.0000",
              "closed_target": "1.0000"
            }
          }
        ]
      },
      "index": {
        "role": "controls the index finger relative to the palm",
        "joints": [
          {
            "name": "rh_A_FFJ4",
            "current": 0.0,
            "range": [
              0.0,
              1.0
            ],
            "meaning": {
              "0.0000..0.2500": "joint/actuator is near extended/open",
              "0.2500..0.5000": "joint/actuator is flexed a small amount",
              "0.5000..0.7500": "joint/actuator is flexed a medium amount",
              "0.7500..1.0000": "joint/actuator is flexed a large amount or near closed",
              "open_target": "0.0000",
              "closed_target": "1.0000"
            }
          }
        ]
      },
      "hand": {
        "role": "appendage-local actuator group",
        "joints": [
          {
            "name": "rh_A_FFJ4",
            "current": 0.0,
            "range": [
              0.0,
              1.0
            ],
            "meaning": {
              "0.0000..0.2500": "joint/actuator is near extended/open",
              "0.2500..0.5000": "joint/actuator is flexed a small amount",
              "0.5000..0.7500": "joint/actuator is flexed a medium amount",
              "0.7500..1.0000": "joint/actuator is flexed a large amount or near closed",
              "open_target": "0.0000",
              "closed_target": "1.0000"
            }
          },
          {
            "name": "rh_A_THJ5",
            "current": 0.0,
            "range": [
              0.0,
              1.0
            ],
            "meaning": {
              "0.0000..0.2500": "joint/actuator is near extended/open",
              "0.2500..0.5000": "joint/actuator is flexed a small amount",
              "0.5000..0.7500": "joint/actuator is flexed a medium amount",
              "0.7500..1.0000": "joint/actuator is flexed a large amount or near closed",
              "open_target": "0.0000",
              "closed_target": "1.0000"
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
            "current": 0.0,
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
  },
  "derived": {
    "object_to_base_distance": 0.0297,
    "object_height": 0.025,
    "object_speed": 0.0,
    "appendage_distances_to_object": {}
  },
  "fingertips": {},
  "contacts": []
}

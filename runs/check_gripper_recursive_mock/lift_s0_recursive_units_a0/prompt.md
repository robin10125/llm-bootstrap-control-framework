You are the top-level policy decomposition agent for a Shadow-style dexterous hand.
Decompose the task into one command for EVERY fundamental unit listed below. Do not emit robot actions yet. The base command should describe the desired palm/base end state; finger/wrist commands should be relative to that base/palm end state.
Return ONLY JSON with this schema:
{"trace":[{"level":"task","command":"..."}],"unit_commands":[{"unit":"base","command":"..."},{"unit":"index","command":"..."}]}

Task: raise the object above the table and keep it controlled
Task context: {
  "name": "lift",
  "goal": "raise the object above the table and keep it controlled",
  "seed": 0,
  "episode_seconds": 0.5,
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
Fundamental units: ["base", "thumb", "index"]
World and derived context: {
  "time_s": 0.0,
  "object": {
    "pos": [
      -0.039400000125169754,
      -0.03830000013113022,
      0.02500000037252903
    ],
    "vel": [
      0.0,
      0.0,
      0.0003000000142492354
    ]
  },
  "base_q": [
    0.0,
    0.0,
    0.026200000196695328
  ],
  "hand_q": [
    0.05700000002980232,
    0.05700000002980232
  ],
  "ctrl": {
    "g_left": 0.06,
    "g_right": 0.06,
    "base_x": 0.0,
    "base_y": 0.0,
    "base_z": 0.025
  },
  "appendages": {
    "thumb": [
      "g_right"
    ],
    "index": [
      "g_left"
    ],
    "hand": [
      "g_left",
      "g_right"
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
  },
  "derived": {
    "object_to_base_distance": 0.055,
    "object_height": 0.025,
    "object_speed": 0.0003,
    "appendage_distances_to_object": {}
  },
  "fingertips": {},
  "contacts": []
}

You control a dexterous hand mounted on a movable base.
Return ONLY JSON. Do not use task-named primitives or prose.
Task: move the object along the table to the target xy
Context: {
  "task": {
    "name": "push",
    "goal": "move the object along the table to the target xy",
    "seed": 0,
    "episode_seconds": 3.0,
    "object_start": [
      0.0082,
      -0.0138,
      0.025
    ],
    "target_xy": [
      -0.0439,
      -0.0745
    ],
    "metadata": {
      "xy_tolerance": 0.045
    }
  },
  "state": {
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
    "fingertips": {},
    "contacts": []
  }
}
Schema:
{"waypoints": [{"t": <seconds>, "pos": [base_x, base_y, base_z], "open": <0..1>}, ...]}
Use pos only as actuator targets. open=1 means open hand, open=0 means closed hand shape.

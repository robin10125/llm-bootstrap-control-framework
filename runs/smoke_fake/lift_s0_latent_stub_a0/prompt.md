Return ONLY JSON containing generic latent action decoder blocks for a dexterous hand.
Tokens are low-level motion/contact modes, not task names. Do not use task-named primitives such as grasp, lift, throw, fold, pick_up, place, or use_chopsticks.
Task: raise the object above the table and keep it controlled
Context: {
  "task": {
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
Schema: {"blocks": [{"op": "latent_decode", "token": "...", "duration_s": 0.5, "gain": 0.5}]}
Useful tokens: approach_object, center_over_object, oppose_and_stabilize, close_around_object, raise, stabilize_height, release, open_and_clear.

Write a hybrid JSON control program for a dexterous robotic hand.
The program may use generic script blocks plus call_controller/call_subscript/call_appendage_agent. Do not use task-named primitives such as grasp, lift, throw, fold, pick_up, place, or use_chopsticks.
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
Generic controllers available: track_frame_to_pose, track_object_relative_offset, close_hand_shape_fraction, hold_contact_count, apply_base_velocity_profile, stabilize_object_height, latent_decode.
Appendage subagents: emit call_appendage_agent with appendage=index|middle|ring|little|thumb|wrist and an inline program limited to that appendage's joints. Example: {"op":"call_appendage_agent","appendage":"index","program":{"blocks":[{"op":"set_joint_target","joint":"rh_A_FFJ4","target":0.4}]}}.
Schema: {"constants": {}, "blocks": [{"op": "...", "...": "..."}]}

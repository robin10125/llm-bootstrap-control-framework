Write a constrained JSON control program for a dexterous robotic hand.
Return ONLY JSON. Do not use task-named action primitives such as grasp, lift, throw, fold, pick_up, place, or use_chopsticks.
Task: raise the object above the table and keep it controlled
Context: {
  "task": {
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
      "height_threshold": 0.07496613748371601
    }
  },
  "state": {
    "time_s": 0.0,
    "object": {
      "pos": [
        -0.039400000125169754,
        -0.03830000013113022,
        0.02500000037252903
      ],
      "vel": [
        -0.0,
        0.0,
        0.0007999999797903001
      ]
    },
    "base_q": [
      -0.0,
      0.0,
      -0.0
    ],
    "hand_q": [
      -0.0,
      0.0,
      -0.0,
      -0.0,
      0.0,
      0.0,
      -0.0,
      -0.0,
      0.0,
      0.0,
      0.0,
      -0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      -0.0,
      0.0,
      0.0,
      -0.0,
      0.0,
      -0.0,
      -0.0,
      -0.0
    ],
    "ctrl": {
      "rh_A_WRJ2": 0.0,
      "rh_A_WRJ1": 0.0,
      "rh_A_THJ5": 0.0,
      "rh_A_THJ4": 0.0,
      "rh_A_THJ3": 0.0,
      "rh_A_THJ2": 0.0,
      "rh_A_THJ1": 0.0,
      "rh_A_FFJ4": 0.0,
      "rh_A_FFJ3": 0.0,
      "rh_A_FFJ0": 0.0,
      "rh_A_MFJ4": 0.0,
      "rh_A_MFJ3": 0.0,
      "rh_A_MFJ0": 0.0,
      "rh_A_RFJ4": 0.0,
      "rh_A_RFJ3": 0.0,
      "rh_A_RFJ0": 0.0,
      "rh_A_LFJ5": 0.0,
      "rh_A_LFJ4": 0.0,
      "rh_A_LFJ3": 0.0,
      "rh_A_LFJ0": 0.0,
      "base_x": 0.0,
      "base_y": 0.0,
      "base_z": 0.0
    },
    "appendages": {
      "wrist": [
        "rh_A_WRJ2",
        "rh_A_WRJ1"
      ],
      "thumb": [
        "rh_A_THJ5",
        "rh_A_THJ4",
        "rh_A_THJ3",
        "rh_A_THJ2",
        "rh_A_THJ1"
      ],
      "index": [
        "rh_A_FFJ4",
        "rh_A_FFJ3",
        "rh_A_FFJ0"
      ],
      "middle": [
        "rh_A_MFJ4",
        "rh_A_MFJ3",
        "rh_A_MFJ0"
      ],
      "ring": [
        "rh_A_RFJ4",
        "rh_A_RFJ3",
        "rh_A_RFJ0"
      ],
      "little": [
        "rh_A_LFJ5",
        "rh_A_LFJ4",
        "rh_A_LFJ3",
        "rh_A_LFJ0"
      ],
      "hand": [
        "rh_A_WRJ2",
        "rh_A_WRJ1",
        "rh_A_THJ5",
        "rh_A_THJ4",
        "rh_A_THJ3",
        "rh_A_THJ2",
        "rh_A_THJ1",
        "rh_A_FFJ4",
        "rh_A_FFJ3",
        "rh_A_FFJ0",
        "rh_A_MFJ4",
        "rh_A_MFJ3",
        "rh_A_MFJ0",
        "rh_A_RFJ4",
        "rh_A_RFJ3",
        "rh_A_RFJ0",
        "rh_A_LFJ5",
        "rh_A_LFJ4",
        "rh_A_LFJ3",
        "rh_A_LFJ0"
      ],
      "base": [
        "base_x",
        "base_y",
        "base_z"
      ]
    },
    "fingertips": {
      "rh_ffdistal": [
        0.009999999776482582,
        -0.032999999821186066,
        0.1379999965429306
      ],
      "rh_mfdistal": [
        0.009999999776482582,
        -0.010999999940395355,
        0.1340000033378601
      ],
      "rh_rfdistal": [
        0.009999999776482582,
        0.010999999940395355,
        0.1379999965429306
      ],
      "rh_lfdistal": [
        0.009999999776482582,
        0.032999999821186066,
        0.14650000631809235
      ],
      "rh_thdistal": [
        0.01860000006854534,
        -0.08349999785423279,
        0.22450000047683716
      ]
    },
    "contacts": []
  }
}
Allowed block ops: set_base_target, move_frame, track_relative_pose, set_joint_targets, set_joint_target, move_joint_delta, set_appendage_joints, set_hand_shape, set_impedance, seek_contact, maintain_contact, apply_wrench_or_impulse, wait, monitor, return.
Use set_joint_target for individual actuator/joint control. Use set_appendage_joints when reasoning about one finger/thumb/wrist as an isolated appendage.
Schema: {"constants": {}, "blocks": [{"op": "...", "...": "..."}]}

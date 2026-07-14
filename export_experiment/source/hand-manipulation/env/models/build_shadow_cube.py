"""Compose the Shadow Hand grab-and-lift scene from the vendored menagerie model.

The stock `shadow_hand/right_hand.xml` mounts the forearm rigidly at the origin, fingers
pointing +x — it can't translate. For the grab-and-lift task (mirroring the gripper env) we
need the hand on a movable base, palm facing down over a table. A textual <include> can't
reparent or rejoint the fixed forearm, so we edit the model with MjSpec instead:

  * reorient the forearm so the fingers point down (-z),
  * give it slide_x / slide_y / slide_z joints + position actuators (the movable base),
  * add a table plane and a free cube to the world,

then write a self-contained `shadow_hand/scene_cube.xml` (kept beside the assets so mesh
paths resolve). Run:  python env/models/build_shadow_cube.py
"""
from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
SRC = HERE / "shadow_hand" / "right_hand.xml"
OUT = HERE / "shadow_hand" / "scene_cube.xml"

SLIDE = mujoco.mjtJoint.mjJNT_SLIDE
PLANE = mujoco.mjtGeom.mjGEOM_PLANE
BOX = mujoco.mjtGeom.mjGEOM_BOX

# World z of the forearm origin at home. Palm-down, the open hand reaches ~0.41 m below the
# forearm origin, so home must be high enough to keep the fingertips above the table.
HOME_Z = 0.55
SLIDE_Z_RANGE = [-0.15, 0.20]   # world forearm z in [0.40, 0.75]; fingertips reach the table
CUBE_HALF = 0.025


def _quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def _axis_angle(axis, deg):
    axis = np.array(axis, float)
    axis /= np.linalg.norm(axis)
    h = np.radians(deg) / 2
    return np.array([np.cos(h), *(np.sin(h) * axis)])


def _palm_down_quat(spec) -> np.ndarray:
    """Search rotations of the forearm for the one that points the fingers most downward."""
    base = np.array([0.0, 1.0, 0.0, 1.0])
    base /= np.linalg.norm(base)
    candidates = {
        "as-is": base,
        "Ry+90": _quat_mul(_axis_angle([0, 1, 0], 90), base),
        "Ry-90": _quat_mul(_axis_angle([0, 1, 0], -90), base),
        "Rx+90": _quat_mul(_axis_angle([1, 0, 0], 90), base),
        "Rx-90": _quat_mul(_axis_angle([1, 0, 0], -90), base),
    }
    best, best_drop = None, 1e9
    for name, q in candidates.items():
        fa = spec.body("rh_forearm")
        fa.quat = q.tolist()
        fa.pos = [0, 0, HOME_Z]
        m = spec.compile()
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        drop = float(d.body("rh_ffdistal").xpos[2] - d.body("rh_palm").xpos[2])
        if drop < best_drop:
            best, best_drop = q, drop
        print(f"  {name:7s} fingertip-vs-palm dz = {drop:+.3f}")
    print(f"chosen palm-down quat (fingertip {best_drop:+.3f} below palm)")
    return best


def build() -> None:
    spec = mujoco.MjSpec.from_file(str(SRC))

    # 1. orient the forearm palm-down and place it at the home height.
    q = _palm_down_quat(spec)
    fa = spec.body("rh_forearm")
    fa.quat = q.tolist()
    fa.pos = [0, 0, HOME_Z]

    # 2. movable base: three world-axis slide joints on the (root) forearm.
    for name, axis, rng in [
        ("slide_x", [1, 0, 0], [-0.30, 0.30]),
        ("slide_y", [0, 1, 0], [-0.30, 0.30]),
        ("slide_z", [0, 0, 1], SLIDE_Z_RANGE),   # world z = HOME_Z + qpos
    ]:
        j = fa.add_joint()
        j.name, j.type, j.axis, j.range = name, SLIDE, axis, rng
        # damping is array-typed in this MjSpec binding; armature (added rotor inertia)
        # damps base oscillation fine alongside the stiff position servos below.
        j.armature, j.frictionloss = 0.5, 0.0

    # Gravity-compensate the whole hand so the base servos hold position instead of sagging
    # under the (heavy) forearm+hand weight — standard for a mounted/floating manipulator.
    for b in spec.bodies:
        if b.name.startswith("rh_"):
            b.gravcomp = 1.0

    # 3. table plane + a free cube to grasp.
    wb = spec.worldbody
    table = wb.add_geom()
    table.name, table.type, table.size = "table", PLANE, [0.5, 0.5, 0.01]
    table.rgba = [0.7, 0.7, 0.7, 1]
    cube = wb.add_body()
    cube.name, cube.pos = "object", [0.0, 0.0, CUBE_HALF]
    cube.add_freejoint()
    g = cube.add_geom()
    g.name, g.type, g.size = "cube", BOX, [CUBE_HALF] * 3
    g.rgba = [0.8, 0.2, 0.2, 1]
    g.condim, g.friction = 4, [1.0, 0.05, 0.001]
    g.mass = 0.05

    # 4. position actuators for the base.
    for name, target, rng, kp in [
        ("base_x", "slide_x", [-0.30, 0.30], 500),
        ("base_y", "slide_y", [-0.30, 0.30], 500),
        ("base_z", "slide_z", SLIDE_Z_RANGE, 1000),
    ]:
        a = spec.add_actuator()
        a.name, a.target = name, target
        a.trntype = mujoco.mjtTrn.mjTRN_JOINT
        a.set_to_position(kp=kp)
        a.ctrlrange, a.ctrllimited = rng, 1
        a.forcerange, a.forcelimited = [-50, 50], 1

    model = spec.compile()
    OUT.write_text(spec.to_xml())
    print(f"\nwrote {OUT}  (nq={model.nq} nu={model.nu})")


if __name__ == "__main__":
    build()

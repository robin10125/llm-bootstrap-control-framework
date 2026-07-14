#!/usr/bin/env python3
"""MJX closed-loop manipulation env, batched on GPU.

This replaces the open-loop primitive-schedule runner for the RL track: instead of
choosing one schedule per episode (a contextual bandit), the policy emits a per-joint
position-target action every control step and the episode is a real multi-step MDP that
PPO can optimize. See `rl_redesign.md`.

The env is functional and JAX-jittable. Build one `MjxEnv` (holds the static model and
index tables), then `jax.vmap` its `reset` / `step` over a batch of RNG keys to run
thousands of envs in parallel on the GPU.

First embodiment/task: parallel-jaw gripper, lift. The action space and observation are
defined generically over the scene's actuators so the Shadow Hand can reuse the same code
once its mesh scene is MJX-portable.
"""
from __future__ import annotations

import functools
import os
from pathlib import Path
from typing import Any

# On 8 GB cards, JAX's default preallocation makes ordinary diagnostics and back-to-back
# experiments look like simulator OOMs. This does not change physics; it only changes the
# allocator strategy. Users can still override these before launching Python.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from flax import struct
from mujoco import mjx

import eval_metrics as EM

HERE = Path(__file__).resolve().parent
GRIPPER_XML = HERE / "models" / "gripper_scene.xml"
SHADOW_XML = HERE.parent / "hand-manipulation" / "env" / "models" / "shadow_hand" / "scene_cube.xml"

# Shadow Hand grasp keyframes (hand-joint qpos, env hand-joint order: WRJ2,WRJ1, FF/MF/RF
# J4..J1, LF J5..J1, TH J5..J1). Open hand = all extended; close hand = full power grasp.
# Source: shadow_hand/keyframes.xml.
_SHADOW_OPEN = tuple([0.0] * 24)
_SHADOW_CLOSE = (0.0, 0.0,
                 0.0, 1.571, 1.571, 1.571,  0.0, 1.571, 1.571, 1.571,
                 0.0, 1.571, 1.571, 1.571,  0.0, 0.0, 1.571, 1.571, 1.571,
                 0.17, 1.2, 0.0, 0.61, 0.52)


_SHADOW_FINGERTIPS = ("rh_ffdistal", "rh_mfdistal", "rh_rfdistal", "rh_lfdistal", "rh_thdistal")


def make_env(task: str = "gripper", reward: Any = None, **overrides) -> "MjxEnv":
    """Build a task env. `task` selects the embodiment/scene and its MJX-portability config.

    - `gripper`: the 5-DOF parallel-jaw scene (cheap, thousands of envs).
    - `shadow`: the 23-DOF Shadow Hand on a slide base lifting a cube. Fingertip-only
      collisions and a 2-substep physics step keep it within an 8 GB GPU at small batch.

    `reward` attaches a curriculum/reward spec (else the builtin shaped reward is used):
    `"default"` = the task's hand-written default spec; a dict = a JSON reward spec; a str =
    a Tier-B reward expression. Returns the env; for `"default"`/dict the env also carries the
    `CompiledReward` on `env.compiled_reward` (for the curriculum controller + judge).
    """
    if task == "gripper":
        cfg = EnvConfig(**overrides)
        env = MjxEnv(GRIPPER_XML, cfg)
    elif task == "shadow":
        base = dict(control_dt=0.025, episode_seconds=2.5, physics_dt=0.01,
                    grasp_geom_substr=("palm", "knuckle", "proximal", "middle", "distal",
                                       "metacarpal", "thbase", "thhub"),
                    fingertip_bodies=_SHADOW_FINGERTIPS,
                    open_pose=_SHADOW_OPEN, close_pose=_SHADOW_CLOSE)
        base.update(overrides)
        env = MjxEnv(SHADOW_XML, EnvConfig(**base))
    else:
        raise ValueError(f"unknown task {task!r}")

    env.compiled_reward = None
    if reward is not None and reward != "builtin":
        import reward_spec as RS
        if reward == "default":
            compiled = RS.compile_spec(RS.DEFAULT_SPECS[task])
        elif isinstance(reward, dict):
            compiled = RS.compile_spec(reward)
        elif isinstance(reward, str):
            compiled = RS.compile_expr(reward, RS.DEFAULT_SPECS[task]["milestones"])
        else:
            raise ValueError(f"bad reward {reward!r}")
        env.compiled_reward = compiled
        env.set_reward_fn(compiled.reward_fn)
    return env

# MJX (3.9) has no cylinder<->box collision routine. `g_mount` is a non-load-bearing
# cosmetic/inertial mount on the base; disabling its contacts is lossless for the task.
_NONCOLLIDING_GEOMS = ("g_mount",)


@struct.dataclass
class EnvState:
    """Per-env state carried through a rollout (a JAX pytree)."""

    data: mjx.Data
    obs: jp.ndarray
    reward: jp.ndarray
    done: jp.ndarray
    step: jp.ndarray
    key: jp.ndarray
    stage: jp.ndarray   # curriculum stage (unlocks reward milestones 0..stage+1)
    metrics: dict[str, jp.ndarray]


@struct.dataclass
class EnvConfig:
    control_dt: float = 0.025         # seconds between policy decisions (40 Hz)
    episode_seconds: float = 2.5      # episode length in (control) time
    action_scale: float = 0.05        # max per-step change in each ctrl target
    settle_seconds: float = 0.15      # physics settle after reset, no action
    # physics cost levers (contact-rich dexterous sims are the throughput bottleneck on
    # an 8 GB 2070). Override the scene's defaults for ~4-5x faster rollouts.
    physics_dt: float = 0.005         # 200 Hz physics (scene authored at 0.002)
    pyramidal_cone: bool = True       # cheaper than the scene's elliptic cone
    solver_iterations: int = 0        # 0 = leave the scene/MJX default
    # object position randomization (metres, uniform box around the scene origin)
    obj_xy_range: float = 0.04
    physics_solver_warmup: bool = False  # reserved
    # Replace large colliding distal fingertip meshes with primitive collision proxies.
    # The visual meshes remain in the model. This removes MJX's high-poly mesh contact
    # intermediates while preserving the articulated hand, actuator layout, and contact role
    # of the fingertips.
    primitive_fingertip_collisions: bool = True
    # Articulated-hand grasp interface. If both are given (hand-joint qpos, in the env's
    # hand-joint order), the env derives open/closed actuator targets from them so the
    # single waypoint `open` scalar and the reward's "close" term work for any hand, not
    # just the 1-DOF gripper. Empty = gripper path (fingers = g_left/g_right).
    open_pose: tuple[float, ...] = ()
    close_pose: tuple[float, ...] = ()
    # collision reduction (Shadow Hand). If non-empty, only geoms whose body name contains
    # one of these substrings stay collidable as the "hand" group, plus the object and the
    # ground plane; everything else has contacts disabled. The kept hand geoms are put in a
    # contype/conaffinity group that collides with the object & plane but NOT itself. This is
    # what makes the 24-DOF mesh hand tractable in MJX on an 8 GB GPU: it removes the all-pairs
    # self-collision constraint blow-up and skips convex-hull precompute for the disabled
    # meshes (notably the large forearm collision mesh). Empty = leave the scene untouched
    # (the gripper, which is already cheap).
    grasp_geom_substr: tuple[str, ...] = ()
    # fingertip bodies (for the eval vector's per-finger DISTANCE term only). Empty =
    # gripper default (the two pad bodies).
    fingertip_bodies: tuple[str, ...] = ()
    contact_eps: float = 0.03
    # A hand region counts as "in contact" (for the eval vector's n_contacts) when its
    # measured normal contact force with the object exceeds this floor (N). Set just above
    # solver noise so a genuine light touch registers but free air / joint-limit load does not.
    contact_force_floor: float = 0.05
    # reward weights (builtin reward only; ignored when a compiled reward spec is attached)
    w_reach: float = 1.0
    w_close: float = 0.25
    w_lift: float = 12.0
    w_success: float = 5.0
    lift_target: float = 0.10         # metres above start counted as full lift reward
    success_height: float = 0.05      # metres above start = task success
    # contact gating (anti-fling). Interaction rewards (close/lift/success) only pay out when the
    # fingers are actually touching the object and it has not been batted sideways. This removes
    # the non-prehensile "fling" optimum (lifting with zero contact), which is a sim-physics
    # artifact that does not transfer to hardware. All gate signals (n_contacts, obj_xy_disp) are
    # observable on a real robot. See policy_bias_lab notes / SOTA contact-gated reward design.
    w_contact: float = 0.5            # dense reward for maintaining fingertip contact
    contact_target: float = 2.0       # contacts for full gate (thumb + >=1 finger ~ a cage)
    fling_xy_thresh: float = 0.08     # object lateral drift (m) above which a lift is a fling
    # Approach-to-contact gradient. Rewarding only palm_obj_dist drives a palm-down hover that
    # splays the fingertips AWAY from the object, so contact is never made. w_finger adds an
    # ungated pull on the *fingertips* (min_finger_dist) -- the gradient to the FIRST touch -- and
    # closure is gated by fingertip nearness (not palm nearness, which caused air-closure), so the
    # hand only curls when the fingertips are actually at the object. Neither is flingable
    # (flinging raises obj_xy and does not reduce fingertip distance to a held object).
    w_finger: float = 1.0             # ungated reward for bringing fingertips to the object
    close_near_scale: float = 0.05    # fingertip-distance scale (m) for the closure gate
    # The approach terms (reach/finger) use POTENTIAL-BASED shaping F = gamma*Phi(s') - Phi(s)
    # with Phi = -distance (Ng/Harada/Russell 1999). Raw distance penalties accumulate to ~-50
    # per episode and drown the sparse contact reward; the potential form telescopes (bounded
    # total ~ initial distance), so it still guides approach but lets the contact/grasp reward
    # dominate -- and it's provably policy-invariant (cannot introduce a new exploit). Use the
    # same gamma as the PPO discount for the invariance guarantee.
    pbrs_gamma: float = 0.99
    # Grasp-to-lift terms. The policy was over-closing into a fist that slides off the cube
    # (closure -> 0.9 while contact -> 0), then never raising the hand. Fixes: (1) the closure
    # reward fades as contact forms (no reward for fisting PAST the object); (2) w_hold rewards
    # SUSTAINED contact (a held grip, not a brief touch); (3) w_lift_pot is a contact-gated
    # potential on object height -- a continuous upward gradient that only pays while the grip is
    # held, so "raise the held object" is reinforced from the current behavior.
    w_hold: float = 0.5               # reward for contact maintained across consecutive steps
    w_lift_pot: float = 20.0          # contact-gated potential on object height (lift progress)
    # Lift-conditioned grasp value. Diagnosis (shaped3_liftfix run): contact+hold paid ~1.0/step
    # regardless of height, forming a flat high plateau the policy settles on -- it grasps the cube
    # on the table and ABANDONS the lifting it had at warm-start (grasp 0.10->0.95 while
    # lift_reached 0.22->0.01). Fix: scale contact/hold from a low floor on the table up to full
    # value when the cube is lifted, so a static on-table grasp is no longer the high optimum; and
    # add a sustained-aloft reward (the held-lift analog of w_hold) that pays for KEEPING the
    # object raised across consecutive steps -- directly reinforcing the measured 'sustained success'.
    contact_lift_floor: float = 0.3   # fraction of contact/hold reward paid when cube is on the table
    w_lift_hold: float = 3.0          # contact-gated reward for sustaining the object aloft


def load_model(xml_path: Path = GRIPPER_XML, cfg: "EnvConfig | None" = None) -> mujoco.MjModel:
    """Load a scene and make it MJX-compatible.

    MJX (3.9) gaps are patched here so the same env code runs the gripper and the Shadow Hand:
    - cylinder collisions are unimplemented for several partners (cyl<->mesh in particular,
      which the Shadow finger/wrist collision cylinders need against the hand meshes and the
      cube). Convert every *colliding* cylinder to a capsule of the same (radius, half-len) —
      MJX has capsule<->{mesh,box,...}. Geometrically near-identical for these thin segments.
    - explicitly listed cosmetic geoms (e.g. the gripper's `g_mount`) have their contacts
      disabled.
    - collision reduction for the contact-rich hand (`cfg.grasp_geom_substr`): see EnvConfig.
    """
    m = mujoco.MjModel.from_xml_path(str(xml_path))

    # --- collision reduction (must run before convex-hull precompute decisions) ----------
    substrs = tuple(cfg.grasp_geom_substr) if cfg else ()
    if substrs:
        plane = mujoco.mjtGeom.mjGEOM_PLANE
        try:
            object_bid = m.body("object").id
        except KeyError:
            object_bid = -1
        # contype/conaffinity groups: hand=2/1, object=1/2, plane=1/3.
        #   hand<->hand : (2&1)|(2&1)=0  -> no self-collision (kills the constraint blow-up)
        #   hand<->obj  : (2&2)|(1&1)!=0 -> collide;  hand<->plane: (2&3)|(1&1)!=0 -> collide
        #   obj <->plane: (1&3)|(1&2)!=0 -> collide
        for gid in range(m.ngeom):
            # only ever keep/drop collisions on geoms that already collide — never promote a
            # visual-only geom (whose mesh may not be convex-decomposable for MJX).
            if not (m.geom_contype[gid] or m.geom_conaffinity[gid]):
                continue
            bname = m.body(m.geom_bodyid[gid]).name
            if m.geom_bodyid[gid] == object_bid:
                m.geom_contype[gid], m.geom_conaffinity[gid] = 1, 2
            elif m.geom_type[gid] == plane:
                m.geom_contype[gid], m.geom_conaffinity[gid] = 1, 3
            elif any(s in bname for s in substrs):
                m.geom_contype[gid], m.geom_conaffinity[gid] = 2, 1
            else:
                m.geom_contype[gid], m.geom_conaffinity[gid] = 0, 0

    if cfg and cfg.primitive_fingertip_collisions:
        _replace_shadow_distal_mesh_collisions(m)

    cyl = mujoco.mjtGeom.mjGEOM_CYLINDER
    cap = mujoco.mjtGeom.mjGEOM_CAPSULE
    for gid in range(m.ngeom):
        collidable = m.geom_contype[gid] or m.geom_conaffinity[gid]
        if collidable and m.geom_type[gid] == cyl:
            m.geom_type[gid] = cap
    for name in _NONCOLLIDING_GEOMS:
        try:
            gid = m.geom(name).id
        except KeyError:
            continue
        m.geom_contype[gid] = 0
        m.geom_conaffinity[gid] = 0
    return m


def _replace_shadow_distal_mesh_collisions(m: mujoco.MjModel) -> None:
    """Approximate Shadow distal collision meshes with MJX-cheap primitive geoms.

    The Shadow XML uses high-poly mesh collisions for the distal pads:
    `f_distal_pst` has 2691 verts / 5374 faces and `th_distal_pst` has 2380 verts /
    4752 faces. Those are fine for MuJoCo CPU inspection but produce large MJX contact
    kernels and memory spikes under batched learning. The neighboring links already use
    capsules/spheres, so this keeps the same style of physical contact model for learning.
    """
    mesh_type = mujoco.mjtGeom.mjGEOM_MESH
    capsule = mujoco.mjtGeom.mjGEOM_CAPSULE
    for gid in range(m.ngeom):
        if not (m.geom_contype[gid] or m.geom_conaffinity[gid]):
            continue
        if m.geom_type[gid] != mesh_type:
            continue
        bname = m.body(m.geom_bodyid[gid]).name
        if bname in {"rh_ffdistal", "rh_mfdistal", "rh_rfdistal", "rh_lfdistal"}:
            m.geom_type[gid] = capsule
            m.geom_dataid[gid] = -1
            m.geom_pos[gid] = (0.0, 0.0, 0.0125)
            m.geom_quat[gid] = (1.0, 0.0, 0.0, 0.0)
            m.geom_size[gid] = (0.0095, 0.0125, 0.0)
        elif bname == "rh_thdistal":
            m.geom_type[gid] = capsule
            m.geom_dataid[gid] = -1
            m.geom_pos[gid] = (0.0, 0.0, 0.012)
            m.geom_quat[gid] = (1.0, 0.0, 0.0, 0.0)
            m.geom_size[gid] = (0.010, 0.012, 0.0)


class MjxEnv:
    """A jittable MJX env. vmap reset/step over a batch of keys to parallelize."""

    def __init__(self, xml_path: Path = GRIPPER_XML, config: EnvConfig | None = None):
        self.cfg = config or EnvConfig()
        self.model = load_model(xml_path, self.cfg)
        # Apply physics cost levers before transferring to the GPU.
        if self.cfg.physics_dt:
            self.model.opt.timestep = self.cfg.physics_dt
        if self.cfg.pyramidal_cone:
            self.model.opt.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
        if self.cfg.solver_iterations:
            self.model.opt.iterations = self.cfg.solver_iterations
        self.mx = mjx.put_model(self.model)

        self.frame_skip = max(1, round(self.cfg.control_dt / self.model.opt.timestep))
        self.settle_steps = max(0, round(self.cfg.settle_seconds / self.model.opt.timestep))
        self.horizon = max(1, round(self.cfg.episode_seconds / self.cfg.control_dt))

        m = self.model
        # Actuator order is the scene's; the action vector follows the same order.
        self.nu = m.nu
        self.ctrl_lo = jp.asarray(m.actuator_ctrlrange[:, 0])
        self.ctrl_hi = jp.asarray(m.actuator_ctrlrange[:, 1])
        # Nominal reset target: midpoint of each actuator's range (open hand, base centred).
        self._ctrl_init = jp.asarray(0.5 * (m.actuator_ctrlrange[:, 0] + m.actuator_ctrlrange[:, 1]))

        # Static index tables (plain python ints; safe inside jit as closed-over constants).
        self.object_bid = int(m.body("object").id)
        self.palm_bid = int(self._first_present(m, ("g_palm", "rh_palm")))
        self.grasp_sid = int(m.site("grasp_site").id)
        obj_jid = int(m.body("object").jntadr[0])
        self.obj_qadr = int(m.jnt_qposadr[obj_jid])
        self.obj_vadr = int(m.jnt_dofadr[obj_jid])
        # Hand (non-base) actuated joint qpos/qvel addresses for the observation.
        base = {"slide_x", "slide_y", "slide_z"}
        self.base_qadr = [int(m.jnt_qposadr[m.joint(n).id]) for n in ("slide_x", "slide_y", "slide_z")]
        self.base_vadr = [int(m.jnt_dofadr[m.joint(n).id]) for n in ("slide_x", "slide_y", "slide_z")]
        hand_jids = [j for j in range(m.njnt)
                     if m.joint(j).name not in base and m.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE]
        self.hand_qadr = [int(m.jnt_qposadr[j]) for j in hand_jids]
        self.hand_vadr = [int(m.jnt_dofadr[j]) for j in hand_jids]

        # Actuator partition: base positioners vs. hand (finger) actuators.
        act_names = [m.actuator(i).name for i in range(self.nu)]
        self.base_act_ids = [i for i, n in enumerate(act_names) if n in ("base_x", "base_y", "base_z")]
        self.hand_act_ids = [i for i in range(self.nu) if i not in self.base_act_ids]

        # Open / closed control targets. For an articulated hand they come from the supplied
        # keyframe poses (mapped through each actuator's transmission via MuJoCo's
        # actuator_length, which handles joint *and* tendon actuators). For the gripper, the
        # finger actuators open at the top of their ctrlrange (0.06) and close at the bottom.
        lo = np.asarray(m.actuator_ctrlrange[:, 0])
        hi = np.asarray(m.actuator_ctrlrange[:, 1])
        if self.cfg.open_pose and self.cfg.close_pose:
            self.ctrl_open = jp.asarray(self._pose_ctrl(self.cfg.open_pose, lo, hi))
            self.ctrl_close = jp.asarray(self._pose_ctrl(self.cfg.close_pose, lo, hi))
        else:
            opn = 0.5 * (lo + hi)  # base centred
            cls = opn.copy()
            for i in self.hand_act_ids:  # gripper fingers: hi = open, lo = closed
                opn[i], cls[i] = hi[i], lo[i]
            self.ctrl_open = jp.asarray(opn)
            self.ctrl_close = jp.asarray(cls)
        # Reset/home posture = open hand, base centred.
        self._ctrl_init = self.ctrl_open

        # Precompute for the generic "close the hand when near" reward term.
        self._use_pose_grasp = bool(self.cfg.open_pose and self.cfg.close_pose)
        self._hand_ids = jp.asarray(self.hand_act_ids, dtype=jp.int32)
        self._open_hand = self.ctrl_open[jp.asarray(self.hand_act_ids)]
        self._close_dir = (self.ctrl_close - self.ctrl_open)[jp.asarray(self.hand_act_ids)]

        # Fingertip bodies for the eval vector's per-finger DISTANCE term.
        ft = self.cfg.fingertip_bodies or ("g_left_finger", "g_right_finger")
        self.fingertip_bids = jp.asarray(
            [int(m.body(n).id) for n in ft if self._has_body(m, n)], dtype=jp.int32)

        # World-frame position of EVERY robot body (all bodies except `world` and `object` -- the
        # object is already exposed as obj_pos). Placed in the observation so the prior AND the
        # policy can read where each part (fingertips, palm, links, base carriage) sits in space.
        # This tuple order defines the obs layout for the body-position block (see _observe).
        self.body_pos_names = tuple(m.body(i).name for i in range(m.nbody)
                                    if m.body(i).name not in ("world", "object"))
        self.body_pos_bids = jp.asarray([int(m.body(n).id) for n in self.body_pos_names],
                                        dtype=jp.int32)

        # --- Real per-region contact force (replaces the qfrc_constraint proxy) ---------------
        # Every collidable geom is tagged as the object, a hand REGION (finger/palm), or non-object
        # environment geometry. The observed object-contact signal is object-vs-region NORMAL force
        # from the constraint solver -- true ground truth: zero in free air AND zero when a joint
        # merely loads its own limit (unlike qfrc_constraint, which sums contact and limit reaction
        # into one per-DOF number). The environment-contact signal mirrors it for hand-region vs
        # non-object/non-hand collidable geometry.
        self.contact_groups = ("thumb", "index", "middle", "ring", "little", "palm")
        self.env_contact_groups = self.contact_groups
        obj_geoms = []
        env_geoms = []
        geom_region = np.full(int(m.ngeom), -1, np.int32)
        for gid in range(int(m.ngeom)):
            if not (m.geom_contype[gid] or m.geom_conaffinity[gid]):
                continue
            bid = int(m.geom_bodyid[gid])
            if bid == self.object_bid:
                obj_geoms.append(gid)
                continue
            reg = self._contact_region(m.body(bid).name)
            if reg is not None:
                geom_region[gid] = self.contact_groups.index(reg)
            else:
                env_geoms.append(gid)
        self._obj_geoms = jp.asarray(obj_geoms, dtype=jp.int32)
        self._env_geoms = jp.asarray(env_geoms, dtype=jp.int32)
        self._geom_region = jp.asarray(geom_region)
        # Pyramidal-cone rows per contact = 2*(condim-1); condim<=6 -> a fixed 10-wide window
        # keeps the per-contact normal-force extraction vmap/jit-safe.
        self._contact_rows = 10

        # Optional compiled reward spec (eval_vec, stage) -> scalar. None = builtin reward.
        self.reward_fn = None

        self.obs_size = int(self.reset(jax.random.PRNGKey(0)).obs.shape[-1])
        self.action_size = int(self.nu)

    @staticmethod
    def _has_body(m: mujoco.MjModel, name: str) -> bool:
        try:
            m.body(name)
            return True
        except KeyError:
            return False

    @staticmethod
    def _contact_region(bname: str) -> str | None:
        """Map a hand body name to a contact region (finger/palm), or None if not a hand body.
        Mirrors the actuator semantic grouping so per-actuator `c_self` lines up with a region."""
        b = bname.lower()
        table = (
            ("thumb",  ("thdistal", "thmiddle", "thproximal", "thhub", "thbase", "thmetacarpal")),
            ("index",  ("ffdistal", "ffmiddle", "ffproximal", "ffknuckle")),
            ("middle", ("mfdistal", "mfmiddle", "mfproximal", "mfknuckle")),
            ("ring",   ("rfdistal", "rfmiddle", "rfproximal", "rfknuckle")),
            ("little", ("lfdistal", "lfmiddle", "lfproximal", "lfknuckle", "lfmetacarpal")),
            ("palm",   ("palm", "wrist", "forearm")),
        )
        for reg, keys in table:
            if any(k in b for k in keys):
                return reg
        return None

    def _contact_forces(self, dx: mjx.Data) -> jp.ndarray:
        """Per-region normal contact force with the object (N), order = self.contact_groups.

        Zero in free air and zero when a joint loads its own limit -- the force is read only from
        contacts whose two geoms are (a hand region, the object). Hand-hand self-collision is
        disabled in the model, so no hand-hand pair can leak in."""
        c = dx._impl.contact
        efc = dx._impl.efc_force
        rows = self._contact_rows

        def normal(efc_address, dim):  # pyramidal: normal = sum of the 2*(dim-1) pyramid rows
            win = jax.lax.dynamic_slice(efc, (jp.maximum(efc_address, 0),), (rows,))
            nrows = jp.where(dim == 1, 1, 2 * (dim - 1))
            keep = jp.arange(rows) < nrows
            return jp.where(efc_address >= 0, jp.sum(jp.where(keep, win, 0.0)), 0.0)

        fn = jax.vmap(normal)(c.efc_address, c.dim)                 # [ncon] normal force
        obj1 = jp.isin(c.geom1, self._obj_geoms)
        obj2 = jp.isin(c.geom2, self._obj_geoms)
        reg1 = self._geom_region[c.geom1]
        reg2 = self._geom_region[c.geom2]
        region = jp.where(obj1, reg2, reg1)                        # the hand side of the pair
        is_obj_hand = (obj1 & (reg2 >= 0)) | (obj2 & (reg1 >= 0))
        fn = jp.where(is_obj_hand, fn, 0.0)
        region = jp.where(is_obj_hand, region, 0)
        return jp.zeros(len(self.contact_groups)).at[region].add(fn)

    def _env_contact_forces(self, dx: mjx.Data) -> jp.ndarray:
        """Per-region normal contact force with non-object environment geometry (N).

        This is separate from object contact: it rises only for hand-region contacts whose other
        geom is collidable but neither the task object nor another hand region.
        """
        c = dx._impl.contact
        efc = dx._impl.efc_force
        rows = self._contact_rows

        def normal(efc_address, dim):
            win = jax.lax.dynamic_slice(efc, (jp.maximum(efc_address, 0),), (rows,))
            nrows = jp.where(dim == 1, 1, 2 * (dim - 1))
            keep = jp.arange(rows) < nrows
            return jp.where(efc_address >= 0, jp.sum(jp.where(keep, win, 0.0)), 0.0)

        fn = jax.vmap(normal)(c.efc_address, c.dim)
        env1 = jp.isin(c.geom1, self._env_geoms)
        env2 = jp.isin(c.geom2, self._env_geoms)
        reg1 = self._geom_region[c.geom1]
        reg2 = self._geom_region[c.geom2]
        region = jp.where(env1, reg2, reg1)
        is_env_hand = (env1 & (reg2 >= 0)) | (env2 & (reg1 >= 0))
        fn = jp.where(is_env_hand, fn, 0.0)
        region = jp.where(is_env_hand, region, 0)
        return jp.zeros(len(self.env_contact_groups)).at[region].add(fn)

    def set_reward_fn(self, reward_fn) -> None:
        """Attach an `(eval_vec, stage) -> scalar` reward (from `reward_spec.compile_*`)."""
        self.reward_fn = reward_fn

    def _pose_ctrl(self, hand_qpos, lo, hi) -> np.ndarray:
        """Actuator targets that hold the given hand-joint pose (in env hand-joint order)."""
        d = mujoco.MjData(self.model)
        qpos = np.asarray(self.model.qpos0).copy()
        for adr, val in zip(self.hand_qadr, hand_qpos):
            qpos[adr] = val
        d.qpos[:] = qpos
        mujoco.mj_forward(self.model, d)
        return np.clip(np.asarray(d.actuator_length).copy(), lo, hi)

    @staticmethod
    def _first_present(m: mujoco.MjModel, names: tuple[str, ...]) -> int:
        for n in names:
            try:
                return m.body(n).id
            except KeyError:
                continue
        raise KeyError(f"none of {names} present in model")

    # --- core API -------------------------------------------------------------

    def reset(self, key: jp.ndarray, stage: int = 0) -> EnvState:
        key, ok = jax.random.split(key)
        dx = mjx.make_data(self.mx)

        # Randomize object xy around the origin; keep it resting on the table.
        dxy = jax.random.uniform(ok, (2,), minval=-self.cfg.obj_xy_range, maxval=self.cfg.obj_xy_range)
        qpos = dx.qpos
        qpos = qpos.at[self.obj_qadr].add(dxy[0])
        qpos = qpos.at[self.obj_qadr + 1].add(dxy[1])
        dx = dx.replace(qpos=qpos, ctrl=self._ctrl_init)
        dx = mjx.forward(self.mx, dx)

        # Settle under gravity with the open-hand nominal target held.
        def settle(dx, _):
            return mjx.step(self.mx, dx), None
        dx, _ = jax.lax.scan(settle, dx, None, length=self.settle_steps)

        obj_pos = dx.xpos[self.object_bid]
        start = {"obj_start_z": obj_pos[2], "obj_start_xy": obj_pos[:2]}
        eval0 = self._eval(dx, start)
        metrics = {
            **start,
            "lift": jp.float32(0.0),
            "success": jp.float32(0.0),
            "reach": eval0[EM.FIELD_INDEX["palm_obj_dist"]],
            "eval": eval0,
        }
        obs = self._observe(dx)
        return EnvState(
            data=dx, obs=obs, reward=jp.float32(0.0), done=jp.float32(0.0),
            step=jp.int32(0), key=key, stage=jp.int32(stage), metrics=metrics,
        )

    def step(self, state: EnvState, action: jp.ndarray) -> EnvState:
        dx = state.data
        # Action in [-1, 1] -> incremental change in each ctrl target, clamped to range.
        action = jp.clip(action, -1.0, 1.0)
        target = jp.clip(dx.ctrl + action * self.cfg.action_scale, self.ctrl_lo, self.ctrl_hi)

        def phys(dx, _):
            return mjx.step(self.mx, dx.replace(ctrl=target)), None
        dx, _ = jax.lax.scan(phys, dx, None, length=self.frame_skip)

        obs = self._observe(dx)
        reward, metrics = self._reward(dx, state.metrics, state.stage)
        step = state.step + 1
        done = jp.where(step >= self.horizon, 1.0, 0.0)
        return state.replace(
            data=dx, obs=obs, reward=reward, done=done, step=step, metrics=metrics,
        )

    # --- observation & reward -------------------------------------------------

    def _observe(self, dx: mjx.Data) -> jp.ndarray:
        base_q = jp.asarray([dx.qpos[a] for a in self.base_qadr])
        base_v = jp.asarray([dx.qvel[a] for a in self.base_vadr])
        hand_q = jp.asarray([dx.qpos[a] for a in self.hand_qadr]) if self.hand_qadr else jp.zeros(0)
        hand_v = jp.asarray([dx.qvel[a] for a in self.hand_vadr]) if self.hand_vadr else jp.zeros(0)
        # Per-region CONTACT force: the true normal force each hand region (finger/palm) exchanges
        # with the object this step -- exactly zero in free air AND zero when a joint merely loads
        # its own limit (it reads the constraint solver only on object<->hand contacts). Real-robot
        # analog: fingertip force / tactile sensing. The environment-contact block is the same
        # measurement for non-object environment geometry. Both are placed between body positions and
        # obj_pos so the trailing [obj_*, palm, obj_rel, ctrl] layout stays valid.
        env_contact = self._env_contact_forces(dx)
        contact = self._contact_forces(dx)
        # World-frame xyz of every robot body, flattened [b0_x,b0_y,b0_z, b1_x,...]. Placed BEFORE
        # the contact block so the trailing [obj_*, palm, obj_rel, ctrl] tail stays the last
        # (12 + nu) elements (n_pre = obs_size - 12 - action_dim) and contact stays immediately
        # before obj_pos -- every existing name->index mapping is preserved.
        body_pos = dx.xpos[self.body_pos_bids].reshape(-1)
        obj_pos = dx.xpos[self.object_bid]
        obj_vel = dx.cvel[self.object_bid, 3:6]  # linear part of spatial velocity
        palm_pos = dx.xpos[self.palm_bid]
        grasp_pos = dx.site_xpos[self.grasp_sid]
        obj_rel = obj_pos - grasp_pos
        return jp.concatenate([
            base_q, base_v, hand_q, hand_v, body_pos, env_contact, contact,
            obj_pos, obj_vel, palm_pos, obj_rel, dx.ctrl,
        ])

    def _closure(self, dx: mjx.Data) -> jp.ndarray:
        """Mean finger-closure fraction in [0,1] (0 = open pose, 1 = closed pose)."""
        ctrl_hand = dx.ctrl[self._hand_ids]
        return jp.mean(jp.clip((ctrl_hand - self._open_hand) / (self._close_dir + 1e-6), 0.0, 1.0))

    def _eval(self, dx: mjx.Data, start: dict[str, jp.ndarray]) -> jp.ndarray:
        """Ground-truth task-progress vector (layout = eval_metrics.EVAL_FIELDS)."""
        obj_pos = dx.xpos[self.object_bid]
        grasp_pos = dx.site_xpos[self.grasp_sid]
        palm_obj_dist = jp.linalg.norm(obj_pos - grasp_pos)
        tip_pos = dx.xpos[self.fingertip_bids]                 # [n_fingers, 3]
        tip_d = jp.linalg.norm(tip_pos - obj_pos[None, :], axis=-1)
        min_finger_dist = tip_d.min()
        # Real contact: count hand regions bearing normal force above the floor (was fingertip
        # geometric proximity, which paid out for a hand hovering near but never touching).
        cforce = self._contact_forces(dx)
        n_contacts = jp.sum((cforce > self.cfg.contact_force_floor).astype(jp.float32))
        closure = self._closure(dx)
        lift = obj_pos[2] - start["obj_start_z"]
        obj_xy_disp = jp.linalg.norm(obj_pos[:2] - start["obj_start_xy"])
        return jp.asarray([palm_obj_dist, min_finger_dist, n_contacts, closure, lift, obj_xy_disp])

    def _builtin_reward(self, dx: mjx.Data, e: jp.ndarray, prev_e: jp.ndarray) -> jp.ndarray:
        reach_d = e[EM.FIELD_INDEX["palm_obj_dist"]]
        lift = e[EM.FIELD_INDEX["lift"]]
        n_contacts = e[EM.FIELD_INDEX["n_contacts"]]
        obj_xy = e[EM.FIELD_INDEX["obj_xy_disp"]]
        # Contact gate (smooth 0->1 over fingertip contacts) and an anti-fling factor. Interaction
        # rewards are multiplied by these so the only way to earn lift/grasp reward is an actual
        # in-hand grasp. The approach terms are potential-based (below), giving a natural curriculum:
        # approach -> (contact unlocks) -> close/lift/hold, without a penalty that drowns the grasp.
        min_finger_dist = e[EM.FIELD_INDEX["min_finger_dist"]]
        prev_reach = prev_e[EM.FIELD_INDEX["palm_obj_dist"]]
        prev_finger = prev_e[EM.FIELD_INDEX["min_finger_dist"]]
        prev_contacts = prev_e[EM.FIELD_INDEX["n_contacts"]]
        prev_lift = prev_e[EM.FIELD_INDEX["lift"]]
        contact_gate = jp.clip(n_contacts / self.cfg.contact_target, 0.0, 1.0)
        prev_contact_gate = jp.clip(prev_contacts / self.cfg.contact_target, 0.0, 1.0)
        in_contact = (n_contacts >= 1.0).astype(jp.float32)
        not_flung = jp.clip(
            (self.cfg.fling_xy_thresh - obj_xy) / (0.5 * self.cfg.fling_xy_thresh + 1e-6), 0.0, 1.0)
        # Smooth fingertip-nearness gate (~1 when a fingertip is within ~close_near_scale of the
        # object, decaying with distance) -- gates closure so the hand curls only as the fingertips
        # arrive at the object, producing contact rather than empty-hand "air-closure".
        finger_near = jp.exp(-(min_finger_dist / self.cfg.close_near_scale) ** 2)
        # Potential-based approach shaping (Phi = -distance): F = gamma*Phi(s') - Phi(s) =
        # prev_dist - gamma*cur_dist. Rewards progress toward the object, telescopes over the
        # episode (bounded total ~ initial distance), does NOT penalize being far (so the policy is
        # not pushed to retreat), and is policy-invariant -- letting the contact/grasp reward dominate.
        g = self.cfg.pbrs_gamma
        r_reach = self.cfg.w_reach * (prev_reach - g * reach_d)
        r_finger = self.cfg.w_finger * (prev_finger - g * min_finger_dist)
        # Lift fraction (0 on the table, 1 at/above success height). Scales the contact/hold rewards
        # so a grip on the table pays only `contact_lift_floor` of full value and a raised grip pays
        # full: this removes the flat high plateau (grasp-on-table) the policy was settling on and
        # makes "raise the held cube" strictly more valuable than holding it grounded.
        lift_frac = jp.clip(lift / self.cfg.success_height, 0.0, 1.0)
        prev_lift_frac = jp.clip(prev_lift / self.cfg.success_height, 0.0, 1.0)
        lift_scale = self.cfg.contact_lift_floor + (1.0 - self.cfg.contact_lift_floor) * lift_frac
        r_contact = self.cfg.w_contact * contact_gate * lift_scale
        # Sustained-contact (hold) reward: pays only when contact is held across consecutive steps,
        # so the policy is rewarded for KEEPING a grip rather than a one-frame touch.
        r_hold = self.cfg.w_hold * contact_gate * prev_contact_gate * lift_scale
        # Closure reward FADES as contact forms (the (1 - contact_gate) factor): the hand is paid to
        # curl toward contact, but NOT to keep closing once it grips -- removing the incentive to
        # fist past the cube and lose it. Once gripping, r_hold (above) takes over.
        if self._use_pose_grasp:
            r_close = self.cfg.w_close * e[EM.FIELD_INDEX["closure"]] * finger_near * (1.0 - contact_gate)
        else:
            sep = jp.sum(jp.asarray([dx.ctrl[i] for i in range(self.nu)
                                     if self.model.actuator(i).name in ("g_left", "g_right")]))
            r_close = -self.cfg.w_close * sep * finger_near * (1.0 - contact_gate)
        # Contact-gated height potential (Phi = object height): F = gamma*Phi(s') - Phi(s), gated by
        # contact -- a continuous upward gradient that only pays while a grip is held, so raising the
        # held object is reinforced (and dropping it penalized) instead of only at the 5cm threshold.
        r_lift_pot = self.cfg.w_lift_pot * contact_gate * not_flung * (g * lift - prev_lift)
        r_lift = self.cfg.w_lift * jp.clip(lift, 0.0, self.cfg.lift_target) * contact_gate * not_flung
        # Sustained-aloft reward: the held-lift analog of r_hold. Pays only when the cube is BOTH
        # currently and previously raised (lift_frac * prev_lift_frac) while gripped and not flung --
        # a dense incentive to KEEP the object up across consecutive steps, not just bounce it.
        r_lift_hold = self.cfg.w_lift_hold * contact_gate * not_flung * lift_frac * prev_lift_frac
        r_success = self.cfg.w_success * jp.where(lift > self.cfg.success_height, 1.0, 0.0) * in_contact * not_flung
        return (r_reach + r_finger + r_contact + r_hold + r_close
                + r_lift_pot + r_lift + r_lift_hold + r_success)

    def _reward(self, dx: mjx.Data, prev: dict[str, jp.ndarray],
                stage: jp.ndarray) -> tuple[jp.ndarray, dict]:
        start = {"obj_start_z": prev["obj_start_z"], "obj_start_xy": prev["obj_start_xy"]}
        e = self._eval(dx, start)
        prev_e = prev["eval"]  # previous-step eval, for potential-based approach shaping
        lift = e[EM.FIELD_INDEX["lift"]]
        if self.reward_fn is not None:
            reward = self.reward_fn(e, stage)            # LLM/curriculum reward spec
        else:
            reward = self._builtin_reward(dx, e, prev_e)
        # Contact-gated success: a real grasp-lift (lifted, touching, not flung) -- mirrors the
        # reward gate so instant/sustained success and best-checkpoint selection cannot be gamed
        # by flinging the object up with no contact.
        n_contacts = e[EM.FIELD_INDEX["n_contacts"]]
        obj_xy = e[EM.FIELD_INDEX["obj_xy_disp"]]
        success = jp.where(
            (lift > self.cfg.success_height) & (n_contacts >= 1.0) & (obj_xy < self.cfg.fling_xy_thresh),
            1.0, 0.0)
        metrics = dict(prev)
        metrics.update(lift=lift, success=success, reach=e[EM.FIELD_INDEX["palm_obj_dist"]], eval=e)
        return reward, metrics


def batched_reset(env: MjxEnv, keys: jp.ndarray) -> EnvState:
    return jax.jit(jax.vmap(env.reset))(keys)


def batched_step(env: MjxEnv):
    """Return a jitted vmapped step closure (compile once, reuse)."""
    return jax.jit(jax.vmap(env.step))


def _selftest(n: int = 64, steps: int = 50) -> None:
    """Random-policy rollout to confirm the env runs and rewards move."""
    import time

    env = MjxEnv()
    print(f"obs_size={env.obs_size} action_size={env.action_size} "
          f"frame_skip={env.frame_skip} horizon={env.horizon}")
    keys = jax.random.split(jax.random.PRNGKey(0), n)
    state = batched_reset(env, keys)
    step_fn = batched_step(env)
    key = jax.random.PRNGKey(1)

    t0 = time.time()
    rewards = []
    for t in range(steps):
        key, ak = jax.random.split(key)
        action = jax.random.uniform(ak, (n, env.action_size), minval=-1.0, maxval=1.0)
        state = step_fn(state, action)
        rewards.append(float(state.reward.mean()))
    jax.block_until_ready(state.reward)
    dt = time.time() - t0
    print(f"ran {n} envs x {steps} ctrl-steps in {dt:.2f}s "
          f"({n*steps*env.frame_skip/dt:,.0f} physics-steps/s)")
    print(f"reward mean first/last: {rewards[0]:.3f} -> {rewards[-1]:.3f}")
    print(f"final lift mean={float(state.metrics['lift'].mean()):.4f} "
          f"success_frac={float(state.metrics['success'].mean()):.3f}")


if __name__ == "__main__":
    _selftest()

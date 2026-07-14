"""Free-form symbolic action priors + robot-spec extraction + DOF-completeness.

For the preliminary experiment comparing a constrained robot-derived DSL against a free-form
symbolic representation (see reports/dsl_vs_freeform.md). A free-form candidate gives, per channel
(a set of named actuators), a symbolic EXPRESSION for the mean-shift over observable signals; a
restricted AST evaluator compiles it into JAX prior functions. Flat/free-form laws remain stateless;
staged laws may also expose a rollout-carried stage cursor so normal stage progress is monotone.

Two shared utilities live here too:
  - ``robot_spec(env)``: enumerate ALL actuators/DOF (name, joint, range, driven body, world-sign)
    into an injectable spec -- the substrate that makes prompting robot-agnostic.
  - ``check_dof_complete`` / ``repair_*``: enforce that a candidate addresses EVERY DOF (the
    guardrail that prevents silent omission, e.g. the wrist).

Safety: free-form expressions are parsed with a strict AST whitelist (numbers, signal names,
+ - * /, unary -, clip/sigmoid/min/max/abs/exp, single comparisons). No attribute access, calls to
arbitrary names, comprehensions, or statements -- so an LLM-authored expression cannot execute
anything but arithmetic over the provided signals.
"""

from __future__ import annotations

import ast
import functools
from typing import Any, Callable

import jax
import jax.numpy as jp
import numpy as np

from policy_bias_lab.composed_priors import make_signal_fn, _calibration, _gates
from policy_bias_lab.symbolic_control import group_masks
from policy_bias_lab.schema import ACTION_GROUPS

# ----------------------------------------------------------------------------------------------
# Robot spec (all DOF enumerated)
# ----------------------------------------------------------------------------------------------

def _target_speed_doc(env: Any, action_scale: float) -> str:
    """Units bridge, mechanically derived: what a channel VALUE means as a physical target speed.
    Authored channels are dimensionless; without this line the author has no quantitative path
    from a gain/clip choice to the m/s it commands (observed: prose said 'small increments' while
    the authored clip commanded ~0.7 m/s)."""
    dt = float(getattr(env.cfg, "control_dt", 0.0) or 0.0)
    if not dt:
        return "control_dt unknown; target speed = action * action_scale per control step"
    rate = float(action_scale) / dt
    return (f"a SUSTAINED action value v moves that actuator's commanded target at "
            f"v * action_scale / control_dt = v * {rate:.2f} ctrl-units per second "
            f"(ctrl units: meters for slide joints, radians for hinge joints) -- e.g. |v| = 0.3 "
            f"held on a slide actuator commands ~{0.3 * rate:.2f} m/s of target motion. Convert "
            f"every channel magnitude and clip bound into these units to know the speed it asks "
            f"for.")


def _actuator_motions(m: Any) -> list[dict[str, Any]]:
    """Per-actuator kinematic motion descriptor, computed at the home pose (qpos0) via one forward
    pass: the WORLD-FRAME axis each actuator's value moves the part about (hinge) or along (slide)
    -- so an author knows which physical DIRECTION a positive q_/ctrl_ value indicates -- plus the
    joint's travel range = the SCALE of the bend. Task-agnostic (pure kinematics)."""
    import mujoco
    data = mujoco.MjData(m)
    data.qpos[:] = m.qpos0
    mujoco.mj_forward(m, data)
    trntype = np.asarray(m.actuator_trntype).reshape(-1)
    kinds = {0: "free", 1: "ball", 2: "slide", 3: "hinge"}
    out: list[dict[str, Any]] = []
    for i in range(m.nu):
        if int(trntype[i]) != int(mujoco.mjtTrn.mjTRN_JOINT):
            out.append({"kind": "coupled",
                        "note": "coupled/tendon transmission -- no single joint axis"})
            continue
        jid = int(m.actuator_trnid[i, 0])
        kind = kinds.get(int(m.jnt_type[jid]), "other")
        axis = [round(float(x), 3) for x in np.asarray(data.xaxis[jid])]
        rng = [round(float(m.jnt_range[jid, 0]), 3), round(float(m.jnt_range[jid, 1]), 3)]
        span = round(rng[1] - rng[0], 3)
        if kind == "hinge":
            deg = [round(rng[0] * 180.0 / np.pi, 1), round(rng[1] * 180.0 / np.pi, 1)]
            span_deg = round(span * 180.0 / np.pi, 1)
            note = (f"+ctrl/+q rotates this part about world axis {axis} by the right-hand rule; "
                    f"travel {rng} rad = [{deg[0]} deg, {deg[1]} deg] ({span_deg} deg full bend)")
        elif kind == "slide":
            note = f"+ctrl/+q translates this part along world axis {axis}; travel {rng} m"
        else:
            note = f"{kind} joint on world axis {axis}"
        out.append({"kind": kind, "world_axis": axis, "range": rng, "note": note})
    return out


def _spawn_attitude(env: Any) -> dict[str, Any] | None:
    """Home-pose ORIENTATION + reorientation capability of each kinematic segment, from FK.

    The per-actuator `motion` field already gives each joint's rotation/translation axis and range.
    What it does NOT convey is the resting ATTITUDE of the hand -- which way each segment points in
    the world at spawn, and which segments can change attitude at all. Without this an author cannot
    tell that (e.g.) the whole hand hangs straight down and the base cannot rotate it. This computes,
    at qpos0: each segment's world pointing axis (from parent->child body origins, convention-free),
    the palm's face normal (perpendicular to the hand axis and the knuckle-spread axis), and whether
    the segment can reorient (hinge joints) or is translation-only. Pure kinematics -- no task info.
    """
    import mujoco
    m = env.model
    try:
        data = mujoco.MjData(m)
    except Exception:
        return None
    data.qpos[:] = m.qpos0
    mujoco.mj_forward(m, data)
    body_names = {m.body(i).name for i in range(m.nbody)}

    def opos(name: str):
        return np.asarray(data.body(name).xpos) if name in body_names else None

    def unit(v):
        if v is None:
            return None
        n = float(np.linalg.norm(v))
        v = v / n if n > 1e-9 else v
        return [round(float(x), 3) for x in v]

    def points(parent: str, child: str):
        a, b = opos(parent), opos(child)
        return unit(b - a) if a is not None and b is not None else None

    # Base carriage: attitude is fixed iff every base actuator is a SLIDE (pure translation).
    base_ids = [int(i) for i in getattr(env, "base_act_ids", ())]
    trntype = np.asarray(m.actuator_trntype).reshape(-1)
    base_all_slide = bool(base_ids) and all(
        int(trntype[i]) == int(mujoco.mjtTrn.mjTRN_JOINT)
        and int(m.jnt_type[int(m.actuator_trnid[i, 0])]) == int(mujoco.mjtJoint.mjJNT_SLIDE)
        for i in base_ids)

    # Convention-free segment pointing axes along the finger/arm chain (skip any missing body).
    forearm_axis = points("rh_forearm", "rh_wrist") or points("rh_wrist", "rh_palm")
    wrist_axis = points("rh_wrist", "rh_palm")
    hand_axis = points("rh_palm", "rh_mfdistal") or wrist_axis
    fingers = {"index": ("rh_ffknuckle", "rh_ffdistal"), "middle": ("rh_mfknuckle", "rh_mfdistal"),
               "ring": ("rh_rfknuckle", "rh_rfdistal"), "little": ("rh_lfknuckle", "rh_lfdistal"),
               "thumb": ("rh_thbase", "rh_thdistal")}
    finger_axes = {k: points(*v) for k, v in fingers.items() if points(*v) is not None}

    # Palm face normal = perpendicular to the hand (long) axis and the knuckle-spread axis.
    palm_normal = None
    ff, lf = opos("rh_ffknuckle"), opos("rh_lfknuckle")
    ha = opos("rh_mfdistal")
    hp = opos("rh_palm")
    if ff is not None and lf is not None and ha is not None and hp is not None:
        palm_normal = unit(np.cross(ha - hp, lf - ff))

    if hand_axis is None:
        return None

    # Wrist joint DIRECTION semantics: perturb each wrist-group hinge at the home pose and measure
    # whether it changes the palm FACING (flexion/extension) or only rolls the hand (deviation), and
    # with which sign -- so the author gets flexion/extension named and directed correctly. Pure FK.
    wrist_dirs: list[dict[str, Any]] = []
    if palm_normal is not None and opos("rh_mfdistal") is not None and opos("rh_palm") is not None:
        pn0 = np.asarray(palm_normal, dtype=float)
        ha0 = opos("rh_mfdistal") - opos("rh_palm"); ha0 = ha0 / (float(np.linalg.norm(ha0)) or 1.0)
        try:
            anames = [m.actuator(i).name for i in range(m.nu)]
        except Exception:
            anames = []
        for i, an in enumerate(anames):
            if _semantic_group(an) != "wrist" or int(trntype[i]) != int(mujoco.mjtTrn.mjTRN_JOINT):
                continue
            jid = int(m.actuator_trnid[i, 0])
            if int(m.jnt_type[jid]) != int(mujoco.mjtJoint.mjJNT_HINGE):
                continue
            jadr = int(m.jnt_qposadr[jid])
            q = np.asarray(m.qpos0).copy(); q[jadr] = float(np.radians(10.0))
            data.qpos[:] = q; mujoco.mj_forward(m, data)
            n = np.cross(opos("rh_mfdistal") - opos("rh_palm"), opos("rh_lfknuckle") - opos("rh_ffknuckle"))
            n = n / (float(np.linalg.norm(n)) or 1.0)
            t = opos("rh_mfdistal") - opos("rh_palm"); t = t / (float(np.linalg.norm(t)) or 1.0)
            dz = float(n[2] - pn0[2])            # +q effect on palm-facing world-z
            flex_dot = float(np.dot(t - ha0, pn0))  # >0 => +q swings fingertips toward palmar side = flexion
            if abs(dz) >= 0.05:                  # this joint changes the palm facing => flexion/extension
                flex_sign = "+q" if flex_dot > 0 else "-q"
                ext_sign = "-q" if flex_dot > 0 else "+q"
                dz_ext = dz if ext_sign == "+q" else -dz
                wrist_dirs.append({"actuator": an, "role": "flexion_extension",
                                   "flexion_sign": flex_sign, "extension_sign": ext_sign,
                                   "extension_tilts_palm_toward": "table (-world_z)" if dz_ext < 0
                                   else "up and back (+world_z)"})
            else:                                # palm facing ~unchanged => deviation / roll
                wrist_dirs.append({"actuator": an, "role": "deviation",
                                   "note": "rocks the hand side-to-side; does not change which way the "
                                           "palm faces"})
        data.qpos[:] = m.qpos0; mujoco.mj_forward(m, data)  # restore home pose

    return {
        "base_translation_only": base_all_slide,
        "forearm_points": forearm_axis,
        "wrist_points": wrist_axis,
        "hand_points": hand_axis,
        "palm_face_normal": palm_normal,
        "finger_points": finger_axes,
        "wrist_directions": wrist_dirs,
        "note": ("world axes are [x, y, z]; +z is up, so [0, 0, -1] points straight down at the "
                 "table. `*_points` is the world direction a segment extends toward at the home "
                 "pose; `palm_face_normal` is the direction the flat of the palm faces."),
    }


def robot_spec(env: Any) -> dict[str, Any]:
    """Injectable, robot-agnostic description of every controllable DOF.

    Includes the world-sign convention for the base carriage so a free-form author can get
    directions right without the DSL's hidden calibration.
    """
    m = env.model
    names = [m.actuator(i).name for i in range(env.nu)]
    cr = np.asarray(m.actuator_ctrlrange)
    cal = _calibration(env)
    sgn = cal["sgn"]  # (x, y, z) sign of d(world)/d(base ctrl)
    base_world = {}
    for axis, idx, s in (("x", cal["bx"], sgn[0]), ("y", cal["by"], sgn[1]), ("z", cal["bz"], sgn[2])):
        if idx >= 0:
            base_world[names[idx]] = {
                "world_axis": axis,
                "ctrl_increases_world": ("+" if s > 0 else "-") + axis,
                "note": f"+{names[idx]} ctrl moves the grasp toward {'+' if s>0 else '-'}world_{axis}",
            }
    # Forward-kinematics at the home pose so each actuator can carry the WORLD-FRAME axis its value
    # moves about/along (the DIRECTION a q_/ctrl_ value indicates) and its travel range (the SCALE
    # of the bend). Pure kinematics -- no task content.
    motions = _actuator_motions(m)
    actuators = []
    gain = np.asarray(m.actuator_gainprm[:, 0])
    for i, n in enumerate(names):
        jid = int(m.actuator_trnid[i, 0])
        actuators.append({
            "name": n,
            "index": i,
            "joint": m.joint(jid).name if jid >= 0 else None,
            "ctrlrange": [float(cr[i, 0]), float(cr[i, 1])],
            "motion": motions[i],
            "servo_gain": float(gain[i]),
            "group": _semantic_group(n),
        })
    # Initial (home) target the servos hold at t=0 -- injected so the author reasons from the
    # ACTUAL spawn configuration instead of assuming the hand starts interaction-ready. Under the
    # incremental-position control law q_<name> settles to ~ this value at spawn.
    ctrl_init = getattr(env, "ctrl_open", None)
    if ctrl_init is None:
        ctrl_init = getattr(env, "_ctrl_init", None)
    initial_pose = None
    if ctrl_init is not None:
        ci = np.asarray(ctrl_init)
        if ci.shape[0] == len(names):
            initial_pose = {names[i]: round(float(ci[i]), 4) for i in range(len(names))}
    return {
        "n_actuators": env.nu,
        "actuators": actuators,
        "initial_pose": initial_pose,
        "spawn_attitude": _spawn_attitude(env),
        "rollout_seconds": round(float(getattr(env.cfg, "episode_seconds", 0.0)) or 20.0, 1),
        "control_law": {
            "type": "incremental_position",
            "rule": "target = current_ctrl + action * action_scale",
            "action_scale": cal["action_scale"],
            "action_range": [-1.0, 1.0],
            "control_dt": float(getattr(env.cfg, "control_dt", 0.0)),
            "target_speed": _target_speed_doc(env, cal["action_scale"]),
            "applied_force": ("each actuator is a position servo: the steady-state force/torque "
                              "it applies to its joint is ~ servo_gain * (ctrl_<name> - q_<name>)"
                              " -- so you control how hard a region presses by how far you drive "
                              "ctrl past the measured q. This is the COMMANDED push; the resulting "
                              "NORMAL force the object actually feels is observed directly as "
                              "c_<region> (zero until real contact), and the two agree only once "
                              "the region is truly loaded against the object"),
        },
        "base_world_sign": base_world,
        "semantic_groups": {g: [a["name"] for a in actuators if a["group"] == g]
                            for g in sorted({a["group"] for a in actuators})},
        # Raw observable enumeration (env detail, mechanically derived): the ONLY signal
        # vocabulary the prompts advertise. Derived signals are LLM-authored per candidate.
        "observables_doc": raw_signal_fn(env)[2],
    }


def _semantic_group(name: str) -> str:
    if name in ("base_x", "base_y"):
        return "base_xy"
    if name == "base_z":
        return "base_z"
    for tag, pre in (("wrist", "rh_A_WR"), ("thumb", "rh_A_TH"), ("index", "rh_A_FF"),
                     ("middle", "rh_A_MF"), ("ring", "rh_A_RF"), ("little", "rh_A_LF")):
        if name.startswith(pre):
            return tag
    return "other"


def _contact_region_of_actuator(name: str) -> str | None:
    """The contact region (finger/palm) an actuator belongs to -- matches env.contact_groups so
    `c_self` resolves to the right c_<region> observable. None = no contact region (base carriage)."""
    g = _semantic_group(name)
    if g in ("thumb", "index", "middle", "ring", "little"):
        return g
    if g == "wrist":
        return "palm"
    return None


def all_actuator_names(env: Any) -> list[str]:
    return [env.model.actuator(i).name for i in range(env.nu)]


# ----------------------------------------------------------------------------------------------
# DOF-completeness
# ----------------------------------------------------------------------------------------------

def _resolve_actuators(env: Any, tokens: list[str]) -> list[int]:
    """Resolve a list of actuator names and/or group names to actuator indices."""
    names = all_actuator_names(env)
    gm = group_masks(env)
    out: list[int] = []
    for tok in tokens:
        tok = str(tok)
        if tok in names:
            out.append(names.index(tok))
        elif tok in ACTION_GROUPS:
            gi = ACTION_GROUPS.index(tok)
            out.extend(int(i) for i in np.nonzero(gm[gi])[0])
        else:
            # also accept semantic group tags (wrist/thumb/...) resolved by prefix
            grp = [i for i, n in enumerate(names) if _semantic_group(n) == tok]
            out.extend(grp)
    return sorted(set(out))


def addressed_actuators(env: Any, candidate: dict[str, Any]) -> set[str]:
    """Set of actuator names a candidate touches (free-form channels)."""
    names = all_actuator_names(env)
    idx: set[int] = set()
    if candidate.get("mode") == "freeform_staged":
        for st in candidate.get("stages", []):
            for ch in st.get("channels", []):
                idx.update(_resolve_actuators(env, ch.get("actuators", [])))
    else:  # freeform: flat list of channels
        for ch in candidate.get("channels", []):
            idx.update(_resolve_actuators(env, ch.get("actuators", [])))
    return {names[i] for i in idx}


def check_dof_complete(env: Any, candidate: dict[str, Any]) -> tuple[bool, list[str]]:
    """Return (ok, missing_actuator_names). A DOF is 'addressed' if any rule/channel touches it."""
    missing = [n for n in all_actuator_names(env) if n not in addressed_actuators(env, candidate)]
    return (len(missing) == 0, missing)


def repair_freeform(candidate: dict[str, Any], missing: list[str]) -> dict[str, Any]:
    """Auto-fallback: hold any unaddressed DOF at neutral (expr '0'). The LLM repair prompt is
    preferred; this guarantees DOF-completeness even if the model still omits some."""
    if not missing:
        return candidate
    out = dict(candidate)
    out["channels"] = list(candidate.get("channels", [])) + [{"actuators": list(missing), "expr": "0",
                                                              "note": "auto-hold (DOF-completeness)"}]
    return out


# ----------------------------------------------------------------------------------------------
# Motion basis: the DSL vocabulary must ACTIVELY drive every DOF (not just hold it). The set of
# motion primitives should form an approximate complete basis over the robot's legal configuration
# space -- every actuator drivable in BOTH directions, so any reachable orientation is a
# combination of primitives. Holding/stabilize is NOT sufficient.
# ----------------------------------------------------------------------------------------------

def derive_motion_basis(env: Any, granularity: str = "group") -> list[dict[str, Any]]:
    """Robot-derived signed articulation primitives spanning every DOF.

    granularity='group' (loose/approximate basis): one signed pair per semantic group -- compact,
    but couples actuators within a group. granularity='actuator' (full basis): one signed pair per
    actuator -- spans joint space exactly. Each primitive drives its actuators toward their + or -
    ctrl limit; combined with the calibrated task operators (toward_object_xy / lower_base /
    raise_base / close_hand / open_hand) these give the DSL its expressive completeness.
    """
    names = all_actuator_names(env)
    if granularity == "actuator":
        units = [(n, [n]) for n in names]
    else:
        groups: dict[str, list[str]] = {}
        for n in names:
            groups.setdefault(_semantic_group(n), []).append(n)
        units = list(groups.items())
    basis: list[dict[str, Any]] = []
    for label, acts in units:
        basis.append({"name": f"{label}_pos", "actuators": acts, "sign": +1.0})
        basis.append({"name": f"{label}_neg", "actuators": acts, "sign": -1.0})
    return basis


def check_basis_complete(env: Any, basis: list[dict[str, Any]]) -> dict[str, Any]:
    """A motion basis is complete when every actuator is ACTIVELY DRIVABLE in both directions.

    Returns {ok, not_drivable, single_sign_only}: `not_drivable` = no primitive moves it at all;
    `single_sign_only` = movable one way but not the other (an incomplete basis -- the joint can't
    reach part of its legal range under the priors).
    """
    names = set(all_actuator_names(env))
    pos: set[str] = set()
    neg: set[str] = set()
    for p in basis:
        acts = set(_names_of(env, p.get("actuators", [])))
        (pos if float(p.get("sign", 0)) > 0 else neg).update(acts)
    drivable = pos | neg
    return {
        "ok": names.issubset(pos) and names.issubset(neg),
        "not_drivable": sorted(names - drivable),
        "single_sign_only": sorted((drivable) - (pos & neg)),
    }


def _names_of(env: Any, tokens: list[str]) -> list[str]:
    names = all_actuator_names(env)
    return [names[i] for i in _resolve_actuators(env, tokens)]


# ----------------------------------------------------------------------------------------------
# Free-form expression compiler (restricted AST safe-eval -> JAX)
# ----------------------------------------------------------------------------------------------

_ALLOWED_FUNCS = {"clip", "sigmoid", "min", "max", "abs", "exp", "sqrt", "tanh", "arrive", "within"}
# Expected argument count per function; checked at validation so a wrong-arity call is rejected
# cleanly instead of crashing at eval time. min/max are variadic (>= 2 args).
_FUNC_ARITY = {"clip": 3, "sigmoid": 1, "abs": 1, "exp": 1, "sqrt": 1,
               "tanh": 1, "arrive": 3, "within": 2}
_VARIADIC_FUNCS = {"min", "max"}


def _validate_ast(node: ast.AST, signals: set[str]) -> None:
    if isinstance(node, ast.Expression):
        _validate_ast(node.body, signals)
    elif isinstance(node, ast.Constant):
        if not isinstance(node.value, (int, float)) or isinstance(node.value, bool):
            raise ValueError(f"only numeric constants allowed, got {node.value!r}")
    elif isinstance(node, ast.Name):
        if node.id not in signals:
            raise ValueError(f"unknown signal {node.id!r}; allowed: {sorted(signals)}")
    elif isinstance(node, ast.BinOp):
        if not isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            raise ValueError(f"operator {type(node.op).__name__} not allowed")
        _validate_ast(node.left, signals); _validate_ast(node.right, signals)
    elif isinstance(node, ast.UnaryOp):
        if not isinstance(node.op, (ast.USub, ast.UAdd)):
            raise ValueError("only unary +/- allowed")
        _validate_ast(node.operand, signals)
    elif isinstance(node, ast.Call):
        if not (isinstance(node.func, ast.Name) and node.func.id in _ALLOWED_FUNCS):
            raise ValueError(f"only calls to {sorted(_ALLOWED_FUNCS)} allowed")
        if node.keywords:
            raise ValueError("keyword args not allowed")
        want = _FUNC_ARITY.get(node.func.id)
        if want is not None and len(node.args) != want:
            raise ValueError(f"{node.func.id}() takes {want} args, got {len(node.args)}")
        if node.func.id in _VARIADIC_FUNCS and len(node.args) < 2:
            raise ValueError(f"{node.func.id}() takes at least 2 args, got {len(node.args)}")
        for a in node.args:
            _validate_ast(a, signals)
    elif isinstance(node, ast.Compare):
        if len(node.ops) != 1 or not isinstance(node.ops[0], (ast.Lt, ast.Gt, ast.LtE, ast.GtE)):
            raise ValueError("only a single < > <= >= comparison allowed")
        _validate_ast(node.left, signals); _validate_ast(node.comparators[0], signals)
    else:
        raise ValueError(f"disallowed expression node: {type(node).__name__}")


def _eval_ast(node: ast.AST, sig: dict[str, jp.ndarray]) -> jp.ndarray:
    if isinstance(node, ast.Expression):
        return _eval_ast(node.body, sig)
    if isinstance(node, ast.Constant):
        return jp.float32(node.value)
    if isinstance(node, ast.Name):
        return sig[node.id]
    if isinstance(node, ast.BinOp):
        l = _eval_ast(node.left, sig); r = _eval_ast(node.right, sig)
        if isinstance(node.op, ast.Add): return l + r
        if isinstance(node.op, ast.Sub): return l - r
        if isinstance(node.op, ast.Mult): return l * r
        return l / r
    if isinstance(node, ast.UnaryOp):
        v = _eval_ast(node.operand, sig)
        return -v if isinstance(node.op, ast.USub) else v
    if isinstance(node, ast.Call):
        f = node.func.id; a = [_eval_ast(x, sig) for x in node.args]
        if f == "clip": return jp.clip(a[0], a[1], a[2])
        if f == "sigmoid": return 1.0 / (1.0 + jp.exp(-a[0]))
        if f == "min": return functools.reduce(jp.minimum, a)
        if f == "max": return functools.reduce(jp.maximum, a)
        if f == "abs": return jp.abs(a[0])
        if f == "sqrt": return jp.sqrt(jp.maximum(a[0], 0.0))
        if f == "tanh": return jp.tanh(a[0])
        if f == "arrive":
            # Cruise-to-target-then-halt: a saturated velocity command toward err=0. The large fixed
            # gain pins the output at +-vmax (constant cruise) until |err| is small, where it drops
            # into the linear zone and the -damp*v term brakes to a clean stop -- a trapezoidal move.
            # a = [err (= target - current), v (the part's velocity), vmax (cruise speed cap)].
            return jp.clip(10.0 * a[0] - 0.6 * a[1], -a[2], a[2])
        if f == "within":
            # Position-only completion: > 0 exactly when |err| < tol. Use in a stage's gate/success so
            # the hand-off fires ON ARRIVAL -- NO velocity term, so a fast stage does not wait to stop.
            return a[1] - jp.abs(a[0])
        return jp.exp(a[0])
    if isinstance(node, ast.Compare):
        l = _eval_ast(node.left, sig); r = _eval_ast(node.comparators[0], sig); op = node.ops[0]
        if isinstance(op, ast.Lt): return (l < r).astype(jp.float32)
        if isinstance(op, ast.Gt): return (l > r).astype(jp.float32)
        if isinstance(op, ast.LtE): return (l <= r).astype(jp.float32)
        return (l >= r).astype(jp.float32)
    raise ValueError(f"uncompilable node {type(node).__name__}")


def compile_expr(src: str, signal_names: set[str]) -> Callable[[dict[str, jp.ndarray]], jp.ndarray]:
    """Validate + compile one expression string into ev(signals) -> scalar (jax-traceable)."""
    tree = ast.parse(str(src), mode="eval")
    _validate_ast(tree, signal_names)
    return lambda sig: _eval_ast(tree, sig)


# Per-actuator SELF observables, available ONLY inside channel expressions: bound, per actuator in
# the channel's set, to that actuator's own commanded target / measured joint position / contact
# region (the same quantities as ctrl_<name>/q_<name>/c_<region>/env_c_<region>, resolved
# mechanically from the obs layout). One channel can thereby give each of its actuators an
# individual reactive response -- without this, per-joint shapes need one channel per actuator,
# which no candidate can afford to author.
CHANNEL_SELF_SIGNALS = {"ctrl_self", "q_self", "v_self", "c_self", "env_c_self"}


def _compile_channel(env: Any, ch: dict, signal_names: set[str],
                     obs_idx: dict[str, int]) -> tuple[jp.ndarray, Callable]:
    """Compile one channel {actuators, expr} -> (idx[int32], ev(obs, sig)).

    ev returns a scalar (broadcast over idx) or, when the expr uses a SELF observable, a vector of
    len(idx) with self bound per actuator. `obs_idx` is the name->index map from raw_obs_entries;
    self indices are resolved through the ctrl_<name>/q_<name> observables, never obs-layout
    arithmetic. An actuator without a q_ observable (e.g. tendon-coupled) makes q_self a compile
    error, so validation rejects the candidate with the actuator names in the message.
    """
    act_ids = _resolve_actuators(env, ch.get("actuators", []))
    idx = jp.asarray(act_ids, dtype=jp.int32)
    src = str(ch.get("expr", "0"))
    used = {n.id for n in ast.walk(ast.parse(src, mode="eval")) if isinstance(n, ast.Name)}
    self_used = used & CHANNEL_SELF_SIGNALS
    ev = compile_expr(src, set(signal_names) | CHANNEL_SELF_SIGNALS)
    if not self_used:
        return idx, lambda obs, sig, _ev=ev: _ev(sig)
    m = env.model
    names = [m.actuator(a).name for a in act_ids]
    ctrl_idx = jp.asarray([obs_idx[f"ctrl_{an}"] for an in names], jp.int32)

    def _self_idx(prefix):
        missing = [an for an in names if f"{prefix}_{an}" not in obs_idx]
        if missing:
            raise ValueError(f"{prefix}_self is not available for actuators without a {prefix}_ "
                             "observable: " + ", ".join(missing))
        return jp.asarray([obs_idx[f"{prefix}_{an}"] for an in names], jp.int32)

    def _contact_idx(prefix: str, self_name: str):
        # *_self binds each actuator to its hand region's CONTACT-force observable
        # (<prefix>_<region>). Base-carriage actuators have no contact region, so this is a compile
        # error there.
        cols, missing = [], []
        for an in names:
            reg = _contact_region_of_actuator(an)
            key = f"{prefix}_{reg}" if reg else None
            if key is None or key not in obs_idx:
                missing.append(an)
            else:
                cols.append(obs_idx[key])
        if missing:
            raise ValueError(f"{self_name} is not available for actuators with no contact region "
                             "(e.g. the base carriage): " + ", ".join(missing))
        return jp.asarray(cols, jp.int32)

    q_idx = _self_idx("q") if "q_self" in self_used else None
    v_idx = _self_idx("v") if "v_self" in self_used else None
    c_idx = _contact_idx("c", "c_self") if "c_self" in self_used else None
    env_c_idx = _contact_idx("env_c", "env_c_self") if "env_c_self" in self_used else None

    def ev_self(obs, sig, _ev=ev, _c=ctrl_idx, _q=q_idx, _v=v_idx, _cf=c_idx, _ecf=env_c_idx):
        s = dict(sig)
        s["ctrl_self"] = obs[_c]
        if _q is not None:
            s["q_self"] = obs[_q]
        if _v is not None:
            s["v_self"] = obs[_v]
        if _cf is not None:
            s["c_self"] = obs[_cf]
        if _ecf is not None:
            s["env_c_self"] = obs[_ecf]
        return _ev(s)

    return idx, ev_self


def freeform_signal_fn(env: Any, *, eps: float = 0.04, kappa: float = 0.6,
                       tau_d: float = 0.02, tau_c: float = 0.15) -> tuple[Callable, set[str]]:
    """signals(obs) -> dict of named scalar signals available to free-form expressions."""
    base, _ = make_signal_fn(env)

    def signals(obs: jp.ndarray) -> dict[str, jp.ndarray]:
        s = base(obs)
        rel = s["obj_rel"]
        near = 1.0 / (1.0 + jp.exp(-((eps - s["palm_obj_dist"]) / tau_d)))
        gripped = 1.0 / (1.0 + jp.exp(-((s["closure"] - kappa) / tau_c)))
        return {
            "palm_obj_dist": s["palm_obj_dist"], "closure": s["closure"], "lift": s["lift"],
            "obj_rel_x": rel[0], "obj_rel_y": rel[1], "obj_rel_z": rel[2],
            "near": near, "gripped": gripped,
        }

    names = {"palm_obj_dist", "closure", "lift", "obj_rel_x", "obj_rel_y", "obj_rel_z", "near", "gripped"}
    return signals, names


def make_freeform_prior_fn(env: Any, program: dict[str, Any]) -> tuple[Callable[..., jp.ndarray], jp.ndarray, dict[str, Any]]:
    """Compile a free-form program {channels:[{actuators, expr}]} into (prior_fn, default_weights, info).

    weights (optional) scale channels; default = ones. prior_fn(obs, weights) -> mean_shift in [-1,1].
    """
    action_dim = int(env.action_size)
    signals, signal_names = program_signal_fn(env, program)
    obs_idx = dict(raw_obs_entries(env)[0])
    channels = list(program.get("channels", []))
    compiled = [_compile_channel(env, ch, signal_names, obs_idx) for ch in channels]
    n_ch = len(compiled)

    def prior_fn(obs, weights):
        gains = jp.ones((n_ch,), jp.float32) if weights is None else jp.asarray(weights, jp.float32)[:n_ch]
        sig = signals(obs)
        out = jp.zeros((action_dim,), jp.float32)
        for i, (idx, ev) in enumerate(compiled):
            val = jp.clip(ev(obs, sig), -1.0, 1.0) * gains[i]
            out = out.at[idx].add(val)
        return jp.clip(out, -1.0, 1.0)

    info = {"mode": "freeform", "n_weights": n_ch, "n_channels": n_ch}
    return prior_fn, jp.ones((n_ch,), jp.float32), info


# ----------------------------------------------------------------------------------------------
# Stacked-gate composition over sub-priors specified as EITHER DSL rules (operator + basis) OR
# free-form channels. Holds the action-prior framework fixed (stacked phase gating) so the
# DSL-vs-free-form comparison isolates the sub-prior representation. Order = [approach, grasp, lift]
# to match _gates("stacked", ...).
# ----------------------------------------------------------------------------------------------

def compile_subprior(env: Any, spec: dict[str, Any]) -> Callable[[jp.ndarray], jp.ndarray]:
    """Compile one phase sub-prior into fn(obs) -> mean_shift.

    Free-form form: {"channels": [{actuators, expr}]} -- each channel is a normalized mean-shift
    expression over the observables for a set of actuators.
    """
    fn, _dw, _info = make_freeform_prior_fn(env, {"channels": spec.get("channels", [])})
    return lambda obs: fn(obs, None)


def make_stacked_prior_fn(env: Any, program: dict[str, Any]):
    """Compile a stacked-gate program {subpriors:[approach, grasp, lift], gates:{}} into
    (prior_fn(obs, weights)->mean_shift, default_weights, info). Each sub-prior is a set of
    free-form channels. The stacked gate (composed_priors._gates) blends them on observable signals."""
    action_dim = int(env.action_size)
    signals, _ = make_signal_fn(env)
    gate_params = dict(program.get("gates", {}))
    subs = [compile_subprior(env, s) for s in program.get("subpriors", [])]
    n = len(subs)

    def prior_fn(obs, weights):
        gains = jp.ones((n,), jp.float32) if weights is None else jp.asarray(weights, jp.float32)[:n]
        g = _gates("stacked", signals(obs), gate_params)  # [approach, grasp, lift]
        out = jp.zeros((action_dim,), jp.float32)
        for i in range(n):
            out = out + (g[i] * gains[i]) * subs[i](obs)
        return jp.clip(out, -1.0, 1.0)

    return prior_fn, jp.ones((n,), jp.float32), {
        "mode": "stacked", "representation": program.get("representation"), "n_weights": n,
        "sub_names": [str(s.get("name", f"sub_{i}")) for i, s in enumerate(program.get("subpriors", []))],
    }


DEFAULT_BLEND_TAU = 0.1


def blend_weights(gates: jp.ndarray, tau: float = DEFAULT_BLEND_TAU) -> jp.ndarray:
    """Stage weights from raw gate values -- the ONE place the blend semantics live, shared by the
    prior itself, make_stage_weight_fn, and stage_occupancy so the executed mixture and every
    diagnostic see identical weights.

    HARD one-stage selection: exactly one stage is ever active, even at a hand-off -- no cross-fade,
    no mixing of adjacent stages' channels. Gates are clipped to [0, 1] before the argmax so a
    later stage whose raw ladder gate saturates large (e.g. 1 - done_k with an unbounded, far-from-
    done measurement) cannot outbid a nearly-finished current stage; among stages that saturate to
    1 the argmax's lowest-index tie-break selects the EARLIEST (first-unfinished) one. `tau` is
    retained for signature compatibility but no longer softens the selection.
    """
    n = int(gates.shape[0])
    if n == 0:
        return gates
    idx = jp.argmax(jp.clip(gates, 0.0, 1.0))  # ties -> lowest index = earliest unfinished stage
    return jax.nn.one_hot(idx, n, dtype=gates.dtype)


def stage_progression_mode(program: dict[str, Any]) -> str:
    """Stage cursor semantics for freeform_staged programs.

    Default is monotone: ordinary stages may advance but not regress. A legacy/reactive mode remains
    available for archived programs or explicit experiments that need pure current-observation gates.
    """
    mode = str(program.get("stage_progression", program.get("progression", "monotone"))).lower()
    return "reactive" if mode in {"reactive", "stateless"} else "monotone"


def advance_stage_index(gates: jp.ndarray, prev_stage: jp.ndarray, tau: float = DEFAULT_BLEND_TAU,
                        *, mode: str = "monotone") -> jp.ndarray:
    """Return the selected stage index under the program's cursor semantics.

    The raw stage is still the authored hard-argmax gate result. Monotone progression treats it as an
    advancement request: it can keep the current stage or move to the next reached frontier, but a
    current-signal regression cannot reopen an already-passed stage.
    """
    n = int(gates.shape[0])
    if n == 0:
        return jp.asarray(0, jp.int32)
    raw = jp.argmax(jp.clip(gates, 0.0, 1.0)).astype(jp.int32)
    if mode == "reactive":
        return raw
    prev = jp.clip(prev_stage.astype(jp.int32), 0, n - 1)
    return jp.minimum(jp.maximum(prev, raw), jp.minimum(prev + 1, n - 1))


def _tau_of(program: dict[str, Any]) -> float:
    return float(program.get("temperature", DEFAULT_BLEND_TAU))


def _compile_staged_components(env: Any, program: dict[str, Any]):
    action_dim = int(env.action_size)
    signals, signal_names = program_signal_fn(env, program)
    obs_idx = dict(raw_obs_entries(env)[0])
    tau = _tau_of(program)
    progression = stage_progression_mode(program)
    stages = list(program.get("stages", []))
    compiled: list[tuple[Callable, list[tuple[jp.ndarray, Callable]]]] = []
    for st in stages:
        gate_ev = compile_expr(str(st.get("gate", "1")), signal_names)
        chans = [_compile_channel(env, ch, signal_names, obs_idx)
                 for ch in st.get("channels", [])]
        compiled.append((gate_ev, chans))
    names = [str(s.get("name", f"stage_{i}")) for i, s in enumerate(stages)]
    return action_dim, signals, signal_names, tau, progression, stages, compiled, names


def make_freeform_staged_prior_fn(env: Any, program: dict[str, Any]):
    """Fully free-form MULTI-STAGE prior: the AUTHOR defines N stages, each with its own free-form
    GATE expression (stage activation, a function of observable signals) AND free-form channels.

    Generalizes the fixed 3-phase near/gripped stacked gate: arbitrary stage count and author-defined
    transitions, expressed symbolically over the same signals as the channels. Stages are combined
    by default with a rollout-carried monotone stage cursor; legacy/reactive mode can still select
    purely from current gates. PPO uses the cursor only during rollout collection, not in minibatch
    loss recomputation.

    program = {"mode":"freeform_staged", "temperature": <tau>,
               "stages":[{"name", "gate":"<expr>", "channels":[{actuators, expr}, ...]}, ...]}
    Stage weights are a HARD one-stage selection over the gate values -- clip to [0,1] then argmax,
    exactly one stage active (the only blend; see blend_weights()). `temperature` is vestigial.
    Per-stage gains are the tunable `weights` vector (default ones).
    """
    action_dim, signals, _signal_names, tau, progression, stages, compiled, names = (
        _compile_staged_components(env, program))
    n = len(compiled)

    def _stage_out(chans, obs, sig):
        out = jp.zeros((action_dim,), jp.float32)
        for idx, ev in chans:
            out = out.at[idx].add(jp.clip(ev(obs, sig), -1.0, 1.0))
        return out

    def prior_fn(obs, weights):
        gains = jp.ones((n,), jp.float32) if weights is None else jp.asarray(weights, jp.float32)[:n]
        sig = signals(obs)
        gates = jp.stack([g(sig) for g, _ in compiled]) if n else jp.zeros((0,), jp.float32)
        gates = jp.asarray(gates, jp.float32)
        w = blend_weights(gates, tau)
        out = jp.zeros((action_dim,), jp.float32)
        for i, (_g, chans) in enumerate(compiled):
            out = out + (w[i] * gains[i]) * _stage_out(chans, obs, sig)
        return jp.clip(out, -1.0, 1.0)

    info = {"mode": "freeform_staged", "blend": "argmax_onehot", "stage_progression": progression,
            "n_stages": n, "n_weights": n, "stage_names": names}
    return prior_fn, jp.ones((n,), jp.float32), info


def make_freeform_staged_step_fn(env: Any, program: dict[str, Any]):
    """Compile a rollout-stateful freeform_staged prior step.

    Returns step_fn(obs, weights, prev_stage) -> (action, new_stage). This is used during actual
    rollouts so stage progress can be monotone without adding task-specific state to the environment
    observation or changing authored signal vocabulary.
    """
    action_dim, signals, _signal_names, tau, progression, _stages, compiled, names = (
        _compile_staged_components(env, program))
    n = len(compiled)

    def _stage_out(chans, obs, sig):
        out = jp.zeros((action_dim,), jp.float32)
        for idx, ev in chans:
            out = out.at[idx].add(jp.clip(ev(obs, sig), -1.0, 1.0))
        return out

    def step_fn(obs, weights, prev_stage):
        gains = jp.ones((n,), jp.float32) if weights is None else jp.asarray(weights, jp.float32)[:n]
        sig = signals(obs)
        gates = jp.stack([g(sig) for g, _ in compiled]) if n else jp.zeros((0,), jp.float32)
        gates = jp.asarray(gates, jp.float32)
        stage_idx = advance_stage_index(gates, jp.asarray(prev_stage, jp.int32), tau, mode=progression)
        w = jax.nn.one_hot(stage_idx, n, dtype=jp.float32) if n else jp.zeros((0,), jp.float32)
        out = jp.zeros((action_dim,), jp.float32)
        for i, (_g, chans) in enumerate(compiled):
            out = out + (w[i] * gains[i]) * _stage_out(chans, obs, sig)
        return jp.clip(out, -1.0, 1.0), stage_idx

    info = {"mode": "freeform_staged", "blend": "argmax_onehot", "stage_progression": progression,
            "n_stages": n, "n_weights": n, "stage_names": names}
    return step_fn, jp.ones((n,), jp.float32), info


def make_stage_weight_fn(env: Any, program: dict[str, Any]):
    """Expose ONLY the per-stage activation weights of a freeform_staged program (not the action).

    Returns (weight_fn, stage_names). weight_fn(obs) -> w[n_stages], the SAME hard one-stage
    (clip+argmax) weights make_freeform_staged_prior_fn uses internally. Used by the diagnostic to localize
    WHICH authored stage a trained policy stalls in -- task-agnostic, since it reads only the model's
    own gates over the shared signals.
    """
    signals, signal_names = program_signal_fn(env, program)
    tau = _tau_of(program)
    progression = stage_progression_mode(program)
    stages = list(program.get("stages", []))
    gate_evs = [compile_expr(str(st.get("gate", "1")), signal_names) for st in stages]
    n = len(gate_evs)
    names = [str(s.get("name", f"stage_{i}")) for i, s in enumerate(stages)]

    def weight_fn(obs):
        sig = signals(obs)
        gates = jp.stack([g(sig) for g in gate_evs]) if n else jp.zeros((0,), jp.float32)
        return blend_weights(jp.asarray(gates, jp.float32), tau)

    return weight_fn, names


# Extra signals available ONLY to diagnostic probes (evaluated offline over whole [T, E] episode
# arrays, so per-episode baselines are allowed -- unlike gate/channel signals, which must stay
# stateless functions of the current obs for PPO). Generic rigid-body quantities, no task nouns.
PROBE_EXTRA_SIGNALS = {
    "obj_disp_xy": "object's horizontal distance from its own position at the episode start (m)",
    "obj_speed": "object's translational speed (m/s)",
}

MAX_PROBES = 8  # measurement slots per candidate (cap keeps prompts bounded, not a hard budget)
MAX_EVALS = 8   # authored acceptance-test slots per candidate (same bound, same reason)


def raw_obs_entries(env: Any) -> tuple[list[tuple[str, int]], list[str]]:
    """(name, obs index) for every raw observable, plus the actuators without a q_ observable.

    THE single place the env's obs layout is decoded into named observables -- everything that
    needs a raw observable's obs index (raw_signal_fn, channel SELF observables, body-motion
    diagnostics) must resolve it through these entries by NAME, never re-derive offsets. Porting
    to an env with a different obs layout means changing only this function.
    """
    _base, info = make_signal_fn(env)  # obs-layout indices only
    nb = len(env.base_qadr)
    n_pre = int(info["n_pre"])
    m = env.model
    act_names = [m.actuator(i).name for i in range(int(env.nu))]
    qmap = actuator_obs_qpos_map(env)
    axes = "xyz"
    entries: list[tuple[str, int]] = []
    for i in range(min(nb, 3)):
        entries.append((f"base_pos_{axes[i]}", i))
    for i in range(3):
        entries.append((f"obj_pos_{axes[i]}", n_pre + i))
    for i in range(3):
        entries.append((f"obj_vel_{axes[i]}", n_pre + 3 + i))
    for i in range(3):
        entries.append((f"palm_pos_{axes[i]}", n_pre + 6 + i))
    for i in range(3):
        entries.append((f"obj_rel_{axes[i]}", n_pre + 9 + i))
    for i, an in enumerate(act_names):
        entries.append((f"ctrl_{an}", n_pre + 12 + i))
    # Per-region CONTACT force blocks (the entries just before obj_pos): object contact
    # c_thumb..c_palm, optionally preceded by non-object environment contact env_c_thumb..env_c_palm.
    groups = list(getattr(env, "contact_groups", ()))
    env_groups = list(getattr(env, "env_contact_groups", ()))
    for k, g in enumerate(env_groups):
        entries.append((f"env_c_{g}", n_pre - len(groups) - len(env_groups) + k))
    for k, g in enumerate(groups):
        entries.append((f"c_{g}", n_pre - len(groups) + k))
    # World-frame BODY POSITION block: <body>_x/y/z for every robot body, sitting immediately BEFORE
    # the contact blocks (see _observe). Indexed tail-relative so it self-corrects with obs_size.
    body_names = list(getattr(env, "body_pos_names", ()))
    body_start = n_pre - len(groups) - len(env_groups) - 3 * len(body_names)
    for j, bn in enumerate(body_names):
        for a in range(3):
            entries.append((f"{bn}_{axes[a]}", body_start + 3 * j + a))
    no_q = []
    vmap = actuator_obs_qvel_map(env)
    for i, an in enumerate(act_names):
        if qmap[i] is not None:
            entries.append((f"q_{an}", int(qmap[i])))
        else:
            no_q.append(an)
        if vmap[i] is not None:
            entries.append((f"v_{an}", int(vmap[i])))
    return entries, no_q


def raw_signal_fn(env: Any) -> tuple[Callable, set[str], str]:
    """RAW observables only, mechanically enumerated from the env/robot: world positions and
    velocities present in obs, contact-force sensors, commanded actuator targets, and measured joint
    positions.

    NO derived quantities live here -- distances, proximity gates, closure fractions, alignment
    measures are TASK STRUCTURE and must be LLM-authored per candidate (`signals` on the program;
    see AGENTS.md). Returns (signals(obs)->dict, names, doc) where doc is the injectable
    description block for prompts.
    """
    entries, no_q = raw_obs_entries(env)
    names = [n for n, _ in entries]
    idxs = jp.asarray([ix for _, ix in entries], jp.int32)

    def signals(obs: jp.ndarray) -> dict[str, jp.ndarray]:
        vals = obs[idxs]
        return {n: vals[k] for k, n in enumerate(names)}

    doc = (
        "OBSERVABLES (raw, per control step -- there are NO predefined derived signals; author "
        "your own from these):\n"
        "  base_pos_x/y/z: base carriage DOF positions (ctrl units of base_x/base_y/base_z)\n"
        "  obj_pos_x/y/z: object position, world frame (m)\n"
        "  obj_vel_x/y/z: object linear velocity, world frame (m/s)\n"
        "  palm_pos_x/y/z: palm position, world frame (m)\n"
        "  obj_rel_x/y/z: object position minus grasp-site position, world frame (m)\n"
        "  <body>_x/y/z: WORLD-FRAME position (m) of every robot body -- so you can read where any "
        "part is in space (e.g. a fingertip's height above the table to keep it from digging in). "
        "+z is up; the table/floor is at world z = 0, so a part's _z IS its height above the table. "
        "The FINGERTIP bodies (the parts most likely to catch on the table -- gate descent so their "
        "_z stays above a small margin) are: "
        + ", ".join(getattr(env.cfg, "fingertip_bodies", ()) or ()) + ". All bodies: "
        + ", ".join(getattr(env, "body_pos_names", ())) + "\n"
        "  ctrl_<actuator>: current commanded target of that actuator (its ctrl units/range)\n"
        "  q_<actuator>: MEASURED position of the joint that actuator drives (differs from "
        "ctrl_<actuator> when the joint is blocked or saturated)\n"
        "  v_<actuator>: MEASURED velocity of that joint (position units per second) -- the "
        "damping/feedback term a position error alone cannot provide\n"
        "  c_<region>: measured NORMAL CONTACT FORCE (N) between that hand region and the object, "
        "regions = " + ", ".join(getattr(env, "contact_groups", ())) + ". This is the GROUND-TRUTH "
        "touch signal: exactly zero in free air, and exactly zero when a joint merely loads its "
        "OWN LIMIT (unlike a joint force, it is read only from real object<->hand contacts). It "
        "rises the instant a region presses the object and grows with how hard it presses. It is "
        "in newtons; use the per-actuator servo_gain values and CONTROL law in this spec (force per "
        "unit tracking error) as the reference for what a light vs firm press means on this hand. "
        "c_self inside a channel binds each actuator to its own region's object-contact force\n"
        "  env_c_<region>: measured NORMAL CONTACT FORCE (N) between that hand region and non-object "
        "environment geometry, regions = " + ", ".join(getattr(env, "env_contact_groups", ())) +
        ". It is separate from c_<region>: object contact does not raise env_c_<region>, and "
        "environment contact does not raise c_<region>. It is sensory data for authoring explicit "
        "environment-contact conditions, budgets, probes, evals, or channels. env_c_self inside a "
        "channel binds each actuator to its own region's environment-contact force"
        + (f"; q_/v_ not available (no single-joint transmission): {', '.join(no_q)}" if no_q else "")
    )
    return signals, set(names), doc


def program_signal_fn(env: Any, program: dict[str, Any]) -> tuple[Callable, set[str]]:
    """The signal vocabulary for ONE program: raw observables + the program's own authored
    `signals: {name: expr}` (evaluated in insertion order; later signals may reference earlier
    ones). `parameters` are scalar constants made available by name to every expression, so an
    exploration/calibration pass can tune thresholds and gains without rewriting the structure.
    Programs WITHOUT authored signals get the legacy derived vocabulary as well, so archived
    candidates replay unchanged -- new candidates are expected to author their own."""
    raw_fn, raw_names, _doc = raw_signal_fn(env)
    params = _parameter_values(program)
    param_names = set(params)
    authored = program.get("signals")
    if isinstance(authored, dict) and authored:
        compiled: list[tuple[str, Callable]] = []
        avail = set(raw_names) | param_names
        for name, expr in authored.items():
            ev = compile_expr(str(expr), avail)  # raises -> validation rejects the candidate
            compiled.append((str(name), ev))
            avail.add(str(name))

        def signals(obs: jp.ndarray) -> dict[str, jp.ndarray]:
            sig = raw_fn(obs)
            for name, value in params.items():
                sig[name] = jp.asarray(value, jp.float32)
            for name, ev in compiled:
                sig[name] = jp.asarray(ev(sig), jp.float32)
            return sig

        return signals, avail
    legacy_fn, _legacy_names = freeform_signal_fn(env)
    legacy_names = {"palm_obj_dist", "closure", "lift", "obj_rel_x", "obj_rel_y", "obj_rel_z",
                    "near", "gripped"}

    def signals(obs: jp.ndarray) -> dict[str, jp.ndarray]:
        sig = raw_fn(obs)
        for name, value in params.items():
            sig[name] = jp.asarray(value, jp.float32)
        sig.update(legacy_fn(obs))
        return sig

    return signals, raw_names | legacy_names | param_names


def _parameter_values(program: dict[str, Any]) -> dict[str, float]:
    """Return scalar parameter values from a candidate's parameter schema.

    Accepted forms:
      {"p": 0.2}
      {"p": {"init": 0.2, "range": [0.0, 1.0]}}
      {"p": {"value": 0.2, "range": [0.0, 1.0]}}
    The compiler uses the current value/init only; calibration code can rewrite these values.
    """
    out: dict[str, float] = {}
    raw = program.get("parameters")
    if not isinstance(raw, dict):
        return out
    for name, spec in raw.items():
        if isinstance(spec, (int, float)):
            out[str(name)] = float(spec)
        elif isinstance(spec, dict):
            out[str(name)] = float(spec.get("value", spec.get("init", 0.0)))
    return out


def actuator_obs_qpos_map(env: Any) -> list[int | None]:
    """Per actuator: the index into the obs qpos block (base_q ++ hand_q) of the joint it drives,
    or None when the transmission is not a single joint (e.g. tendon-coupled actuators). Derived
    from the mujoco model, so it is robot-agnostic."""
    import mujoco
    m = env.model
    base_qadr = [int(a) for a in env.base_qadr]
    hand_qadr = [int(a) for a in getattr(env, "hand_qadr", ())]
    hand_off = len(base_qadr) + len(env.base_vadr)
    out: list[int | None] = []
    for i in range(int(env.nu)):
        if int(m.actuator_trntype[i]) != int(mujoco.mjtTrn.mjTRN_JOINT):
            out.append(None)
            continue
        qadr = int(m.jnt_qposadr[int(m.actuator_trnid[i, 0])])
        if qadr in base_qadr:
            out.append(base_qadr.index(qadr))
        elif qadr in hand_qadr:
            out.append(hand_off + hand_qadr.index(qadr))
        else:
            out.append(None)
    return out


def actuator_obs_qvel_map(env: Any) -> list[int | None]:
    """Per actuator: the index into the obs of the MEASURED VELOCITY of the joint it drives
    (base_v ++ hand_v blocks), or None when the transmission is not a single joint. Mirrors
    actuator_obs_qpos_map; derived from the mujoco model, so it is robot-agnostic."""
    import mujoco
    m = env.model
    base_vadr = [int(a) for a in env.base_vadr]
    hand_vadr = [int(a) for a in getattr(env, "hand_vadr", ())]
    base_off = len(env.base_qadr)
    hand_off = len(env.base_qadr) + len(base_vadr) + len(getattr(env, "hand_qadr", ()))
    out: list[int | None] = []
    for i in range(int(env.nu)):
        if int(m.actuator_trntype[i]) != int(mujoco.mjtTrn.mjTRN_JOINT):
            out.append(None)
            continue
        vadr = int(m.jnt_dofadr[int(m.actuator_trnid[i, 0])])
        if vadr in base_vadr:
            out.append(base_off + base_vadr.index(vadr))
        elif vadr in hand_vadr:
            out.append(hand_off + hand_vadr.index(vadr))
        else:
            out.append(None)
    return out


def stage_transition_stats(active: "np.ndarray", n: int, dwell: int = 3) -> dict:
    """Dwell-qualified, ORDERED stage-transition statistics from the active-stage array [T, E].

    Gate dominance (>=1 winning step) certifies gate arithmetic, not accomplishment. The success
    predicate here is the model's OWN hand-off: stage k SUCCEEDED in an episode iff stage k+1
    subsequently stays dominant for >= `dwell` consecutive steps after k was dominant. This kills
    one-step transient dominance and correlated "air chain" false positives, and attributes a broken
    hand-off to the correct side (k entered but never converts vs k+1 converts but k+2 never fires).

    Returns per-stage lists (index i; hand-off entries are for i -> i+1, so the last is None):
      entered  -- fraction of episodes with a dwell-qualified run of stage i
      handoff  -- fraction of episodes with an ordered i -> i+1 dwell-qualified transition
      reverse  -- fraction of episodes that FALL BACK: a dwell run of i starting after i+1's first
    Pure numpy over the precomputed `active` array; unit-testable without an env.
    """
    import numpy as _np

    T, E = active.shape
    dwell = max(1, min(int(dwell), T))
    onehot = active[:, :, None] == _np.arange(n)[None, None, :]              # [T,E,n]
    cpad = _np.concatenate([_np.zeros((1, E, n), int), _np.cumsum(onehot, axis=0)], axis=0)
    runs = (cpad[dwell:] - cpad[:-dwell]) >= dwell                           # dwell run STARTS at j
    has_run = runs.any(axis=0)                                               # [E,n]
    first_run = _np.where(has_run, runs.argmax(axis=0), T + 1)               # [E,n]
    first_occ = _np.where(onehot.any(axis=0), onehot.argmax(axis=0), T + 1)  # [E,n]
    entered = [float(has_run[:, i].mean()) for i in range(n)]
    jidx = _np.arange(runs.shape[0])[:, None, None]
    if n > 1:
        # hand-off i->i+1: a dwell run of i+1 starting at/after i's first dominance (ordered)
        handoff_ok = (runs[:, :, 1:] & (jidx >= first_occ[None, :, :-1])).any(axis=0)  # [E,n-1]
        handoff_ok &= first_occ[:, :-1] <= T
        # reverse i+1->i: a dwell run of i starting AFTER i+1's first dwell run (fell back)
        reverse_ok = (runs[:, :, :-1] & (jidx > first_run[None, :, 1:])).any(axis=0)   # [E,n-1]
        handoff = [float(handoff_ok[:, i].mean()) for i in range(n - 1)] + [None]
        reverse = [float(reverse_ok[:, i].mean()) for i in range(n - 1)] + [None]
    else:
        handoff, reverse = [None], [None]
    return {"dwell": dwell, "entered": entered, "handoff": handoff, "reverse": reverse,
            "first_occ": first_occ}  # [E,n] first dominance step (T+1 = never); for success windows


def _failure_attribution(failure: "np.ndarray", active: "np.ndarray",
                         names: list[str], n: int) -> dict:
    """Localize the task's per-step mistake indicator [T, E] to the authored stages: for each
    failing episode, the dominant stage at the FIRST failing step -- which stage was in control
    when the episode went wrong. The indicator is injected task data (task_failure_signal); this
    only attributes it."""
    f = np.asarray(failure, dtype=bool)
    fired = f.any(axis=0)
    outd: dict[str, Any] = {"failure_rate": round(float(fired.mean()), 4),
                            "by_stage_at_first_failure": {}}
    if fired.any():
        eps = np.nonzero(fired)[0]
        first = f.argmax(axis=0)[eps]
        counts = np.bincount(active[first, eps], minlength=n)
        outd["by_stage_at_first_failure"] = {
            names[i]: round(float(c / len(eps)), 4) for i, c in enumerate(counts) if c > 0}
    return outd


def stage_occupancy(env: Any, program: dict[str, Any], obs: "np.ndarray",
                    failure: "np.ndarray | None" = None) -> dict:
    """Localize the stalling stage from observations VISITED by the trained policy.

    obs: array [T, E, obs_dim] of states the policy actually occupied. For each state we recompute
    the model's own stage weights and take the argmax as the "active" stage. We report, per stage,
    the time-share (occupancy), raw dominance reach (reached_frac), and the TRANSITION-BASED success
    stats from stage_transition_stats (entered/handoff/reverse/conversion). The stall is the FIRST
    BROKEN HAND-OFF: the shallowest reliably-entered stage whose successor rarely gets a
    dwell-qualified run. When no hand-off is broken (the chain completes), `weakest_stage` names the
    lowest-conversion hand-off so the endgame still gets stage-localized improvement pressure.

    For the stall stage we additionally surface DIRECTION evidence -- localization says WHERE the
    policy stalls, but not which way to fix it (an LLM told only "you stall in stage k" defaults to
    pushing stage k harder, which is exactly wrong when the stage is diverging the signal it acts
    on). So, restricted to timesteps where the stall stage is active, we report each generic
    signal's mean over the first vs second half of the episode, the raw gate values of the stall
    stage and its successor over the same split, and a self-lock flag (stall occupies ~all steps
    and the successor is ~never reached => under the hard one-stage selection the gate never wins).
    Purely structural -- no task labels -- so it generalizes to any staged prior on any task.
    """
    import re
    import numpy as _np

    signals, signal_names = program_signal_fn(env, program)
    sig_keys = sorted(signal_names)
    tau = _tau_of(program)
    progression = stage_progression_mode(program)
    stages = list(program.get("stages", []))
    names = [str(s.get("name", f"stage_{i}")) for i, s in enumerate(stages)]
    gate_exprs = [str(st.get("gate", "1")) for st in stages]
    gate_evs = [compile_expr(g, signal_names) for g in gate_exprs]
    n = len(names)
    if n == 0:
        return {"stage_names": [], "occupancy": [], "reached_frac": [], "stall_stage": None}
    action_dim = int(env.action_size)
    _obs_idx0 = dict(raw_obs_entries(env)[0])
    chan_evs = [[_compile_channel(env, ch, signal_names, _obs_idx0)
                 for ch in st.get("channels", [])] for st in stages]
    # Optional LLM-authored per-stage `success` expressions (the author's explicit post-condition
    # over the same signals). They are a cross-check, not the primary predicate -- a bad one must
    # not kill the eval, so compile failures are reported, never raised.
    succ_evs, succ_errors = [], {}
    for i, st in enumerate(stages):
        sx = st.get("success")
        if sx is None:
            succ_evs.append(None)
            continue
        try:
            succ_evs.append(compile_expr(str(sx), signal_names))
        except Exception as e:  # noqa: BLE001
            succ_evs.append(None)
            succ_errors[names[i]] = str(e)

    def _all(o):
        sig = signals(o)
        gates = jp.asarray(jp.stack([g(sig) for g in gate_evs]), jp.float32)
        w = blend_weights(gates, tau)
        succ = jp.stack([(jp.asarray(ev(sig), jp.float32) if ev is not None
                          else jp.asarray(0.0, jp.float32)) for ev in succ_evs])
        stage_act = []
        for chans in chan_evs:
            sa = jp.zeros((action_dim,), jp.float32)
            for idx, ev in chans:
                sa = sa.at[idx].add(jp.clip(ev(o, sig), -1.0, 1.0))
            stage_act.append(sa)
        return (w, gates, jp.stack([jp.asarray(sig[k], jp.float32) for k in sig_keys]), succ,
                jp.stack(stage_act))

    arr = _np.asarray(obs)
    T, E = arr.shape[0], arr.shape[1]
    flat = arr.reshape(T * E, -1)
    w, gates, sigv, succv, actv = (
        _np.asarray(x) for x in jax.jit(jax.vmap(_all))(jp.asarray(flat)))
    gates_te = gates.reshape(T, E, n)
    raw_active = w.argmax(axis=1).reshape(T, E)                       # raw gate winner per step
    if progression == "monotone":
        active = _np.zeros((T, E), dtype=_np.int32)
        prev = _np.zeros((E,), dtype=_np.int32)
        hi = max(n - 1, 0)
        for t in range(T):
            cur = _np.minimum(_np.maximum(prev, raw_active[t]), _np.minimum(prev + 1, hi))
            active[t] = cur
            prev = cur
        w_te = _np.eye(n, dtype=_np.float32)[active]
    else:
        active = raw_active
        w_te = w.reshape(T, E, n)
    sig_te = sigv.reshape(T, E, len(sig_keys))
    succ_te = succv.reshape(T, E, n)
    act_te = actv.reshape(T, E, n, action_dim)                       # per-stage channel intents
    occupancy = [float((active == i).mean()) for i in range(n)]      # time-share
    reached = [float(((active == i).any(axis=0)).mean()) for i in range(n)]  # episodes reaching i

    # BODY MOTION report: kinematics of every observed body, mechanically read from the obs layout
    # and attributed to whichever stage is dominant at each step. Uninterpreted measurements under
    # the env's own field names (speeds, net displacement from the episode-start pose) -- whether a
    # displacement is progress or disturbance is the author's call, not the framework's.
    def _triple(prefix):
        return arr[:, :, [_obs_idx0[f"{prefix}_{ax}"] for ax in "xyz"]]
    obj_pos, obj_vel, palm_pos = _triple("obj_pos"), _triple("obj_vel"), _triple("palm_pos")
    dt = float(getattr(env.cfg, "control_dt", 1.0))
    obj_speed = _np.linalg.norm(obj_vel, axis=-1)                              # [T,E] m/s
    obj_disp_xy = _np.linalg.norm(obj_pos[:, :, :2] - obj_pos[:1, :, :2], axis=-1)
    obj_disp = _np.linalg.norm(obj_pos - obj_pos[:1], axis=-1)
    palm_speed = _np.zeros_like(obj_speed)
    if T > 1:
        palm_speed[1:] = _np.linalg.norm(_np.diff(palm_pos, axis=0), axis=-1) / max(dt, 1e-9)

    def _masked(v, m):
        c = m.sum()
        return round(float((v * m).sum() / c), 4) if c else None

    per_stage_motion = []
    for i in range(n):
        m = active == i
        per_stage_motion.append({
            "stage": names[i],
            "obj_speed_mean": _masked(obj_speed, m),
            "obj_speed_max": round(float(obj_speed[m].max()), 4) if m.any() else None,
            "palm_speed_mean": _masked(palm_speed, m),
            "obj_disp_xy_from_start_mean": _masked(obj_disp_xy, m),
        })
    body_motion = {
        "obj_disp_xy_from_start_final": {"mean": round(float(obj_disp_xy[-1].mean()), 4),
                                         "max": round(float(obj_disp_xy[-1].max()), 4)},
        "obj_disp_from_start_final": {"mean": round(float(obj_disp[-1].mean()), 4),
                                      "max": round(float(obj_disp[-1].max()), 4)},
        "obj_speed": {"mean": round(float(obj_speed.mean()), 4),
                      "max": round(float(obj_speed.max()), 4)},
        "per_stage_while_dominant": per_stage_motion,
    }

    # COMMANDED MOTION report: the target speed the CHANNELS asked for, per dominant stage and
    # actuator group (|delta ctrl| / dt, ctrl-units/s, from the obs ctrl block). body_motion says
    # what the bodies DID; this says what the authoring commanded -- separating commanded
    # aggression from passive drift, and directly comparable to the control-law units bridge.
    m_act = env.model
    act_names_cm = [m_act.actuator(i).name for i in range(int(env.nu))]
    ctrl_cols = _np.asarray([_obs_idx0[f"ctrl_{an}"] for an in act_names_cm])
    groups_cm = sorted({_semantic_group(an) for an in act_names_cm})
    gmask = {g: _np.asarray([_semantic_group(an) == g for an in act_names_cm]) for g in groups_cm}
    ctrl_arr = arr[:, :, ctrl_cols]                                        # [T, E, nu]
    if T > 1:
        cmd_speed = _np.abs(_np.diff(ctrl_arr, axis=0)) / max(dt, 1e-9)   # [T-1, E, nu]
        act_cm = active[1:]                                                # stage at the step taken
        commanded_motion = []
        for i in range(n):
            sm = act_cm == i
            row: dict[str, Any] = {"stage": names[i]}
            for g in groups_cm:
                if sm.any():
                    row[g] = round(float(cmd_speed[:, :, gmask[g]][sm].mean()), 4)
                else:
                    row[g] = None
            commanded_motion.append(row)
    else:
        commanded_motion = []

    # CONTACT-FORCE report: per-region object contact and non-object environment contact, attributed
    # to whichever authored stage is dominant. This is purely structural force telemetry; whether a
    # contact is intended or unwanted belongs to the authored stages/probes.
    contact_groups = list(getattr(env, "contact_groups", ()))
    env_contact_groups = list(getattr(env, "env_contact_groups", ()))
    c_cols = [(g, _obs_idx0[f"c_{g}"]) for g in contact_groups if f"c_{g}" in _obs_idx0]
    ec_cols = [(g, _obs_idx0[f"env_c_{g}"]) for g in env_contact_groups if f"env_c_{g}" in _obs_idx0]

    def _force_rows(cols):
        rows = []
        for i in range(n):
            m = active == i
            row: dict[str, Any] = {"stage": names[i]}
            maxes = []
            for g, ix in cols:
                v = arr[:, :, ix]
                row[g] = {
                    "mean": _masked(v, m),
                    "max": round(float(v[m].max()), 4) if m.any() else None,
                }
                if row[g]["max"] is not None:
                    maxes.append(row[g]["max"])
            row["max_any_region"] = max(maxes) if maxes else None
            rows.append(row)
        return rows

    contact_forces = {
        "units": "N",
        "object_contact": {"regions": [g for g, _ix in c_cols], "per_stage": _force_rows(c_cols)},
        "environment_contact": {"regions": [g for g, _ix in ec_cols], "per_stage": _force_rows(ec_cols)},
    }

    ts = stage_transition_stats(active, n)
    dwell, entered, handoff, reverse = ts["dwell"], ts["entered"], ts["handoff"], ts["reverse"]
    raw_ts = stage_transition_stats(raw_active, n) if progression == "monotone" else None
    conversion = [(round(handoff[i] / max(reached[i], 1e-9), 4) if handoff[i] is not None else None)
                  for i in range(n)]

    # Authored-success evaluation: expr > 0 for >= dwell consecutive steps, starting at/after the
    # stage's first dominance (the episode suffix from entry). Cross-check vs the hand-off
    # predicate: disagreement in either direction is the DOMINANCE-SUCCESS DISCREPANCY signal.
    authored = [None] * n
    if any(ev is not None for ev in succ_evs):
        sb = succ_te > 0.0
        cpad2 = _np.concatenate([_np.zeros((1, E, n), int), _np.cumsum(sb, axis=0)], axis=0)
        runs2 = (cpad2[dwell:] - cpad2[:-dwell]) >= dwell
        jidx2 = _np.arange(runs2.shape[0])[:, None, None]
        ok = (runs2 & (jidx2 >= ts["first_occ"][None, :, :])).any(axis=0)   # [E,n]
        for i in range(n):
            if succ_evs[i] is not None:
                authored[i] = float(ok[:, i].mean())
    discrepancy = [None] * n
    for i in range(n):
        if authored[i] is None:
            continue
        if handoff[i] is not None:
            if handoff[i] >= 0.1 and authored[i] < 0.05:
                discrepancy[i] = "handoff_without_success"
            elif authored[i] >= 0.1 and handoff[i] < 0.05:
                discrepancy[i] = "success_without_handoff"
        elif entered[i] >= 0.1 and authored[i] < 0.05:  # terminal stage: entered but no success
            discrepancy[i] = "entered_without_success"

    # Stall = the FIRST broken hand-off: the shallowest stage reliably (dwell-)entered whose
    # successor rarely gets a dwell-qualified run. The chain completes when no hand-off is broken.
    # A SKIPPED stage (never entered while DEEPER stages run) is NOT a stall: which stage wins at
    # t=0 is gate arithmetic, and directing revisions to "fix" a skipped entry stage of a working
    # chain destroys the working behavior (observed: agentic_v3 burned its refine budget that way).
    # Skipped stages are reported separately; only an entirely-dead chain (nothing ever entered)
    # localizes to stage 0's entry.
    reach_thresh, handoff_thresh = 0.1, 0.1
    stall = None
    for i in range(n - 1):
        if entered[i] >= reach_thresh and (handoff[i] or 0.0) < handoff_thresh:
            stall = i
            break
    if stall is None and max(entered) < reach_thresh:
        stall = 0  # no stage ever reliably occupies; the chain is dead from the start
    reaches_terminal = stall is None and (n == 1 or entered[n - 1] >= reach_thresh)
    deepest = max((i for i in range(n) if entered[i] >= reach_thresh), default=-1)
    skipped = [i for i in range(deepest) if entered[i] < reach_thresh]
    if stall is None and not reaches_terminal:
        stall = n - 2 if n > 1 else None  # defensive; shouldn't occur with the rules above
    # Endgame leak analysis: when the chain completes, the improvement target is the WEAKEST
    # hand-off (lowest conversion), so stage-localized pressure survives past the last stall.
    weakest = None
    if reaches_terminal and n > 1:
        weakest = int(_np.argmin([h if h is not None else 1.0 for h in handoff[:-1]]))

    # TIMING report: how long each stage actually occupies the rollout vs the author's estimate, so
    # the reviser can tell "this hand-off never fires" (broken) apart from "this stage just runs too
    # long and the rollout ends inside it" (slow -- speed it up / loosen its tolerance, don't rewrite
    # the gate). Per-stage measured seconds = mean over episodes of dominant-step count * control_dt.
    dt_s = float(getattr(env.cfg, "control_dt", 1.0))
    episode_seconds = round(T * dt_s, 2)
    active_counts = _np.stack([(active == i).sum(axis=0) for i in range(n)], axis=1)  # [E, n]
    measured_seconds = [round(float(active_counts[:, i].mean()) * dt_s, 2) for i in range(n)]
    ended_in = active[-1]                                           # [E] stage active at the last step
    ended_in_frac = [round(float((ended_in == i).mean()), 4) for i in range(n)]
    est_seconds = []
    for st in stages:
        v = st.get("est_seconds", st.get("expected_seconds"))
        est_seconds.append(float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None)
    have_est = any(x is not None for x in est_seconds)
    total_est = round(sum(x for x in est_seconds if x is not None), 2) if have_est else None
    # Per-stage OVERRUN ratio (measured / author estimate): the reviser's clearest signal that a
    # stage is far slower than intended -- a high ratio on a stage that DID run means its channel is
    # crawling asymptotically toward its target, not that its gate is broken. Only meaningful where
    # the stage was actually reached (measured > 0) and the author gave an estimate.
    measured_vs_est = [
        (round(measured_seconds[i] / est_seconds[i], 1)
         if (est_seconds[i] and est_seconds[i] > 0 and measured_seconds[i] > 0) else None)
        for i in range(n)]
    worst_overrun = None
    cand = [(measured_vs_est[i], i) for i in range(n) if measured_vs_est[i] is not None]
    if cand:
        r, i = max(cand)
        if r >= 1.5:
            worst_overrun = {"stage": i, "name": names[i], "ratio": r,
                             "measured_seconds": measured_seconds[i], "est_seconds": est_seconds[i]}
    time_report = {
        "rollout_seconds": episode_seconds,
        "per_stage_measured_seconds": measured_seconds,
        "ended_in_stage_frac": ended_in_frac,
        "authored_est_seconds": est_seconds,
        "authored_total_est_seconds": total_est,
        "fits_rollout": (bool(total_est <= episode_seconds) if total_est is not None else None),
        "measured_vs_est_ratio": measured_vs_est,
        "worst_overrun": worst_overrun,
    }
    # Time-limited stall: the flagged hand-off is likely SLOW not broken when the episode still ends
    # inside the stall stage (or shallower) in a large fraction of runs, or the author's own estimate
    # over-runs the budget. Surfaced so the reviser speeds stages up instead of rewriting a fine gate.
    if stall is not None:
        ended_at_or_before_stall = round(float((ended_in <= stall).mean()), 4)
        time_report["stall_measured_seconds"] = measured_seconds[stall]
        time_report["stall_ends_within_frac"] = ended_at_or_before_stall
        time_report["stall_time_limited"] = bool(
            ended_at_or_before_stall >= 0.4 or (total_est is not None and total_est > episode_seconds))

    out = {
        "stage_names": names,
        "stage_progression": progression,
        "occupancy": [round(x, 4) for x in occupancy],
        "reached_frac": [round(x, 4) for x in reached],
        "dwell": int(dwell),
        "entered_frac": [round(x, 4) for x in entered],
        "handoff_frac": [round(x, 4) if x is not None else None for x in handoff],
        "conversion": conversion,
        "reverse_frac": [round(x, 4) if x is not None else None for x in reverse],
        **({} if raw_ts is None else {
            "raw_gate_reverse_frac": [
                round(x, 4) if x is not None else None for x in raw_ts["reverse"]
            ],
            "cursor_blocked_regression_frac": round(float((raw_active < active).mean()), 4),
        }),
        "stall_stage": stall,
        "stall_name": (names[stall] if stall is not None else None),
        "reaches_terminal": bool(reaches_terminal),
        "skipped_entry_stages": skipped,
        "weakest_stage": weakest,
        "weakest_name": (names[weakest] if weakest is not None else None),
        "authored_success_frac": [round(x, 4) if x is not None else None for x in authored],
        "success_discrepancy": discrepancy,
        "time_report": time_report,
        "body_motion": body_motion,
        "contact_forces": contact_forces,
        "commanded_motion": {"units": "ctrl-units per second, |delta ctrl|/dt, mean over steps "
                                      "where the stage is dominant",
                             "per_stage": commanded_motion},
        **({} if failure is None
           else {"failure_attribution": _failure_attribution(failure, active, names, n)}),
    }
    if succ_errors:
        out["success_compile_errors"] = succ_errors

    # LLM-authored diagnostic PROBES: named expressions the author asked the framework to measure
    # on the trained policy's visited states (optionally masked to one stage). Evaluated offline
    # over the whole [T, E] episode array, so the vocabulary adds the episode-relative
    # PROBE_EXTRA_SIGNALS. A bad probe is reported, never fatal.
    probes = list(program.get("probes") or [])[:MAX_PROBES]
    if probes:
        probe_env = {s: sig_te[:, :, j] for j, s in enumerate(sig_keys)}
        probe_env["obj_disp_xy"] = obj_disp_xy
        probe_env["obj_speed"] = obj_speed
        probe_names = set(signal_names) | set(PROBE_EXTRA_SIGNALS)
        phalf = max(T // 2, 1)
        preport: dict[str, dict] = {}
        for p in probes:
            pname = str(p.get("name") or p.get("expr") or f"probe{len(preport)}")
            entry: dict[str, Any] = {"expr": str(p.get("expr", "")), "stage": p.get("stage")}
            try:
                ev = compile_expr(str(p.get("expr", "0")), probe_names)
                val = _np.broadcast_to(_np.asarray(ev(probe_env), dtype=_np.float64), (T, E))
                st = p.get("stage")
                if st is None:
                    pm = _np.ones((T, E), bool)
                else:
                    si = names.index(str(st)) if str(st) in names else int(st)
                    pm = active == si

                def _mm(sl, pm=pm, val=val):
                    m = pm[sl]
                    return round(float((val[sl] * m).sum() / m.sum()), 4) if m.sum() else None

                entry.update({
                    "early_late_mean": [_mm(slice(0, phalf)), _mm(slice(phalf, None))],
                    "mean": _mm(slice(None)),
                    "min": round(float(val[pm].min()), 4) if pm.any() else None,
                    "max": round(float(val[pm].max()), 4) if pm.any() else None,
                })
            except Exception as e:  # noqa: BLE001 -- a bad probe must not kill the eval
                entry["error"] = str(e)
            preport[pname] = entry
        out["probe_report"] = preport

    # LLM-authored EVALS: named acceptance tests over the same vocabulary as probes. Each is
    # {name, expr, when: "ever"|"end"} scored per EPISODE as pass/fail -- "ever": expr > 0 holds
    # for >= dwell consecutive steps anywhere in the episode; "end": expr > 0 holds over the final
    # dwell steps. pass_frac feeds the revision loop's tie-break acceptance (an authored-eval
    # improvement can carry a revision whose objective is within noise, never one that regresses).
    # A bad eval is reported, never fatal.
    authored_evals = list(program.get("evals") or [])[:MAX_EVALS]
    if authored_evals:
        eval_env = {s: sig_te[:, :, j] for j, s in enumerate(sig_keys)}
        eval_env["obj_disp_xy"] = obj_disp_xy
        eval_env["obj_speed"] = obj_speed
        allowed = set(signal_names) | set(PROBE_EXTRA_SIGNALS)
        dwell_e = min(3, T)
        ereport: dict[str, dict] = {}
        for it in authored_evals:
            ename = str(it.get("name") or it.get("expr") or f"eval{len(ereport)}")
            when = str(it.get("when", "ever"))
            entry: dict[str, Any] = {"expr": str(it.get("expr", "")), "when": when}
            try:
                ev = compile_expr(str(it.get("expr", "0")), allowed)
                val = _np.broadcast_to(_np.asarray(ev(eval_env), dtype=_np.float64), (T, E))
                pos = val > 0.0
                if when == "end":
                    passed = pos[-dwell_e:].all(axis=0)
                else:
                    c = _np.concatenate([_np.zeros((1, E), int), _np.cumsum(pos, axis=0)], axis=0)
                    passed = ((c[dwell_e:] - c[:-dwell_e]) >= dwell_e).any(axis=0)
                entry["pass_frac"] = round(float(passed.mean()), 4)
            except Exception as e:  # noqa: BLE001 -- a bad eval must not kill the evaluation
                entry["error"] = str(e)
            ereport[ename] = entry
        out["eval_report"] = ereport

    # Direction evidence for the FOCUS stage: the stall, or (endgame) the weakest hand-off.
    focus = stall if stall is not None else weakest
    if focus is None or focus >= n - 1:
        return out
    k, nk = focus, focus + 1
    mask = active == k
    half = max(T // 2, 1)

    def _early_late(v):  # masked means over episode halves; None where the stage is never active
        pair = []
        for sl in (slice(0, half), slice(half, None)):
            m = mask[sl]
            c = m.sum()
            pair.append(round(float((v[sl] * m).sum() / c), 4) if c else None)
        return pair

    # Intent vs execution + command vs measured, restricted to steps where the focus stage is
    # dominant. "stage_cmd" = what the focus stage's own channels command; "executed" = the blended
    # prior actually applied (dilution/cancellation across stages shows up as a gap). "tracking" =
    # commanded ctrl target vs the measured joint position, as a fraction of the actuator's range
    # (a persistent gap = the joint is saturated, blocked by contact, or physically stopped).
    msum = int(mask.sum())
    if msum:
        act_names_all = [env.model.actuator(i).name for i in range(int(env.nu))]
        intended = act_te[:, :, k, :]
        executed = _np.clip((w_te[:, :, :, None] * act_te).sum(axis=2), -1.0, 1.0)
        mi = (intended * mask[:, :, None]).sum(axis=(0, 1)) / msum
        me = (executed * mask[:, :, None]).sum(axis=(0, 1)) / msum
        gap = (_np.abs(executed - intended) * mask[:, :, None]).sum(axis=(0, 1)) / msum
        rows_ie = [{"actuator": act_names_all[a], "stage_cmd": round(float(mi[a]), 3),
                    "executed": round(float(me[a]), 3), "gap": round(float(gap[a]), 3)}
                   for a in range(action_dim) if abs(mi[a]) >= 0.02 or gap[a] >= 0.02]
        rows_ie.sort(key=lambda r: -r["gap"])
        out["intent_execution"] = rows_ie[:8]
        qmap = actuator_obs_qpos_map(env)
        lo = _np.asarray(env.ctrl_lo, dtype=_np.float64)
        hi = _np.asarray(env.ctrl_hi, dtype=_np.float64)
        ctrl = arr[:, :, -action_dim:]
        rows_tr = []
        for a in range(action_dim):
            if qmap[a] is None:
                continue
            err = _np.abs(ctrl[:, :, a] - arr[:, :, qmap[a]]) / max(float(hi[a] - lo[a]), 1e-6)
            rows_tr.append({"actuator": act_names_all[a],
                            "cmd_vs_measured_frac_of_range":
                                round(float((err * mask).sum() / msum), 3)})
        rows_tr.sort(key=lambda r: -r["cmd_vs_measured_frac_of_range"])
        out["tracking"] = rows_tr[:6]

    next_sigs = [s for s in sig_keys if re.search(rf"\b{re.escape(s)}\b", gate_exprs[nk])]
    # Trend table restricted to signals the PROGRAM references (authored signals + observables it
    # actually uses) -- with per-candidate vocabularies the full observable set would flood the
    # revision directive.
    prog_text = str(program.get("stages", [])) + str(program.get("signals") or {})
    ref_keys = [s for s in sig_keys if re.search(rf"\b{re.escape(s)}\b", prog_text)]
    out.update({
        "stall_signal_trend": {s: _early_late(sig_te[:, :, sig_keys.index(s)]) for s in ref_keys},
        "stall_gate": {"expr": gate_exprs[k], "value_early_late": _early_late(gates_te[:, :, k])},
        "next_gate": {"index": nk, "name": names[nk], "expr": gate_exprs[nk],
                      "signals": next_sigs, "value_early_late": _early_late(gates_te[:, :, nk])},
        "self_lock": bool(occupancy[k] >= 0.95 and reached[nk] <= 0.05),
    })
    return out

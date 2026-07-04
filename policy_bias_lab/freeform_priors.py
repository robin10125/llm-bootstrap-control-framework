"""Free-form symbolic action priors + robot-spec extraction + DOF-completeness.

For the preliminary experiment comparing a constrained robot-derived DSL against a free-form
symbolic representation (see PRELIM_dsl_vs_freeform.md). A free-form candidate gives, per channel
(a set of named actuators), a symbolic EXPRESSION for the mean-shift over observable signals; a
restricted AST evaluator compiles it into a stateless JAX ``fn(obs, weights) -> mean_shift`` (same
contract as composed_priors, so PPO/the scorer use it unchanged).

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
from typing import Any, Callable

import jax
import jax.numpy as jp
import numpy as np

from policy_bias_lab.composed_priors import make_signal_fn, _calibration, _gates
from policy_bias_lab.symbolic_control import group_masks, make_rule_action_fn, encode_rules
from policy_bias_lab.schema import ACTION_GROUPS

# ----------------------------------------------------------------------------------------------
# Robot spec (all DOF enumerated)
# ----------------------------------------------------------------------------------------------

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
    actuators = []
    for i, n in enumerate(names):
        jid = int(m.actuator_trnid[i, 0])
        actuators.append({
            "name": n,
            "index": i,
            "joint": m.joint(jid).name if jid >= 0 else None,
            "ctrlrange": [float(cr[i, 0]), float(cr[i, 1])],
            "group": _semantic_group(n),
        })
    return {
        "n_actuators": env.nu,
        "actuators": actuators,
        "control_law": {
            "type": "incremental_position",
            "rule": "target = current_ctrl + action * action_scale",
            "action_scale": cal["action_scale"],
            "action_range": [-1.0, 1.0],
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
    """Set of actuator names a candidate touches (DSL rules or free-form channels)."""
    names = all_actuator_names(env)
    idx: set[int] = set()
    if candidate.get("mode") == "freeform":
        for ch in candidate.get("channels", []):
            idx.update(_resolve_actuators(env, ch.get("actuators", [])))
    elif candidate.get("mode") == "freeform_staged":
        for st in candidate.get("stages", []):
            for ch in st.get("channels", []):
                idx.update(_resolve_actuators(env, ch.get("actuators", [])))
    else:  # DSL: rules with a 'group'
        rules = candidate.get("rules", [])
        for sub in candidate.get("library", []):  # gated library form
            rules = rules + list(sub.get("rules", []))
        for r in rules:
            idx.update(_resolve_actuators(env, [r.get("group", "")]))
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

_ALLOWED_FUNCS = {"clip", "sigmoid", "min", "max", "abs", "exp"}


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
        if f == "min": return jp.minimum(a[0], a[1])
        if f == "max": return jp.maximum(a[0], a[1])
        if f == "abs": return jp.abs(a[0])
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
    channels = list(program.get("channels", []))
    compiled = []
    for ch in channels:
        idx = _resolve_actuators(env, ch.get("actuators", []))
        ev = compile_expr(ch.get("expr", "0"), signal_names)
        compiled.append((jp.asarray(idx, dtype=jp.int32), ev))
    n_ch = len(compiled)

    def prior_fn(obs, weights):
        gains = jp.ones((n_ch,), jp.float32) if weights is None else jp.asarray(weights, jp.float32)[:n_ch]
        sig = signals(obs)
        out = jp.zeros((action_dim,), jp.float32)
        for i, (idx, ev) in enumerate(compiled):
            val = jp.clip(ev(sig), -1.0, 1.0) * gains[i]
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

    DSL form: {"rules": [{kind:"operator", group, direction, weight} | {kind:"basis", group, sign,
    weight}]}. Operator rules use the calibrated symbolic operators; basis rules drive a group's
    actuators toward +/- ctrl (the robot-derived motion basis -- makes any DOF, incl. the wrist,
    movable). Free-form form: {"channels": [{actuators, expr}]}.
    """
    action_dim = int(env.action_size)
    if "channels" in spec:
        fn, _dw, _info = make_freeform_prior_fn(env, {"channels": spec["channels"]})
        return lambda obs: fn(obs, None)
    rules = list(spec.get("rules", []))
    op_rules = [r for r in rules if r.get("kind", "operator") == "operator" and r.get("direction")]
    basis_rules = [r for r in rules if r.get("kind") == "basis"]
    rule_action_fn, _ = make_rule_action_fn(env)
    enc = None
    if op_rules:
        e = encode_rules([{ "group": r.get("group", "all"), "direction": r["direction"],
                            "weight": float(r.get("weight", 0.0))} for r in op_rules])
        enc = (jp.asarray(e["group_ids"], jp.int32), jp.asarray(e["direction_ids"], jp.int32),
               jp.asarray(e["weights"], jp.float32))
    basis = [(jp.asarray(_resolve_actuators(env, [r.get("group", "")]), jp.int32),
              float(r.get("sign", 1.0)) * float(r.get("weight", 0.0))) for r in basis_rules]

    def fn(obs):
        out = jp.zeros((action_dim,), jp.float32)
        if enc is not None:
            out = out + rule_action_fn(obs, enc[0], enc[1], enc[2])
        for idx, val in basis:
            out = out.at[idx].add(val)
        return out

    return fn


def make_stacked_prior_fn(env: Any, program: dict[str, Any]):
    """Compile a stacked-gate program {subpriors:[approach, grasp, lift], gates:{}} into
    (prior_fn(obs, weights)->mean_shift, default_weights, info). Each sub-prior may be DSL or
    free-form. The stacked gate (composed_priors._gates) blends them on observable signals."""
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


DEFAULT_BLEND = "softmax"
DEFAULT_BLEND_TAU = 0.1


def blend_weights(gates: jp.ndarray, blend: str, tau: float = DEFAULT_BLEND_TAU) -> jp.ndarray:
    """Normalized stage weights from raw gate values -- the ONE place the blend semantics live,
    shared by the prior itself, make_stage_weight_fn, and stage_occupancy so the executed mixture
    and every diagnostic see identical weights.

    softmax (default): sharp but continuous hand-offs. Invariant to a shared additive gate offset,
    so authored constant gate floors cannot leak permanent weight to every stage (the relu-norm
    failure mode where a 4-stage program degenerates into one static averaged pose).
    soft: legacy relu(gate)/sum. hard: one-hot on argmax (ties split evenly).
    """
    if blend == "hard":
        top = (gates >= jp.max(gates)).astype(jp.float32)
        return top / (jp.sum(top) + 1e-8)
    if blend == "soft":
        pos = jp.maximum(gates, 0.0)
        return pos / (jp.sum(pos) + 1e-8)
    return jax.nn.softmax(gates / jp.maximum(tau, 1e-4))


def _blend_of(program: dict[str, Any]) -> tuple[str, float]:
    return (str(program.get("blend", DEFAULT_BLEND)),
            float(program.get("temperature", DEFAULT_BLEND_TAU)))


def make_freeform_staged_prior_fn(env: Any, program: dict[str, Any]):
    """Fully free-form MULTI-STAGE prior: the AUTHOR defines N stages, each with its own free-form
    GATE expression (stage activation, a function of observable signals) AND free-form channels.

    Generalizes the fixed 3-phase near/gripped stacked gate: arbitrary stage count and author-defined
    transitions, expressed symbolically over the same signals as the channels. Stages are combined
    STATELESSLY -- pure functions of the CURRENT signals, no phase latch/pointer/hysteresis (PPO
    recomputes the prior on shuffled minibatches, so any hidden state breaks the importance ratio).

    program = {"mode":"freeform_staged", "blend":"softmax"|"soft"|"hard", "temperature": <tau>,
               "stages":[{"name", "gate":"<expr>", "channels":[{actuators, expr}, ...]}, ...]}
    Blend semantics live in blend_weights(); softmax (sharp, continuous) is the default.
    Per-stage gains are the tunable `weights` vector (default ones).
    """
    action_dim = int(env.action_size)
    signals, signal_names = program_signal_fn(env, program)
    blend, tau = _blend_of(program)
    stages = list(program.get("stages", []))
    compiled: list[tuple[Callable, list[tuple[jp.ndarray, Callable]]]] = []
    for st in stages:
        gate_ev = compile_expr(str(st.get("gate", "1")), signal_names)
        chans = [(jp.asarray(_resolve_actuators(env, ch.get("actuators", [])), dtype=jp.int32),
                  compile_expr(str(ch.get("expr", "0")), signal_names))
                 for ch in st.get("channels", [])]
        compiled.append((gate_ev, chans))
    n = len(compiled)

    def _stage_out(chans, sig):
        out = jp.zeros((action_dim,), jp.float32)
        for idx, ev in chans:
            out = out.at[idx].add(jp.clip(ev(sig), -1.0, 1.0))
        return out

    def prior_fn(obs, weights):
        gains = jp.ones((n,), jp.float32) if weights is None else jp.asarray(weights, jp.float32)[:n]
        sig = signals(obs)
        gates = jp.stack([g(sig) for g, _ in compiled]) if n else jp.zeros((0,), jp.float32)
        gates = jp.asarray(gates, jp.float32)
        w = blend_weights(gates, blend, tau)
        out = jp.zeros((action_dim,), jp.float32)
        for i, (_g, chans) in enumerate(compiled):
            out = out + (w[i] * gains[i]) * _stage_out(chans, sig)
        return jp.clip(out, -1.0, 1.0)

    info = {"mode": "freeform_staged", "blend": blend, "n_stages": n, "n_weights": n,
            "stage_names": [str(s.get("name", f"stage_{i}")) for i, s in enumerate(stages)]}
    return prior_fn, jp.ones((n,), jp.float32), info


def make_stage_weight_fn(env: Any, program: dict[str, Any]):
    """Expose ONLY the per-stage activation weights of a freeform_staged program (not the action).

    Returns (weight_fn, stage_names). weight_fn(obs) -> w[n_stages], the SAME soft/hard-normalized
    stage weights make_freeform_staged_prior_fn uses internally. Used by the diagnostic to localize
    WHICH authored stage a trained policy stalls in -- task-agnostic, since it reads only the model's
    own gates over the shared signals.
    """
    signals, signal_names = program_signal_fn(env, program)
    blend, tau = _blend_of(program)
    stages = list(program.get("stages", []))
    gate_evs = [compile_expr(str(st.get("gate", "1")), signal_names) for st in stages]
    n = len(gate_evs)
    names = [str(s.get("name", f"stage_{i}")) for i, s in enumerate(stages)]

    def weight_fn(obs):
        sig = signals(obs)
        gates = jp.stack([g(sig) for g in gate_evs]) if n else jp.zeros((0,), jp.float32)
        return blend_weights(jp.asarray(gates, jp.float32), blend, tau)

    return weight_fn, names


# Extra signals available ONLY to diagnostic probes (evaluated offline over whole [T, E] episode
# arrays, so per-episode baselines are allowed -- unlike gate/channel signals, which must stay
# stateless functions of the current obs for PPO). Generic rigid-body quantities, no task nouns.
PROBE_EXTRA_SIGNALS = {
    "obj_disp_xy": "object's horizontal distance from its own position at the episode start (m)",
    "obj_speed": "object's translational speed (m/s)",
}

MAX_PROBES = 8  # measurement slots per candidate (cap keeps prompts bounded, not a hard budget)


def raw_signal_fn(env: Any) -> tuple[Callable, set[str], str]:
    """RAW observables only, mechanically enumerated from the env/robot: world positions and
    velocities present in obs, commanded actuator targets, and measured joint positions.

    NO derived quantities live here -- distances, proximity gates, closure fractions, alignment
    measures are TASK STRUCTURE and must be LLM-authored per candidate (`signals` on the program;
    see AGENTS.md). Returns (signals(obs)->dict, names, doc) where doc is the injectable
    description block for prompts.
    """
    _base, info = make_signal_fn(env)  # obs-layout indices only
    nb = len(env.base_qadr)
    n_pre = int(info["n_pre"])
    nu = int(env.nu)
    m = env.model
    act_names = [m.actuator(i).name for i in range(nu)]
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
    no_q = []
    for i, an in enumerate(act_names):
        if qmap[i] is not None:
            entries.append((f"q_{an}", int(qmap[i])))
        else:
            no_q.append(an)
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
        "  ctrl_<actuator>: current commanded target of that actuator (its ctrl units/range)\n"
        "  q_<actuator>: MEASURED position of the joint that actuator drives (differs from "
        "ctrl_<actuator> when the joint is blocked or saturated)"
        + (f"; q_ not available for tendon-coupled: {', '.join(no_q)}" if no_q else "")
    )
    return signals, set(names), doc


def program_signal_fn(env: Any, program: dict[str, Any]) -> tuple[Callable, set[str]]:
    """The signal vocabulary for ONE program: raw observables + the program's own authored
    `signals: {name: expr}` (evaluated in insertion order; later signals may reference earlier
    ones). Programs WITHOUT authored signals get the legacy derived vocabulary as well, so
    archived candidates replay unchanged -- new candidates are expected to author their own."""
    raw_fn, raw_names, _doc = raw_signal_fn(env)
    authored = program.get("signals")
    if isinstance(authored, dict) and authored:
        compiled: list[tuple[str, Callable]] = []
        avail = set(raw_names)
        for name, expr in authored.items():
            ev = compile_expr(str(expr), avail)  # raises -> validation rejects the candidate
            compiled.append((str(name), ev))
            avail.add(str(name))

        def signals(obs: jp.ndarray) -> dict[str, jp.ndarray]:
            sig = raw_fn(obs)
            for name, ev in compiled:
                sig[name] = jp.asarray(ev(sig), jp.float32)
            return sig

        return signals, avail
    legacy_fn, _legacy_names = freeform_signal_fn(env)
    legacy_names = {"palm_obj_dist", "closure", "lift", "obj_rel_x", "obj_rel_y", "obj_rel_z",
                    "near", "gripped"}

    def signals(obs: jp.ndarray) -> dict[str, jp.ndarray]:
        sig = raw_fn(obs)
        sig.update(legacy_fn(obs))
        return sig

    return signals, raw_names | legacy_names


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


def stage_occupancy(env: Any, program: dict[str, Any], obs: "np.ndarray") -> dict:
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
    stage and its successor over the same split, and a hard-blend self-lock flag (stall occupies
    ~all steps and the successor is ~never reached => the argmax gate never yields).
    Purely structural -- no task labels -- so it generalizes to any staged prior on any task.
    """
    import re
    import numpy as _np

    signals, signal_names = program_signal_fn(env, program)
    sig_keys = sorted(signal_names)
    blend, tau = _blend_of(program)
    stages = list(program.get("stages", []))
    names = [str(s.get("name", f"stage_{i}")) for i, s in enumerate(stages)]
    gate_exprs = [str(st.get("gate", "1")) for st in stages]
    gate_evs = [compile_expr(g, signal_names) for g in gate_exprs]
    n = len(names)
    if n == 0:
        return {"stage_names": [], "occupancy": [], "reached_frac": [], "stall_stage": None}
    action_dim = int(env.action_size)
    chan_evs = [[(jp.asarray(_resolve_actuators(env, ch.get("actuators", [])), dtype=jp.int32),
                  compile_expr(str(ch.get("expr", "0")), signal_names))
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
        w = blend_weights(gates, blend, tau)
        succ = jp.stack([(jp.asarray(ev(sig), jp.float32) if ev is not None
                          else jp.asarray(0.0, jp.float32)) for ev in succ_evs])
        stage_act = []
        for chans in chan_evs:
            sa = jp.zeros((action_dim,), jp.float32)
            for idx, ev in chans:
                sa = sa.at[idx].add(jp.clip(ev(sig), -1.0, 1.0))
            stage_act.append(sa)
        return (w, gates, jp.stack([jp.asarray(sig[k], jp.float32) for k in sig_keys]), succ,
                jp.stack(stage_act))

    arr = _np.asarray(obs)
    T, E = arr.shape[0], arr.shape[1]
    flat = arr.reshape(T * E, -1)
    w, gates, sigv, succv, actv = (
        _np.asarray(x) for x in jax.jit(jax.vmap(_all))(jp.asarray(flat)))
    active = w.argmax(axis=1).reshape(T, E)                          # dominant stage per step
    w_te = w.reshape(T, E, n)
    gates_te = gates.reshape(T, E, n)
    sig_te = sigv.reshape(T, E, len(sig_keys))
    succ_te = succv.reshape(T, E, n)
    act_te = actv.reshape(T, E, n, action_dim)                       # per-stage channel intents
    occupancy = [float((active == i).mean()) for i in range(n)]      # time-share
    reached = [float(((active == i).any(axis=0)).mean()) for i in range(n)]  # episodes reaching i

    ts = stage_transition_stats(active, n)
    dwell, entered, handoff, reverse = ts["dwell"], ts["entered"], ts["handoff"], ts["reverse"]
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

    out = {
        "stage_names": names,
        "occupancy": [round(x, 4) for x in occupancy],
        "reached_frac": [round(x, 4) for x in reached],
        "dwell": int(dwell),
        "entered_frac": [round(x, 4) for x in entered],
        "handoff_frac": [round(x, 4) if x is not None else None for x in handoff],
        "conversion": conversion,
        "reverse_frac": [round(x, 4) if x is not None else None for x in reverse],
        "stall_stage": stall,
        "stall_name": (names[stall] if stall is not None else None),
        "reaches_terminal": bool(reaches_terminal),
        "skipped_entry_stages": skipped,
        "weakest_stage": weakest,
        "weakest_name": (names[weakest] if weakest is not None else None),
        "authored_success_frac": [round(x, 4) if x is not None else None for x in authored],
        "success_discrepancy": discrepancy,
    }
    if succ_errors:
        out["success_compile_errors"] = succ_errors

    # LLM-authored diagnostic PROBES: named expressions the author asked the framework to measure
    # on the trained policy's visited states (optionally masked to one stage). Evaluated offline
    # over the whole [T, E] episode array, so the vocabulary adds the episode-relative
    # PROBE_EXTRA_SIGNALS. A bad probe is reported, never fatal.
    probes = list(program.get("probes") or [])[:MAX_PROBES]
    if probes:
        _sigfn, _info = make_signal_fn(env)
        n_pre = int(_info["n_pre"])
        obj_pos = arr[:, :, n_pre:n_pre + 3]
        obj_vel = arr[:, :, n_pre + 3:n_pre + 6]
        probe_env = {s: sig_te[:, :, j] for j, s in enumerate(sig_keys)}
        probe_env["obj_disp_xy"] = _np.linalg.norm(obj_pos[:, :, :2] - obj_pos[:1, :, :2], axis=-1)
        probe_env["obj_speed"] = _np.linalg.norm(obj_vel, axis=-1)
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
        "self_lock": bool(blend in ("hard", "softmax") and occupancy[k] >= 0.95 and reached[nk] <= 0.05),
    })
    return out

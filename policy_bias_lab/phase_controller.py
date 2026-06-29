"""Closed-loop, curriculum-structured phase controller (the "teacher").

A static action prior is one constant mean-shift vector; it cannot express a task that
requires "do A until a condition holds, *then* do B". This module provides a small
state machine over the same symbolic action vocabulary (see symbolic_control), where each
PHASE has its own set of weak rules and advances to the next phase when a GATE condition on
observable signals is met. For a grasp-and-lift task the natural curriculum is
approach -> (contact) -> close -> (closure) -> lift, but the structure is task-agnostic.

Why this is portable and real-world usable
------------------------------------------
* Tasks/robots: a program is plain data (phases of {group, direction, weight} rules plus
  {signal, op, value} gates). Gates reference observable signals by NAME, resolved against a
  task-specific field index, so any task that defines its own observable fields and action
  groups works without code changes.
* Real-world: phase advancement is driven only by signals a physical robot can sense
  (e.g. fingertip contact count, object height, lateral drift) -- never privileged simulator
  state. `PhaseController.step_single` is a pure per-step feedback function suitable for a
  real robot control loop; the JAX `rollout` is just its batched twin for sim validation.
* No clock-phasing: phases switch on achieved subgoals, not on a fixed timestep, so the same
  program transfers across object placements and reset distributions.

This module covers Phase A (the controller + staged validator + warm-start dataset). The
LLM authoring of programs and curriculum-aware selection among candidates is Phase B; see
PHASE_B_curriculum_aware_selection.md.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jp
import numpy as np

from policy_bias_lab.schema import ACTION_GROUPS, FIELD_INDEX, PRIOR_DIRECTIONS
from policy_bias_lab.symbolic_control import encode_rules, make_rule_action_fn, sustained_bool

# Comparison operators allowed in gate conditions, with stable integer codes for JAX.
GATE_OPS = (">=", "<=", ">", "<")
OP_CODE = {op: i for i, op in enumerate(GATE_OPS)}

MAX_RULES = 12
MAX_GATES = 4


# --------------------------------------------------------------------------------------
# Program schema + validation
# --------------------------------------------------------------------------------------
def default_phase_program(task: str) -> dict[str, Any]:
    """Hand-written fallback teacher. For 'lift' this is the approach->close->lift curriculum.

    Other tasks fall back to a single inert phase; real per-task programs come from the LLM
    (Phase B). Weights are conservative -- the teacher only needs to reach a good basin, not
    be optimal, and RL polishes it.
    """
    if task == "lift":
        # Tuned + validated against the real env config (see policy-bias-lab notes). Key facts
        # this encodes: (1) approach advances on PROXIMITY (palm_obj_dist), not contact -- the
        # hand is open/curling and cannot make contact until it closes, so a contact gate would
        # deadlock; (2) fingers PRE-CLOSE while descending (concurrent), because the base can only
        # descend ~0.17 m and the object sits ~0.19 m down, so there are too few steps to align
        # then close sequentially; (3) the base directions rely on the Jacobian calibration in
        # symbolic_control (this base body is world-inverted on y/z). Weights run up to 1.0: the
        # teacher needs full control authority, unlike weak exploration priors.
        return {
            "name": "approach_close_lift",
            "task": "lift",
            "phases": [
                {
                    "name": "approach",
                    "rules": [
                        {"group": "base_xy", "direction": "toward_object_xy", "weight": 0.60},
                        {"group": "base_z", "direction": "lower_base", "weight": 1.00},
                        {"group": "hand", "direction": "close_hand", "weight": 0.70},
                    ],
                    "advance_when": [{"signal": "palm_obj_dist", "op": "<=", "value": 0.06}],
                    "min_steps": 2,
                },
                {
                    "name": "close",
                    "rules": [
                        {"group": "hand", "direction": "close_hand", "weight": 1.00},
                        {"group": "base_xy", "direction": "toward_object_xy", "weight": 0.50},
                        {"group": "base_z", "direction": "lower_base", "weight": 1.00},
                    ],
                    "advance_when": [
                        {"signal": "n_contacts", "op": ">=", "value": 1.0},
                        {"signal": "closure", "op": ">=", "value": 0.25},
                    ],
                    "min_steps": 3,
                },
                {
                    "name": "lift",
                    "rules": [
                        {"group": "hand", "direction": "close_hand", "weight": 1.00},
                        {"group": "base_z", "direction": "raise_base", "weight": 0.60},
                        {"group": "base_xy", "direction": "toward_object_xy", "weight": 0.50},
                    ],
                    "advance_when": [],  # terminal
                    "min_steps": 0,
                },
            ],
        }
    return {
        "name": "inert_single_phase",
        "task": task,
        "phases": [
            {
                "name": "stabilize",
                "rules": [{"group": "all", "direction": "stabilize", "weight": 0.0}],
                "advance_when": [],
                "min_steps": 0,
            }
        ],
    }


def sanitize_phase_program(
    raw: Any,
    *,
    max_weight: float = 1.0,  # a teacher controller needs full authority (actions clip to [-1,1])
    field_index: dict[str, int] | None = None,
    task: str = "lift",
) -> dict[str, Any]:
    """Validate/clamp an LLM- or human-authored program; fall back to default on failure.

    Invalid rules/gates are dropped rather than raising, so a partially malformed program
    still yields a runnable teacher. Gates referencing unknown signals are skipped.
    """
    field_index = field_index or FIELD_INDEX
    if not isinstance(raw, dict) or not isinstance(raw.get("phases"), list) or not raw["phases"]:
        return default_phase_program(task)
    phases_out: list[dict[str, Any]] = []
    for p_idx, raw_phase in enumerate(raw["phases"]):
        if not isinstance(raw_phase, dict):
            continue
        rules_out: list[dict[str, Any]] = []
        for rule in raw_phase.get("rules", []) or []:
            if not isinstance(rule, dict):
                continue
            group = str(rule.get("group", ""))
            direction = str(rule.get("direction", ""))
            if group not in ACTION_GROUPS or direction not in PRIOR_DIRECTIONS:
                continue
            try:
                weight = float(rule.get("weight", 0.0))
            except (TypeError, ValueError):
                continue
            rules_out.append({
                "group": group,
                "direction": direction,
                "weight": float(min(max(weight, 0.0), max_weight)),
            })
        gates_out: list[dict[str, Any]] = []
        for gate in raw_phase.get("advance_when", []) or []:
            if not isinstance(gate, dict):
                continue
            signal = str(gate.get("signal", ""))
            op = str(gate.get("op", ""))
            if signal not in field_index or op not in OP_CODE:
                continue
            try:
                value = float(gate.get("value"))
            except (TypeError, ValueError):
                continue
            gates_out.append({"signal": signal, "op": op, "value": value})
        if not rules_out:
            rules_out = [{"group": "all", "direction": "stabilize", "weight": 0.0}]
        try:
            min_steps = max(0, int(raw_phase.get("min_steps", 0)))
        except (TypeError, ValueError):
            min_steps = 0
        phases_out.append({
            "name": str(raw_phase.get("name", f"phase_{p_idx}")),
            "rules": rules_out[:MAX_RULES],
            "advance_when": gates_out[:MAX_GATES],
            "min_steps": min_steps,
        })
    if not phases_out:
        return default_phase_program(task)
    return {"name": str(raw.get("name", "phase_program")), "task": str(raw.get("task", task)), "phases": phases_out}


@dataclass
class CompiledProgram:
    """Fixed-shape arrays for the JAX rollout, plus python-side metadata."""
    n_phases: int
    phase_names: list[str]
    group_ids: np.ndarray       # [P, MAX_RULES] int32
    direction_ids: np.ndarray   # [P, MAX_RULES] int32
    weights: np.ndarray         # [P, MAX_RULES] float32
    gate_signal: np.ndarray     # [P, MAX_GATES] int32
    gate_op: np.ndarray         # [P, MAX_GATES] int32
    gate_value: np.ndarray      # [P, MAX_GATES] float32
    gate_active: np.ndarray     # [P, MAX_GATES] float32 (1 real gate, 0 pad)
    min_steps: np.ndarray       # [P] int32
    program: dict[str, Any] = field(default_factory=dict)


def compile_phase_program(program: dict[str, Any], *, field_index: dict[str, int] | None = None) -> CompiledProgram:
    field_index = field_index or FIELD_INDEX
    phases = program["phases"]
    n = len(phases)
    group_ids = np.zeros((n, MAX_RULES), dtype=np.int32)
    direction_ids = np.full((n, MAX_RULES), PRIOR_DIRECTIONS.index("stabilize"), dtype=np.int32)
    weights = np.zeros((n, MAX_RULES), dtype=np.float32)
    gate_signal = np.zeros((n, MAX_GATES), dtype=np.int32)
    gate_op = np.zeros((n, MAX_GATES), dtype=np.int32)
    gate_value = np.zeros((n, MAX_GATES), dtype=np.float32)
    gate_active = np.zeros((n, MAX_GATES), dtype=np.float32)
    min_steps = np.zeros((n,), dtype=np.int32)
    for p_idx, phase in enumerate(phases):
        enc = encode_rules(phase["rules"], max_rules=MAX_RULES)
        group_ids[p_idx] = enc["group_ids"]
        direction_ids[p_idx] = enc["direction_ids"]
        weights[p_idx] = enc["weights"]
        for g_idx, gate in enumerate(phase["advance_when"][:MAX_GATES]):
            gate_signal[p_idx, g_idx] = field_index[gate["signal"]]
            gate_op[p_idx, g_idx] = OP_CODE[gate["op"]]
            gate_value[p_idx, g_idx] = float(gate["value"])
            gate_active[p_idx, g_idx] = 1.0
        min_steps[p_idx] = int(phase["min_steps"])
    return CompiledProgram(
        n_phases=n,
        phase_names=[str(p["name"]) for p in phases],
        group_ids=group_ids,
        direction_ids=direction_ids,
        weights=weights,
        gate_signal=gate_signal,
        gate_op=gate_op,
        gate_value=gate_value,
        gate_active=gate_active,
        min_steps=min_steps,
        program=program,
    )


def _gates_satisfied_np(signals: np.ndarray, sig_idx: np.ndarray, op: np.ndarray, val: np.ndarray, active: np.ndarray) -> bool:
    """Single-robot gate check (numpy). signals: [F]; the rest: [MAX_GATES]."""
    ok = True
    for g in range(sig_idx.shape[0]):
        if active[g] < 0.5:
            continue
        s = float(signals[int(sig_idx[g])])
        v = float(val[g])
        code = int(op[g])
        cond = (s >= v) if code == 0 else (s <= v) if code == 1 else (s > v) if code == 2 else (s < v)
        ok = ok and bool(cond)
    return ok


# --------------------------------------------------------------------------------------
# Controller
# --------------------------------------------------------------------------------------
class PhaseController:
    """Stateful symbolic teacher usable in (a) batched JAX sim rollouts, (b) BC dataset
    generation for the warm-start, and (c) a real per-step robot control loop."""

    def __init__(self, env: Any, program: dict[str, Any], *, field_index: dict[str, int] | None = None):
        self.env = env
        self.field_index = field_index or FIELD_INDEX
        self.program = program
        self.compiled = compile_phase_program(program, field_index=self.field_index)
        self.action_from_obs, info = make_rule_action_fn(env)
        self.action_dim = int(info["action_dim"])
        self.n_phases = self.compiled.n_phases
        self.phase_names = self.compiled.phase_names
        # Single-robot runtime state (for step_single).
        self._phase = 0
        self._steps_in_phase = 0

    # ---- batched sim rollout (validation + BC dataset) -------------------------------
    def _rollout_fn(self, envs: int, steps: int):
        c = self.compiled
        reset = jax.jit(lambda keys: jax.vmap(self.env.reset)(keys))
        step_fn = jax.vmap(self.env.step)
        gids = jp.asarray(c.group_ids)
        dids = jp.asarray(c.direction_ids)
        wts = jp.asarray(c.weights)
        g_sig = jp.asarray(c.gate_signal)
        g_op = jp.asarray(c.gate_op)
        g_val = jp.asarray(c.gate_value)
        g_act = jp.asarray(c.gate_active)
        min_steps = jp.asarray(c.min_steps)
        n_phases = c.n_phases
        action_from_obs = self.action_from_obs

        def gates_satisfied(signals_e, phase_e):  # signals_e: [F], phase_e: scalar
            sig_idx = g_sig[phase_e]
            sig_val = signals_e[sig_idx]
            v = g_val[phase_e]
            op = g_op[phase_e]
            ge = sig_val >= v
            le = sig_val <= v
            gt = sig_val > v
            lt = sig_val < v
            sat = jp.where(op == 0, ge, jp.where(op == 1, le, jp.where(op == 2, gt, lt)))
            sat = jp.where(g_act[phase_e] > 0.5, sat, True)
            return jp.all(sat)

        def rollout(key):
            state = reset(jax.random.split(key, int(envs)))
            phase0 = jp.zeros((int(envs),), dtype=jp.int32)
            steps0 = jp.zeros((int(envs),), dtype=jp.int32)

            def body(carry, _):
                state, phase, in_steps = carry
                signals = state.metrics["eval"]  # [E, F]
                sat = jax.vmap(gates_satisfied)(signals, phase)  # [E]
                min_ok = in_steps >= min_steps[phase]
                can = phase < (n_phases - 1)
                advance = sat & min_ok & can
                new_phase = phase + advance.astype(jp.int32)
                new_steps = jp.where(advance, 0, in_steps + 1)
                act = jax.vmap(action_from_obs)(state.obs, gids[new_phase], dids[new_phase], wts[new_phase])
                nstate = step_fn(state, act)
                out = (state.obs, act, nstate.reward, nstate.metrics["eval"], new_phase)
                return (nstate, new_phase, new_steps), out

            _final, traj = jax.lax.scan(body, (state, phase0, steps0), None, length=int(steps))
            return traj  # (obs[T,E,O], act[T,E,A], reward[T,E], eval[T,E,F], phase[T,E])

        return jax.jit(rollout)

    def rollout(self, key, *, envs: int, steps: int | None = None):
        steps = int(steps or self.env.horizon)
        return self._rollout_fn(envs, steps)(key)

    def bc_dataset(self, key, *, envs: int, gamma: float, steps: int | None = None):
        """Flattened (obs, target_action, discounted_return) over a teacher rollout, for the
        BC warm-start. Targets are the teacher's own actions on the states it visits."""
        steps = int(steps or self.env.horizon)
        obs, act, reward, _eval, _phase = self.rollout(key, envs=envs, steps=steps)

        def disc(carry, r):
            ret = r + gamma * carry
            return ret, ret

        _last, returns = jax.lax.scan(disc, jp.zeros((int(envs),), jp.float32), reward, reverse=True)
        flat = lambda x: x.reshape((-1,) + x.shape[2:])
        return flat(obs), flat(act), returns.reshape((-1,))

    # ---- single-robot real-world control loop ----------------------------------------
    def reset_single(self) -> None:
        self._phase = 0
        self._steps_in_phase = 0

    def step_single(self, signals: np.ndarray, obs: np.ndarray) -> np.ndarray:
        """One control step for a single physical (or sim) robot.

        `signals` is the observable field vector (same layout as field_index, supplied by the
        robot's perception stack); `obs` is the policy observation. Returns a normalized action.
        Maintains phase state internally -- call reset_single() at episode start.
        """
        c = self.compiled
        advance = (
            self._phase < (self.n_phases - 1)
            and self._steps_in_phase >= int(c.min_steps[self._phase])
            and _gates_satisfied_np(
                np.asarray(signals), c.gate_signal[self._phase], c.gate_op[self._phase],
                c.gate_value[self._phase], c.gate_active[self._phase],
            )
        )
        if advance:
            self._phase += 1
            self._steps_in_phase = 0
        else:
            self._steps_in_phase += 1
        p = self._phase
        act = self.action_from_obs(
            jp.asarray(obs), jp.asarray(c.group_ids[p]), jp.asarray(c.direction_ids[p]), jp.asarray(c.weights[p])
        )
        return np.asarray(act)


# --------------------------------------------------------------------------------------
# Curriculum-aware staged validation
# --------------------------------------------------------------------------------------
def staged_score(
    eval_traj: jp.ndarray,
    phase_traj: jp.ndarray,
    *,
    n_phases: int,
    control_dt: float,
    field_index: dict[str, int] | None = None,
    contact_field: str = "n_contacts",
    lift_field: str = "lift",
    xy_field: str = "obj_xy_disp",
    min_contacts: float = 1.0,
    max_xy_disp: float = 0.08,
    lift_threshold: float = 0.05,
    hold_seconds: float = 0.5,
) -> dict[str, Any]:
    """Curriculum-aware, contact-gated score from a teacher rollout.

    Gives graded partial credit for *depth of curriculum progress* (how far through the
    phases each env got) plus full contact-gated grasp-lift success, and penalizes flinging.
    Because phase-reach is monotone, summing reach fractions rewards reaching deeper phases --
    this is what discriminates candidates even before any of them completes a full lift, which
    was the null-signal failure of end-to-end-only scoring.

    All inputs are observable signals, so the same score is reproducible from a handful of
    real-robot rollouts (sim not required).
    """
    field_index = field_index or FIELD_INDEX
    hold_steps = max(1, int(round(float(hold_seconds) / max(float(control_dt), 1e-9))))

    # Depth of curriculum progress: fraction of envs that ever reached phase >= k.
    max_phase = jp.max(phase_traj, axis=0)  # [E]
    phase_reach = {}
    reach_sum = 0.0
    for k in range(1, n_phases):
        frac = jp.mean((max_phase >= k).astype(jp.float32))
        phase_reach[k] = round(float(frac), 6)
        reach_sum = reach_sum + float(frac)

    contacts_t = eval_traj[:, :, field_index[contact_field]]
    lift_t = eval_traj[:, :, field_index[lift_field]]
    xy_t = eval_traj[:, :, field_index[xy_field]]

    in_contact = contacts_t >= float(min_contacts)
    lifted = lift_t > float(lift_threshold)
    not_flung = xy_t <= float(max_xy_disp)
    grasp_lift = in_contact & lifted & not_flung
    fling_only = lifted & jp.logical_not(in_contact)

    contact_gated_success = float(sustained_bool(grasp_lift, hold_steps))
    fling_fraction = float(sustained_bool(fling_only, hold_steps))
    contact_engagement = float(jp.mean(in_contact.astype(jp.float32)))
    contact_conditioned_lift = float(jp.mean(jp.where(in_contact & not_flung, lift_t, 0.0)))

    score = (
        reach_sum                       # graded curriculum depth (monotone partial credit)
        + 4.0 * contact_gated_success   # full contact-grasp-lift
        + 1.0 * contact_conditioned_lift
        - 3.0 * fling_fraction          # suppress non-prehensile flinging
    )
    return {
        "staged_score": round(float(score), 6),
        "phase_reach": {str(k): v for k, v in phase_reach.items()},
        "final_phase_frac": round(float(reach_sum / max(1, n_phases - 1)), 6),
        "contact_gated_success": round(contact_gated_success, 6),
        "contact_conditioned_lift": round(contact_conditioned_lift, 6),
        "contact_engagement": round(contact_engagement, 6),
        "fling_fraction": round(fling_fraction, 6),
    }


def validate_phase_program(
    controller: PhaseController,
    *,
    envs: int,
    seed: int = 0,
    steps: int | None = None,
    min_contacts: float = 1.0,
    max_xy_disp: float = 0.08,
    lift_threshold: float = 0.05,
    hold_seconds: float = 0.5,
) -> dict[str, Any]:
    """Run a few rollouts and score the teacher with the staged contact-gated metric.

    This is the real-world-realizable validation gate: a controller that does not actually
    reach contact-grasp-lift scores low and can be rejected before committing it as the
    warm-start teacher.
    """
    _obs, _act, _reward, eval_traj, phase_traj = controller.rollout(
        jax.random.PRNGKey(seed), envs=envs, steps=steps
    )
    eval_traj.block_until_ready()
    record = staged_score(
        eval_traj,
        phase_traj,
        n_phases=controller.n_phases,
        control_dt=float(controller.env.cfg.control_dt),
        field_index=controller.field_index,
        min_contacts=min_contacts,
        max_xy_disp=max_xy_disp,
        lift_threshold=lift_threshold,
        hold_seconds=hold_seconds,
    )
    record["program_name"] = str(controller.program.get("name", "phase_program"))
    record["phase_names"] = controller.phase_names
    record["rollout_envs"] = int(envs)
    record["rollout_steps"] = int(steps or controller.env.horizon)
    return record


def load_phase_teacher(
    env: Any,
    *,
    task: str,
    program: dict[str, Any] | None = None,
    log_dir: Any = None,
    field_index: dict[str, int] | None = None,
    validate_envs: int = 64,
    **validate_kwargs: Any,
) -> tuple[PhaseController, dict[str, Any]]:
    """Build + validate a phase teacher. Phase A uses the hand-written default program when
    none is supplied; Phase B will pass an LLM-authored (and candidate-selected) program here.
    Returns (controller, validation_record)."""
    program = sanitize_phase_program(
        program if program is not None else default_phase_program(task),
        field_index=field_index,
        task=task,
    )
    controller = PhaseController(env, program, field_index=field_index)
    record = validate_phase_program(controller, envs=validate_envs, **validate_kwargs)
    if log_dir is not None:
        from pathlib import Path

        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / "phase_program.json").write_text(json.dumps(program, indent=2) + "\n")
        (log_dir / "phase_program_validation.json").write_text(json.dumps(record, indent=2) + "\n")
    return controller, record

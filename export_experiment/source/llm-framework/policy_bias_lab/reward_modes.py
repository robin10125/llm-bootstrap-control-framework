"""Experimental reward modes for long PPO runs with a fixed selected prior.

Motivated by the longppo_20260702-234528 diagnosis: the builtin base reward's dense, state-based
closure term (`w_close * closure * finger_near * (1 - contact_gate)`) pays ~12/episode for
air-closure hovering (fingers curled at ~3 cm, no contact, no lift), and 2 h of PPO collapsed the
warm-started policy into exactly that optimum while base_return rose 5x. Design principles (user):
intermediate/shaping rewards must target CURRENT weaknesses, not blindly pay for already-mastered
behavior; lift and sustained lift must DOMINATE intermediate income; intermediate rewards should be
gated (like the action prior's stages) so premature behavior (e.g. grasping before reach) is never
paid.

Three arms, all implemented WITHOUT touching bootstrapping/mjx_env.py:
  - env-side: dense intermediate base terms are zeroed via `make_env(**overrides)` (EnvConfig
    fields). The contact-gated lift terms (w_lift, w_lift_pot, w_lift_hold, w_success) are kept
    everywhere -- they ARE the task.
  - shaping-side: replacement intermediate terms are computed in the collect loop via a
    `shaping_fn(prev_eval, eval_vec, obs) -> (reward, contrib)` hook, where prev/cur eval vectors
    enable PROGRESS (telescoping) forms and `obs` enables prior-stage gating.

  lift_only    -- reward ONLY lifting/sustained lifting (all contact-gated): every intermediate
                  base term zeroed, shaping zeroed. The action prior alone must carry the policy
                  to contact.
  adjusted     -- the diagnosis fixes: closure paid on PROGRESS (bounded telescoping, tight
                  0.02 m fingertip gate) instead of per-step state; contact paid on progress;
                  empty-hand squeezing PENALIZED at the same magnitude the old exploit paid
                  (-0.25); env keeps its (safe, potential-based) approach terms.
  stage_gated  -- same adjusted intermediate terms, but each multiplied by the prior program's
                  OWN stage weights (freeform_priors.make_stage_weight_fn -- identical soft/hard
                  blend the prior acts with): approach terms pay only while a base-driving stage
                  is active, closure/contact terms only while a hand-driving stage is active.
                  Stage->term mapping is mechanical (which actuators the stage's channels drive),
                  so it stays task-agnostic. Env intermediate terms all zeroed (they cannot see
                  the stages); penalty stays ungated (premature squeezing is never OK).

Eval-vector fields (mjx_env EM.FIELD_INDEX order):
  [0] palm_obj_dist  [1] min_finger_dist  [2] n_contacts  [3] closure  [4] lift  [5] obj_xy_disp
"""
from __future__ import annotations

from typing import Any, Callable

import jax.numpy as jp

from policy_bias_lab.bias import REWARD_TEMPLATE_COUNT

REWARD_MODES = ("default", "lift_only", "adjusted", "stage_gated")

# EnvConfig weight overrides per mode (passed through make_env(**overrides)).
ENV_OVERRIDES: dict[str, dict[str, float]] = {
    "default": {},
    # only w_lift / w_lift_pot / w_lift_hold / w_success (+ their contact/anti-fling gates) remain
    "lift_only": dict(w_reach=0.0, w_finger=0.0, w_close=0.0, w_contact=0.0, w_hold=0.0),
    # keep the env's potential-based approach terms (telescoping, not exploitable); kill the dense
    # state-based closure/contact/hold payments -- replaced by progress forms in the shaping layer
    "adjusted": dict(w_close=0.0, w_contact=0.0, w_hold=0.0),
    # everything intermediate moves into the (stage-gated) shaping layer
    "stage_gated": dict(w_reach=0.0, w_finger=0.0, w_close=0.0, w_contact=0.0, w_hold=0.0),
}

# contrib-vector slot names (first 5 of REWARD_TEMPLATE_COUNT; rest zero) for metrics.jsonl
CONTRIB_NAMES = ("approach_potential", "closure_progress", "contact_progress",
                 "empty_squeeze_penalty", "hand_gate_mean")

# term weights (see module docstring for rationale; totals are bounded so the env lift terms --
# up to ~0.6/step held at 5 cm plus the height potential -- always dominate)
W_APPROACH = 1.0          # matches env w_reach/w_finger; telescopes to ~initial distance
W_CLOSE_PROG = 0.25       # matches old w_close but on Δclosure: bounded ~0.25 total, not 12
W_CONTACT_PROG = 0.5      # matches old w_contact but on Δcontact_gate: bounded per contact event
W_SQUEEZE_PEN = 0.25      # matches the magnitude the old exploit PAID; now it costs that instead
CLOSE_NEAR_SCALE = 0.02   # fingertips must be genuinely at the surface (old 0.05 passed at 3.3cm)
CONTACT_TARGET = 2.0      # same as EnvConfig.contact_target
CLIP = (-0.75, 0.60)      # same per-step clip as bias.dynamic_shaped_reward


def _terms(prev_e: jp.ndarray, e: jp.ndarray) -> dict[str, jp.ndarray]:
    """The adjusted intermediate terms, shared by `adjusted` and `stage_gated`."""
    contact_gate = jp.clip(e[2] / CONTACT_TARGET, 0.0, 1.0)
    prev_contact_gate = jp.clip(prev_e[2] / CONTACT_TARGET, 0.0, 1.0)
    finger_near = jp.exp(-((e[1] / CLOSE_NEAR_SCALE) ** 2))
    approach = W_APPROACH * ((prev_e[0] - e[0]) + (prev_e[1] - e[1]))  # telescoping potentials
    closure_prog = (W_CLOSE_PROG * (e[3] - prev_e[3]) * finger_near * (1.0 - contact_gate))
    contact_prog = W_CONTACT_PROG * (contact_gate - prev_contact_gate)
    squeeze_pen = -W_SQUEEZE_PEN * jp.maximum(e[3] - 0.55, 0.0) * (1.0 - contact_gate)
    return {"approach": approach, "closure_prog": closure_prog,
            "contact_prog": contact_prog, "squeeze_pen": squeeze_pen}


def _pack(approach, closure_prog, contact_prog, squeeze_pen, hand_gate) -> tuple:
    contrib = jp.zeros((REWARD_TEMPLATE_COUNT,), dtype=jp.float32)
    contrib = contrib.at[0].set(approach).at[1].set(closure_prog)
    contrib = contrib.at[2].set(contact_prog).at[3].set(squeeze_pen).at[4].set(hand_gate)
    reward = jp.clip(approach + closure_prog + contact_prog + squeeze_pen, *CLIP)
    return reward, contrib


def _stage_masks(program: dict, stage_names: list[str]) -> tuple[list[float], list[float]]:
    """Per-stage 0/1 masks: does the stage's channel set drive base actuators / hand actuators?
    Mechanical read of the program (actuator-name tokens containing 'base' = base group), so no
    task knowledge enters the framework."""
    base_mask, hand_mask = [], []
    for st in program.get("stages", []):
        toks = [str(a).lower() for ch in st.get("channels", []) for a in ch.get("actuators", [])]
        base_mask.append(1.0 if any("base" in t for t in toks) else 0.0)
        hand_mask.append(1.0 if any("base" not in t for t in toks) else 0.0)
    return base_mask, hand_mask


def build_shaping_fn(mode: str, env: Any, program: dict | None
                     ) -> Callable[[jp.ndarray, jp.ndarray, jp.ndarray], tuple] | None:
    """Returns a shaping_fn(prev_eval, eval_vec, obs) -> (reward, contrib[16]) for the mode, or
    None for `default` (the builtin template shaping stays in effect)."""
    if mode == "default":
        return None
    if mode == "lift_only":
        z = jp.zeros((REWARD_TEMPLATE_COUNT,), dtype=jp.float32)

        def zero_fn(prev_e, e, obs):
            return jp.float32(0.0), z
        return zero_fn
    if mode == "adjusted":
        def adjusted_fn(prev_e, e, obs):
            t = _terms(prev_e, e)
            # env keeps its own approach potentials in this mode -- don't pay them twice
            return _pack(jp.float32(0.0), t["closure_prog"], t["contact_prog"],
                         t["squeeze_pen"], jp.float32(1.0))
        return adjusted_fn
    if mode == "stage_gated":
        from policy_bias_lab.freeform_priors import make_stage_weight_fn
        if not (program and program.get("stages")):
            raise ValueError("stage_gated reward mode needs a staged prior program")
        weight_fn, names = make_stage_weight_fn(env, program)
        bm, hm = _stage_masks(program, names)
        base_mask = jp.asarray(bm, dtype=jp.float32)
        hand_mask = jp.asarray(hm, dtype=jp.float32)

        def gated_fn(prev_e, e, obs):
            w = weight_fn(obs)                       # the prior's OWN stage weights on this state
            base_gate = jp.sum(w * base_mask)
            hand_gate = jp.sum(w * hand_mask)
            t = _terms(prev_e, e)
            return _pack(t["approach"] * base_gate,
                         t["closure_prog"] * hand_gate,
                         t["contact_prog"] * hand_gate,
                         t["squeeze_pen"],            # premature squeezing is never OK: ungated
                         hand_gate)
        return gated_fn
    raise ValueError(f"unknown reward mode {mode!r}; choose from {REWARD_MODES}")

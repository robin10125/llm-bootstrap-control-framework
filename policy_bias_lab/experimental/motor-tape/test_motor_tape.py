"""Unit tests for motor_tape (run: .venv/bin/python -m pytest policy_bias_lab/experimental/motor-tape/test_motor_tape.py -x -q).

Part 1 needs no env (pure interpolation math). Part 2 builds the real shadow env once.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for p in (str(HERE), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from motor_tape import (
    compile_tape,
    expand_keyframes,
    interp_weights,
    probe_base_calibration,
    render_calibration_block,
    tape_grounding_report,
    tape_report,
    validate_motor_tape,
)


# ------------------------------------------------------------------------------------------
# Part 1: interpolation math, no env
# ------------------------------------------------------------------------------------------

def test_linear_segment_exact():
    W = interp_weights(np.array([0.0, 1.0]), ["linear", "linear"], n_steps=40, control_dt=0.025)
    v = np.array([2.0, 6.0])
    q = W @ v
    assert q[0] == pytest.approx(2.0)
    assert q[20] == pytest.approx(4.0)  # t = 0.5
    assert q[40] == pytest.approx(6.0)


def test_minjerk_endpoints_zero_slope_monotone_bounded():
    W = interp_weights(np.array([0.0, 1.0]), ["minjerk", "minjerk"], n_steps=100,
                       control_dt=0.01)
    v = np.array([-1.0, 3.0])
    q = W @ v
    assert q[0] == pytest.approx(-1.0)
    assert q[-1] == pytest.approx(3.0)
    assert q[50] == pytest.approx(1.0)  # ease is symmetric: midpoint = midvalue
    d = np.diff(q)
    assert np.all(d >= -1e-6)                        # monotone
    assert abs(d[0]) < abs(d[50]) * 0.1              # zero slope at start
    assert abs(d[-1]) < abs(d[50]) * 0.1             # zero slope at end
    assert q.min() >= -1.0 - 1e-6 and q.max() <= 3.0 + 1e-6  # never leaves [v0, v1]


def test_hold_before_first_and_after_last():
    W = interp_weights(np.array([1.0, 2.0]), ["linear", "linear"], n_steps=120,
                       control_dt=0.025)
    v = np.array([5.0, 7.0])
    q = W @ v
    assert np.all(q[:41] == pytest.approx(5.0))   # t <= 1.0 holds first knot
    assert np.all(q[80:] == pytest.approx(7.0))   # t >= 2.0 holds last knot
    assert np.all(np.abs(W.sum(axis=1) - 1.0) < 1e-6)


def test_mixed_interp_per_segment():
    # segment 0->1 linear, 1->2 minjerk: interp on the knot ENDING the segment
    W = interp_weights(np.array([0.0, 1.0, 2.0]), ["minjerk", "linear", "minjerk"],
                       n_steps=80, control_dt=0.025)
    v = np.array([0.0, 1.0, 2.0])
    q = W @ v
    assert q[20] == pytest.approx(0.5)            # linear midpoint of segment 0
    d2 = np.diff(q[40:])
    assert abs(d2[0]) < 1e-3                       # minjerk zero slope entering segment 1


# ------------------------------------------------------------------------------------------
# Part 2: real env
# ------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def env():
    from experiment_runtime.environment import make_env
    return make_env("shadow", control_dt=0.025, episode_seconds=20.0, physics_dt=0.01,
                    obj_xy_range=0.04)


def _prog(keyframes, **kw):
    return {"mode": "motor_tape", "name": "t", "keyframes": keyframes, **kw}


def test_group_expansion_and_member_precedence(env):
    prog = _prog([{"t": 2.0, "targets": {"thumb": "0.5", "rh_A_THJ4": "0.9"}}])
    per = expand_keyframes(env, prog)
    names = [env.model.actuator(i).name for i in range(env.nu)]
    thj4 = names.index("rh_A_THJ4")
    thumb_ids = [i for i, n in enumerate(names) if n.startswith("rh_A_TH")]
    assert set(per) == set(thumb_ids)
    assert per[thj4][-1]["expr"] == "0.9"          # exact name wins over group
    others = [i for i in thumb_ids if i != thj4]
    assert all(per[i][-1]["expr"] == "0.5" for i in others)
    # implicit t=0 anchor prepended
    assert all(per[i][0]["t"] == 0.0 and per[i][0]["expr"] is None for i in thumb_ids)


def test_compile_tape_matches_numpy_reference(env):
    import jax
    prog = _prog([
        {"t": 2.0, "targets": {"base_x": "obj_pos_x", "base_z": "obj_pos_z + 0.10"},
         "interp": "linear"},
        {"t": 4.0, "targets": {"base_z": "obj_pos_z + 0.02"}, "interp": "minjerk"},
    ])
    compiled = compile_tape(env, prog)
    obs0 = env.reset(jax.random.PRNGKey(3)).obs
    tape = np.asarray(compiled.tape_from_obs(obs0))
    assert tape.shape == (env.horizon + 1, env.nu)

    from policy_bias_lab.freeform_priors import raw_obs_entries, all_actuator_names
    obs_idx = dict(raw_obs_entries(env)[0])
    o = np.asarray(obs0)
    names = all_actuator_names(env)
    bx, bz = names.index("base_x"), names.index("base_z")
    lo = np.asarray(env.ctrl_lo)
    hi = np.asarray(env.ctrl_hi)
    obj_x, obj_z = o[obs_idx["obj_pos_x"]], o[obs_idx["obj_pos_z"]]
    init_bx = o[obs_idx["ctrl_base_x"]]
    init_bz = o[obs_idx["ctrl_base_z"]]

    # base_x: linear from reset ctrl to obj_x by t=2, hold after
    tgt = np.clip(obj_x, lo[bx], hi[bx])
    assert tape[0, bx] == pytest.approx(init_bx, abs=1e-5)
    assert tape[40, bx] == pytest.approx(0.5 * (init_bx + tgt), abs=1e-5)  # t=1.0
    assert tape[80, bx] == pytest.approx(tgt, abs=1e-5)
    assert tape[-1, bx] == pytest.approx(tgt, abs=1e-5)
    # base_z: two segments; knot at t=2 then minjerk to t=4
    hover = np.clip(obj_z + 0.10, lo[bz], hi[bz])
    low = np.clip(obj_z + 0.02, lo[bz], hi[bz])
    assert tape[80, bz] == pytest.approx(hover, abs=1e-5)
    assert tape[120, bz] == pytest.approx(0.5 * (hover + low), abs=1e-4)  # minjerk midpoint
    assert tape[160, bz] == pytest.approx(low, abs=1e-5)
    # untouched actuator holds reset ctrl for the whole tape
    wr = names.index("rh_A_WRJ1")
    assert np.allclose(tape[:, wr], o[obs_idx["ctrl_rh_A_WRJ1"]], atol=1e-6)


def test_ctrl_init_self_binds_per_actuator(env):
    import jax
    prog = _prog([{"t": 3.0, "targets":
                   {"index": "ctrl_init_self + 0.5 * (ctrl_hi_self - ctrl_init_self)"}}])
    compiled = compile_tape(env, prog)
    obs0 = env.reset(jax.random.PRNGKey(1)).obs
    tape = np.asarray(compiled.tape_from_obs(obs0))
    from policy_bias_lab.freeform_priors import raw_obs_entries, all_actuator_names
    obs_idx = dict(raw_obs_entries(env)[0])
    o = np.asarray(obs0)
    hi = np.asarray(env.ctrl_hi)
    for n in compiled.covered:
        a = all_actuator_names(env).index(n)
        init = o[obs_idx[f"ctrl_{n}"]]
        assert tape[-1, a] == pytest.approx(init + 0.5 * (hi[a] - init), abs=1e-5)


def test_validation_rejects_bad_candidates(env):
    cases = [
        ({"mode": "freeform_staged", "keyframes": [{"t": 1, "targets": {"base_x": "0"}}]},
         "mode"),
        ({"mode": "motor_tape", "keyframes": []}, "keyframes"),
        (_prog([{"t": 2.0, "targets": {"base_x": "0"}}, {"t": 1.0, "targets": {"base_x": "0"}}]),
         "decreases"),
        (_prog([{"t": 999.0, "targets": {"base_x": "0"}}]), "outside"),
        (_prog([{"t": 1.0, "targets": {"no_such_actuator": "0"}}]), "matches no actuator"),
        (_prog([{"t": 1.0, "targets": {"base_x": "not_a_signal + 1"}}]), "compile"),
        (_prog([{"t": 1.0, "targets": {"base_x": "0"}},
                {"t": 1.0, "targets": {"base_x": "0.1"}}]), "duplicate"),
        (_prog([{"t": 1.0, "targets": {"base_x": "0"}, "interp": "cubic"}]), "interp"),
    ]
    for cand, needle in cases:
        errs: list[str] = []
        assert validate_motor_tape(env, cand, errs) is None, cand
        assert any(needle in e for e in errs), (needle, errs)


def test_validation_accepts_and_reports(env):
    cand = _prog(
        [{"t": 2.0, "label": "hover",
          "targets": {"base_x": "obj_pos_x", "base_y": "obj_pos_y",
                      "base_z": "obj_pos_z + 0.10"}},
         {"t": 5.0, "label": "close",
          "targets": {"thumb": "ctrl_init_self + close_frac * (ctrl_hi_self - ctrl_init_self)"}}],
        signals={"obj_high": "obj_pos_z + 0.25"},
        parameters={"close_frac": {"init": 0.6, "range": [0.0, 1.0]}},
    )
    errs: list[str] = []
    prog = validate_motor_tape(env, cand, errs)
    assert prog is not None, errs
    rep = tape_report(env, prog, envs=4, seed=0)
    assert rep["covered"] >= 8  # 3 base + thumb group
    assert rep["last_keyframe_t"] == 5.0


def test_probe_calibration_signs_match_spec(env):
    from policy_bias_lab.freeform_priors import robot_spec
    cal = probe_base_calibration(env)
    assert set(cal) == {"base_x", "base_y", "base_z"}
    bws = robot_spec(env)["base_world_sign"]
    for act, deltas in cal.items():
        axis = bws[act]["world_axis"]
        spec_sign = 1.0 if bws[act]["ctrl_increases_world"].startswith("+") else -1.0
        palm = deltas.get(f"palm_pos_{axis}")
        assert palm is not None, (act, deltas)
        assert palm * spec_sign > 0, (act, palm, bws[act])
        # obj_rel = object - grasp point: moving the hand toward +axis makes obj_rel_axis fall
        rel = deltas.get(f"obj_rel_{axis}")
        assert rel is not None and rel * palm < 0, (act, deltas)
    block = render_calibration_block(cal)
    assert "base_y" in block and "HOW TO USE" in block


def test_grounding_report_on_reaching_plan(env):
    import json
    prog = json.loads((HERE / "example_tape.json").read_text())
    errs: list[str] = []
    prog = validate_motor_tape(env, prog, errs)
    assert prog is not None, errs
    rep = tape_grounding_report(env, prog, envs=3, seed=0)
    assert len(rep["spawns"]) == 3
    for s in rep["spawns"]:
        # the example plan reaches: horizontal obj_rel must shrink from spawn to end
        for k in ("obj_rel_x", "obj_rel_y"):
            assert abs(s["end_obj_rel"][k]) <= abs(s["spawn_obj_rel"][k]) + 0.02, s
        assert s["min_palm_obj_dist"] < 0.15


# ------------------------------------------------------------------------------------------
# Part 3: contact-harm mitigations (grasp-gated reward, feedforward handoff, near-miss score)
# ------------------------------------------------------------------------------------------

FB_PROGRAM = ROOT / "runs" / "motor_tape_genstudy_20260716" / "fb" / "best_program.json"


def test_nearmiss_score_properties():
    from motor_tape import nearmiss_score
    lo, hi = 0.05, 0.065
    in_band = {"palm_obj_dist_min": 0.057, "contact_engagement": 0.0}
    contacting = {"palm_obj_dist_min": 0.042, "contact_engagement": 0.014}
    far = {"palm_obj_dist_min": 0.075, "contact_engagement": 0.0}
    s_in = nearmiss_score(in_band, lo, hi)
    s_contact = nearmiss_score(contacting, lo, hi)
    s_far = nearmiss_score(far, lo, hi)
    assert s_in > 0.9
    assert s_contact < 0.05          # contact penalty drives it to ~0 despite proximity
    assert s_far < s_in and s_far < 0.2
    assert nearmiss_score({"palm_obj_dist_min": 0.057, "contact_engagement": 0.02},
                          lo, hi) == 0.0


@pytest.mark.skipif(not FB_PROGRAM.is_file(), reason="fb study artifact not present")
def test_grasp_gated_matches_env_lift_terms(env):
    """The re-implemented lift terms must equal the env's own: playing the SAME contact-making
    tape on the default env and on an env with lift income zeroed, the per-step reward diff must
    equal make_gated_lift_fn(closure_target=0)."""
    import jax
    import jax.numpy as jp
    import json
    from experiment_runtime.environment import EnvConfig, make_env
    from motor_tape import compile_tape
    from motor_tape_ppo import make_gated_lift_fn

    env_z = make_env("shadow", control_dt=0.025, episode_seconds=20.0, physics_dt=0.01,
                     obj_xy_range=0.04, w_lift=0.0, w_lift_pot=0.0, w_lift_hold=0.0,
                     w_success=0.0)
    prog = json.loads(FB_PROGRAM.read_text())
    errs: list[str] = []
    prog = validate_motor_tape(env, prog, errs)
    assert prog is not None, errs
    compiled = compile_tape(env, prog)
    E, T = 4, int(env.horizon)
    scale = float(env.cfg.action_scale)
    keys = jax.random.split(jax.random.PRNGKey(3), E)
    st = jax.jit(jax.vmap(env.reset))(keys)
    st_z = jax.jit(jax.vmap(env_z.reset))(keys)
    tape = jax.jit(jax.vmap(compiled.tape_from_obs))(st.obs)
    step = jax.jit(jax.vmap(env.step))
    step_z = jax.jit(jax.vmap(env_z.step))

    d = EnvConfig()
    gg = dict(w_lift=d.w_lift, w_lift_pot=d.w_lift_pot, w_lift_hold=d.w_lift_hold,
              w_success=d.w_success, lift_target=d.lift_target, success_height=d.success_height,
              contact_target=d.contact_target, fling_xy_thresh=d.fling_xy_thresh,
              pbrs_gamma=d.pbrs_gamma, closure_target=0.0)
    ungated = make_gated_lift_fn(gg)
    gated = make_gated_lift_fn({**gg, "closure_target": 0.5})

    max_err, lift_income_seen, gated_lt_ungated = 0.0, 0.0, True
    for t in range(T):
        idx = jp.full((E,), float(t))
        i0 = jp.clip(jp.floor(idx).astype(jp.int32), 0, T)
        q = jp.take_along_axis(tape, i0[:, None, None], axis=1)[:, 0]
        a = jp.clip((q - st.data.ctrl) / scale, -1.0, 1.0)
        prev_e = st.metrics["eval"]
        st, st_z = step(st, a), step_z(st_z, a)
        diff = np.asarray(st.reward - st_z.reward)
        pred = np.asarray(ungated(prev_e, st.metrics["eval"]))
        pred_gated = np.asarray(gated(prev_e, st.metrics["eval"]))
        max_err = max(max_err, float(np.abs(diff - pred).max()))
        lift_income_seen += float(np.abs(pred).sum())
        if np.any(np.abs(pred_gated) > np.abs(pred) + 1e-5):
            gated_lt_ungated = False
    assert max_err < 1e-3, max_err
    assert lift_income_seen > 0.01, "tape never produced lift income; test is vacuous"
    assert gated_lt_ungated


def test_handoff_masks_feedforward(env):
    """With the handoff band far above any reachable distance (full mask), the tape cannot drive
    the approach: the hand stays near spawn, while the unmasked run closes in. Plan features are
    identical at t=0 (the plan stays visible; only actuation is masked)."""
    import jax
    import jax.numpy as jp
    import json
    from motor_tape import compile_tape
    from motor_tape_ppo import Actor, Critic, MotorTapePPOConfig, action_layout, feat_dim, \
        make_tape_collect

    prog = json.loads((HERE / "example_tape.json").read_text())
    errs: list[str] = []
    prog = validate_motor_tape(env, prog, errs)
    assert prog is not None, errs
    compiled = compile_tape(env, prog)
    E = 4

    def run(handoff):
        cfg = MotorTapePPOConfig(envs=E, fragment_steps=100, use_rate=False,
                                 handoff_lo=handoff[0], handoff_hi=handoff[1])
        lay = action_layout(cfg, int(env.action_size))
        actor = Actor(out_dim=lay["out_dim"], hidden=cfg.hidden)
        critic = Critic(hidden=cfg.hidden)
        k = jax.random.PRNGKey(0)
        params = {
            "actor": actor.init(k, jp.zeros((1, env.obs_size)),
                                jp.zeros((1, feat_dim(cfg, int(env.action_size))))),
            "critic": critic.init(k, jp.zeros((1, env.obs_size)),
                                  jp.zeros((1, feat_dim(cfg, int(env.action_size))))),
        }
        collect = make_tape_collect(env=env, actor=actor, critic=critic, compiled=compiled,
                                    cfg=cfg, deterministic=True)
        st = jax.jit(jax.vmap(env.reset))(jax.random.split(jax.random.PRNGKey(5), E))
        tape = jax.jit(jax.vmap(compiled.tape_from_obs))(st.obs)
        s = jp.zeros((E,), jp.float32)
        min_dist, feats0 = None, None
        for frag in range(4):                  # 10s of the plan -- well into the approach
            st, s, traj, _lv, _es = collect(params, st, s, tape, jax.random.PRNGKey(6 + frag))
            ev = np.asarray(traj[10])          # [T, E, 6]
            md = ev[:, :, 0].min(axis=0)
            min_dist = md if min_dist is None else np.minimum(min_dist, md)
            if frag == 0:
                feats0 = np.asarray(traj[1][0])
        return min_dist.mean(), feats0

    dist_off, feats_off = run((0.0, 0.0))
    dist_masked, feats_masked = run((5.0, 10.0))   # dist << lo always -> a_ff fully masked
    np.testing.assert_allclose(feats_off, feats_masked, atol=1e-6)
    assert dist_masked > dist_off + 0.02, (dist_masked, dist_off)

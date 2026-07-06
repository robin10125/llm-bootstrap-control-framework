from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
os.environ.setdefault("MUJOCO_GL", "egl")

import jax
import jax.numpy as jp
import mujoco
import numpy as np

from policy_bias_lab.bias import CompiledBias, compile_bias
from policy_bias_lab.es import BIAS_ARMS

ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAPPING = ROOT.parent / "bootstrapping"
if str(BOOTSTRAPPING) not in sys.path:
    sys.path.insert(0, str(BOOTSTRAPPING))

import ppo
import eval_metrics as EM
from mjx_env import EnvConfig, _SHADOW_CLOSE, _SHADOW_FINGERTIPS, _SHADOW_OPEN, load_model

SHADOW_XML = ROOT.parent / "hand-manipulation" / "env" / "models" / "shadow_hand" / "scene_cube.xml"


@dataclass(frozen=True)
class RolloutStats:
    seed: int
    success: bool
    lift_max: float
    base_return: float
    final_lift: float


class CpuShadowEnv:
    def __init__(self, *, control_dt: float, episode_seconds: float, physics_dt: float, obj_xy_range: float):
        self.cfg = EnvConfig(
            control_dt=control_dt,
            episode_seconds=episode_seconds,
            physics_dt=physics_dt,
            obj_xy_range=obj_xy_range,
            grasp_geom_substr=("palm", "knuckle", "proximal", "middle", "distal", "metacarpal", "thbase", "thhub"),
            fingertip_bodies=_SHADOW_FINGERTIPS,
            open_pose=_SHADOW_OPEN,
            close_pose=_SHADOW_CLOSE,
        )
        self.model = load_model(SHADOW_XML, self.cfg)
        if self.cfg.physics_dt:
            self.model.opt.timestep = self.cfg.physics_dt
        if self.cfg.pyramidal_cone:
            self.model.opt.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
        if self.cfg.solver_iterations:
            self.model.opt.iterations = self.cfg.solver_iterations
        self.data = mujoco.MjData(self.model)
        self.frame_skip = max(1, round(self.cfg.control_dt / self.model.opt.timestep))
        self.settle_steps = max(0, round(self.cfg.settle_seconds / self.model.opt.timestep))
        self.horizon = max(1, round(self.cfg.episode_seconds / self.cfg.control_dt))
        self.nu = self.model.nu
        self.action_size = self.nu
        self.ctrl_lo = np.asarray(self.model.actuator_ctrlrange[:, 0], dtype=np.float32)
        self.ctrl_hi = np.asarray(self.model.actuator_ctrlrange[:, 1], dtype=np.float32)

        self.object_bid = int(self.model.body("object").id)
        self.palm_bid = int(self.model.body("rh_palm").id)
        self.grasp_sid = int(self.model.site("grasp_site").id)
        obj_jid = int(self.model.body("object").jntadr[0])
        self.obj_qadr = int(self.model.jnt_qposadr[obj_jid])
        self.base_qadr = [int(self.model.jnt_qposadr[self.model.joint(n).id]) for n in ("slide_x", "slide_y", "slide_z")]
        self.base_vadr = [int(self.model.jnt_dofadr[self.model.joint(n).id]) for n in ("slide_x", "slide_y", "slide_z")]
        base = {"slide_x", "slide_y", "slide_z"}
        hand_jids = [
            j for j in range(self.model.njnt)
            if self.model.joint(j).name not in base and self.model.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE
        ]
        self.hand_qadr = [int(self.model.jnt_qposadr[j]) for j in hand_jids]
        self.hand_vadr = [int(self.model.jnt_dofadr[j]) for j in hand_jids]
        # prefix (q + v + constraint-force blocks) + [obj_pos, obj_vel, palm_pos, obj_rel] + ctrl
        self.obs_size = (len(self.base_qadr) + len(self.base_vadr) + len(self.hand_qadr)
                         + len(self.hand_vadr) + len(self.base_vadr) + len(self.hand_vadr)
                         + 12 + self.nu)
        act_names = [self.model.actuator(i).name for i in range(self.nu)]
        self.base_act_ids = [i for i, n in enumerate(act_names) if n in ("base_x", "base_y", "base_z")]
        self.hand_act_ids = [i for i in range(self.nu) if i not in self.base_act_ids]
        self.fingertip_bids = np.asarray([int(self.model.body(n).id) for n in _SHADOW_FINGERTIPS], dtype=np.int32)

        lo = np.asarray(self.model.actuator_ctrlrange[:, 0])
        hi = np.asarray(self.model.actuator_ctrlrange[:, 1])
        self.ctrl_open = self._pose_ctrl(_SHADOW_OPEN, lo, hi).astype(np.float32)
        self.ctrl_close = self._pose_ctrl(_SHADOW_CLOSE, lo, hi).astype(np.float32)
        self._hand_ids = np.asarray(self.hand_act_ids, dtype=np.int32)
        self._open_hand = self.ctrl_open[self._hand_ids]
        self._close_dir = self.ctrl_close[self._hand_ids] - self.ctrl_open[self._hand_ids]
        self._obj_start_z = 0.0
        self._obj_start_xy = np.zeros(2, dtype=np.float32)

    def _pose_ctrl(self, hand_qpos, lo, hi) -> np.ndarray:
        d = mujoco.MjData(self.model)
        qpos = np.asarray(self.model.qpos0).copy()
        for adr, val in zip(self.hand_qadr, hand_qpos):
            qpos[adr] = val
        d.qpos[:] = qpos
        mujoco.mj_forward(self.model, d)
        return np.clip(np.asarray(d.actuator_length).copy(), lo, hi)

    def reset(self, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.model.qpos0
        dxy = rng.uniform(-self.cfg.obj_xy_range, self.cfg.obj_xy_range, size=2)
        self.data.qpos[self.obj_qadr] += dxy[0]
        self.data.qpos[self.obj_qadr + 1] += dxy[1]
        self.data.ctrl[:] = self.ctrl_open
        mujoco.mj_forward(self.model, self.data)
        for _ in range(self.settle_steps):
            self.data.ctrl[:] = self.ctrl_open
            mujoco.mj_step(self.model, self.data)
        obj_pos = self.data.xpos[self.object_bid].copy()
        self._obj_start_z = float(obj_pos[2])
        self._obj_start_xy = obj_pos[:2].copy()
        return self.observe()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, dict[str, float]]:
        action = np.clip(action, -1.0, 1.0)
        target = np.clip(self.data.ctrl + action * self.cfg.action_scale, self.ctrl_lo, self.ctrl_hi)
        for _ in range(self.frame_skip):
            self.data.ctrl[:] = target
            mujoco.mj_step(self.model, self.data)
        eval_vec = self.eval_vec()
        reward = self.builtin_reward(eval_vec)
        return self.observe(), reward, {
            "success": float(eval_vec[EM.FIELD_INDEX["lift"]] > self.cfg.success_height),
            "lift": float(eval_vec[EM.FIELD_INDEX["lift"]]),
        }

    def observe(self) -> np.ndarray:
        base_q = self.data.qpos[self.base_qadr]
        base_v = self.data.qvel[self.base_vadr]
        hand_q = self.data.qpos[self.hand_qadr]
        hand_v = self.data.qvel[self.hand_vadr]
        base_f = self.data.qfrc_constraint[self.base_vadr]
        hand_f = self.data.qfrc_constraint[self.hand_vadr]
        obj_pos = self.data.xpos[self.object_bid]
        obj_vel = self.data.cvel[self.object_bid, 3:6]
        palm_pos = self.data.xpos[self.palm_bid]
        grasp_pos = self.data.site_xpos[self.grasp_sid]
        obj_rel = obj_pos - grasp_pos
        return np.concatenate([base_q, base_v, hand_q, hand_v, base_f, hand_f,
                               obj_pos, obj_vel, palm_pos, obj_rel, self.data.ctrl]).astype(np.float32)

    def eval_vec(self) -> np.ndarray:
        obj_pos = self.data.xpos[self.object_bid]
        grasp_pos = self.data.site_xpos[self.grasp_sid]
        palm_obj_dist = np.linalg.norm(obj_pos - grasp_pos)
        tip_pos = self.data.xpos[self.fingertip_bids]
        tip_d = np.linalg.norm(tip_pos - obj_pos[None, :], axis=-1)
        min_finger_dist = tip_d.min()
        n_contacts = float(np.sum(tip_d < self.cfg.contact_eps))
        closure = float(np.mean(np.clip((self.data.ctrl[self._hand_ids] - self._open_hand) / (self._close_dir + 1e-6), 0.0, 1.0)))
        lift = obj_pos[2] - self._obj_start_z
        obj_xy_disp = np.linalg.norm(obj_pos[:2] - self._obj_start_xy)
        return np.asarray([palm_obj_dist, min_finger_dist, n_contacts, closure, lift, obj_xy_disp], dtype=np.float32)

    def builtin_reward(self, eval_vec: np.ndarray) -> float:
        reach_d = eval_vec[EM.FIELD_INDEX["palm_obj_dist"]]
        lift = eval_vec[EM.FIELD_INDEX["lift"]]
        near = np.exp(-(reach_d / 0.05) ** 2)
        r_reach = -self.cfg.w_reach * reach_d
        r_close = self.cfg.w_close * eval_vec[EM.FIELD_INDEX["closure"]] * near
        r_lift = self.cfg.w_lift * np.clip(lift, 0.0, self.cfg.lift_target)
        r_success = self.cfg.w_success * (1.0 if lift > self.cfg.success_height else 0.0)
        return float(r_reach + r_close + r_lift + r_success)


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir
    config = json.loads((run_dir / "config.json").read_text())
    bias_spec = json.loads((run_dir / "bias_spec.json").read_text())
    env = CpuShadowEnv(
        control_dt=float(config["ppo"].get("control_dt", 0.025) if "control_dt" in config["ppo"] else args.control_dt),
        episode_seconds=float(config["ppo"].get("episode_seconds", args.episode_seconds)),
        physics_dt=float(args.physics_dt),
        obj_xy_range=float(args.obj_xy_range),
    )
    net = ppo.ActorCritic(action_dim=env.action_size, hidden=tuple(config["ppo"]["hidden"]))
    action_transform = str(config["ppo"].get("action_transform", "raw"))
    prior_logit_clip = float(config["ppo"].get("prior_logit_clip", 0.95))
    out_dir = args.out or run_dir / "checkpoint_videos"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    for arm in arms:
        if arm not in BIAS_ARMS:
            raise KeyError(f"unknown arm {arm!r}")
        # Per-arm bias: inject the arm's situation-dependent prior program (saved at train time)
        # so the rendered policy applies the SAME prior it trained with. Without this the shared
        # bias_spec has no prior_program and the composed-prior arms would render with no prior.
        arm_spec = dict(bias_spec)
        prog_path = run_dir / f"lift_s0_{arm}" / "prior_program.json"
        if prog_path.exists():
            arm_spec["prior_program"] = json.loads(prog_path.read_text())
        bias = compile_bias(arm_spec, env)
        ckpt_root = run_dir / f"lift_s0_{arm}" / "checkpoints"
        if args.checkpoints == "final":
            checkpoints = sorted(ckpt_root.glob("params_t*_final.pkl")) or sorted(ckpt_root.glob("params_t*_iter*.pkl"))[-1:]
        elif args.checkpoints == "best":
            checkpoints = sorted(ckpt_root.glob("params_best_iter*.pkl"))
        else:
            checkpoints = sorted(ckpt_root.glob("params_t*_iter*.pkl"))
        # Jit the per-step action (net + prior) once per arm -- eager per-step JAX dispatch on CPU
        # is ~100x slower and makes rendering take tens of minutes.
        act_fn = make_act_fn(net, bias, arm, args.task, action_transform, prior_logit_clip)
        for ckpt in checkpoints:
            label = ckpt.stem.replace("params_", "")
            with ckpt.open("rb") as f:
                params = pickle.load(f)
            attempts = [
                rollout_stats(env, act_fn, params, args.seed_base + i)
                for i in range(args.max_attempts)
            ]
            successes = [s for s in attempts if s.success]
            selected = successes[:args.successes_per_checkpoint]
            if not selected and attempts:
                selected = [max(attempts, key=lambda s: (s.success, s.lift_max, s.base_return))]
            ckpt_dir = out_dir / arm / label
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            for idx, stats in enumerate(selected):
                video_path = ckpt_dir / f"rollout_{idx:02d}_seed{stats.seed}_success{int(stats.success)}_lift{stats.lift_max:.3f}.mp4"
                if video_path.exists() and not args.overwrite:
                    pass
                else:
                    render_rollout(env, act_fn, params, stats.seed, video_path, fps=args.fps, width=args.width, height=args.height)
                rec = {
                    "arm": arm,
                    "checkpoint": label,
                    "checkpoint_path": str(ckpt),
                    "video": str(video_path),
                    "seed": stats.seed,
                    "success": stats.success,
                    "lift_max": stats.lift_max,
                    "base_return": stats.base_return,
                    "final_lift": stats.final_lift,
                    "attempts": [s.__dict__ for s in attempts],
                }
                manifest.append(rec)
                print(json.dumps({k: rec[k] for k in ("arm", "checkpoint", "video", "success", "lift_max", "base_return")}))
            (ckpt_dir / "attempts.json").write_text(json.dumps([s.__dict__ for s in attempts], indent=2) + "\n")
            (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return 0


def make_act_fn(
    net: Any,
    bias: CompiledBias,
    arm: str,
    task: str,
    action_transform: str,
    prior_logit_clip: float,
) -> Any:
    """Jitted per-step action: net mean + (optional) the arm's situation-dependent prior.

    The prior is applied exactly as in training (bias.action_prior == weighted_action_prior with
    default weights, which dispatches to the composed prior_fn when a prior_program is present).
    """
    _, use_action_prior, _, _ = BIAS_ARMS[arm]

    def _act(params, obs):
        mean, _log_std, _value = net.apply(params, obs[None, :])
        action = mean[0]
        if use_action_prior:
            prior = bias.action_prior(obs, task)
            if action_transform == "tanh":
                prior = _atanh_clipped(prior, prior_logit_clip)
            action = action + prior
        if action_transform == "tanh":
            action = jp.tanh(action)
        return jp.clip(action, -1.0, 1.0)

    return jax.jit(_act)


def rollout_stats(env: CpuShadowEnv, act_fn: Any, params: Any, seed: int) -> RolloutStats:
    obs = env.reset(seed)
    lift_max = -1e9
    total = 0.0
    success = False
    final_lift = 0.0
    for _ in range(env.horizon):
        action = np.asarray(act_fn(params, jp.asarray(obs)), dtype=np.float32)
        obs, reward, metrics = env.step(action)
        total += reward
        final_lift = metrics["lift"]
        lift_max = max(lift_max, final_lift)
        success = success or bool(metrics["success"])
    return RolloutStats(seed=seed, success=success, lift_max=float(lift_max), base_return=float(total), final_lift=float(final_lift))


def _atanh_clipped(value: jp.ndarray, limit: float) -> jp.ndarray:
    limit = float(min(max(limit, 0.0), 0.999999))
    clipped = jp.clip(value, -limit, limit)
    return 0.5 * (jp.log1p(clipped) - jp.log1p(-clipped))


def render_rollout(
    env: CpuShadowEnv,
    act_fn: Any,
    params: Any,
    seed: int,
    video_path: Path,
    *,
    fps: int,
    width: int,
    height: int,
) -> None:
    obs = env.reset(seed)
    renderer = mujoco.Renderer(env.model, height=height, width=width)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = np.asarray([0.0, 0.0, 0.13])
    camera.distance = 0.75
    camera.azimuth = 135.0
    camera.elevation = -28.0
    render_every = max(1, round((1.0 / fps) / env.cfg.control_dt))
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "-",
        "-an",
        "-vcodec", "libx264",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(video_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        for t in range(env.horizon):
            if t % render_every == 0:
                renderer.update_scene(env.data, camera=camera)
                frame = np.asarray(renderer.render(), dtype=np.uint8)
                assert proc.stdin is not None
                proc.stdin.write(frame.tobytes())
            action = np.asarray(act_fn(params, jp.asarray(obs)), dtype=np.float32)
            obs, _reward, _metrics = env.step(action)
        renderer.update_scene(env.data, camera=camera)
        frame = np.asarray(renderer.render(), dtype=np.uint8)
        assert proc.stdin is not None
        proc.stdin.write(frame.tobytes())
        proc.stdin.close()
        stderr = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
        rc = proc.wait()
        if rc != 0:
            raise RuntimeError(stderr[-2000:])
    finally:
        renderer.close()
        if proc.poll() is None:
            proc.kill()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--arms", default="baseline,reward,action_prior,exploration,supervised_init")
    parser.add_argument("--task", default="lift")
    parser.add_argument("--successes-per-checkpoint", type=int, default=2)
    parser.add_argument("--max-attempts", type=int, default=12)
    parser.add_argument("--seed-base", type=int, default=10000)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--episode-seconds", type=float, default=2.5)
    parser.add_argument("--control-dt", type=float, default=0.025)
    parser.add_argument("--physics-dt", type=float, default=0.01)
    parser.add_argument("--obj-xy-range", type=float, default=0.04)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--checkpoints", choices=["final", "best", "all"], default="final",
                        help="Which checkpoint(s) per arm to render. 'final' = end-of-training policy.")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())

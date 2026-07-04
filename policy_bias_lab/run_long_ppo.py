"""Long-duration PPO training with a FIXED selected prior program.

The agentic selection loop (run_agentic_selection) spends its fixed LLM/eval budget picking a good
prior; this runner answers the follow-up question -- what happens when a policy trains under that
prior for a long time -- with ZERO further LLM calls. One env, one arm, one seed, one continuous
PPO run (single XLA compile), terminating only on:

  - prolonged training plateau: the best-checkpoint metric (sustained contact-gated success +
    gated-progress tie-breaks, the same metric train_ppo_arm uses to pick checkpoints) has not
    improved by > --plateau-eps for --plateau-hours hours of ACTIVE run time, gated by
    --min-hours; or
  - sustained success: the rolling mean (--success-window iters) of the training sustained
    contact-gated hold rate reaches --success-stop.

Pause/resume: SIGINT/SIGTERM (or --stop-after-iters) end the session after the current PPO
iteration and write `resume.pkl` (policy params, Adam optimizer state, iteration count, run-time
clocks, global best); `--resume` continues exactly there, and paused time never counts toward the
wall-clock criteria. `resume.pkl` is also refreshed every --save-every-iters as crash insurance.
Per-iteration metrics append to metrics.jsonl across sessions.

Typical use, after a selection run:
  python -m policy_bias_lab.run_long_ppo --out runs/long1 \
      --program runs/<selection>/best_program.json \
      --min-hours 8 --plateau-hours 2 --success-stop 0.8
  # optional warm start from the selection arbiter's trained weights:
  #   --init-params runs/<selection>/best_params.pkl
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import signal
import sys
import time
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_triton_gemm=false")
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", str(Path(".xla_cache").resolve()))
import jax  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAPPING = ROOT.parent / "bootstrapping"


class RunController:
    """Per-iteration run control: wall-clock criteria, rolling-success stop, resume snapshots,
    and signal-driven pause. Passed to train_ppo_arm as `control_fn`."""

    def __init__(self, args, out: Path, resume_state: dict | None):
        self.args = args
        self.out = out
        self.metrics = (out / "metrics.jsonl").open("a")
        rs = resume_state or {}
        self.wall_prev = float(rs.get("wall_elapsed", 0.0))       # active seconds, prior sessions
        self.t_improve = float(rs.get("t_improve_elapsed", 0.0))  # total-elapsed stamp of last gain
        self.best_score = float(rs.get("best_score", -1.0))       # global across sessions
        self.best_success = float(rs.get("best_success", -1.0))
        self.best_iter = int(rs.get("best_iter", -1))
        self.best_params = rs.get("best_params")
        self.succ_window: list[float] = list(rs.get("succ_window") or [])
        self.iter_offset = int(rs.get("iter_next", 0))
        self.session_iters = 0
        self.last_save_iter = -1
        self.pause = False
        self._last_info = None

        def handler(signum, frame):
            if self.pause:
                raise KeyboardInterrupt
            self.pause = True
            print(f"\n[pause] signal {signum}: finishing the current PPO iteration, then saving "
                  f"resume.pkl and exiting (signal again to abort; last periodic save stands)")
        for s in (signal.SIGINT, signal.SIGTERM):
            signal.signal(s, handler)

    def total_elapsed(self, session_elapsed: float) -> float:
        return self.wall_prev + session_elapsed

    def __call__(self, info: dict) -> str | None:
        self._last_info = info
        self.session_iters += 1
        self.metrics.write(json.dumps(info["row"] | {"seed": self.args.seed}) + "\n")
        self.metrics.flush()
        total = self.total_elapsed(info["elapsed_seconds"])
        if self.session_iters == 1:
            # The first iteration of a session carries the XLA compile (~minutes); that is
            # overhead, not training that failed to improve -- keep it out of the stall window.
            self.t_improve = min(self.t_improve + info["elapsed_seconds"], total)

        # Global best across sessions (train_ppo_arm restarts its own tracker on resume).
        if info["best_score"] > self.best_score + self.args.plateau_eps:
            self.best_score = float(info["best_score"])
            self.best_success = float(info["best_success"])
            self.best_iter = int(info["best_iter"])
            self.best_params = jax.device_get(info["best_params"])
            self.t_improve = total
            print(f"  [best] iter {info['iter']}: metric {self.best_score:.6f} "
                  f"(sustained success {self.best_success:.3f}) at {total / 3600.0:.2f}h")

        self.succ_window.append(float(info["row"].get("success", 0.0)))
        self.succ_window = self.succ_window[-self.args.success_window:]

        if info["iter"] % 25 == 0:
            r = info["row"]
            print(f"  [it {info['iter']}] {total / 3600.0:5.2f}h success={r['success']:.3f} "
                  f"grasp={r['grasp_rate']:.3f} reach={r['reach_rate']:.3f} "
                  f"lift={r['lift_reached_rate']:.3f} base_ret={r['base_return']:+.3f} "
                  f"stalled={(total - self.t_improve) / 3600.0:.2f}h", flush=True)

        if (info["iter"] - self.last_save_iter >= self.args.save_every_iters) or self.pause:
            self.save_resume(info, total)

        if self.pause:
            return "pause"
        if (self.args.success_stop is not None and len(self.succ_window) >= self.args.success_window
                and sum(self.succ_window) / len(self.succ_window) >= self.args.success_stop):
            return (f"success criterion met: sustained hold success mean "
                    f"{sum(self.succ_window) / len(self.succ_window):.3f} over the last "
                    f"{self.args.success_window} iters >= {self.args.success_stop}")
        stall_h = (total - self.t_improve) / 3600.0
        if (self.args.plateau_hours is not None and total >= self.args.min_hours * 3600.0
                and stall_h >= self.args.plateau_hours):
            return (f"training plateau: best metric unimproved for {stall_h:.2f}h "
                    f"(threshold {self.args.plateau_hours}h) after {total / 3600.0:.2f}h run time")
        if self.args.max_hours is not None and total >= self.args.max_hours * 3600.0:
            return f"max run time reached ({self.args.max_hours}h)"
        return None

    def save_resume(self, info: dict, total: float) -> None:
        self.last_save_iter = info["iter"]
        state = {
            "params": jax.device_get(info["params"]),
            "opt_state": jax.device_get(info["opt_state"]),
            "iter_next": int(info["iter"]) + 1,
            "wall_elapsed": total, "t_improve_elapsed": self.t_improve,
            "best_score": self.best_score, "best_success": self.best_success,
            "best_iter": self.best_iter, "best_params": self.best_params,
            "succ_window": self.succ_window,
        }
        tmp = self.out / "resume.pkl.tmp"
        with tmp.open("wb") as f:
            pickle.dump(state, f)
        os.replace(tmp, self.out / "resume.pkl")


def main() -> int:
    args = parse_args()
    if str(BOOTSTRAPPING) not in sys.path:
        sys.path.insert(0, str(BOOTSTRAPPING))
    from mjx_env import make_env

    from policy_bias_lab.bias import compile_bias, default_reward_template_weights
    from policy_bias_lab.es import BIAS_ARMS
    from policy_bias_lab.ppo_bias import PPOBiasConfig, evaluate_ppo_policy, train_ppo_arm

    args.out.mkdir(parents=True, exist_ok=True)
    resume_state = None
    if args.resume:
        rp = args.out / "resume.pkl"
        if not rp.exists():
            print(f"[resume] no {rp}", file=sys.stderr)
            return 1
        with rp.open("rb") as f:
            resume_state = pickle.load(f)
        print(f"[resume] iter {resume_state['iter_next']}, "
              f"{resume_state['wall_elapsed'] / 3600.0:.2f}h run time, "
              f"best metric {resume_state['best_score']:.6f}")

    from policy_bias_lab.reward_modes import CONTRIB_NAMES, ENV_OVERRIDES, build_shaping_fn

    program = json.loads(Path(args.program).read_text())
    (args.out / "prior_program.json").write_text(json.dumps(program, indent=2) + "\n")
    env = make_env("shadow", control_dt=args.control_dt, episode_seconds=args.episode_seconds,
                   physics_dt=args.physics_dt, obj_xy_range=args.obj_xy_range,
                   **ENV_OVERRIDES[args.reward_mode])
    shaping_fn = build_shaping_fn(args.reward_mode, env, program)
    if args.reward_mode != "default":
        print(f"[reward] mode={args.reward_mode}: env overrides {ENV_OVERRIDES[args.reward_mode]}"
              f"; shaping contrib slots {CONTRIB_NAMES}")
    bias = compile_bias({"name": "long_ppo", "action_priors": [], "prior_program": program}, env)
    use_reward, _use_prior, _, _ = BIAS_ARMS[args.arm]
    reward_weights = default_reward_template_weights(args.task)
    if not use_reward:
        reward_weights = reward_weights * 0.0

    cfg = PPOBiasConfig(
        envs=args.envs, lr=args.lr, hidden=tuple(args.hidden), ent_coef=args.ent_coef,
        iters=10 ** 7, target_train_seconds=None, checkpoint_count=0,  # our controller governs
        action_transform=args.action_transform, prior_logit_clip=args.prior_logit_clip,
        success_hold_seconds=args.success_hold_seconds,
        success_lift_threshold=args.success_lift_threshold, warmup_compile=False,
        success_terminate_seconds=args.terminate_on_success,
    )
    if not args.resume:
        (args.out / "config.json").write_text(json.dumps(
            {"learner": "long_ppo", "task": args.task, "arm": args.arm, "seed": args.seed,
             "reward_mode": args.reward_mode,
             "reward_env_overrides": ENV_OVERRIDES[args.reward_mode],
             "reward_contrib_names": list(CONTRIB_NAMES) if args.reward_mode != "default" else None,
             "program": str(args.program), "init_params": str(args.init_params or ""),
             "episode_seconds": args.episode_seconds,
             "terminate_on_success": args.terminate_on_success,
             "criteria": {"min_hours": args.min_hours, "plateau_hours": args.plateau_hours,
                          "plateau_eps": args.plateau_eps, "success_stop": args.success_stop,
                          "success_window": args.success_window, "max_hours": args.max_hours},
             "ppo": cfg.__dict__}, indent=2, default=str) + "\n")

    initial_params = None
    initial_opt_state = None
    iter_offset = 0
    if resume_state is not None:
        initial_params = jax.device_put(resume_state["params"])
        initial_opt_state = jax.device_put(resume_state["opt_state"])
        iter_offset = int(resume_state["iter_next"])
    elif args.init_params:
        with Path(args.init_params).open("rb") as f:
            initial_params = jax.device_put(pickle.load(f))
        print(f"[init] warm start from {args.init_params}")

    ctrl = RunController(args, args.out, resume_state)
    msg = f"[long-ppo] stop on {args.plateau_hours}h plateau after >= {args.min_hours}h"
    if args.success_stop is not None:
        msg += f", or rolling success >= {args.success_stop}"
    if ctrl.wall_prev:
        msg += f" (resuming at {ctrl.wall_prev / 3600.0:.2f}h)"
    print(msg, flush=True)

    if args.stop_after_iters:  # planned pause for this session (also the test hook)
        limit = iter_offset + args.stop_after_iters

        def control_fn(info, _orig=ctrl.__call__):
            if info["iter"] + 1 >= limit:
                ctrl.pause = True  # triggers save_resume + "pause" inside the controller
            return _orig(info)
    else:
        control_fn = ctrl

    state_out: dict = {}
    _params, _rows, _bp, _bs, _bi = train_ppo_arm(
        env=env, bias=bias, task=args.task, arm=args.arm, seed=args.seed + iter_offset, cfg=cfg,
        checkpoint_dir=None, reward_weights=reward_weights, base_reward_weight=1.0,
        action_prior_weights=bias.default_action_prior_weights(),
        initial_params=initial_params, iter_offset=iter_offset,
        initial_opt_state=initial_opt_state, control_fn=control_fn, state_out=state_out,
        shaping_fn=shaping_fn,
    )
    reason = state_out.get("stop_reason") or "iteration cap"
    ctrl.metrics.close()

    if reason == "pause":
        print(f"[paused] at iter {ctrl._last_info['iter'] if ctrl._last_info else iter_offset}, "
              f"{ctrl.total_elapsed(ctrl._last_info['elapsed_seconds']) / 3600.0:.2f}h run time "
              f"-> {args.out / 'resume.pkl'}")
        print(f"         resume with: python -m policy_bias_lab.run_long_ppo --out {args.out} "
              f"--program {args.program} --resume")
        return 0

    print(f"[stop] {reason}")
    total = ctrl.total_elapsed(ctrl._last_info["elapsed_seconds"]) if ctrl._last_info else 0.0
    if ctrl._last_info is not None:
        # Keep resume.pkl current with the FINAL state: a later --resume (e.g. with relaxed
        # criteria) must not replay iterations already logged to metrics.jsonl.
        ctrl.save_resume(ctrl._last_info, total)
    best_params = ctrl.best_params if ctrl.best_params is not None else jax.device_get(_bp)
    with (args.out / "best_params.pkl").open("wb") as f:
        pickle.dump(best_params, f)
    ev = evaluate_ppo_policy(
        env=env, params=jax.device_put(best_params), bias=bias, task=args.task, arm=args.arm,
        seed=args.seed + 10_000, n_envs=args.eval_envs, cfg=cfg, reward_weights=reward_weights,
        base_reward_weight=1.0, action_prior_weights=bias.default_action_prior_weights())
    report = {"stop_reason": reason, "wall_hours": round(total / 3600.0, 3),
              "iters": (ctrl._last_info or {}).get("iter"),
              "best_metric": ctrl.best_score, "best_train_success": ctrl.best_success,
              "best_iter": ctrl.best_iter, "eval": ev}
    (args.out / "final_report.json").write_text(json.dumps(report, indent=2, default=float) + "\n")
    print(f"[done] {total / 3600.0:.2f}h, best sustained success {ctrl.best_success:.3f}, "
          f"eval success {ev.get('eval_success_rate')} -> {args.out / 'final_report.json'}")
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--program", type=Path, required=True,
                   help="selected prior program JSON (e.g. runs/<selection>/best_program.json)")
    p.add_argument("--task", default="lift")
    p.add_argument("--arm", default="freeform_encourage",
                   help="BIAS_ARMS entry; default matches the selection arbiter.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--init-params", type=Path, default=None,
                   help="warm-start weights pkl (e.g. the selection run's best_params.pkl; "
                   "requires the same --hidden as the arbiter: 256 256).")
    p.add_argument("--resume", action="store_true", help="continue from <out>/resume.pkl")
    p.add_argument("--reward-mode", choices=["default", "lift_only", "adjusted", "stage_gated"],
                   default="default",
                   help="experimental reward arms (reward_modes.py): lift_only = env pays only "
                   "contact-gated lift/sustained-lift/success; adjusted = progress-based closure/"
                   "contact + empty-squeeze penalty (the air-closure exploit fixes); stage_gated = "
                   "adjusted terms gated by the prior program's own stage weights. On --resume, "
                   "pass the SAME mode as the original run.")
    # termination criteria
    p.add_argument("--min-hours", type=float, default=8.0,
                   help="no plateau stop before this much ACTIVE run time (pauses excluded).")
    p.add_argument("--plateau-hours", type=float, default=2.0,
                   help="stop when the best-checkpoint metric has not improved this long.")
    p.add_argument("--plateau-eps", type=float, default=1e-5,
                   help="metric gain that resets the plateau clock. At the default, new sustained "
                   "success or gated-lift progress resets it; bare grasp-rate wiggle does not.")
    p.add_argument("--success-stop", type=float, default=0.8,
                   help="stop when rolling mean training sustained-hold success reaches this "
                   "(the 'policy is very successful' criterion).")
    p.add_argument("--success-window", type=int, default=10)
    p.add_argument("--max-hours", type=float, default=None, help="optional hard cap.")
    # session control
    p.add_argument("--stop-after-iters", type=int, default=None,
                   help="pause (save resume.pkl + exit) after this many PPO iters this session.")
    p.add_argument("--save-every-iters", type=int, default=100,
                   help="refresh resume.pkl this often (crash insurance).")
    # PPO/env knobs (defaults match the selection arbiter so warm starts line up)
    p.add_argument("--envs", type=int, default=256)
    p.add_argument("--eval-envs", type=int, default=256)
    p.add_argument("--hidden", type=int, nargs="+", default=[256, 256])
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--ent-coef", type=float, default=0.0)
    p.add_argument("--action-transform", choices=["raw", "tanh"], default="tanh")
    p.add_argument("--prior-logit-clip", type=float, default=0.95)
    p.add_argument("--success-hold-seconds", type=float, default=0.5)
    p.add_argument("--success-lift-threshold", type=float, default=0.05)
    p.add_argument("--episode-seconds", type=float, default=2.5)
    p.add_argument("--control-dt", type=float, default=0.025)
    p.add_argument("--terminate-on-success", type=float, default=None, metavar="SECONDS",
                   help="early termination for credit assignment: once the per-step success "
                        "metric holds for this many consecutive seconds, later steps in the "
                        "episode carry no reward/value/loss (fixed-horizon stepping continues)")
    p.add_argument("--physics-dt", type=float, default=0.01)
    p.add_argument("--obj-xy-range", type=float, default=0.04)
    return p.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())

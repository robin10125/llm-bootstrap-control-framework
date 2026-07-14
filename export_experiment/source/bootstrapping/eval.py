#!/usr/bin/env python3
"""Evaluate trained policies on held-out setups and compare arms.

Loads one or more `params.pkl` checkpoints (with their `config.json`), runs each policy
deterministically on a fixed batch of held-out object positions, and reports success rate
and lift height with bootstrap 95% confidence intervals. This is how the arms
(`rl_only` vs `rl_codex` vs ...) are compared on equal footing.

    python eval.py runs_rl/rl_only/params.pkl runs_rl/rl_codex/params.pkl
    python eval.py runs_rl/*/params.pkl --n 512 --xy 0.06     # wider (generalization)
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import jax
import jax.numpy as jp
import numpy as np

import ppo
from mjx_env import MjxEnv, EnvConfig


def load_policy(params_path: Path):
    params = pickle.load(open(params_path, "rb"))
    cfg_path = params_path.parent / "config.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    hidden = tuple(cfg.get("hidden", (256, 256)))
    return params, cfg, hidden


def make_eval(env: MjxEnv, net):
    step = jax.vmap(env.step)

    def rollout(params, state):
        def body(state, _):
            mean, _ls, _v = net.apply(params, state.obs)
            nstate = step(state, mean)
            return nstate, (nstate.metrics["success"], nstate.metrics["lift"])
        _, (succ, lift) = jax.lax.scan(body, state, None, length=env.horizon)
        return (succ.max(axis=0) > 0.5), lift.max(axis=0)  # per-env

    return jax.jit(rollout)


def bootstrap_ci(x: np.ndarray, reps: int = 2000, seed: int = 0):
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(reps, len(x)))].mean(axis=1)
    return float(x.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def evaluate(params_path: Path, n: int, seed: int, xy: float | None):
    params, cfg, hidden = load_policy(params_path)
    overrides = {}
    if cfg.get("control_dt"):
        overrides["control_dt"] = cfg["control_dt"]
    if cfg.get("episode_seconds"):
        overrides["episode_seconds"] = cfg["episode_seconds"]
    if xy is not None:
        overrides["obj_xy_range"] = xy
    env = MjxEnv(config=EnvConfig(**overrides))
    net = ppo.ActorCritic(action_dim=env.action_size, hidden=hidden)
    reset = jax.jit(jax.vmap(env.reset))
    roll = make_eval(env, net)
    keys = jax.random.split(jax.random.PRNGKey(seed), n)  # held-out stream
    state = reset(keys)
    succ, lift = roll(params, state)
    succ = np.asarray(jax.device_get(succ)).astype(np.float32)
    lift = np.asarray(jax.device_get(lift))
    return succ, lift, cfg


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("params", nargs="+", type=Path)
    p.add_argument("--n", type=int, default=256, help="held-out eval setups")
    p.add_argument("--seed", type=int, default=999, help="held-out RNG seed (!= train)")
    p.add_argument("--xy", type=float, default=None, help="object xy range (default=train)")
    args = p.parse_args()

    print(f"held-out eval: n={args.n} seed={args.seed} "
          f"xy_range={'train-default' if args.xy is None else args.xy}\n")
    print(f"{'arm / path':40s} {'success [95% CI]':28s} {'lift_m [95% CI]':24s}")
    print("-" * 92)
    for pp in args.params:
        succ, lift, cfg = evaluate(pp, args.n, args.seed, args.xy)
        s_m, s_lo, s_hi = bootstrap_ci(succ)
        l_m, l_lo, l_hi = bootstrap_ci(lift)
        label = cfg.get("arm", pp.parent.name)
        print(f"{label:40s} {f'{s_m:.3f} [{s_lo:.3f},{s_hi:.3f}]':28s} "
              f"{f'{l_m:.3f} [{l_lo:.3f},{l_hi:.3f}]':24s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

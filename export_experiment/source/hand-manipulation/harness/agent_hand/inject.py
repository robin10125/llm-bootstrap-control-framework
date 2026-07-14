"""Inject a hand-written control script — the project-constitution loop, done by hand.

Two steps:

  # 1. start the sim and dump all relevant state to a text file for the author to read
  python -m agent_hand.inject dump [--cube-pos X Y] [--out ../inject/state.txt]

  # 2. inject an authored script (a Python function body) and run it against that state
  python -m agent_hand.inject run --script ../scripts/grab_and_lift.py [--cube-pos X Y]

The script runs in the same primitive scope as a generated skill (see prompts.PRIMITIVES_DOC
and sim.py). It is expected to (a) drive toward a clear end state, (b) verify that end state,
and (c) carry its OWN failsafe so it returns a failure instead of running forever — the bridge
also enforces a hard step budget as a backstop. `run` records a trajectory so the result can
be inspected with `python -m agent_hand.stepper --latest`.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from . import prompts
from .bridge import Bridge

REPO = Path(__file__).resolve().parents[2]
DEFAULT_STATE = REPO / "inject" / "state.txt"


def _model_facts(sim) -> str:
    m = sim.model
    lines = ["Actuators (name: ctrlrange):"]
    for i in range(m.nu):
        a = m.actuator(i)
        lines.append(f"  {a.name}: [{a.ctrlrange[0]:.3f}, {a.ctrlrange[1]:.3f}]")
    cube = m.geom("cube").size
    lines.append(f"Cube: box half-extents {cube.round(3).tolist()} (a {2*cube[0]*100:.0f} cm cube).")
    lines.append("Table surface is at z=0; the cube rests with its centre at z≈0.02.")
    lines.append("Fingers hang below the palm; palm height ≈ slide_z. A palm height around")
    lines.append("0.11–0.13 puts the fingers straddling a cube on the table (lower jams them")
    lines.append("into the table). Commanding fingers toward 0.0 squeezes; the cube blocks")
    lines.append("them and that creates the grip force needed to lift it.")
    return "\n".join(lines)


def dump(args) -> None:
    bridge = Bridge()
    cube_pos = args.cube_pos
    bridge.reset(cube_pos=cube_pos)
    obs = bridge.observe()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    text = f"""# MuJoCo hand — simulation state dump
# written {time.strftime('%Y-%m-%d %H:%M:%S')}

## GOAL
Make the gripper grab the red cube and lift it at least 10 cm off the table, keeping it
gripped. Success end state: grasped('cube') is True AND obj_pos('cube')[2] >= 0.12.

## RESET
cube_pos = {cube_pos if cube_pos else '[0.0, 0.0] (default, centred)'}
(Re-run `inject run` with the same --cube-pos so the script sees this exact state.)

## CURRENT OBSERVATION
```json
{json.dumps(obs, indent=2)}
```

## MODEL FACTS
{_model_facts(bridge.sim)}

## CONTROL API (the script is the BODY of a Python function; these are in scope)
{prompts.PRIMITIVES_DOC}
"""
    out.write_text(text)
    print(f"wrote sim state -> {out}")
    print(f"  cube at {obs['cube']['pos']}, palm at {obs['palm']['pos']}, grasped={obs['grasped']}")


def run(args) -> None:
    code = Path(args.script).read_text()
    bridge = Bridge()
    bridge.reset(cube_pos=args.cube_pos)
    bridge.enable_recording()
    feedback = bridge.execute(code, timeout_ms=args.timeout_ms)

    run_dir = Path(args.runs) / f"{time.strftime('%Y%m%d-%H%M%S')}-inject_{Path(args.script).stem}"
    run_dir.mkdir(parents=True, exist_ok=True)
    bridge.save_trajectory(run_dir / "trajectory.npz")
    (run_dir / "script.py").write_text(code)
    (run_dir / "feedback.json").write_text(json.dumps(feedback, indent=2))

    after = feedback["observationAfter"]
    end_state_ok = bool(after["grasped"] and after["cube"]["z"] >= 0.12)
    ran_clean = feedback["success"]

    print("\n=== inject result ===")
    print(f"script: {args.script}")
    print(f"ran without error: {ran_clean}")
    if feedback["error"]:
        print(f"  error: {feedback['error']['type']}: {feedback['error']['message']}")
    print(f"returned: {feedback['result']}")
    for line in feedback["logs"]:
        print(f"  log: {line}")
    print(f"final: cube_z={after['cube']['z']} grasped={after['grasped']} "
          f"steps={feedback['stepsUsed']}")
    print(f"END STATE satisfied (grasped & z>=0.12): {end_state_ok}")
    print(f"trajectory -> {run_dir/'trajectory.npz'}  (view: python -m agent_hand.stepper --latest)")
    raise SystemExit(0 if (ran_clean and end_state_ok) else 1)


def main() -> None:
    ap = argparse.ArgumentParser(description="Dump sim state / inject a control script.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("dump", help="start the sim and write its state to a text file")
    d.add_argument("--cube-pos", type=float, nargs=2, default=None, metavar=("X", "Y"))
    d.add_argument("--out", default=str(DEFAULT_STATE))
    d.set_defaults(func=dump)

    r = sub.add_parser("run", help="inject and execute an authored control script")
    r.add_argument("--script", required=True, help="path to a Python function-body script")
    r.add_argument("--cube-pos", type=float, nargs=2, default=None, metavar=("X", "Y"))
    r.add_argument("--runs", default=str(REPO / "runs"))
    r.add_argument("--timeout-ms", type=int, default=0, help="hard step-budget backstop (0=default)")
    r.set_defaults(func=run)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

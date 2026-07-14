"""CLI entry point.

  python -m agent_hand.run --task ../tasks/lift_cube.yaml            # real LLM
  python -m agent_hand.run --task ../tasks/lift_cube.yaml --mock     # canned skill, no API
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .bridge import Bridge
from .controller import Controller
from .llm import AnthropicLLM, MockLLM
from .skills import SkillLibrary
from .tasks import Task

REPO = Path(__file__).resolve().parents[2]


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run a code-as-action episode on the MuJoCo hand.")
    ap.add_argument("--task", required=True, help="path to a task YAML")
    ap.add_argument("--skills", default=str(REPO / "skills"))
    ap.add_argument("--runs", default=str(REPO / "runs"))
    ap.add_argument("--mock", action="store_true", help="use the canned MockLLM (no API key)")
    ap.add_argument("--model", default=None, help="override the Anthropic model id")
    ap.add_argument("--timeout-ms", type=int, default=0,
                    help="per-skill wall budget in ms (0 = sim default step budget)")
    ap.add_argument("--gen-mode", default="scratch",
                    choices=["scratch", "edit-nearest", "skeleton"],
                    help="starting point for generation (comparable knob)")
    ap.add_argument("--max-depth", type=int, default=3,
                    help="max subgoal-decomposition depth before forcing a leaf script")
    ap.add_argument("--no-record", action="store_true",
                    help="disable control-timeline recording for the step-through viewer")
    args = ap.parse_args()

    task = Task.load(args.task)
    bridge = Bridge()
    library = SkillLibrary(args.skills)
    llm = MockLLM() if args.mock else AnthropicLLM(model=args.model)

    controller = Controller(bridge, library, llm, args.runs,
                            gen_mode=args.gen_mode, max_depth=args.max_depth,
                            exec_timeout_ms=args.timeout_ms,
                            record=not args.no_record)
    result = controller.run_episode(task)

    print("\n=== episode result ===")
    print(f"task={result.task} success={result.success} "
          f"iterations={result.iterations}\nlogs: {result.run_dir}")
    raise SystemExit(0 if result.success else 1)


if __name__ == "__main__":
    main()

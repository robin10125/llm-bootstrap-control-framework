from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from llm_framework.adapters.experiment_runtime import make_runtime_env
from llm_framework.core.metrics import summarize, write_json, write_metrics_csv
from llm_framework.core.state import RolloutResult, SafetyLimits
from llm_framework.core.tasks import build_task_context
from llm_framework.core.world import world_from_env_state
from llm_framework.interfaces.base import interface_by_name
from llm_framework.llm.worker import complete_for_interface
from llm_framework.runtime.rollout import rollout_compiled


def main() -> int:
    args = parse_args()
    out_dir = args.out
    if out_dir is None:
        out_dir = Path("runs") / f"compare_{time.strftime('%Y%m%d-%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    env_overrides: dict[str, Any] = {}
    if args.episode_seconds is not None:
        env_overrides["episode_seconds"] = args.episode_seconds
    if args.control_dt is not None:
        env_overrides["control_dt"] = args.control_dt
    env = make_runtime_env(args.env, **env_overrides)
    interfaces = [interface_by_name(name.strip()) for name in args.interfaces.split(",") if name.strip()]
    tasks = [name.strip() for name in args.tasks.split(",") if name.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    limits = SafetyLimits(max_episode_seconds=float(env.cfg.episode_seconds))

    results: list[RolloutResult] = []
    manifest: dict[str, Any] = {
        "env": args.env,
        "interfaces": [i.name for i in interfaces],
        "tasks": tasks,
        "seeds": seeds,
        "backend": args.backend,
        "llm_model": args.llm_model,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "runs": [],
    }

    for task_name in tasks:
        for seed in seeds:
            state = reset_env(env, seed)
            world = world_from_env_state(env, state)
            ctx = build_task_context(task_name, seed, world, float(env.cfg.episode_seconds))
            for interface in interfaces:
                for attempt in range(args.budget_per_interface):
                    tag = f"{task_name}_s{seed}_{interface.name}_a{attempt}"
                    run_dir = out_dir / tag
                    run_dir.mkdir(parents=True, exist_ok=True)
                    write_json(run_dir / "context.json", {
                        "task": ctx.compact(),
                        "world": world.compact(),
                    })
                    prompt = interface.build_prompt(ctx, world)
                    (run_dir / "prompt.md").write_text(prompt)
                    completion = complete_for_interface(
                        interface,
                        ctx,
                        world,
                        backend=args.backend,
                        model=args.llm_model,
                        log_dir=run_dir / "llm",
                        tag=tag,
                    )
                    (run_dir / "completion.txt").write_text(completion.text)
                    record: dict[str, Any] = {
                        "task": ctx.compact(),
                        "interface": interface.name,
                        "seed": seed,
                        "attempt": attempt,
                        "llm": {
                            "source": completion.source,
                            "ok": completion.ok,
                            "error": completion.error,
                            "metadata": completion.metadata,
                        },
                    }
                    program = None
                    if not completion.ok:
                        result = error_result(interface.name, ctx.name, seed, ctx.object_start, completion.error or "LLM call failed")
                    else:
                        try:
                            program = interface.parse(completion.text)
                            validation = interface.validate(program, limits)
                            write_json(run_dir / "program.json", program.source)
                            record["validation"] = {
                                "ok": validation.ok,
                                "errors": list(validation.errors),
                                "warnings": list(validation.warnings),
                            }
                            if not validation.ok:
                                result = error_result(interface.name, ctx.name, seed, ctx.object_start, "; ".join(validation.errors))
                            else:
                                result = compile_and_rollout(interface, program, ctx, world, env, seed, run_dir)
                        except Exception as exc:  # keep the compare run going
                            result = error_result(interface.name, ctx.name, seed, ctx.object_start, f"{type(exc).__name__}: {exc}")
                    record["result"] = result.row()
                    record["result_metadata"] = result.metadata
                    write_json(run_dir / "result.json", record)
                    results.append(result)
                    manifest["runs"].append({"tag": tag, "result": result.row()})
                    if completion.ok and program is not None:
                        repair_results = run_repairs(
                            args,
                            interface,
                            program,
                            result,
                            ctx,
                            world,
                            env,
                            seed,
                            out_dir,
                            tag,
                            limits,
                        )
                        for repair_tag, repair_result in repair_results:
                            results.append(repair_result)
                            manifest["runs"].append({"tag": repair_tag, "result": repair_result.row()})

    write_metrics_csv(out_dir / "metrics.csv", results)
    summary = summarize(results)
    write_json(out_dir / "summary.json", {"summary": summary, "manifest": manifest})
    (out_dir / "report.md").write_text(report_markdown(summary, results))
    print(f"Wrote {out_dir}")
    print(json.dumps(summary, indent=2))
    return 0


def compile_and_rollout(interface, program, ctx, world, env, seed: int, run_dir: Path) -> RolloutResult:
    compiled = interface.compile(program, ctx, world, env)
    np.save(run_dir / "action_stream.npy", compiled.action_stream)
    write_json(run_dir / "compiled_summary.json", {
        "interface": compiled.interface,
        "action_shape": list(compiled.action_stream.shape),
        "metadata": compiled.metadata,
        "action_min": float(compiled.action_stream.min()),
        "action_max": float(compiled.action_stream.max()),
    })
    rollout_state = reset_env(env, seed)
    return rollout_compiled(env, rollout_state, compiled, ctx, seed=seed)


def run_repairs(args, interface, program, result, ctx, world, env, seed: int, out_dir: Path,
                parent_tag: str, limits: SafetyLimits) -> list[tuple[str, RolloutResult]]:
    if args.repair_attempts <= 0 or result.success:
        return []
    repair_fn = getattr(interface, "repair_with_llm", None)
    if repair_fn is None:
        return []
    out: list[tuple[str, RolloutResult]] = []
    prev_program = program
    prev_result = result
    for repair_idx in range(args.repair_attempts):
        repair_tag = f"{parent_tag}_repair{repair_idx + 1}"
        repair_dir = out_dir / repair_tag
        repair_dir.mkdir(parents=True, exist_ok=True)
        completion = repair_fn(
            ctx,
            world,
            prev_program,
            prev_result,
            backend=args.backend,
            model=args.llm_model,
            log_dir=repair_dir / "llm",
            tag=repair_tag,
        )
        (repair_dir / "completion.txt").write_text(completion.text)
        record: dict[str, Any] = {
            "task": ctx.compact(),
            "interface": interface.name,
            "seed": seed,
            "repair_attempt": repair_idx + 1,
            "parent_result": prev_result.row(),
            "parent_result_metadata": prev_result.metadata,
            "llm": {
                "source": completion.source,
                "ok": completion.ok,
                "error": completion.error,
                "metadata": completion.metadata,
            },
        }
        if not completion.ok:
            repair_result = error_result(interface.name, ctx.name, seed, ctx.object_start, completion.error or "repair LLM call failed")
        else:
            try:
                repaired_program = interface.parse(completion.text)
                validation = interface.validate(repaired_program, limits)
                write_json(repair_dir / "program.json", repaired_program.source)
                record["validation"] = {
                    "ok": validation.ok,
                    "errors": list(validation.errors),
                    "warnings": list(validation.warnings),
                }
                if not validation.ok:
                    repair_result = error_result(interface.name, ctx.name, seed, ctx.object_start, "; ".join(validation.errors))
                else:
                    repair_result = compile_and_rollout(interface, repaired_program, ctx, world, env, seed, repair_dir)
                    prev_program = repaired_program
            except Exception as exc:
                repair_result = error_result(interface.name, ctx.name, seed, ctx.object_start, f"{type(exc).__name__}: {exc}")
        record["result"] = repair_result.row()
        record["result_metadata"] = repair_result.metadata
        write_json(repair_dir / "result.json", record)
        out.append((repair_tag, repair_result))
        prev_result = repair_result
        if repair_result.success:
            break
    return out


def reset_env(env: Any, seed: int):
    import jax

    return env.reset(jax.random.PRNGKey(seed))


def error_result(interface: str, task: str, seed: int, object_start: tuple[float, float, float], error: str) -> RolloutResult:
    return RolloutResult(
        interface=interface,
        task=task,
        seed=seed,
        success=False,
        score=0.0,
        total_return=0.0,
        final_object_pos=object_start,
        max_object_z=object_start[2],
        errors=(error,),
    )


def report_markdown(summary: dict[str, Any], results: list[RolloutResult]) -> str:
    lines = ["# Interface Comparison", "", "## Summary", ""]
    for name, row in summary.items():
        lines.append(f"- `{name}`: success_rate={row['success_rate']}, mean_score={row['mean_score']}, n={row['n']}")
    lines += ["", "## Runs", ""]
    for result in results:
        lines.append(
            f"- `{result.interface}` task=`{result.task}` seed={result.seed} "
            f"success={result.success} score={result.score} errors={'; '.join(result.errors)}"
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="shadow", choices=["shadow"])
    parser.add_argument("--interfaces", default="waypoint,script_dsl,hybrid,latent_stub")
    parser.add_argument("--backend", default="mock", choices=["mock", "codex", "claude-code", "anthropic"])
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--tasks", default="lift")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--budget-per-interface", type=int, default=1)
    parser.add_argument("--repair-attempts", type=int, default=0)
    parser.add_argument("--episode-seconds", type=float, default=None)
    parser.add_argument("--control-dt", type=float, default=None)
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())

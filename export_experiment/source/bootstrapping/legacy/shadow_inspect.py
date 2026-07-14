#!/usr/bin/env python3
"""Create human/video inspection artifacts for a Shadow Hand primitive policy."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import mujoco
from PIL import Image, ImageDraw

from shadow_policy_runner import DEFAULT_KEYFRAMES, DEFAULT_MODEL, ShadowPolicyRunner, evaluate_checks
from shadow_residual_prompt import build_residual_prompt


DEFAULT_OUT_ROOT = Path("runs")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--keyframes", type=Path, default=DEFAULT_KEYFRAMES)
    args = parser.parse_args()

    policy = json.loads(args.policy.read_text())
    out_dir = args.out_dir or DEFAULT_OUT_ROOT / f"{time.strftime('%Y%m%d-%H%M%S')}-{policy.get('name', 'shadow')}-inspect"
    out_dir.mkdir(parents=True, exist_ok=True)

    runner = ShadowPolicyRunner(args.model, args.keyframes)
    result = runner.run(policy)
    (out_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    (out_dir / "control_instructions.json").write_text(json.dumps(policy, indent=2) + "\n")
    (out_dir / "control_instructions.md").write_text(control_instructions_markdown(policy, result))
    (out_dir / "primitive_report.md").write_text(primitive_report_markdown(runner, policy))
    primitives = json.loads(Path("shadow_primitives.json").read_text())
    residual_prompt = build_residual_prompt(
        policy,
        result,
        primitives,
        "Edit the primitive schedule as a residual policy to improve the next rollout.",
    )
    (out_dir / "residual_policy_prompt.md").write_text(residual_prompt)

    video_path = out_dir / "shadow_policy.mp4"
    render_policy_video(policy, runner, video_path, fps=args.fps, width=args.width, height=args.height)

    summary = {
        "out_dir": str(out_dir),
        "video": str(video_path),
        "success": result["success"],
        "control_instructions": str(out_dir / "control_instructions.md"),
        "primitive_report": str(out_dir / "primitive_report.md"),
        "residual_prompt": str(out_dir / "residual_policy_prompt.md"),
        "result": str(out_dir / "result.json"),
    }
    print(json.dumps(summary, indent=2))
    return 0 if result["success"] else 1


def render_policy_video(
    policy: dict[str, Any],
    runner: ShadowPolicyRunner,
    video_path: Path,
    *,
    fps: int,
    width: int,
    height: int,
) -> None:
    renderer = mujoco.Renderer(runner.model, height=height, width=width)
    frames_written = 0
    runner.reset(policy.get("setup", {}))

    with tempfile.TemporaryDirectory() as tmp:
        frame_dir = Path(tmp)

        def capture(label: str) -> None:
            nonlocal frames_written
            renderer.update_scene(runner.data)
            image = Image.fromarray(renderer.render())
            draw_overlay(image, label, runner.observe())
            image.save(frame_dir / f"frame_{frames_written:06d}.png")
            frames_written += 1

        capture("initial")
        for idx, step in enumerate(policy.get("steps", []), start=1):
            runner._validate_step(step, idx)
            runner._command_step(step)
            label = f"{idx}. {step['primitive']} {json.dumps(step.get('params', {}), separators=(',', ':'))}"
            runner._step_seconds(float(step["duration_s"]), frame_callback=lambda label=label: capture(label), fps=fps)
            capture(label + " done")

        if frames_written == 1:
            capture("final")

        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", str(frame_dir / "frame_%06d.png"),
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(video_path),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def draw_overlay(image: Image.Image, label: str, obs: dict[str, Any]) -> None:
    draw = ImageDraw.Draw(image)
    lines = [
        label,
        f"t={obs['time_s']:.2f}s base=({obs['base']['x']:.3f},{obs['base']['y']:.3f},{obs['base']['z']:.3f})",
        f"object=({obs['object']['x']:.3f},{obs['object']['y']:.3f},{obs['object']['z']:.3f}) "
        f"contacts={obs['contacts']['hand_object_count']} grasped={obs['grasped']} lifted={obs['lifted']}",
    ]
    pad = 10
    line_h = 18
    box_w = max(draw.textlength(line) for line in lines) + pad * 2
    box_h = line_h * len(lines) + pad * 2
    draw.rectangle([0, 0, box_w, box_h], fill=(0, 0, 0))
    for i, line in enumerate(lines):
        draw.text((pad, pad + i * line_h), line, fill=(255, 255, 255))


def control_instructions_markdown(policy: dict[str, Any], result: dict[str, Any]) -> str:
    lines = [
        "# Control Instructions",
        "",
        f"Policy: `{policy.get('name', 'unnamed')}`",
        "",
        f"Goal: {policy.get('goal', '')}",
        "",
        f"Result: `{'success' if result.get('success') else 'failure'}`",
        "",
        "## Primitive Schedule",
        "",
        "| Step | Primitive | Parameters | Duration | Assertions |",
        "| ---: | --- | --- | ---: | --- |",
    ]
    for idx, step in enumerate(policy.get("steps", []), start=1):
        assertions = step.get("assert", [])
        lines.append(
            f"| {idx} | `{step['primitive']}` | `{json.dumps(step.get('params', {}), separators=(',', ':'))}` "
            f"| {step['duration_s']}s | `{json.dumps(assertions, separators=(',', ':'))}` |"
        )
    lines.extend([
        "",
        "## Success Checks",
        "",
        "```json",
        json.dumps(policy.get("success", []), indent=2),
        "```",
        "",
        "## Raw LLM/Policy JSON",
        "",
        "```json",
        json.dumps(policy, indent=2),
        "```",
    ])
    return "\n".join(lines) + "\n"


def primitive_report_markdown(runner: ShadowPolicyRunner, policy: dict[str, Any]) -> str:
    primitive_spec = json.loads(Path("shadow_primitives.json").read_text())
    lines = [
        "# Shadow Primitive Report",
        "",
        "## Summary",
        "",
        "The control system exposes a parameterized primitive action space. The LLM or policy chooses "
        "a primitive, its parameters, and a duration. The runner validates the command, applies "
        "targets in MuJoCo, steps physics, and records observation/contact feedback.",
        "",
        "## Available Primitives",
        "",
    ]
    for name, spec in primitive_spec["primitives"].items():
        lines.extend([
            f"### `{name}`",
            "",
            spec["description"],
            "",
            f"- Duration: `{spec['duration_s']['min']}..{spec['duration_s']['max']}s`",
        ])
        params = spec.get("params", {})
        if params:
            for param, meta in params.items():
                if "enum" in meta:
                    lines.append(f"- `{param}`: one of `{', '.join(meta['enum'])}`")
                else:
                    lines.append(f"- `{param}`: `{meta['min']}..{meta['max']} {meta.get('unit', '')}`".rstrip())
        else:
            lines.append("- Parameters: none")
        lines.append("")

    lines.extend([
        "## MuJoCo Actuator Surface",
        "",
        "| Actuator | Control Range |",
        "| --- | ---: |",
    ])
    for i in range(runner.model.nu):
        actuator = runner.model.actuator(i)
        lo, hi = actuator.ctrlrange
        lines.append(f"| `{actuator.name}` | `{lo:.4f} .. {hi:.4f}` |")

    lines.extend([
        "",
        "## Named Hand Poses",
        "",
    ])
    for pose_name in sorted(runner.pose_targets):
        lines.append(f"- `{pose_name}`")

    lines.extend([
        "",
        "## Observation Metrics Used For Checks",
        "",
        "- `contacts.hand_object_count`: number of distinct Shadow Hand bodies touching the cube.",
        "- `grasped`: true when at least two distinct hand bodies contact the cube.",
        "- `lifted`: true when object z exceeds 0.06m.",
        "- `object.z`: cube center height in metres.",
        "",
        "## Current Policy Success Criteria",
        "",
        "```json",
        json.dumps(policy.get("success", []), indent=2),
        "```",
        "",
        "## Notes",
        "",
        "- Positive `set_base.z` lowers the palm in the current palm-down scene.",
        "- `hand_pose` changes only Shadow Hand actuators and preserves the current base target.",
        "- The current baseline verifies contact grasp, not lift stability.",
    ])
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())

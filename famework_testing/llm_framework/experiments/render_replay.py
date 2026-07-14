from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from llm_framework.adapters.experiment_runtime import make_runtime_env


def main() -> int:
    args = parse_args()
    action_path = args.action_stream or args.run_dir / "action_stream.npy"
    result_path = args.run_dir / "result.json"
    if not action_path.exists():
        raise FileNotFoundError(f"missing action stream: {action_path}")
    if not result_path.exists():
        raise FileNotFoundError(f"missing result record: {result_path}")

    record = json.loads(result_path.read_text())
    seed = int(record.get("seed", args.seed))
    env = make_runtime_env(
        "shadow",
        **{k: v for k, v in {
            "episode_seconds": args.episode_seconds,
            "control_dt": args.control_dt,
        }.items() if v is not None},
    )
    actions = np.load(action_path)
    out = args.out or args.run_dir / "replay.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    render_action_stream(env, actions, seed=seed, video_path=out, fps=args.fps, width=args.width, height=args.height)
    print(out)
    return 0


def render_action_stream(
    env: Any,
    actions: np.ndarray,
    *,
    seed: int,
    video_path: Path,
    fps: int,
    width: int,
    height: int,
) -> None:
    import jax
    import jax.numpy as jp
    import mujoco
    from mujoco import mjx
    state = env.reset(jax.random.PRNGKey(seed))
    step = jax.jit(env.step)
    renderer = mujoco.Renderer(env.model, height=height, width=width)
    control_dt = float(env.cfg.control_dt)
    capture_stride = max(1, round(1.0 / (fps * control_dt)))
    frames_written = 0

    def capture(frame_dir: Path, label: str) -> None:
        nonlocal frames_written
        data = mjx.get_data(env.model, state.data)
        renderer.update_scene(data)
        image = np.asarray(renderer.render(), dtype=np.uint8)
        _write_ppm(frame_dir / f"frame_{frames_written:06d}.ppm", image)
        frames_written += 1

    with tempfile.TemporaryDirectory() as tmp:
        frame_dir = Path(tmp)
        capture(frame_dir, "initial")
        for idx, action in enumerate(actions):
            state = step(state, jp.asarray(action))
            if idx % capture_stride == 0 or idx == len(actions) - 1:
                capture(frame_dir, f"step {idx + 1}/{len(actions)}")
        _write_mp4(frame_dir, video_path, fps)

    renderer.close()


def _write_ppm(path: Path, image: np.ndarray) -> None:
    height, width, channels = image.shape
    if channels != 3:
        raise ValueError(f"expected RGB image, got shape {image.shape}")
    with path.open("wb") as f:
        f.write(f"P6\n{width} {height}\n255\n".encode("ascii"))
        f.write(image.tobytes())


def _write_mp4(frame_dir: Path, video_path: Path, fps: int) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(frame_dir / "frame_%06d.ppm"),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(video_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--action-stream", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episode-seconds", type=float, default=None)
    parser.add_argument("--control-dt", type=float, default=None)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())

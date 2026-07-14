#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "source"
FRAMEWORK = SOURCE / "llm-framework"
BOOTSTRAPPING = SOURCE / "bootstrapping"
HAND = SOURCE / "hand-manipulation"

for path in (FRAMEWORK, FRAMEWORK / "famework_testing", BOOTSTRAPPING, HAND / "harness"):
    sys.path.insert(0, str(path))


def ok(message: str) -> None:
    print(f"OK: {message}")


def fail(message: str, errors: list[str]) -> None:
    print(f"FAIL: {message}")
    errors.append(message)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expect", choices=("cpu", "gpu", "any"), default="any")
    args = parser.parse_args()
    errors: list[str] = []

    ok(f"platform is {platform.system()} {platform.machine()}")
    if args.expect == "gpu" and platform.system() != "Linux":
        fail("the NVIDIA GPU setup in this bundle requires Linux", errors)

    if sys.version_info >= (3, 11):
        ok(f"Python {platform.python_version()}")
    else:
        fail(f"expected Python 3.11 or newer, found {platform.python_version()}", errors)

    modules = (
        "numpy",
        "jax",
        "mujoco",
        "mujoco.mjx",
        "flax",
        "optax",
        "yaml",
        "anthropic",
        "PIL",
        "matplotlib",
        "pytest",
        "mjx_env",
        "ppo",
        "eval_metrics",
        "llm_backend",
        "policy_bias_lab.agentic_orchestrator",
        "llm_framework.experiments.compare_interfaces",
        "agent_hand.sim",
    )
    for name in modules:
        try:
            importlib.import_module(name)
            ok(f"import {name}")
        except Exception as exc:  # noqa: BLE001 - installation diagnostic
            fail(f"import {name}: {type(exc).__name__}: {exc}", errors)

    try:
        import jax

        devices = jax.devices()
        platforms = {device.platform for device in devices}
        ok("JAX devices: " + ", ".join(map(str, devices)))
        if args.expect == "gpu" and "gpu" not in platforms:
            fail("GPU mode requested but JAX found no GPU device", errors)
        if args.expect == "cpu" and "cpu" not in platforms:
            fail("CPU mode requested but JAX found no CPU device", errors)
    except Exception as exc:  # noqa: BLE001
        fail(f"JAX device discovery: {type(exc).__name__}: {exc}", errors)

    model_paths = (
        HAND / "env/models/gripper_cube.xml",
        HAND / "env/models/shadow_hand/scene_cube.xml",
    )
    try:
        import mujoco

        for model_path in model_paths:
            mujoco.MjModel.from_xml_path(str(model_path))
            ok(f"MuJoCo model and assets: {model_path.relative_to(ROOT)}")
    except Exception as exc:  # noqa: BLE001
        fail(f"MuJoCo model loading: {type(exc).__name__}: {exc}", errors)

    for command in ("ffmpeg",):
        path = shutil.which(command)
        if path:
            ok(f"external command {command}: {path}")
        else:
            fail(f"external command not found: {command}", errors)

    if args.expect == "gpu":
        nvidia_smi = shutil.which("nvidia-smi")
        if not nvidia_smi:
            fail("nvidia-smi not found", errors)
        else:
            result = subprocess.run(
                [nvidia_smi, "--query-gpu=name,driver_version", "--format=csv,noheader"],
                check=False,
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                ok("NVIDIA driver: " + result.stdout.strip().replace("\n", "; "))
            else:
                fail("nvidia-smi failed: " + result.stderr.strip(), errors)

    for optional in ("codex", "claude"):
        path = shutil.which(optional)
        if path:
            ok(f"optional LLM CLI {optional}: {path}")
        else:
            print(f"NOTE: optional LLM CLI not installed: {optional}")
    if os.environ.get("ANTHROPIC_API_KEY"):
        ok("ANTHROPIC_API_KEY is set (value not displayed)")
    else:
        print("NOTE: ANTHROPIC_API_KEY is not set; mock/fixture modes still work")

    print()
    if errors:
        print(f"Verification failed with {len(errors)} problem(s).")
        return 1
    print("Verification passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

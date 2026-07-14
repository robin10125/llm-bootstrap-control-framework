# Agent setup guide

This archive contains the source and assets for a robot control experiment. Python and CUDA packages are
downloaded on the destination so they match its hardware and operating system.

## 1. Get administrator setup

Give the administrator `README_ADMIN.md` and the extracted archive.

Ask for GPU setup if the experiments will use an NVIDIA GPU:

```bash
sudo ./setup_system_ubuntu.sh --gpu
```

Ask for CPU setup otherwise:

```bash
sudo ./setup_system_ubuntu.sh --cpu
```

If a driver was installed or changed, reboot before continuing. Confirm that
this succeeds after reboot:

```bash
nvidia-smi
```

The project needs Python 3.11 or newer. On older Ubuntu releases, the
administrator may need to install a newer Python and its matching `venv`
package separately.

## 2. Install Python dependencies

Administrator privileges are not needed for this step.

For an NVIDIA GPU:

```bash
./setup_python.sh --gpu
```

For CPU-only use:

```bash
./setup_python.sh --cpu
```

To select a specific Python 3.11+ interpreter:

```bash
./setup_python.sh --gpu --python /usr/bin/python3.11
```

The script creates `.venv` inside this directory and downloads packages from
the configured pip index. It respects `PIP_INDEX_URL`, `HTTPS_PROXY`, and other
normal pip settings.

## 3. Activate and verify

Run this in every new shell:

```bash
source ./env.sh
```

Verify the installation:

```bash
python ./verify_install.py --expect gpu
# or
python ./verify_install.py --expect cpu
```

Run the framework tests:

```bash
cd "$LLM_FRAMEWORK_ROOT"
python -m pytest famework_testing/tests
```

## 4. Configure an LLM backend

Fixture and mock modes need no credentials.

For the Anthropic Python backend:

```bash
export ANTHROPIC_API_KEY='...'
```

For Codex and Claude command-line backends:

```bash
cd "$EXPERIMENT_EXPORT_ROOT"
./setup_llm_clis.sh --all
source ./env.sh
```

Complete each CLI's local login flow. Never store credentials in this archive.

## 5. Run experiments

Load the environment first:

```bash
source /path/to/export_experiment/env.sh
cd "$LLM_FRAMEWORK_ROOT"
```

Experiment commands are documented in:

- `source/llm-framework/README.md`
- `source/llm-framework/policy_bias_lab/README.md`
- `source/bootstrapping/README.md`
- `source/hand-manipulation/README.md`

Hand-manipulation mock smoke test:

```bash
cd "$HAND_MANIPULATION_ROOT/harness"
python -m agent_hand.run --task ../tasks/lift_cube.yaml --mock
```

Run long GPU jobs sequentially on an 8 GB GPU. Initial MJX/JAX/XLA compilation
can take several minutes.

## Platform notes

- Ubuntu and Xubuntu: use the included admin script.
- Other Linux distributions: install equivalent system packages manually.
- A different x86-64 CPU is normally fine.
- ARM versus x86-64 is a platform change; pip must provide compatible wheels.
- NVIDIA support depends on the destination GPU, driver, architecture, and
  available JAX wheels.
- A complete CUDA toolkit is normally unnecessary. The GPU Python installation
  downloads CUDA and cuDNN user-space libraries into `.venv`.

If the destination is offline, download wheels on a machine matching its OS,
architecture, and Python version. Do not reuse wheels from unrelated hardware.

## Rendering

- Desktop: `MUJOCO_GL=glfw`
- Headless NVIDIA: `MUJOCO_GL=egl`
- Headless CPU: `MUJOCO_GL=osmesa`, where supported

FFmpeg must be available on `PATH`.

## Archive integrity

From the extracted directory:

```bash
sha256sum -c MANIFEST.sha256
```


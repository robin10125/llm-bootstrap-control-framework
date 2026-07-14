# Administrator setup

## Purpose

This script installs system packages for a Python MuJoCo/JAX experiment.

Supported system: Ubuntu or Xubuntu.

## Run

CPU:

```bash
sudo ./setup_system_ubuntu.sh --cpu
```

NVIDIA GPU:

```bash
sudo ./setup_system_ubuntu.sh --gpu
```

## Packages installed

Both modes install:

- `build-essential`
- `ca-certificates`
- `ffmpeg`
- `libegl1`
- `libgl1`
- `libglfw3`
- `libosmesa6`
- `nodejs`
- `npm`
- `python3`
- `python3-pip`
- `python3-venv`
- `unzip`

GPU mode also installs:

- `ubuntu-drivers-common`
- Headers for the running Linux kernel, when available
- Ubuntu's recommended NVIDIA driver, if no working driver is detected

`apt-get` may install normal transitive dependencies.

## Python requirement

Python 3.11 or newer is required.

The script stops if Ubuntu's default Python is older. Install Python 3.11+ and
its matching `venv` package from an approved source. Tell the user its path.

## GPU completion

If the NVIDIA driver changes:

1. Reboot.
2. Complete Secure Boot MOK enrollment if prompted.
3. Run `nvidia-smi` and confirm that it succeeds.

## Not installed by this script

- CUDA toolkit
- Python packages inside the project virtual environment
- Codex or Claude CLI packages
- API keys or login credentials

The user installs Python and CUDA user-space packages without root access by
running `setup_python.sh`.


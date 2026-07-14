#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage: sudo setup_system_ubuntu.sh --cpu|--gpu

Installs Ubuntu/Xubuntu system packages. In GPU mode it also asks Ubuntu's
hardware detector to install the recommended NVIDIA kernel driver when needed.
EOF
}

mode=""
while (($#)); do
  case "$1" in
    --cpu) mode="cpu" ;;
    --gpu) mode="gpu" ;;
    -h|--help) usage; exit 0 ;;
    *) usage; exit 2 ;;
  esac
  shift
done

if [[ -z "$mode" ]]; then
  usage
  exit 2
fi

if [[ ${EUID:-$(id -u)} -ne 0 ]]; then
  echo "ERROR: run this system-package script with sudo." >&2
  exit 1
fi

if [[ ! -r /etc/os-release ]]; then
  echo "ERROR: /etc/os-release not found; this script supports Ubuntu derivatives." >&2
  exit 1
fi
. /etc/os-release
if [[ "${ID:-}" != "ubuntu" && -z "${UBUNTU_CODENAME:-}" ]]; then
  echo "ERROR: this script supports Ubuntu/Xubuntu; detected ${PRETTY_NAME:-unknown}." >&2
  exit 1
fi

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y \
  build-essential \
  ca-certificates \
  ffmpeg \
  libegl1 \
  libgl1 \
  libglfw3 \
  libosmesa6 \
  nodejs \
  npm \
  python3 \
  python3-pip \
  python3-venv \
  unzip

if ! python3 - <<'PY'
import sys
raise SystemExit(0 if sys.version_info >= (3, 11) else 1)
PY
then
  echo >&2
  echo "ERROR: this Ubuntu release's default Python is older than 3.11." >&2
  echo "Install Python 3.11+ and its matching venv package according to your" >&2
  echo "organization's approved repository, then run setup_python.sh --python PATH." >&2
  exit 1
fi

if [[ "$mode" == "gpu" ]]; then
  if [[ "$(uname -s)" != "Linux" ]]; then
    echo "ERROR: NVIDIA setup requires Linux." >&2
    exit 1
  fi
  apt-get install -y ubuntu-drivers-common
  apt-get install -y "linux-headers-$(uname -r)" || true
  if nvidia-smi >/dev/null 2>&1; then
    echo "A working NVIDIA driver is already present:"
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
  else
    echo "No working NVIDIA driver was detected; installing Ubuntu's recommended driver."
    ubuntu-drivers install
    echo
    echo "IMPORTANT: reboot, then confirm that nvidia-smi succeeds."
    echo "Secure Boot systems may require enrolling a prompted MOK key during reboot."
  fi
fi

echo "Ubuntu system dependencies installed for $mode mode."

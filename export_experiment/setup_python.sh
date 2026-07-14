#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage: setup_python.sh --cpu|--gpu [--recreate] [--python PATH]

Downloads Python packages for the destination machine into a local virtualenv.
No administrator privileges are required. Network access to the configured pip
index is required.
EOF
}

mode=""
recreate=0
python_bin="${PYTHON_BIN:-}"
while (($#)); do
  case "$1" in
    --cpu) mode="cpu" ;;
    --gpu) mode="gpu" ;;
    --recreate) recreate=1 ;;
    --python)
      shift
      [[ $# -gt 0 ]] || { usage; exit 2; }
      python_bin="$1"
      ;;
    -h|--help) usage; exit 0 ;;
    *) usage; exit 2 ;;
  esac
  shift
done

if [[ -z "$mode" ]]; then
  usage
  exit 2
fi

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
venv="$root/.venv"
requirements="$root/requirements/requirements-$mode.txt"

if [[ -z "$python_bin" ]]; then
  for candidate in python3.12 python3.11 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
      version="$($candidate -c 'import sys; print(sys.version_info >= (3, 11))')"
      if [[ "$version" == "True" ]]; then
        python_bin="$candidate"
        break
      fi
    fi
  done
fi

if [[ -z "$python_bin" ]] || ! command -v "$python_bin" >/dev/null 2>&1; then
  echo "ERROR: Python 3.11 or newer with venv support is required." >&2
  echo "Ask the administrator to run setup_system_ubuntu.sh first." >&2
  exit 1
fi

if ! "$python_bin" - <<'PY'
import sys
raise SystemExit(0 if sys.version_info >= (3, 11) else 1)
PY
then
  found="$($python_bin --version 2>&1)"
  echo "ERROR: Python 3.11 or newer is required; found $found." >&2
  exit 1
fi

if ((recreate)) && [[ -e "$venv" ]]; then
  rm -rf -- "$venv"
fi

if [[ ! -x "$venv/bin/python" ]]; then
  "$python_bin" -m venv "$venv"
fi

"$venv/bin/python" -m pip install --upgrade pip setuptools wheel
"$venv/bin/python" -m pip install -r "$requirements"
"$venv/bin/python" -m pip check

printf '%s\n' "$mode" > "$venv/experiment-install-mode"

echo
echo "Python environment installed in $venv ($mode mode)."
echo "Next: source '$root/env.sh'"
echo "Then: python '$root/verify_install.py' --expect $mode"


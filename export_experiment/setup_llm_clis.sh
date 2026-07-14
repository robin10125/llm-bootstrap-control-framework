#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 --codex|--claude|--all" >&2
  echo "This optional step requires network access and Node.js/npm." >&2
}

install_codex=0
install_claude=0
while (($#)); do
  case "$1" in
    --codex) install_codex=1 ;;
    --claude) install_claude=1 ;;
    --all) install_codex=1; install_claude=1 ;;
    -h|--help) usage; exit 0 ;;
    *) usage; exit 2 ;;
  esac
  shift
done

if ((install_codex == 0 && install_claude == 0)); then
  usage
  exit 2
fi

if ! command -v npm >/dev/null 2>&1; then
  echo "ERROR: npm is required. Install a currently supported Node.js release first." >&2
  exit 1
fi

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
prefix="$root/.tools"
mkdir -p "$prefix"

packages=()
((install_codex)) && packages+=("@openai/codex@0.144.3")
((install_claude)) && packages+=("@anthropic-ai/claude-code@2.1.208")

npm install --prefix "$prefix" "${packages[@]}"

echo "CLI tools installed under $prefix. Run: source '$root/env.sh'"
echo "Authentication remains local to the destination machine; follow each CLI's login flow."


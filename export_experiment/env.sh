#!/usr/bin/env bash
# Source this file; do not execute it.

_experiment_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export EXPERIMENT_EXPORT_ROOT="$_experiment_root"
export LLM_FRAMEWORK_ROOT="$_experiment_root/source/llm-framework"
export BOOTSTRAPPING_ROOT="$_experiment_root/source/bootstrapping"
export HAND_MANIPULATION_ROOT="$_experiment_root/source/hand-manipulation"

if [[ -f "$_experiment_root/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$_experiment_root/.venv/bin/activate"
else
  echo "WARNING: no bundle virtualenv found; run setup_python.sh first." >&2
fi

export PYTHONPATH="$LLM_FRAMEWORK_ROOT:$LLM_FRAMEWORK_ROOT/famework_testing:$BOOTSTRAPPING_ROOT:$HAND_MANIPULATION_ROOT/harness${PYTHONPATH:+:$PYTHONPATH}"
if [[ -d "$_experiment_root/.tools/node_modules/.bin" ]]; then
  export PATH="$_experiment_root/.tools/node_modules/.bin:$PATH"
fi

unset _experiment_root


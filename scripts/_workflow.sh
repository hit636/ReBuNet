#!/usr/bin/env bash
# Shared launcher: local sources, current Python environment, logged exit status.
set -euo pipefail

run_workflow() {
  local mode="$1"
  shift
  local root python_bin log
  root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
  cd "$root"
  python_bin="${PYTHON_BIN:-python}"
  if ! command -v "$python_bin" >/dev/null 2>&1; then
    if [[ -z "${PYTHON_BIN:-}" ]] && command -v python3 >/dev/null 2>&1; then
      python_bin=python3
    else
      echo "Python executable not found: $python_bin. Activate an environment or set PYTHON_BIN." >&2
      return 127
    fi
  fi
  mkdir -p logs
  log="$(mktemp "logs/${mode}_$(date +%Y%m%d_%H%M%S)_XXXXXX.log")"
  echo "Project directory: $root"
  echo "Run log: $log"
  "$python_bin" -u "$root/scripts/workflows.py" "$mode" "$@" 2>&1 | tee "$log"
}

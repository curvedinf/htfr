#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${ROOT}/.venv"
ENV_FILE="${ROOT}/.env"

if [[ ! -d "${VENV}" ]]; then
  echo "[benchmark_htfr] Virtual environment missing; running setup."
  "${ROOT}/agents/setup_env.sh"
fi

if [[ -f "${ENV_FILE}" ]]; then
  echo "[benchmark_htfr] Loading environment variables from ${ENV_FILE}"
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi

# shellcheck disable=SC1090
source "${VENV}/bin/activate"
export PYTHONUNBUFFERED=1

if [[ $# -eq 0 ]]; then
  DEFAULT_OUTPUT="${ROOT}/benchmarks/benchmark_results.json"
  DEFAULT_TEST_MODEL="${ROOT}/checkpoints/test_htfr_checkpoint.npz"
  mkdir -p "$(dirname "${DEFAULT_OUTPUT}")"
  args=("--output" "${DEFAULT_OUTPUT}")
  if [[ -f "${DEFAULT_TEST_MODEL}" ]]; then
    args+=("--test-model" "${DEFAULT_TEST_MODEL}")
    echo "[benchmark_htfr] Found ${DEFAULT_TEST_MODEL}; will benchmark it by default."
  else
    echo "[benchmark_htfr] No arguments provided; defaulting to --output ${DEFAULT_OUTPUT}"
  fi
  set -- "${args[@]}"
fi

echo "[benchmark_htfr] Launching gemma3_benchmark.py with args: $*"
python "${ROOT}/examples/gemma3_benchmark.py" "$@"

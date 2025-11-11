#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${ROOT}/.venv"
ENV_FILE="${ROOT}/.env"

if [[ ! -d "${VENV}" ]]; then
  echo "[smoke_cli] Virtual environment missing; running setup."
  "${ROOT}/agents/setup_env.sh"
fi

if [[ -f "${ENV_FILE}" ]]; then
  echo "[smoke_cli] Loading environment variables from ${ENV_FILE}"
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi

# shellcheck disable=SC1090
source "${VENV}/bin/activate"
export PYTHONUNBUFFERED=1

SCRIPTS=(
  "python ${ROOT}/examples/gemma3_benchmark.py --help"
  "python ${ROOT}/examples/train_test_htfr.py --help"
)

for cmd in "${SCRIPTS[@]}"; do
  echo "[smoke_cli] Running: ${cmd}"
  eval "${cmd}" >/dev/null
done

echo "[smoke_cli] CLI smoke tests completed."

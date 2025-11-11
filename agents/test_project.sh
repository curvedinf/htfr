#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${ROOT}/.venv"
ENV_FILE="${ROOT}/.env"

if [[ ! -d "${VENV}" ]]; then
  echo "[test_project] Virtual environment missing; running setup."
  "${ROOT}/agents/setup_env.sh"
fi

if [[ -f "${ENV_FILE}" ]]; then
  echo "[test_project] Loading environment variables from ${ENV_FILE}"
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi

# shellcheck disable=SC1090
source "${VENV}/bin/activate"
export PYTHONUNBUFFERED=1

echo "[test_project] Running python -m compileall"
python -m compileall "${ROOT}/htfr" "${ROOT}/examples"

echo "[test_project] Running pytest with coverage"
pytest --cov=htfr --cov-report=term-missing --cov-report=xml "$@"

echo "[test_project] All tests passed. Coverage report written to coverage.xml."

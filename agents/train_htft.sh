#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${ROOT}/.venv"
ENV_FILE="${ROOT}/.env"

if [[ ! -d "${VENV}" ]]; then
  echo "[train_htft] Virtual environment missing; running setup."
  "${ROOT}/agents/setup_env.sh"
fi

if [[ -f "${ENV_FILE}" ]]; then
  echo "[train_htft] Loading environment variables from ${ENV_FILE}"
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi

# shellcheck disable=SC1090
source "${VENV}/bin/activate"
export PYTHONUNBUFFERED=1

echo "[train_htft] Starting training via examples/train_htft.py"
python "${ROOT}/examples/train_htft.py" "$@"

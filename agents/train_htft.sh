#!/usr/bin/env bash
set -euo pipefail

timestamp() {
  date "+%Y-%m-%d %H:%M:%S"
}

log() {
  printf '[train_htft][%s] %s\n' "$(timestamp)" "$*"
}

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${ROOT}/.venv"
ENV_FILE="${ROOT}/.env"

log "Initializing training helper (root=${ROOT})"

if [[ ! -d "${VENV}" ]]; then
  log "Virtual environment missing; invoking setup_env.sh"
  "${ROOT}/agents/setup_env.sh"
else
  log "Reusing existing virtual environment at ${VENV}"
fi

if [[ -f "${ENV_FILE}" ]]; then
  log "Loading environment variables from ${ENV_FILE}"
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
else
  log "No .env file detected; proceeding with current shell environment"
fi

log "Activating virtual environment"
# shellcheck disable=SC1090
source "${VENV}/bin/activate"
export PYTHONUNBUFFERED=1

log "Launching examples/train_htft.py with args: $*"
python "${ROOT}/examples/train_htft.py" "$@"
log "Training script exited with status $?"

#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${ROOT}/.venv"
ENV_FILE="${ROOT}/.env"

if [[ ! -d "${VENV}" ]]; then
  echo "[train_htfr] Virtual environment missing; running setup."
  "${ROOT}/agents/setup_env.sh"
fi

if [[ -f "${ENV_FILE}" ]]; then
  echo "[train_htfr] Loading environment variables from ${ENV_FILE}"
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi

# shellcheck disable=SC1090
source "${VENV}/bin/activate"
export PYTHONUNBUFFERED=1

if [[ $# -eq 0 ]]; then
  DEFAULT_CHECKPOINT="${ROOT}/checkpoints/test_htfr_checkpoint.npz"
  mkdir -p "$(dirname "${DEFAULT_CHECKPOINT}")"
  echo "[train_htfr] No arguments provided; defaulting to --output ${DEFAULT_CHECKPOINT}"
  set -- "--output" "${DEFAULT_CHECKPOINT}"
fi

echo "[train_htfr] Launching train_test_htfr.py with args: $*"
START_TIME=$(date +%s)
python -u "${ROOT}/examples/train_test_htfr.py" "$@" &
TRAIN_PID=$!

progress_tick() {
  local pid=$1
  local origin=$2
  while true; do
    if ! kill -0 "${pid}" 2>/dev/null; then
      break
    fi
    local now elapsed mins secs
    now=$(date +%s)
    elapsed=$((now - origin))
    mins=$((elapsed / 60))
    secs=$((elapsed % 60))
    printf '[train_htfr] Training in progress... %02d:%02d elapsed\n' "${mins}" "${secs}"
    sleep 60
  done
}

progress_tick "${TRAIN_PID}" "${START_TIME}" &
PROGRESS_PID=$!

set +e
wait "${TRAIN_PID}"
STATUS=$?
set -e

wait "${PROGRESS_PID}" 2>/dev/null || true

END_TIME=$(date +%s)
TOTAL=$((END_TIME - START_TIME))
printf '[train_htfr] Training finished with status %s after %02d:%02d elapsed\n' "${STATUS}" "$((TOTAL / 60))" "$((TOTAL % 60))"

exit "${STATUS}"

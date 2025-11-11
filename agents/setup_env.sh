#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${ROOT}/.venv"
ENV_FILE="${ROOT}/.env"
PYTHON_BIN="${PYTHON_BIN:-python3}"
EXTRAS="${HTFR_SETUP_EXTRAS:-benchmark,dev,rocm}"
INSTALL_ROCM_TORCH="${HTFR_INSTALL_ROCM_TORCH:-1}"
ROCM_INDEX_URL="${HTFR_ROCM_INDEX_URL:-https://download.pytorch.org/whl/nightly/rocm7.0}"

echo "[setup_env] Using repository root: ${ROOT}"

if [[ -f "${ENV_FILE}" ]]; then
  echo "[setup_env] Loading environment variables from ${ENV_FILE}"
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi

if [[ ! -d "${VENV}" ]]; then
  echo "[setup_env] Creating virtual environment at ${VENV}"
  "${PYTHON_BIN}" -m venv "${VENV}"
else
  echo "[setup_env] Reusing existing virtual environment at ${VENV}"
fi

# shellcheck disable=SC1090
source "${VENV}/bin/activate"
export PYTHONUNBUFFERED=1

echo "[setup_env] Upgrading pip"
python -m pip install --upgrade pip

echo "[setup_env] Installing project with extras: ${EXTRAS}"
python -m pip install -e "${ROOT}[${EXTRAS}]"

if [[ "${INSTALL_ROCM_TORCH}" == "1" ]]; then
  echo "[setup_env] Reinforcing ROCm PyTorch stack from ${ROCM_INDEX_URL}"
  python -m pip uninstall -y torch torchvision torchaudio >/dev/null 2>&1 || true
  python -m pip install --pre --force-reinstall \
    --index-url "${ROCM_INDEX_URL}" \
    torch torchvision torchaudio
  echo "[setup_env] Removing incompatible CPU-only extensions (e.g., xformers)"
  python -m pip uninstall -y xformers >/dev/null 2>&1 || true
fi

echo "[setup_env] Environment ready."

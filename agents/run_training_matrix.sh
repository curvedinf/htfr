#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ROOT}/.env"
VENV="${ROOT}/.venv"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${ROOT}/logs/training_matrix/${STAMP}"

mkdir -p "${RUN_ROOT}"

if [[ -f "${ENV_FILE}" ]]; then
  echo "[run_training_matrix] Loading ${ENV_FILE}"
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
fi

if [[ ! -d "${VENV}" ]]; then
  echo "[run_training_matrix] Missing .venv — bootstrapping with ROCM_AVAILABLE=${ROCM_AVAILABLE:-1}"
  ROCM_AVAILABLE="${ROCM_AVAILABLE:-1}" "${ROOT}/agents/setup_env.sh"
fi

# shellcheck disable=SC1090
source "${VENV}/bin/activate"
export PYTHONUNBUFFERED=1

declare -a RUN_MATRIX=(
  "seq_baseline::--max-train-examples 128 --max-eval-examples 32 --batch-size 8 --batch-workers 2 --log-interval-seconds 15 --stage1-device hip:0 --stage2-device hip:0"
  "pipeline_dual::--max-train-examples 256 --max-eval-examples 64 --batch-size 16 --batch-workers 4 --pipeline --stage1-device hip:0 --stage2-device hip:1 --log-interval-seconds 10"
  "large_batch_pipeline::--max-train-examples 384 --max-eval-examples 96 --batch-size 32 --batch-workers 6 --pipeline --stage1-device hip:0 --stage2-device hip:1 --tail-tokens 32 --log-interval-seconds 10"
)

TARGET_FILTER="${HTFR_MATRIX_TARGET:-}"

summaries=()
for entry in "${RUN_MATRIX[@]}"; do
  name="${entry%%::*}"
  args="${entry#*::}"
  if [[ -n "${TARGET_FILTER}" && "${name}" != *"${TARGET_FILTER}"* ]]; then
    echo "[run_training_matrix] Skipping ${name} (filter=${TARGET_FILTER})"
    continue
  fi
  RUN_DIR="${RUN_ROOT}/${name}"
  mkdir -p "${RUN_DIR}"
  metrics_path="${RUN_DIR}/metrics.jsonl"
  checkpoint_path="${RUN_DIR}/checkpoint.npz"
  log_path="${RUN_DIR}/train.log"
  bench_log="${RUN_DIR}/benchmark.log"
  profile_path="${RUN_DIR}/profile.jsonl"
  echo "[run_training_matrix] >>> Starting ${name} (logs -> ${RUN_DIR})"
  (
    cd "${ROOT}"
    python examples/train_htft.py \
      --metrics-path "${metrics_path}" \
      --output "${checkpoint_path}" \
      --profile-output "${profile_path}" \
      ${args} | tee "${log_path}"
  )
  (
    cd "${ROOT}"
    python examples/benchmark_htft.py "${metrics_path}" | tee "${bench_log}"
  )
  summaries+=("${name}: ${RUN_DIR}")
done

echo "[run_training_matrix] Completed runs stored under ${RUN_ROOT}"
if [[ ${#summaries[@]} -gt 0 ]]; then
  printf ' - %s\n' "${summaries[@]}"
else
  echo "[run_training_matrix] No runs executed (filter eliminated all targets)."
fi

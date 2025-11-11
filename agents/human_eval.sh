#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ROOT}/.env"
LOG_DIR="${ROOT}/logs/human_eval"
METRICS_PATH="${LOG_DIR}/metrics.jsonl"
CHECKPOINT_PATH="${ROOT}/checkpoints/human_eval_htft.npz"

mkdir -p "${LOG_DIR}" "${ROOT}/checkpoints"

if [[ -f "${ENV_FILE}" ]]; then
  echo "[human_eval] Loading ${ENV_FILE}"
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
fi

cat <<'SUMMARY'
[human_eval] ROCm validation checklist
1. Ensure ROCm drivers (e.g., Radeon RX 7900 XTX on gfx1100) and firmware are installed.
2. Export ROCm env vars in .env: ROCM_VISIBLE_DEVICES, HSA_OVERRIDE_GFX_VERSION, HIP_VISIBLE_DEVICES.
3. Run: ROCM_AVAILABLE=1 agents/setup_env.sh
4. Launch a short training smoke test:
   python examples/train_htft.py \
     --max-train-examples 512 \
     --max-eval-examples 128 \
     --batch-size 32 \
     --batch-workers 8 \
     --pipeline \
     --stage1-device hip:0 \
     --stage2-device hip:0 \
     --log-interval-seconds 5 \
     --profile-output logs/human_eval/profile.jsonl \
     --metrics-path logs/human_eval/metrics.jsonl \
     --output checkpoints/human_eval_htft.npz
5. Inspect logs for:
   - Teacher capture line showing the hip device descriptor from htfr.devices.
   - Stage1/Stage2 device strings mentioning hip:*
   - Batch stats showing >0 stage1/stage2 rates and reasonable loss deltas.
   - Trainer progress lines reporting host↔device MiB transfer counts.
6. Verify metrics JSONL trends with agents/benchmark_htft.sh logs/human_eval/metrics.jsonl
7. Kick off automated training-issue sweeps via: agents/run_training_matrix.sh
   - Creates timestamped subdirectories under logs/training_matrix/, each with train/benchmark logs, metrics, and checkpoints.
   - Use HTFR_MATRIX_TARGET=<substring> to rerun a specific scenario (e.g., pipeline_dual).
SUMMARY

if [[ "${RUN_HUMAN_EVAL:-0}" != "1" ]]; then
  echo "[human_eval] Set RUN_HUMAN_EVAL=1 to execute the quick ROCm smoke test automatically."
  exit 0
fi

(
  cd "${ROOT}"
  ROCM_AVAILABLE=1 \
    HTFR_INSTALL_ROCM_TORCH="${HTFR_INSTALL_ROCM_TORCH:-1}" \
    HTFR_MINIMAL_SETUP=1 \
    agents/setup_env.sh
  # shellcheck disable=SC1090
  source ".venv/bin/activate"
  python examples/train_htft.py \
    --max-train-examples 512 \
    --max-eval-examples 128 \
    --batch-size 32 \
    --batch-workers 8 \
    --pipeline \
    --stage1-device hip:0 \
    --stage2-device hip:0 \
    --log-interval-seconds 5 \
    --profile-output "${LOG_DIR}/profile.jsonl" \
    --metrics-path "${METRICS_PATH}" \
    --output "${CHECKPOINT_PATH}"
)

echo "[human_eval] Smoke test completed. Inspect ${METRICS_PATH} and ${CHECKPOINT_PATH} for diagnostics."

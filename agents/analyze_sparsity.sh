#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <checkpoint.npz>"
  exit 1
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CKPT="$1"

if [[ ! -f "${CKPT}" ]]; then
  echo "[analyze_sparsity] Checkpoint not found: ${CKPT}" >&2
  exit 1
fi

if [[ -f "${ROOT}/.env" ]]; then
  # shellcheck disable=SC1090
  source "${ROOT}/.env"
fi

if [[ ! -d "${ROOT}/.venv" ]]; then
  echo "[analyze_sparsity] Missing .venv; bootstrap via agents/setup_env.sh first." >&2
  exit 1
fi

# shellcheck disable=SC1090
source "${ROOT}/.venv/bin/activate"

python - "$CKPT" <<'PYCODE'
import argparse
import numpy as np
from pathlib import Path

from htfr.checkpoint import load_htft_checkpoint


def summarize(name: str, model) -> None:
    total = len(model.tensors)
    if total == 0:
        print(f"[sparsity] {name}: no tensors")
        return
    controls = np.stack([tensor.C.astype(np.float32) for tensor in model.tensors], axis=0)
    weights = controls.reshape(total, -1)
    total_params = weights.size
    nonzero_params = int(np.count_nonzero(weights))
    sparsity = 1.0 - nonzero_params / max(total_params, 1)
    per_tensor_nonzero = np.count_nonzero(weights, axis=1)
    per_tensor_sparsity = 1.0 - per_tensor_nonzero / weights.shape[1]
    norms = np.linalg.norm(weights, axis=1)
    print(f"[sparsity] {name}: tensors={total} params={total_params} sparsity={sparsity:.4f}")
    print(
        f"           per_tensor_sparsity(mean/median)=({per_tensor_sparsity.mean():.4f}/"
        f"{np.median(per_tensor_sparsity):.4f})"
    )
    print(
        f"           weight_norm(mean/median/max)=({norms.mean():.4f}/{np.median(norms):.4f}/{norms.max():.4f})"
    )


parser = argparse.ArgumentParser()
parser.add_argument("checkpoint", type=Path)
args = parser.parse_args()

ckpt = load_htft_checkpoint(args.checkpoint)
summarize("stage1", ckpt.stage1.model)
summarize("stage2", ckpt.stage2.model)
PYCODE

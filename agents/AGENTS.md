# Agent Automation Helpers

The scripts in this directory provide a reproducible interface for setting up environments, running tests, training the Hypertensor Field Transformer, and summarizing metrics. They assume you launch them from the repository root (or that `agents/` is reachable relative to the root) and follow the top-level `AGENTS.md` guidelines.

## Conventions
- All scripts enable `set -euo pipefail` so failures terminate immediately.
- `.env` (if present) is loaded before activating the virtual environment; place secrets like `HF_TOKEN`, `HUGGINGFACE_HUB_CACHE`, or GPU selection variables here.
- Python commands run with `PYTHONUNBUFFERED=1` so long-running jobs stream logs as they happen.
- The shared virtual environment lives at `.venv`. Scripts call `agents/setup_env.sh` automatically if it is missing.

## Scripts
### `setup_env.sh`
- Creates or refreshes `.venv` using `${PYTHON_BIN:-python3}`.
- Installs the project in editable mode with extras from `HTFR_SETUP_EXTRAS` (defaults to `benchmark,dev,rocm`).
- When `HTFR_INSTALL_ROCM_TORCH=1` (default), uninstalls existing torch wheels and reinstalls ROCm nightlies from `${HTFR_ROCM_INDEX_URL}`. Set the variable to `0` to keep the standard PyPI wheels.

### `test_project.sh`
- Ensures `.venv` exists, activates it, and loads `.env`.
- Runs `python -m compileall htfr examples` followed by `pytest --cov=htfr --cov-report=term-missing --cov-report=xml`.
- Accepts extra pytest arguments (e.g., `agents/test_project.sh -k trainer`).

### `train_htft.sh`
- Lazily bootstraps the environment, sources `.env`, and runs `examples/train_htft.py "$@"`.
- Provides timestamped logs showing the exact CLI forwarded to the trainer. Use this wrapper for long runs so reproducible settings end up in your shell history.

### `benchmark_htft.sh`
- Loads the environment and invokes `examples/benchmark_htft.py`, forwarding arguments such as the metrics JSONL path.
- Useful for quick summaries/CI checks without re-running teacher or student models.

## Tips
- To customize extras or skip ROCm wheels for CPU-only development, export `HTFR_SETUP_EXTRAS` / `HTFR_INSTALL_ROCM_TORCH=0` before calling `setup_env.sh`.
- If you run training on shared clusters, set `HF_TOKEN`, dataset cache locations, and `ROCM_VISIBLE_DEVICES` in `.env` so every helper script inherits the same context.

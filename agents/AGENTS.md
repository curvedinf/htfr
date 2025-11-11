# Agent Automation Helpers

The scripts in this directory provide a reproducible interface for setting up environments, running tests, training the Hypertensor Field Transformer, and summarizing metrics. They assume you launch them from the repository root (or that `agents/` is reachable relative to the root) and follow the top-level `AGENTS.md` guidelines.

## Conventions
- All scripts enable `set -euo pipefail` so failures terminate immediately.
- `.env` (if present) is loaded before activating the virtual environment; place secrets like `HF_TOKEN`, `HUGGINGFACE_HUB_CACHE`, or GPU selection variables here.
- Python commands run with `PYTHONUNBUFFERED=1` so long-running jobs stream logs as they happen.
- The shared virtual environment lives at `.venv`. Scripts call `agents/setup_env.sh` automatically if it is missing.
- Use `ROCM_AVAILABLE=1` when invoking `agents/setup_env.sh` on HIP-capable hosts; leave it unset to keep CUDA/CPU wheels.
- Set `HTFR_MINIMAL_SETUP=1` to skip pip reinstall and ROCm wheel refresh when you just need the existing `.venv` activated quickly (e.g., during repeated ROCm smoke tests).

## Scripts
### `setup_env.sh`
- Creates or refreshes `.venv` using `${PYTHON_BIN:-python3}`.
- Installs the project in editable mode with extras from `HTFR_SETUP_EXTRAS` (defaults to `benchmark,dev,rocm`).
- When `HTFR_INSTALL_ROCM_TORCH=1` (default) **and** `ROCM_AVAILABLE=1`, uninstalls existing torch wheels and reinstalls ROCm nightlies from `${HTFR_ROCM_INDEX_URL}`. Set the variable to `0` (or unset `ROCM_AVAILABLE`) to keep the standard PyPI wheels.
- Honor `HTFR_MINIMAL_SETUP=1` to skip pip upgrades and wheel reinstalls; the script will only ensure `.venv` exists and is activated.

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

### `human_eval.sh`
- Prints the ROCm validation checklist (driver requirements, `.env` expectations, short training command) so GPU-only steps can be verified manually. Set `RUN_HUMAN_EVAL=1` to execute the abbreviated smoke test on a developer workstation.

### `run_training_matrix.sh`
- Automates a small suite of HTFT training runs (sequential baseline, dual-GPU pipeline, large-batch pipeline) to surface ROCm training issues.
- Produces timestamped folders under `logs/training_matrix/` with per-run logs, metrics JSONL files, checkpoints, and benchmark summaries.
- Set `HTFR_MATRIX_TARGET=<substring>` to restrict execution to matching scenario names (`seq_baseline`, `pipeline_dual`, `large_batch_pipeline`).

### `analyze_sparsity.sh`
- Summarizes Hypertensor usage/update statistics for a saved checkpoint.
- Usage: `agents/analyze_sparsity.sh checkpoints/your_model.npz`
- Reports tensor counts, % updated, mean/max usage, and Gini coefficient per stage to guide pruning/tuning decisions.

## Tips
- To customize extras or skip ROCm wheels for CPU-only development, export `HTFR_SETUP_EXTRAS` / `HTFR_INSTALL_ROCM_TORCH=0` before calling `setup_env.sh`.
- If you run training on shared clusters, set `HF_TOKEN`, dataset cache locations, and `ROCM_VISIBLE_DEVICES` in `.env` so every helper script inherits the same context.

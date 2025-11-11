# Agent Guidelines for `htfr`

Welcome to the Hypertensor Field Regressor project. This repo pairs research-grade Hypertensor math with a reproducible HTFT training pipeline. Please follow the conventions below before contributing changes.

## Before You Start
- Always search for nested `AGENTS.md` files in the directories you touch (e.g., `agents/AGENTS.md` describes the automation helpers).
- Work inside the repo virtual environment (`.venv`). Run `agents/setup_env.sh` when dependencies drift—it installs the project in editable mode with the `benchmark`, `dev`, and ROCm extras by default.
- Keep an `.env` file with required secrets (e.g., `HF_TOKEN`) so scripts such as `train_htft.sh` and `test_project.sh` inherit them automatically.

## Coding Standards
- Target Python 3.10+ and follow PEP 8. Use explicit imports from `htfr` modules—avoid `from module import *`.
- Hypertensor math is numerically sensitive: keep new operations in `float32` unless there is a compelling reason to widen. Match the dtype conventions in existing modules (`np.float16` for stored tensors, `np.float32` for accumulators).
- Prefer short, single-line log messages; the training scripts log progress every few seconds, so verbosity matters.
- Reuse helpers from `htfr.feature_ops`, `htfr.context`, and `htfr.trainer` instead of duplicating logic in examples or scripts. If an abstraction is missing, add it to the package and document it.

## Validation Checklist
Run these before requesting review or committing substantial changes:
1. `python -m compileall htfr examples` for any Python files you touched.
2. `python examples/train_htft.py --help` and `python examples/benchmark_htft.py --help` whenever CLI arguments change.
3. `agents/test_project.sh` to execute `pytest --cov` (writes `coverage.xml`) after the compileall pass. Tests rely on lightweight fixtures; do not skip them.
4. For CLI or automation changes, execute the wrapper script you touched (`agents/train_htft.sh`, `agents/benchmark_htft.sh`, etc.) long enough to hit argument parsing and environment activation.

## Documentation & Metadata
- Update `README.md`, module docstrings, or example comments whenever you add a feature, change dependencies, or adjust the recommended workflow.
- When new runtime dependencies are introduced, declare them in `pyproject.toml`—use optional extras (`benchmark`, `dev`, `rocm`, or a new extra) when appropriate, and document installation steps.
- Keep explanations of projection parameters, tensor counts, or dataset expectations near the relevant code (e.g., `htfr/context.py`, `htfr/feature_ops.py`) so downstream users can reason about dimensionality changes.

## Pull Requests & Experiments
- Summarize functional changes plus the exact commands you ran (training, benchmarking, tests) in the PR description. Include dataset/model IDs and token budgets for reproducibility.
- Favor lightweight smoke tests and small token counts when demonstrating new training behaviors. Long-running experiments should log metrics to JSONL so reviewers can inspect perplexity trends via `examples/benchmark_htft.py`.
- When touching checkpoint serialization, note the version in `StageState.metadata` so saved artifacts remain compatible with previous releases.

Thanks for keeping the HTFR toolkit cohesive and reproducible!

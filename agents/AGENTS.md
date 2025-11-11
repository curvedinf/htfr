# Agent Automation Scripts

This directory provides reusable helpers for common maintenance tasks.
All scripts assume they are executed from the repository root (or that
`$REPO_ROOT/agents` is on disk) and rely on the top-level `AGENTS.md`
guidelines.

## Available helpers

- `setup_env.sh` — creates/refreshes `.venv` using the system
  `python3`, upgrades `pip`, and installs `htfr` in editable mode with
  the `benchmark`, `dev`, and ROCm-aligned extras. Set
  `HTFR_INSTALL_ROCM_TORCH=1` (optionally overriding
  `HTFR_ROCM_INDEX_URL`) before running to install the ROCm PyTorch
  stack; use `HTFR_SETUP_EXTRAS` to customize which extras are installed.
- `test_project.sh` — ensures the virtual environment exists, activates
  it, runs `python -m compileall` over the `htfr` and `examples`
  packages, and finally executes `pytest` with coverage reporting.

Each script enables `set -euo pipefail` so unexpected failures stop the
automation early. Python invocations run with `PYTHONUNBUFFERED=1` so
long-running tasks stream their logs immediately. Before activating the
virtual environment, the helpers load `.env` (if present) into the shell
environment so GPU selection flags (`ROCM_VISIBLE_DEVICES`, etc.) and
credentials (e.g. `HF_TOKEN`) apply to subprocesses as well as the
Python-level dotenv integration.

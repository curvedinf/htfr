# Agent Guidelines for `htfr`

Welcome! This repository contains Hypertensor Field Regressor (HTFR) research code and reproducible examples. Please read the following conventions before making changes.

## General Workflow
- Always search for nested `AGENTS.md` files inside subdirectories you plan to modify. They may refine these rules.
- Keep changes focused and include tests or scripts needed to validate the behavior whenever possible.
- Run `python -m compileall <path>` on any new or modified Python scripts to ensure they are syntax-error free. For CLI changes, also run the relevant `--help` command to confirm argument parsing remains valid.

## Code Style
- Follow [PEP 8](https://peps.python.org/pep-0008/) guidelines unless project-specific guidance overrides them.
- Prefer explicit imports from within the `htfr` package rather than wildcard imports.
- Keep logging and CLI output concise; favor structured helper functions when emitting repeated status messages.

## Documentation
- Update `README.md` or module-level docstrings when adding new features or dependencies so downstream users understand how to run examples.

## Dependencies
- Declare optional extras in `pyproject.toml` when adding new runtime requirements and document how to install them.

## Testing
- Use lightweight smoke tests for expensive workflows. For example, limit dataset sizes or token counts when validating training scripts in CI.

## Pull Requests
- Summarize functional changes and tests executed in the PR description. Include commands exactly as executed so maintainers can reproduce them.

Happy hacking!

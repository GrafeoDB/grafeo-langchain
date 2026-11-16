# Contributing to grafeo-langchain

Thank you for your interest in contributing!

## Getting started

1. Fork and clone the repository
2. Install dependencies with [uv](https://docs.astral.sh/uv/). The lockfile builds grafeo from a
   checkout of [grafeo](https://github.com/GrafeoDB/grafeo) next to this repository (`../grafeo`).
   Without one, install grafeo from PyPI instead, as CI does (this updates `uv.lock`; leave that
   change out of your pull request):
   ```bash
   uv sync --all-extras                         # with ../grafeo checked out
   uv sync --all-extras --no-sources -P grafeo  # grafeo from PyPI
   ```
3. Run the tests:
   ```bash
   uv run pytest
   ```
4. Run the linter, formatter check and type checker:
   ```bash
   uv run ruff check src/ tests/ examples/
   uv run ruff format --check src/ tests/ examples/
   uv run ty check
   ```

## Development guidelines

- Python 3.12+ with modern syntax
- Format with `ruff format`, lint with `ruff check`
- Type check with `ty`
- Line length: 120 characters
- Google-style docstrings
- All new code should include tests

## Submitting changes

1. Create a branch for your change
2. Make sure tests pass and linting is clean
3. Open a pull request with a clear description of what changed and why

## License

By contributing, you agree that your contributions will be licensed under the Apache-2.0 license.

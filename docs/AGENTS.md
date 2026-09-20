# Repository Guidelines

## Project Structure & Module Organization

Application code lives in `src/timbre/`. The student-built perception pipeline is divided
into `features/`, `reduction/`, `hmm/`, `bayesian/`, and `eval/`; these modules must remain
free of network calls. External LLM and TTS integrations belong only in `glue/`, while
FastAPI endpoints live in `api/`. Command-line workflows are in `scripts/`, browser assets
are in `frontend/`, tests are in `tests/`, and research notes are in `docs/`. Treat
`DESIGN.md` as the detailed architecture specification.

## Build, Test, and Development Commands

- `make setup` creates `.venv` and installs development dependencies.
- `make data-download`, `make data-extract`, and `make data-split` prepare datasets in that
  exact order. Extraction writes cached `.npz` files; training and evaluation must reuse them.
- `make train` fits reduction and per-emotion HMM artifacts under `models/`.
- `make evaluate` writes reproducible metrics under `experiments/`.
- `make serve` starts FastAPI and the static frontend at `http://localhost:8000`.
- `make ci` runs Ruff and pytest and must pass before work is considered complete.

## Coding Style & Naming Conventions

Use four-space indentation, type hints, short docstrings, `snake_case` for functions and
modules, and `PascalCase` for classes. Ruff enforces Python 3.10 style with a 100-character
line limit. Keep `viterbi_decode` hand-written and free of decoder-library calls. Mark every
LLM or TTS call with `# EXTERNAL SERVICE`; never present external services as original ML.

## Testing Guidelines

Use pytest and name files `tests/test_<module>.py`. Prefer deterministic synthetic inputs.
Tests should pin feature shapes, train-only preprocessing, split leakage guards, Viterbi
results against brute force, posterior validity, and evaluation metrics. Add regression tests
with every bug fix.

## Commit & Pull Request Guidelines

No repository Git history is available to infer an established convention. Use short,
imperative commit subjects such as `Fix actor-disjoint data split`. Keep commits focused.
Pull requests should explain motivation, implementation, tests run, research impact, and
linked issues; include screenshots for frontend changes and metrics for model changes.

## Security & Research Integrity

Never commit `.env`, recordings, `data/`, `models/`, or `experiments/`. Preserve the workflow
`download → extract → split → train → evaluate`, prevent train/test leakage, report measured
results honestly, and distinguish vocal-affect predictions from a person's true internal state.

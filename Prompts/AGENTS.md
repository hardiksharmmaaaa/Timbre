# Agent Instructions

## Project Boundaries

- Keep ML algorithms in `ml/`; keep FastAPI integration in `src/timbre/`.
- Keep external LLM/TTS calls in `src/timbre/glue/` and label them `# EXTERNAL SERVICE`.
- Treat vocal affect as an uncertain acoustic estimate, never a person's true emotion, diagnosis, or lie detection.
- Do not commit `.env`, recordings, `data/`, `models/`, or `experiments/`.

## Commands

| Task | Command |
| --- | --- |
| Set up environment | `make setup` |
| Run lint and tests | `make ci` |
| Run Python tests | `make test` |
| Run lint | `make lint` |
| Verify prepared data | `make data-ready` |
| Train and calibrate models | `make train-all` |
| Evaluate frozen test sets | `make test-models` |
| Start the app | `make serve` |

## Working Rules

- Inspect the relevant route, UI, model, and tests before editing.
- Preserve unrelated working-tree changes; stage only task-relevant files.
- Add a deterministic regression test for every bug fix or behavior change.
- Use speaker-disjoint, training-only selection; never tune using held-out or TESS results.
- Fit scalers, PCA, calibration, and thresholds only on their allowed development partition.
- Make browser microphone lifecycle explicit: request on join, stop every track on end/error, and reset for a new session.
- Return truthful errors: malformed/silent audio is a client error; unavailable artifacts are a service-unavailable error.

## Primary References

| Need | File |
| --- | --- |
| Architecture and API contract | `docs/DESIGN.md` |
| Code ownership map | `docs/REPOSITORY_MAP.md` |
| Reproducible ML workflow | `docs/TRAINING_RUNBOOK.md` |
| ML walkthrough | `docs/PROFESSOR_ML_CODE_GUIDE.md` |
| Prompt demonstration pack | `Prompts/README.md` |

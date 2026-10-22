# Timbre Repository Map

Use this as the starting point when explaining the project. Every source file has one
clear owner; runtime data and presentation outputs are deliberately kept outside source
packages.

## Source code

| Folder | Owns | Start here |
| --- | --- | --- |
| `ml/` | All machine-learning code: data preparation, feature extraction, HMM/Viterbi, Bayesian uncertainty, training, calibration, and evaluation | `ml/core/pipeline/pipeline.py` |
| `src/timbre/` | Application integration only: FastAPI routes and LLM/session glue; it calls `ml/` but contains no ML algorithm | `src/timbre/api/main.py` |
| `frontend/` | Browser interface, microphone capture, charts, and session UI | `frontend/session.html`, `frontend/app.js` |
| `tests/` | Regression tests for ML, API, and frontend behavior | `tests/` |
| `scripts/` | Non-ML engineering utilities only: tracker refresh and guarded shipping | `scripts/update_tracker.py` |

## Non-source material

| Folder | Purpose | Why it stays separate |
| --- | --- | --- |
| `data/`, `models/`, `experiments/`, `recordings/` | Local runtime inputs and generated ML outputs | Large/reproducible files; ignored by Git and never imported as code |
| `docs/` | Research report, runbook, architecture, and presentation guide | Human-readable project evidence |
| `practice/` | Professor preparation questions and answers | Study material, not application code |
| `Hardik_Report/` | Submitted/report deliverables | Final document artifacts, not source code |
| `brag-output/` | Pitch-video composition and generated media | A separate presentation deliverable, not part of the app runtime |

## Typical flows

```text
Training: data/ -> ml/data -> ml/training -> models/ and experiments/
Serving:  frontend/ -> src/timbre/api -> ml/core/pipeline -> models/
Testing:  tests/ -> ml/ and src/timbre/ and frontend/
```

## Commands

```bash
make ci          # lint every source area and run all tests
make train-all   # data-ready -> ML training -> calibration
make serve       # FastAPI application plus frontend at localhost:8000
```

For the ML presentation walkthrough, use `docs/PROFESSOR_ML_CODE_GUIDE.md` alongside
the `ml/` folder. For the full engineering structure, use `docs/DESIGN.md`.

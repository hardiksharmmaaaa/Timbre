# Timbre

A voice-emotion-aware companion prototype that estimates vocal affect from prosodic and
spectral voice features and uses that uncertain estimate to adapt an LLM reply. Vocal affect
does not establish a person's internal emotional state. TTS integration is a planned part of
the end-to-end workflow. Research deliverable for an ML course.

```
voice → [1. features] → [2. PCA/LLE] → [3. HMM/Viterbi] → [4. Bayesian uncertainty]
      → [5. prompt injection] → [6. LLM] → [7. TTS] → [8. mood graph]
```

Boxes **1–4 are the student's own ML contribution** (graded). Boxes **5–7 are off-the-shelf
glue** (LLM/TTS APIs), used transparently and never claimed as own work. See
[`docs/AGENTS.md`](docs/AGENTS.md) (design intent) and
[`docs/DESIGN.md`](docs/DESIGN.md) (engineering spec).

## Setup

```bash
make setup                      # create .venv + install deps
cp .env.example .env            # add your OpenAI API key / data URLs
```

## Data → train → serve

```bash
make data-download              # fetch RAVDESS (fill RAVDESS_URL in .env)
make data-prepare               # screen/hash audio; freeze canonical manifest + splits
make data-extract               # cache framewise features for all accepted clips
make data-ready                 # verify manifest, split hash, and complete feature cache
make train-all                  # train baselines + six-class HMM + validation calibration
make test-models                # in-domain and untouched TESS evaluation -> experiments/
make serve                      # FastAPI + static frontend at http://localhost:8000
```

The first main experiment pools RAVDESS and CREMA-D into six shared vocal-affect labels.
Speaker-disjoint partitions are created independently inside each development dataset. TESS is
locked to `external_test` and cannot enter training or validation through these commands. See
[`docs/TRAINING_RUNBOOK.md`](docs/TRAINING_RUNBOOK.md) for the exact staged workflow.

Baselines include logistic regression, random forest, KNN, AdaBoost, and SVM. KNN, AdaBoost, and SVM
hyperparameters are selected by five-fold speaker-disjoint cross-validation on training data.
KNN/SVM scaling is fitted inside each fold; AdaBoost tree learners use unscaled clip statistics.
The saved winners are evaluated on both test splits. Search reports are stored in
`models/baselines/knn_cv.json` and `models/baselines/adaboost_cv.json`. AdaBoost searches tree
depth, estimator count, and learning rate. SVM searches 30 linear/RBF configurations across
regularization, kernel width, and class weighting; its report is `models/baselines/svm_cv.json`.
SVM evaluation uses class predictions; probability calibration metrics are unavailable.

The chat integration uses OpenAI first through the official `openai` SDK. Set
`OPENAI_API_KEY` in `.env`. Gemini can be configured later with `GEMINI_API_KEY` as an
automatic backup; leave it blank for OpenAI-only operation. With neither key, the app stays
usable in demo mode and returns local canned replies without making network calls.

## Tests & lint

```bash
make test      # pytest
make lint      # ruff
make ci        # lint + test
```

Node.js 18+ enables the frontend runtime regression tests within `make ci`. They exercise
microphone lifecycle, WAV encoding, chat failures, and rendering fallbacks with synthetic
browser inputs; the frontend test is skipped when Node.js is unavailable.

Live microphone clips are converted to PCM WAV in the browser and resampled to the configured
training sample rate by the backend. The full feature contract is stored with cached features
and trained models and is validated before artifacts are reused or served.

## Project layout

```
src/timbre/    Python package (features, reduction, hmm, bayesian, glue, api, eval)
frontend/      static HTML/JS mic + mood-graph app
scripts/       CLI entrypoints (download, extract, split, train, evaluate)
tests/         pytest suite
data/          raw / processed features / splits (gitignored)
models/        trained artifacts (gitignored)
experiments/   eval outputs (gitignored)
```

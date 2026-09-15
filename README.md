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
[`AGENTS.md`](AGENTS.md) (design intent) and [`DESIGN.md`](DESIGN.md) (engineering spec).

## Setup

```bash
make setup                      # create .venv + install deps
cp .env.example .env            # add your OpenAI API key / data URLs
```

## Data → train → serve

```bash
make data-download              # fetch RAVDESS (fill RAVDESS_URL in .env)
make data-extract               # framewise features -> data/processed
make data-split                 # seeded train/val/test -> data/splits
make train                      # fit reduction + per-state HMMs -> models/
make evaluate                   # accuracy, confusion matrix, calibration -> experiments/
make serve                      # FastAPI + static frontend at http://localhost:8000
```

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
training sample rate by the backend. Keep feature settings consistent between extraction,
training, and serving; existing model artifacts do not store the full feature configuration.

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

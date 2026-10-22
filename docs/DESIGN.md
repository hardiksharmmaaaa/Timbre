# Timbre — Engineering Design Document

This document is the **executable specification** for the Timbre engineering project. It
complements `AGENTS.md` (the what/why) with the how: concrete module boundaries, data flows,
interfaces, and operational details. When code lands, this doc must stay reconciled with the
actual repository — treat discrepancies as bugs and fix the more correct side.

---

## 1. Project layout

```
Timbre/
├── pyproject.toml            # build config, deps, tool config (ruff, pytest)
├── Makefile                  # one-stop dev commands (see §9)
├── .env.example              # required env vars (copy to .env, never commit)
├── .gitignore
├── README.md
├── docs/                     # report, research notes, and REPOSITORY_MAP.md
├── data/                     # gitignored
│   ├── raw/                  # downloaded datasets (RAVDESS, CREMA-D, TESS)
│   ├── processed/            # extracted feature arrays per clip (.npz)
│   └── splits/               # seeded train/val/test index files (.json)
├── models/                   # gitignored; serialized pipeline artifacts (.joblib/.pkl)
├── experiments/              # reproducible run logs, notebooks, ablation outputs
├── ml/                       # all ML data/core/training/evaluation code (see §2)
├── src/timbre/               # application API and LLM/session glue only
├── frontend/                 # static HTML/JS app (mic + mood graph)
├── scripts/                  # engineering utilities (tracker and guarded shipping)
└── tests/                    # pytest suite
```

## 2. Source-package boundaries

The **perception core** (stages that are the student's own ML contribution) lives entirely in
`ml/`. Application glue is intentionally isolated in `src/timbre/`.

| Module | Pipeline stage | Ownership | Responsibility |
|--------|----------------|-----------|----------------|
| `ml/core/features/` | 1 | **Built by me** | framewise feature extraction (F0, RMS, MFCCs, voiced proportion) |
| `ml/core/reduction/` | 2 | **Built by me** | PCA + LLE dimensionality reduction, fit/transform |
| `ml/core/hmm/` | 3 | **Built by me** | HMM (emission/transition) + hand-written Viterbi decoder |
| `ml/core/bayesian/` | 4 | **Built by me** | posterior over emotion states + credible intervals |
| `ml/core/pipeline/` | 1–4 | **Built by me** | serving composition of the perception stages |
| `ml/data/`, `ml/training/`, `ml/eval/`, `ml/config/` | ML workflow | **Built by me** | data lineage, experiments, metrics, and reproducibility |
| `src/timbre/glue/` | 5–7 | off-the-shelf | LLM prompt injection + TTS API + mood log persistence |
| `src/timbre/api/` | — | glue | FastAPI app: streaming endpoint, health check, model loading |

Rules:
- `ml/core/` and `ml/eval/` contain **no API/network calls** — pure
  computation, fully unit-testable.
- `src/timbre/glue/` and `src/timbre/api/` may call external services but must **never**
  import-claim them as research.
- Every public function in the "built by me" modules carries a leading docstring noting whether
  it is hand-implemented or wraps a library.

## 3. Core data model (shared types)

`ml/config/types.py`:

- `FrameFeatures`: a single frame's feature vector (np.ndarray) + timestamp.
- `Clip`: raw audio path + parsed emotion label + feature matrix (nframes × nfeatures).
- `FeatureSequence`: a clip's framewise feature matrix + frame metadata.
- `EmotionState`: enum of the 8 RAVDESS emotions + unknown.
- `EmotionPosterior`: dict[EmotionState, probability] + credible interval per state.
- `SessionLogPoint`: timestamp + posterior → consumed by mood-graph logger.

## 4. Data pipeline

### 4.1 Sources
| Dataset | Use | Size | Emotion encoding |
|---------|-----|------|------------------|
| RAVDESS | pooled development | 1,056 six-class candidates | 3rd filename field |
| CREMA-D | pooled development | 7,442 candidates | filename token |
| TESS | untouched external test | 2,400 six-class candidates | filename suffix |

Emotion label must be **parsed from filenames** — never from folder placement assumptions.
RAVDESS discovery validates all seven filename fields and selects audio-only speech (`03-01`)
regardless of whether the archive contains an `Audio_Speech` parent directory.

### 4.2 Feature extraction (frame semantics)
- Window: **25 ms**; stride: **10 ms** (defaults; configurable in config `feature`).
- Per frame: **F0/pitch** (energy-gated `librosa.yin`, 50 ms analysis window), **RMS energy**,
  **MFCCs (20 coefficients)**, and **voiced proportion**.
- Output: one `FeatureSequence` per clip, written to `data/processed/<dataset>/<id>.npz`
  as `{"features": (nframes, nf), "sr": int, "label": int}`.

### 4.3 Train/val/test split
- **Deterministic, seeded, and speaker-independent** (SEED in config, default 42). Assign each
  RAVDESS actor wholly to train, validation, or test; never split at frame level or allow an
  actor to cross partitions.
- Index file: `data/splits/six_class_v1.json` — frozen clip IDs, manifest hash, seed, and
  disjoint speaker groups per development dataset.
- Personal/native-language test set lives as its own split json and is excluded from training.

### 4.4 Repro wiring
- `ml/data/download.py` — downloads RAVDESS (Zenodo) / TESS / CREMA-D, verifies hashes.
- `ml/training/extract_features.py` — reads raw → writes `data/processed`.
- `ml/training/make_splits.py` — writes `data/splits/*.json` with the fixed seed.
- A `Makefile` target chains them in order. **Do not run extraction on raw audio more than
  once** — the processed `.npz` cache is the source for all downstream training.

## 5. Perception pipeline (built by me)

### 5.1 Reduction (`reduction/`)
- `fit_reduction(features, method=pca|lle, n_components)` → artifact.
- Standardize Hz, energy, MFCC, and ratio columns with `StandardScaler`, then fit PCA on
  **concatenated training frames** only (never on validation/test frames) — leakage guard.
- LLE: `sklearn.manifold.LocallyLinearEmbedding`; compare separation in report.
- Store fitted object in `models/` for reuse at inference.

### 5.2 HMM + Viterbi (`hmm/`)
- Fit one Gaussian HMM per emotion with `hmmlearn` (Baum-Welch). Each HMM's hidden states are
  acoustic regimes within that emotion, not the eight emotion labels.
- **`viterbi.py` — fully hand-written Viterbi decoder** (`argmax` over log-probs with
  backpointers). This is the graded inference contribution; no library call in this file.
- API: `decode(features_2d, model) -> (states_seq, log_score)`. Classification compares the
  hand-written Viterbi best-path score from every emotion model after dividing by frame count;
  forward likelihood remains an evaluation ablation.
- Design note (known research caveat): "one HMM per emotion class" is effectively N separate
  sequence classifiers. Document this modeling choice explicitly in `docs/report.md`; if time
  permits, implement a single HMM with 8 latent states as an ablation.

### 5.3 Bayesian uncertainty (`bayesian/`)
- On top of HMM (log-)likelihoods, estimate a **posterior distribution over emotion states**
  rather than a single argmax.
- Implementation options, in order of preference given real-time constraints:
  1. **Hybrid**: closed-form Gaussian posterior over emission params (fast), 2. **VI** with a
     mean-field surrogate (recommended middle ground), 3. **MCMC/Gibbs** (full rigor, slow —
     batch/offline only, not real-time).
- Output: `EmotionPosterior` with per-state probability **and credible intervals**.
- Current implementation uses uncertainty surrogates: `hybrid` and `vi` return heuristic
  widths, while the compatibility selector `mcmc` samples independent probability vectors
  from a Dirichlet surrogate. It is not full HMM posterior inference or Gibbs sampling.
  See `docs/report.md` for the pseudo-count model and its calibration limitations.
- **Real-time mandate**: streaming inference must stay under a latency budget (see §7). If
  MCMC exceeds it, ship the VI path in `api` and keep MCMC as an offline eval artifact.

## 6. Evaluation (`eval/`)
Hand-written scoring + visualizations, all deterministic:
- Accuracy & balanced accuracy (8-class).
- **Confusion matrix** (which emotion pairs confound).
- **Calibration curve**: bin predicted confidence vs. actual correctness; report expected
  calibration error (ECE). This is the "does 72% ≈ 72%?" test.
- **Ablations**: PCA vs. LLE; ±Bayesian layer; ±Kalman smoothing; HMM vs. logistic-regression
  vs. random-forest baselines.
- Script: `ml/training/evaluate.py` → writes figures + metrics into `experiments/<run>/`.

## 7. Backend & inference (`api/` + `glue/`)

### 7.1 FastAPI app (`api/main.py`)
Endpoints:
- `POST /api/inference` — accepts an uploaded audio file (the browser sends PCM WAV), runs
  features → reduction → HMM → Bayesian, returns `EmotionPosterior`.
- `POST /api/chat` — takes posterior + user text → injects prompt → returns LLM reply + TTS URL.
- `GET /api/health` — model loaded? version? latency stats.

### 7.2 Streaming / latency contract
- Browser sends audio chunks; backend buffers a **sliding window** (config `inference.window_ms`,
  default ~2000 ms). Inference runs on the window.
- Latency budget: **< ~300 ms** end-to-end for perception. If PCA+HMM+VI exceeds it, reduce
  window, trim MFCC count, or downsample frames before decode.
- Decode and perception execute in a worker thread so they do not block unrelated API
  requests. Live waveforms are resampled to the configured training sample rate before
  extracting features. Empty, non-finite, and unreadable audio receives HTTP 400; incomplete
  model artifacts receive HTTP 503. Health checks load available models before reporting mode.
- HMM/Viterbi state **resets at utterance boundaries** (detected by VAD/silence) — documented
  decision so the sequence model doesn't carry state across turns.

### 7.3 LLM/TTS glue (`glue/`)
- `prompt.py`: builds system prompt from `EmotionPosterior` (e.g. "user sounds stressed 72%").
- `companion.py`: calls OpenAI first, with optional Gemini fallback. **Marked
  `# EXTERNAL SERVICE`**.
- `tts.py`: calls external TTS API; may slow speaking rate under stress (conversational mirroring).
- `mood_logger.py`: appends `SessionLogPoint` to a session store (JSON/DB) for the graph.
- **API keys**: read from environment via `utils/config.py`. Never hardcode or commit.

## 8. Frontend (`frontend/`)
- `index.html` + `app.js` + `style.css` — single-page app, no build step.
- Browser mic via `getUserMedia()`; `MediaRecorder` → browser audio decoding → mono 16-bit
  PCM WAV → POST chunk to `/api/inference`. This avoids sending WebM/MP4 containers to the
  backend's libsndfile decoder. Microphone startup, pause, failure cleanup, and overlapping
  requests have deterministic JavaScript regression tests.
- Renders reply from `/api/chat` and plays TTS audio.
- **Mood graph**: Chart.js or Plotly line/area chart — X = conversation turns, Y = emotion
  probability, confidence-interval bands. Fed by `/api/inference` responses.
- Served from FastAPI static mount (no separate dev server) for demo simplicity.

## 9. Developer commands (Makefile)
```make
setup        # python -m venv .venv && pip install -e ".[dev]"
data-download # ml/data/download.py
data-extract  # ml/training/extract_features.py
data-split    # ml/training/make_splits.py
train         # ml/training/train.py            -> models/
evaluate      # ml/training/evaluate.py          -> experiments/<run>/
serve         # uvicorn timbre.api.main:app  (serves frontend too)
lint          # ruff check .
typecheck     # (none yet; optionally mypy)
test          # pytest -q
ci            # lint && test
```
All commands run from repo root. Python 3.10+.

## 10. Env / config
- `.env` (from `.env.example`): `RAVDESS_URL`, `CREMA_URL`, `TESS_URL`, `OPENAI_API_KEY`,
  optional `GEMINI_API_KEY`, provider model names, `TTS_API_KEY`, `TTS_VOICE_ID`, `SEED`,
  `LOG_LEVEL`.
- All defaults in `ml/config/config.py` (pydantic-settings); overridable via env.
- `.gitignore`: `data/`, `models/`, `.env`, `*.npz`, `*.wav`, `*.joblib`, `experiments/`,
  `.venv/`, `__pycache__/`.

## 11. Reproducibility rules
- **Seed everything** (feature extraction, split, PCA, HMM init, baselines).
- Never fit reduction or scalers on test/val frames.
- Cache processed features; never recompute raw → features in training/eval scripts.
- Every experiment records: seed, config snapshot, dataset split json hash, commit hash →
  `experiments/<run>/manifest.json`.

## 12. Testing strategy
- `tests/test_features.py` — known-signal synthetic clips → expected feature ranges.
- `tests/test_viterbi.py` — hand-computed tiny HMM → assert exact best-path (the proof the
  hand-written decoder is correct).
- `tests/test_reduction.py` — PCA/LLE transform shapes; no-leakage guard.
- `tests/test_bayesian.py` — posterior sums to 1; intervals shrink with more frames.
- `tests/test_eval.py` — calibration/ECE on synthetic well-calibrated data → near-zero ECE.

## 13. Known research caveats (to document, not hide)
1. Per-emotion HMM == N sequence classifiers (single-8-state-HMM ablation planned).
2. RAVDESS is acted/English; cross-lingual transfer drops — expected, report honestly.
3. Real-time MCMC is hard → VI primary, MCMC offline.
4. 65–75% target: reasonable for classical methods; never chase ~100%.

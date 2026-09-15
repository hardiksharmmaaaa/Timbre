# Application bug review — 2026-09-15

## Scope and method

Reviewed the frontend, API, perception modules, service glue, configuration, scripts,
tests, and project documentation. The original Ruff/pytest baseline passed 35 tests.
Each code fix below was driven by a failing deterministic regression. No datasets were
re-extracted, models retrained, external LLM/TTS providers called, or Git changes published.

## Confirmed failures and fixes

| Area | Reproduced failure | Correction |
| --- | --- | --- |
| Live inference | Trained HMM requests returned HTTP 500 | Corrected reversed Viterbi scorer arguments; added an endpoint test with a real synthetic HMM |
| Audio transport | Browser WebM/MP4 was incompatible with the backend decoder | Convert browser recordings to mono PCM WAV and resample to the configured training rate |
| API availability | Inference blocked other asynchronous requests | Run audio decoding and perception in a worker thread |
| API validation | Invalid audio caused 500s; demo chat bypassed payload validation | Validate waveforms and chat payloads before use; return client errors |
| Model health | Available models were reported as demo until first inference; partial artifacts crashed | Load available models for health checks; report incomplete artifacts as HTTP 503 |
| Microphone | Duplicate permission requests, leaked tracks, stale stop callbacks, inactive start CTA | Guard startup and recorder ownership, release tracks on failure, connect session start to capture |
| Chat and status | Overlapping submissions, misleading HTTP errors, stale health and inference displays | Serialize chat submissions, show current errors, ignore older inference results |
| Optional graphics | Missing chart, canvas, or WebGL support interrupted the UI | Allow the useful interface to operate when optional rendering is unavailable |
| Dataset preparation | Valid actor folders yielded zero clips; malformed filenames crashed discovery | Validate filename metadata and discover speech independently of folder names |
| Downloads | Cached archives were not extracted; interrupted downloads poisoned the cache | Extract cached archives and publish validated downloads atomically |
| Splits | Rounding holdouts could consume every training actor | Reserve at least one actor for each partition |
| Evaluation | Zero-recall classes inflated balanced accuracy; zero-confidence predictions disappeared from ECE | Include these valid boundary values |
| Evaluation orchestration | Predictions for a class absent from test labels overflowed the confusion matrix | Include model classes in the matrix dimensions |
| Evaluation consistency | Labels came from raw HMM scores while confidence came from the posterior | Use the posterior prediction and confidence together, matching the live API |
| Viterbi | Tiny substituted probabilities allowed forbidden starts/transitions | Preserve exact probability-zero exclusions in log space |
| Features | Digital silence was counted as voiced | Give all-zero waveforms zero voiced fraction |
| Uncertainty | Invalid evidence produced NaNs; sampling intervals ignored frame count and measured sampling error | Validate inputs; use a documented Dirichlet surrogate with probability-draw intervals |
| Configuration | Zero dimensions/timings reached runtime | Reject invalid nonpositive configuration values early |
| Repository workflow | Session data could be tracked; the public environment template could not pass shipping checks | Ignore private/generated directories, extend tracking guards, allow only `.env.example` |
| Test reproducibility | A feature test depended on downloadable sample audio | Use a deterministic local waveform |

## Verification

Final `make ci`: Ruff passed and 124 pytest tests passed (8.94 seconds). The included
frontend runtime suite passed all 14 JavaScript cases. `git diff --check` passed.
Four existing dependency warnings remain: three Python audio-module deprecations and
joblib falling back from physical to logical CPU counts.

Run `make ci` for Ruff and the full pytest suite. With Node.js 18+, pytest also runs
`node --test tests/frontend_runtime.cjs` against the actual shipped frontend scripts.
The browser tests replace media/network boundaries with deterministic inputs.

Additional checks confirmed frontend-generated WAV bytes decode with `soundfile` and reach
the inference API, and the locally served health endpoint returns HTTP 200. Read-only
discovery finds 1,440 RAVDESS speech clips across eight labels in the existing dataset.

## Limits and remaining project work

- No connected browser was available for a visual or real microphone hardware test.
- Provider network behavior and full dataset training/evaluation were not exercised.
- Model artifacts still require consistent feature settings across extraction, training,
  and serving; they do not retain the complete feature configuration.
- Existing evaluation results should be regenerated because metric and uncertainty
  calculations changed. No accuracy, calibration, coverage, or latency improvement is claimed.
- Combined-dataset training, STT, provider TTS wiring, calibration, and other planned
  research milestones remain described in the model/data plan.

The uncertainty implementation and assumptions are detailed in `docs/report.md`.

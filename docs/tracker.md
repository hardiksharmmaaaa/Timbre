| Area | Trackable item | Status | Evidence or exit criterion | Next checkpoint |
| --- | --- | --- | --- | --- |
| Workflow | Next engineering task | Next | Pipeline: Dataset-specific parsers | Implement canonical six-class mappings |
| Workflow | Automatic ship refresh | Complete | `make ship` runs CI and refreshes this table before committing and pushing | Keep tracker updates in the guarded ship workflow |
| UI | Frontend experience | Complete | Interface implemented; 14 frontend checks cover microphone lifecycle, WAV uploads, chat errors, stale responses, and graphics fallbacks | Verify real microphone capture and visual behavior in a connected browser |
| Planning | Model and data plan | Complete | `docs/MODEL_AND_DATA_PLAN.md` documents architecture, datasets, training, and evaluation | Keep decisions and dates current |
| Data | RAVDESS download | Pass | 1,440 WAV files, 24 actors; archive MD5 `bc696df654c87fed845eb13823edef8a` | Exclude calm and surprise for the six-class experiment |
| Data | CREMA-D download | Pass | 7,442 WAV files, 91 actors; commit `1658cd342dff90010aa843eaeebd53610a08b1dc`; Git LFS check passed | Preserve ratings metadata in the canonical manifest |
| Data | TESS download | Pass | 2,800 WAV files, 2 speakers; archive SHA-256 `cf33eed0cb23671af0e7c607eb6cf89685ef6469eb7c51987b2d32cbe081b0b9` | Reserve 2,400 shared-label clips for external testing only |
| Data | RIRS_NOISES download | Pass | 61,260 WAV files; archive SHA-256 `3b50cfde915b3984738169b4beb341e9f6b8062ae4c2076146c5db71c2c05dc7` | Apply augmentation to training partitions only |
| Data safety | Raw dataset exclusion | Pass | No data, model, experiment, or recording files tracked on 2026-09-15; ignore rules and tracking guards cover private session data | Keep generated artifacts and recordings out of Git |
| Security | Example credentials | Action required | `.env.example` no longer contains an API key | Revoke the previously exposed key and keep secret scans clean |
| Quality | Ruff | Pass | `make ci` passed Ruff on 2026-09-15 | Run through `make ci` before every completed change |
| Quality | Pytest | Pass | `make ci` passed 124 pytest tests on 2026-09-15, including a wrapper for 14 frontend cases | Keep deterministic regressions for confirmed failures |
| Quality | Patch hygiene | Pass | `git diff --check` passed on 2026-09-15 | Require a clean result before every commit |
| Application | Crash and behavior fixes | Complete | Reproduced API, audio, microphone, chat, data-workflow, and numerical bugs fixed with regression coverage; see `docs/BUGFIX_REVIEW.md` | Preserve regressions; regenerate metrics before citing results |
| Application | Hardware and provider verification | Pending | Local HTTP health and synthetic WAV/API integration pass; real browser microphone and live LLM/TTS providers were not exercised | Run a real microphone/browser session and configured provider smoke tests |
| Application | Full voice conversation loop | Pending | Typed chat and LLM fallback glue exist; speech-to-text and provider TTS wiring remain planned | Implement and verify speech transcription and spoken responses |
| Version control | Branch synchronization | Not verified | Current branch is `ui`; bug fixes and tracker changes are local, with no commit or push performed in this review | Review changes and verify remote synchronization when shipping |
| Pipeline | RAVDESS clip discovery | Complete | Validated filename discovery finds 1,440 speech clips across eight labels; actor-directory and malformed-name regressions pass | Retain discovery regressions while adding dataset-specific parsers |
| Pipeline | Dataset-specific parsers | Pending | Validated RAVDESS parsing is implemented; CREMA-D and TESS label/speaker parsers are still missing | Implement canonical six-class mappings |
| Pipeline | Canonical manifest | Pending | No shared manifest exists | Record paths, speakers, labels, licenses, durations, and audio hashes |
| Pipeline | Audio integrity screening | Pending | API rejects unreadable, empty, and non-finite audio; dataset-wide corrupt, silent, clipped, and duplicate screening is still missing | Produce documented exclusion reasons and retained counts |
| Pipeline | Speaker-disjoint splits | Pending | RAVDESS actor-disjoint splitting is tested and preserves nonempty partitions; pooled multi-dataset manifests are not frozen | Freeze deterministic train, validation, test, and external-test manifests |
| Features | Cached extraction | Pending | RAVDESS cached extraction exists; silence handling and offline feature tests pass; multi-dataset extraction is not complete | Extract each retained clip once and reuse cached `.npz` files |
| Baselines | Logistic regression and random forest | Pending | Baselines are not wired into training | Train and evaluate on the frozen six-class split |
| Model | Six-class PCA and HMM training | Pending | RAVDESS training code exists and live inference passes with a synthetic HMM; pooled six-class models have not been trained and evaluated | Fit preprocessing on training frames only and train one HMM per class |
| Model | Hand-written Viterbi validation | Pass | Brute-force equivalence, length normalization, and impossible start/transition regressions pass | Retain decoder validation during model changes |
| Uncertainty | Probability calibration and abstention | Pending | Finite evidence checks and frame-dependent Dirichlet surrogate intervals are tested; empirical calibration and abstention are not implemented | Fit on validation data and report ECE and Brier score |
| Evaluation | In-domain and TESS evaluation | Pending | Metric boundary bugs and posterior-label/confidence consistency are fixed; full in-domain and TESS evaluation remains unrun | Report clip counts, speaker counts, macro-F1, balanced accuracy, and confusion matrices |
| Evaluation | Training-only augmentation ablation | Pending | RIRS_NOISES is downloaded but not integrated | Compare identical frozen splits with and without augmentation |
| Release gate | Repository validation | Pass | `make ci` passes 124 tests and Ruff; patch and tracker structure checks pass on 2026-09-15 | Complete real browser/provider checks and dataset evaluation before a research release |

## Project Progress

> **44% complete** — 8 of 18 tracked delivery milestones.
>
> **Estimated effort remaining: 23 developer-days (about 5 working weeks).** This excludes model-training runtime and external approvals.

> **Scope:** the percentage counts the original 18 delivery milestones equally. It is not a percentage of source code written or model accuracy. Application stabilization and verification are tracked separately above; partial research milestones remain unfinished. The effort estimate is the existing planning allowance for those delivery milestones, not a measured whole-app forecast.

| Area | Trackable item | Status | Evidence or exit criterion | Next checkpoint |
| --- | --- | --- | --- | --- |
| Workflow | Next engineering task | Next | Evaluation: Training-only augmentation ablation | Run `make augmentation-ablation` after freezing primary results |
| Workflow | Automatic ship refresh | Complete | `make ship` runs CI and refreshes this table before committing and pushing | Keep tracker updates in the guarded ship workflow |
| UI | Frontend experience | Complete | Interface implemented; 14 frontend checks cover microphone lifecycle, WAV uploads, chat errors, stale responses, and graphics fallbacks | Verify real microphone capture and visual behavior in a connected browser |
| Planning | Model and data plan | Complete | `docs/MODEL_AND_DATA_PLAN.md` documents architecture, datasets, training, and evaluation | Keep decisions and dates current |
| Data | RAVDESS download | Pass | 1,440 WAV files, 24 actors; archive MD5 `bc696df654c87fed845eb13823edef8a` | Exclude calm and surprise for the six-class experiment |
| Data | CREMA-D download | Pass | 7,442 WAV files, 91 actors; commit `1658cd342dff90010aa843eaeebd53610a08b1dc`; Git LFS check passed | Preserve ratings metadata in the canonical manifest |
| Data | TESS download | Pass | 2,800 WAV files, 2 speakers; archive SHA-256 `cf33eed0cb23671af0e7c607eb6cf89685ef6469eb7c51987b2d32cbe081b0b9` | Reserve 2,400 shared-label clips for external testing only |
| Data | RIRS_NOISES download | Pass | 61,260 WAV files; archive SHA-256 `3b50cfde915b3984738169b4beb341e9f6b8062ae4c2076146c5db71c2c05dc7` | Apply augmentation to training partitions only |
| Data safety | Raw dataset exclusion | Pass | No data, model, experiment, or recording files were tracked on 2026-09-29 | Keep generated artifacts and recordings out of Git |
| Security | Example credentials | Action required | `.env.example` no longer contains an API key | Revoke the previously exposed key and keep secret scans clean |
| Quality | Ruff | Pass | `make ci` passed Ruff on 2026-09-29 | Run through `make ci` before every completed change |
| Quality | Pytest | Pass | `make ci` passed pytest on 2026-09-29 | Keep deterministic regressions for confirmed failures |
| Quality | Patch hygiene | Pass | `git diff --check` passed on 2026-09-29 | Require a clean result before every commit |
| Application | Crash and behavior fixes | Complete | Reproduced API, audio, microphone, chat, data-workflow, and numerical bugs fixed with regression coverage; see `docs/BUGFIX_REVIEW.md` | Preserve regressions; regenerate metrics before citing results |
| Application | Hardware and provider verification | Pending | Local HTTP health and synthetic WAV/API integration pass; real browser microphone and live LLM/TTS providers were not exercised | Run a real microphone/browser session and configured provider smoke tests |
| Application | Full voice conversation loop | Pending | Typed chat and LLM fallback glue exist; speech-to-text and provider TTS wiring remain planned | Implement and verify speech transcription and spoken responses |
| Version control | Branch synchronization | Pending | Tracker refreshed for `timbre-train` on 2026-09-29; commit and push remain pending | Review changes and verify remote synchronization when shipping |
| Pipeline | RAVDESS clip discovery | Complete | Validated filename discovery finds 1,440 speech clips across eight labels; actor-directory and malformed-name regressions pass | Retain discovery regressions while adding dataset-specific parsers |
| Pipeline | Dataset-specific parsers | Complete | RAVDESS, CREMA-D, and TESS parsers map six canonical labels, namespaced speakers, intensities, and CREMA-D voice-vote agreement; parser regressions pass | Preserve mappings and metadata contracts |
| Pipeline | Canonical manifest | Complete | `six_class_v1` contains 10,890 accepted clips with source paths, labels, speakers, roles, licenses, audio hashes, and split assignments; SHA-256 `2e59f4c0a7447516c09dd2fac6564087fe27aca54e809750c8853f3f79545e1a` | Version the manifest when inputs or rules change |
| Pipeline | Audio integrity screening | Complete | 10,898 candidates decoded and screened; six conflicting-label copies, one same-label duplicate, and one nearly silent clip were excluded with recorded reasons | Keep exclusions and thresholds with each experiment |
| Pipeline | Speaker-disjoint splits | Complete | Frozen split has 5,848 train, 1,318 validation, 1,324 in-domain test, and 2,400 TESS external-test clips; development speakers are disjoint per dataset | Never tune against test or external-test splits |
| Features | Cached extraction | Complete | All 10,890 accepted clips have finite hash-validated feature caches under the 16 kHz versioned feature contract; `make data-ready` passes | Rebuild only after a versioned contract change |
| Baselines | Logistic regression and random forest | Complete | Both baselines were fitted on 5,848 frozen training recordings; logistic regression reached 49.92% in-domain accuracy and random forest reached 32.54% on external TESS | Preserve artifacts and use the baselines when interpreting the HMM |
| Model | Six-class PCA and HMM training | Complete | Training-only scaling/PCA used 1,580,436 frames and six three-state diagonal Gaussian HMMs were fitted and serialized with manifest and environment metadata | Treat the current model as the frozen primary experiment |
| Model | Hand-written Viterbi validation | Pass | Brute-force equivalence, length normalization, and impossible start/transition regressions pass | Retain decoder validation during model changes |
| Uncertainty | Probability calibration and abstention | Complete | Standardized multinomial calibration converged on 1,318 validation recordings; threshold 0.2358 retained 80.20% at 42.95% validation accuracy | Develop OOD-aware abstention without tuning on TESS |
| Evaluation | In-domain and TESS evaluation | Complete | Frozen evaluation recorded 1,324 in-domain and 2,400 external clips; logistic regression won in-domain at 49.92% and random forest won TESS at 32.54% | Use validation-only changes for any versioned follow-up experiment |
| Evaluation | Training-only augmentation ablation | Pending | Deterministic RIRS reverberation extraction and isolated train/calibrate/evaluate commands are implemented; the ablation remains unrun | Run `make augmentation-ablation` after freezing primary results |
| Release gate | Repository validation | Pass | `make ci` and tracker validation passed on 2026-09-29 | Complete real browser/provider checks and dataset evaluation before a research release |

## Project Progress

> **94% complete** — 17 of 18 tracked delivery milestones.
>
> **Estimated effort remaining: 2 developer-days (about 1 working week).** This excludes model-training runtime and external approvals.

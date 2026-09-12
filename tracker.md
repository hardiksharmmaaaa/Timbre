| Area | Trackable item | Status | Evidence or exit criterion | Next checkpoint |
| --- | --- | --- | --- | --- |
| Workflow | Next engineering task | Next | Pipeline: RAVDESS clip discovery | Fix discovery using validated filename fields and add a regression test |
| Workflow | Automatic ship refresh | Complete | `make ship` runs CI and refreshes this table before committing and pushing | Keep tracker updates in the guarded ship workflow |
| UI | Frontend experience | Complete | Browser interface, microphone flow, mood graph, and animations are implemented | Keep UI integration stable while connecting the trained pipeline |
| Planning | Model and data plan | Complete | `docs/MODEL_AND_DATA_PLAN.md` documents architecture, datasets, training, and evaluation | Keep decisions and dates current |
| Data | RAVDESS download | Pass | 1,440 WAV files, 24 actors; archive MD5 `bc696df654c87fed845eb13823edef8a` | Exclude calm and surprise for the six-class experiment |
| Data | CREMA-D download | Pass | 7,442 WAV files, 91 actors; commit `1658cd342dff90010aa843eaeebd53610a08b1dc`; Git LFS check passed | Preserve ratings metadata in the canonical manifest |
| Data | TESS download | Pass | 2,800 WAV files, 2 speakers; archive SHA-256 `cf33eed0cb23671af0e7c607eb6cf89685ef6469eb7c51987b2d32cbe081b0b9` | Reserve 2,400 shared-label clips for external testing only |
| Data | RIRS_NOISES download | Pass | 61,260 WAV files; archive SHA-256 `3b50cfde915b3984738169b4beb341e9f6b8062ae4c2076146c5db71c2c05dc7` | Apply augmentation to training partitions only |
| Data safety | Raw dataset exclusion | Pass | No raw dataset files were tracked when checked on 2026-09-12 | Recheck with `git status --ignored --short data/raw` before commits |
| Security | Example credentials | Action required | `.env.example` no longer contains an API key | Revoke the previously exposed key and keep secret scans clean |
| Quality | Ruff | Pass | `make ci` passed Ruff on 2026-09-12 | Run through `make ci` before every completed change |
| Quality | Pytest | Pass | `make ci` passed pytest on 2026-09-12 | Add regression tests with every bug fix |
| Quality | Patch hygiene | Pass | `git diff --check` passed on 2026-09-12 | Require a clean result before every commit |
| Version control | Branch synchronization | Pass | Tracker refreshed for `ui` before commit and push on 2026-09-12 | Keep generated data out of future commits |
| Pipeline | RAVDESS clip discovery | Blocked | Current extractor returns zero clips because downloaded paths do not contain `Audio_Speech` | Fix discovery using validated filename fields and add a regression test |
| Pipeline | Dataset-specific parsers | Pending | RAVDESS, CREMA-D, and TESS require separate filename and speaker parsers | Implement canonical six-class mappings |
| Pipeline | Canonical manifest | Pending | No shared manifest exists | Record paths, speakers, labels, licenses, durations, and audio hashes |
| Pipeline | Audio integrity screening | Pending | Counts are verified, but corrupt, silent, clipped, and duplicate checks are not implemented | Produce documented exclusion reasons and retained counts |
| Pipeline | Speaker-disjoint splits | Pending | Current split generation is RAVDESS-specific | Freeze deterministic train, validation, test, and external-test manifests |
| Features | Cached extraction | Pending | Current extraction supports only RAVDESS-style labels | Extract each retained clip once and reuse cached `.npz` files |
| Baselines | Logistic regression and random forest | Pending | Baselines are not wired into training | Train and evaluate on the frozen six-class split |
| Model | Six-class PCA and HMM training | Pending | Training currently reads only processed RAVDESS files | Fit preprocessing on training frames only and train one HMM per class |
| Model | Hand-written Viterbi validation | Pass | Existing Viterbi tests pass in the current test suite | Retain brute-force equivalence coverage during model changes |
| Uncertainty | Probability calibration and abstention | Pending | HMM score calibration is not implemented | Fit on validation data and report ECE and Brier score |
| Evaluation | In-domain and TESS evaluation | Pending | Downloaded data is ready, but the combined pipeline is not | Report clip counts, speaker counts, macro-F1, balanced accuracy, and confusion matrices |
| Evaluation | Training-only augmentation ablation | Pending | RIRS_NOISES is downloaded but not integrated | Compare identical frozen splits with and without augmentation |
| Release gate | Repository validation | Pass | `make ci` and tracker validation passed on 2026-09-12 | Do not mark future work complete unless Ruff and pytest both pass |

## Project Progress

> **39% complete** — 7 of 18 tracked delivery milestones.
>
> **Estimated effort remaining: 24 developer-days (about 5 working weeks).** This excludes model-training runtime and external approvals.

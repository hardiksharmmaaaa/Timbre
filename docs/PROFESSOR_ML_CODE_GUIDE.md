# Timbre ML Code Guide for the Professor Demonstration

This document separates the machine-learning work from the application backend, browser UI,
external services, and project tooling. Use it as the code-reading checklist before presenting
Timbre to an ML professor.

## The most important facts

- Timbre currently uses **classical machine learning**, not a custom deep-learning network.
- The principal model is a **six-class, per-emotion Gaussian Hidden Markov Model (HMM)**
  pipeline.
- Logistic regression and random forest are comparison baselines.
- The browser interface is **vanilla HTML, CSS, and JavaScript**, not React.
- OpenAI and Gemini are external application services. They are not part of the ML model.
- The model estimates **vocal affect from speech acoustics**. It does not establish a person's
  true internal emotion or mental state.

## End-to-end system map

```text
Audio datasets
    |
    v
Canonical manifest, quality screening, and speaker-disjoint splits
    |
    v
Framewise MFCC + RMS energy + F0/pitch + voiced proportion
    |
    +--------------------------+
    |                          |
    v                          v
Clip-level aggregation     StandardScaler + PCA
    |                          |
    v                          v
Logistic regression        One Gaussian HMM per affect class
Random forest                  |
                               v
                       Hand-written Viterbi scoring
                               |
                               v
                     Calibration and abstention
                               |
                               v
                  Vocal-affect probability distribution
                               |
                               v
                 FastAPI -> browser UI / optional LLM
```

## Study-depth labels

- **Depth A — understand line by line:** Core model logic you should be ready to explain on a
  whiteboard.
- **Depth B — understand the design and important functions:** Data correctness, leakage
  prevention, reproducibility, and experiment construction.
- **Depth C — know its purpose and boundary:** Supporting code that connects the ML system to
  files, configuration, or the live application.

## Depth A: core ML files to understand deeply

### 1. `ml/core/features/extractor.py`

**Purpose:** Converts a mono waveform into a time sequence of acoustic feature vectors.

Understand:

- `feature_contract()` and why preprocessing parameters are stored with features and models.
- `frame_lengths()` and the 25 ms analysis window / 10 ms stride.
- `extract_frame_features()`.
- MFCCs as spectral-envelope descriptors.
- RMS as frame energy.
- YIN F0 as a pitch estimate.
- Why F0 is set to zero for frames with insufficient voiced activity.
- Why the output is shaped `(number_of_frames, number_of_features)` instead of one vector per
  clip.
- How `load_and_extract()` resamples clips to the configured 16 kHz sample rate.

Library-backed parts: Librosa's MFCC, RMS, split, framing, YIN, loading, and resampling
operations.

Our contribution: the feature contract, selected feature combination, alignment rules, voiced
gate, validation, and integration into a sequence-classification pipeline.

### 2. `ml/core/features/aggregation.py`

**Purpose:** Converts a variable-length frame sequence into one fixed-width vector for the
non-temporal baseline models.

Understand:

- Why logistic regression and random forest cannot directly consume variable-length sequences.
- The mean, standard deviation, median, minimum, maximum, 25th percentile, and 75th percentile
  summaries.
- What temporal information is lost by aggregation.
- Why this makes the baselines a useful comparison against the sequence-aware HMM.

### 3. `ml/core/reduction/reducer.py`

**Purpose:** Standardizes frame features and reduces dimensionality before HMM training.

Understand:

- Why pitch, energy, MFCC, and ratio features need standardization.
- `StandardScaler -> PCA` as the main reduction pipeline.
- PCA versus optional Locally Linear Embedding (LLE).
- Why the scaler and reducer must be fitted using **training frames only**.
- How `reduce_sequences()` preserves each clip's time sequence.

Library-backed parts: scikit-learn `StandardScaler`, `PCA`, `LocallyLinearEmbedding`, and
`Pipeline`.

### 4. `ml/core/hmm/decoder.py`

**Purpose:** Trains the Gaussian HMMs and scores observation sequences with Viterbi.

This is one of the most important files in the project.

Understand:

- An HMM's hidden states, observations, start probabilities, transition probabilities, and
  Gaussian emission probabilities.
- Why Timbre trains one HMM per vocal-affect class.
- `train_hmm()` and how multiple clip sequences are concatenated with their lengths.
- `viterbi_decode()` line by line.
- The dynamic-programming matrices `delta` and `psi`.
- Why calculations occur in log space.
- How impossible transitions remain negative infinity.
- The backpointer traceback that reconstructs the best hidden-state sequence.
- Why `score_viterbi_sequences()` divides by the number of frames.
- Why length normalization is needed when recordings have different durations.

Library-backed part: `hmmlearn.GaussianHMM.fit()` estimates HMM parameters using Baum-Welch.

Hand-written part: `viterbi_decode()` and the length-normalized Viterbi scoring path.

### 5. `ml/training/train.py`

**Purpose:** Executes the main six-class PCA + per-emotion HMM training experiment.

Understand:

- Loading only the frozen training split.
- The optional addition of training-only reverberated samples.
- Concatenating training frames before fitting the reducer.
- Transforming each clip while retaining sequence boundaries.
- Grouping reduced sequences by label.
- Training one HMM per class.
- Saving `reducer.joblib`, `state_models.joblib`, and `meta.json`.
- Why the manifest hash, feature contract, seed, Git commit, package versions, and training
  counts are stored with the artifacts.

### 6. `ml/training/train_baselines.py`

**Purpose:** Trains logistic-regression and random-forest baselines.

Understand:

- Why the sequences are summarized by `aggregate_sequences()`.
- Why logistic regression includes a `StandardScaler`.
- Why class weighting is used.
- What a linear baseline tells us about the need for a temporal model.
- What a random forest contributes as a nonlinear, non-temporal comparison.
- Why baseline artifacts store the same manifest and feature-contract evidence.

Library-backed parts: scikit-learn logistic regression, random forest, scaling, and pipeline.

### 7. `ml/eval/scoring.py`

**Purpose:** Provides one consistent HMM class-scoring rule for calibration and evaluation.

Understand:

- Applying the trained reducer to cached frame features.
- Running every class-specific HMM on the same clip.
- Producing a score matrix shaped `(clips, classes)`.
- Numerically stable softmax using subtraction of the row maximum.
- Why training, calibration, evaluation, and live inference must use the same scoring contract.

### 8. `ml/training/calibrate.py`

**Purpose:** Converts raw HMM class scores into better probability estimates and selects an
abstention threshold.

Understand:

- Why raw HMM/Viterbi scores are not automatically calibrated probabilities.
- Why calibration is fitted on validation data, never test data.
- Multinomial logistic score calibration.
- Expected calibration error and multiclass Brier score.
- `select_abstention_threshold()`.
- Coverage versus retained accuracy.
- How abstention lets the system decline low-confidence predictions.

Library-backed part: scikit-learn multinomial logistic regression.

### 9. `ml/core/bayesian/posterior.py`

**Purpose:** Supplies fallback probability and uncertainty surrogates when a fitted calibrator is
not available.

Understand:

- Stable normalization of log evidence with softmax.
- The `hybrid`, `vi`, and compatibility `mcmc` paths.
- Why the code calls these **uncertainty surrogates**.
- Why the `mcmc` compatibility name is not an actual Markov-chain Monte Carlo algorithm.
- The use of independent Dirichlet samples and pseudo-counts.
- Why these intervals must not be presented as empirically calibrated confidence intervals.

Research-integrity point: fitted validation calibration is stronger evidence than the fallback
surrogate intervals.

### 10. `ml/eval/metrics.py`

**Purpose:** Implements the principal evaluation and calibration metrics.

Understand:

- Accuracy.
- Balanced accuracy and why it matters for class imbalance.
- Confusion matrices.
- Expected calibration error (ECE).
- Multiclass Brier score.
- Writing versioned experiment results under `experiments/`.

### 11. `ml/training/evaluate.py`

**Purpose:** Evaluates the HMM and available baselines on frozen test partitions.

Understand:

- Evaluation on the in-domain `test` split.
- Separate external evaluation on TESS's `external_test` split.
- Clip counts, speaker counts, accuracy, balanced accuracy, macro precision/recall/F1,
  confusion matrices, calibration, abstention coverage, and latency.
- Comparing the HMM against logistic regression and random forest.
- Recording calibrated and uncalibrated references.
- Verifying that model artifacts match the current manifest hash.
- Why test and external-test results must never tune the model.

### 12. `src/timbre/api/pipeline.py`

**Purpose:** Reconstructs the trained ML artifacts and runs the exact live inference path.

Understand:

- `PerceptionPipeline.from_trained()` and artifact loading.
- Checking that the saved feature contract matches serving configuration.
- Optional calibrator and abstention artifact loading.
- Resampling live audio to the training sample rate.
- The exact inference order: features -> reduction -> all HMMs -> calibration/posterior.
- `should_abstain()`.
- Why this file is ML orchestration even though it lives under `api/`.

## Depth B: ML data and experiment files

### 13. `ml/data/manifest.py`

**Purpose:** Defines the canonical six-class dataset and frozen split contract.

Understand the design and important functions:

- Mapping RAVDESS, CREMA-D, and TESS labels into six shared classes.
- Namespaced speaker IDs.
- Dataset roles: RAVDESS/CREMA-D for development and TESS for external testing.
- File hashing and metadata capture.
- Audio decoding and quality screening.
- Silence, clipping, corruption, and duplicate checks.
- Exact-duplicate rejection.
- Speaker-disjoint split assignment.
- Why the canonical manifest is shared by extraction, training, calibration, and evaluation.
- Why changing the manifest changes the experiment identity.

You do not need to memorize every CSV-writing line, but you must be able to explain how this
file prevents label inconsistency and speaker leakage.

### 14. `ml/data/cache.py`

**Purpose:** Loads cached features only when they match the manifest.

Understand:

- The `CachedClip` data structure.
- Manifest SHA-256 calculation.
- Validation of audio hashes, labels, array shapes, and finite values.
- Missing/stale cache rejection.
- Separate loading of original and augmented training clips.
- Why cached extraction improves reproducibility and avoids recomputing audio features during
  every training run.

### 15. `ml/core/features/augmentation.py`

**Purpose:** Applies deterministic, label-preserving room reverberation to training audio.

Understand:

- Deterministic room impulse response selection from clip ID plus seed.
- FFT convolution.
- RMS matching of dry and reverberated signals.
- Wet/dry mixing and clipping prevention.
- Why augmentation is applied to training data only.

Library-backed part: SciPy FFT convolution.

### 16. `ml/data/prepare.py`

**Purpose:** Command-line entry point for canonical manifest generation and frozen splits.

Understand how it calls `prepare_manifest()` and why data preparation happens before feature
extraction.

### 17. `ml/training/extract_features.py`

**Purpose:** Extracts and caches features for every accepted manifest clip.

Understand:

- Manifest-driven extraction.
- Idempotent cache reuse.
- Feature-contract and source-audio hash checks.
- Atomic cache writes.
- Parallel extraction.
- Why labels and split names are stored with each cache.

### 18. `ml/training/extract_augmented.py`

**Purpose:** Creates a separate feature cache for reverberated training clips.

Understand:

- Selecting training rows only.
- Using RIRS_NOISES room impulse responses.
- Keeping augmentation metadata and hashes.
- Never replacing original features or augmenting validation/test data.

### 19. `ml/data/verify.py`

**Purpose:** Fails unless the manifest, frozen split document, and feature caches agree.

Understand the manifest-hash check, required nonempty splits, full cache coverage, and optional
augmentation verification.

### 20. `ml/config/types.py`

**Purpose:** Defines the shared emotion labels and ML input/output data structures.

Understand `EmotionState`, `Clip`, `EmotionPosterior`, `most_likely()`, and `confidence`.

### 21. `ml/config/config.py`

**Purpose:** Centralizes experiment and serving hyperparameters.

Understand:

- Sample rate and feature-window parameters.
- PCA/LLE component count.
- HMM hidden-state count and training iterations.
- Seed and artifact directories.
- F0/Nyquist validation.
- Why training and serving must share these settings.

### 22. `ml/config/common.py`

**Purpose:** Provides deterministic seeding, logging, and RAVDESS filename parsing.

Understand how RAVDESS filename fields encode labels and speakers and why the seed matters.

### 23. `ml/config/repro.py`

**Purpose:** Records the Git commit, Python/platform information, and ML-library versions in
model metadata.

Understand why reproducibility metadata is part of an ML experiment.

## Depth C: know the boundary and purpose

### 24. `ml/data/download.py`

Downloads and verifies dataset ZIP files. This is data acquisition, not an ML algorithm. Know
which datasets enter the experiment and their roles; you do not need to memorize the HTTP and
ZIP handling.

### 25. `ml/training/make_splits.py`

This is the older RAVDESS-only split helper. Study `split_by_actor()` as a simple example of
speaker-independent splitting, but do not present this as the current canonical workflow.
The current `make data-split` target is a compatibility alias for `ml.data.prepare`, and
the pooled split logic lives in `ml/data/manifest.py`.

### 26. `src/timbre/api/main.py`

This is primarily FastAPI application code, but understand its ML boundary:

- It loads `PerceptionPipeline` lazily.
- It decodes uploaded audio and sends the waveform to the ML pipeline.
- It handles model-unavailable, invalid-audio, and inference errors.
- Its mock inference path is for application demonstration and is not trained-model evidence.

Do not spend professor-preparation time memorizing routes, HTTP models, or static-file serving.

### 27. `Makefile`

Not an ML algorithm, but it is the executable experiment map. Understand these targets:

```text
make data-prepare
make data-extract
make data-ready
make train-all
make test-models
make augmentation-ablation
make serve
```

### 28. `pyproject.toml` and `.env.example`

Know the major ML dependencies and hyperparameters:

- NumPy and SciPy.
- Librosa and SoundFile.
- scikit-learn.
- hmmlearn.
- Feature, HMM, sample-rate, seed, and inference settings.

They are configuration/dependency files, not algorithms.

## ML tests you should use while studying

The tests are valuable because they show the expected behavior with small examples.

| Test file | What it teaches |
|---|---|
| `tests/test_features.py` | Feature shapes, silence handling, and file/direct extraction consistency |
| `tests/test_reduction.py` | PCA/LLE fitting and dimensionality reduction |
| `tests/test_viterbi.py` | Hand-written Viterbi versus brute force, impossible transitions, and normalization |
| `tests/test_bayesian.py` | Valid posterior probabilities and uncertainty-surrogate edge cases |
| `tests/test_eval.py` | Accuracy, confusion matrix, ECE, and Brier-score behavior |
| `tests/test_evaluate_script.py` | End-to-end evaluation summaries and artifact checks |
| `tests/test_manifest.py` | Dataset parsing, quality screening, deduplication, and frozen splits |
| `tests/test_feature_cache.py` | Cache coverage, hashes, labels, and stale-artifact rejection |
| `tests/test_augmentation.py` | Deterministic RIR choice and reverberation behavior |
| `tests/test_data_workflow.py` | Download, extraction, and split workflow regressions |
| `tests/test_splits.py` | Speaker-independent split behavior of the older helper |
| `tests/test_config.py` | Hyperparameter validation |
| `tests/test_api.py` | ML/application boundary, trained inference, resampling, and failures |

The following tests are not necessary for understanding the ML algorithms:

- `tests/test_frontend.py`
- `tests/frontend_runtime.cjs`
- `tests/test_companion.py`
- `tests/test_ship.py`
- `tests/test_tracker.py`

## Files that are not ML code

| Location | Category | What to know |
|---|---|---|
| `frontend/` | Browser UI | Static HTML/CSS/JavaScript microphone and graph interface; not React |
| `src/timbre/glue/companion.py` | External-service glue | Prompt construction and OpenAI/Gemini calls; not original ML |
| `src/timbre/glue/mood_logger.py` | Application persistence | Stores points for the mood graph |
| `src/timbre/api/main.py` | Backend/API | HTTP, audio upload, static UI serving, and model boundary |
| `scripts/ship.sh` | Git workflow | Tests, commits, and pushes code |
| `scripts/update_tracker.py` | Project management | Updates milestone status |
| `brag-output/` | Presentation media | Pitch-video composition, not the product or model |
| `docs/` | Documentation | Design, experiment plan, runbook, report, and this guide |

The package `__init__.py` files contain no ML algorithm. They only mark Python packages and may
contain short package descriptions.

## What is hand-written versus library-backed

| Component | Status |
|---|---|
| Feature selection, feature contract, voiced gating, and pipeline composition | Designed in this project |
| MFCC, RMS, YIN, loading, and resampling | Librosa-backed |
| StandardScaler, PCA, and LLE implementations | scikit-learn-backed |
| HMM parameter fitting / Baum-Welch | hmmlearn-backed |
| Viterbi dynamic programming and traceback | Hand-written in this project |
| Logistic regression and random forest | scikit-learn-backed baselines |
| Calibration logistic regression | scikit-learn-backed |
| ECE, Brier score integration, abstention selection, and experiment harness | Implemented in this project |
| Reverberation experiment design and guards | Implemented in this project |
| FFT convolution | SciPy-backed |
| OpenAI/Gemini language generation | External service, not ML contribution |

Be precise with the phrase "built by us." The project owns the experiment design,
preprocessing contract, leakage guards, hand-written Viterbi implementation, integration, and
evaluation workflow. It uses established libraries for standard numerical algorithms.

## Recommended reading order

Read the code in this order:

1. `README.md` — overall goal and system stages.
2. `docs/MODEL_AND_DATA_PLAN.md` — research question and experiment design.
3. `ml/config/types.py` and `ml/config/config.py` — labels and parameters.
4. `ml/data/manifest.py` — data sources, canonical labels, and splits.
5. `ml/data/cache.py` — trusted feature loading.
6. `ml/core/features/extractor.py` — audio-to-feature conversion.
7. `ml/core/features/aggregation.py` and `ml/training/train_baselines.py` — baselines.
8. `ml/core/reduction/reducer.py` — scaling and PCA.
9. `ml/core/hmm/decoder.py` — HMM training and Viterbi.
10. `ml/training/train.py` — complete training orchestration.
11. `ml/eval/scoring.py` and `ml/training/calibrate.py` — scores and probabilities.
12. `ml/core/bayesian/posterior.py` — fallback uncertainty surrogates.
13. `ml/eval/metrics.py` and `ml/training/evaluate.py` — measured results.
14. `src/timbre/api/pipeline.py` — live inference.
15. `src/timbre/api/main.py` — application boundary only.

After each file, read its corresponding test before moving to the next stage.

## Short explanation to give the professor

> Timbre estimates vocal affect from speech. We first extract a frame-by-frame sequence of
> MFCC, energy, pitch, and voiced-activity features. Scaling and PCA are fitted only on training
> frames to prevent leakage. We train one Gaussian HMM per affect class. At prediction time,
> our hand-written log-space Viterbi algorithm scores the most likely hidden-state path under
> every class model, normalized by recording length. Validation-only calibration converts those
> scores into class probabilities and selects an abstention threshold. We compare the temporal
> HMM with logistic-regression and random-forest baselines and evaluate separately on an
> in-domain test set and untouched TESS external-test data.

## Claims you must not make

- Do not call the system a deep neural network.
- Do not claim that the application reads a person's true emotion.
- Do not claim `hmmlearn` training or scikit-learn models were implemented from scratch.
- Do not call the Dirichlet surrogate genuine MCMC.
- Do not claim final accuracy or calibration numbers until actual training and evaluation
  artifacts have been generated and inspected.
- Do not use test or external-test performance to tune hyperparameters.
- Do not present mock API predictions or canned LLM replies as trained-model results.

## Exact experiment commands

Run these from the repository root:

```bash
make data-ready       # verify frozen manifest, splits, and feature cache
make train-all        # train baselines, PCA/HMM models, and validation calibration
make test-models      # evaluate in-domain test and external TESS data
make serve            # run FastAPI and the static browser UI
```

Optional augmentation experiment:

```bash
make augmentation-ablation
```

Run quality checks before demonstrating or changing the code:

```bash
make ci
```

## Final study checklist

- [ ] I can explain every stage from waveform to probabilities.
- [ ] I understand why frames and sequences are different from clip-level vectors.
- [ ] I can explain MFCC, RMS, F0, and voiced proportion in simple language.
- [ ] I understand why scaling and PCA are fitted only on training data.
- [ ] I can draw an HMM and identify hidden states, observations, transitions, and emissions.
- [ ] I can derive and trace the Viterbi recurrence and traceback.
- [ ] I understand why Viterbi scores are normalized by sequence length.
- [ ] I can explain the logistic-regression and random-forest baselines.
- [ ] I can explain calibration, Brier score, ECE, coverage, and abstention.
- [ ] I can explain speaker-disjoint splitting and external TESS evaluation.
- [ ] I know which functions are hand-written and which use established libraries.
- [ ] I can clearly separate ML, FastAPI backend, external LLM glue, and browser UI.
- [ ] I will only quote metrics produced by a completed, preserved evaluation run.

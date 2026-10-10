# Timbre Training Run Record

This document records the primary six-class Timbre training run, the complete audio-corpus
contract used by the run, the commands executed, and the artifacts produced. It is intended to
make the training process visible and reproducible for the ML professor demonstration.

## Run identity

| Field | Value |
|---|---|
| Run status | Primary training and held-out evaluation complete; augmentation ablation pending |
| Start date | 2026-09-29 |
| Git branch | `timbre-train` |
| Code commit at start | `e5f4c1b55cb4383e00d27c1829b429efc9f5c767` |
| Experiment version | `six_class_v1` |
| Random seed | `42` |
| Canonical manifest SHA-256 | `2e59f4c0a7447516c09dd2fac6564087fe27aca54e809750c8853f3f79545e1a` |
| Working audio contract | Mono, 16 kHz |
| Frame contract | 25 ms windows, 10 ms stride, 50 ms F0 window |
| Main model | StandardScaler + PCA + one diagonal Gaussian HMM per affect class |
| Baselines | Scaled logistic regression and class-balanced random forest |

## Complete recording inventory

The canonical per-recording inventory is:

```text
data/manifests/all_clips.csv
```

That CSV contains one row for **every accepted recording** used anywhere in the experiment. It
records the clip ID, dataset, relative audio path, speaker ID, canonical label, numeric label,
original label, sample rate, duration, split, dataset role, intensity, annotation agreement,
license, source URL, audio SHA-256, channel count, RMS, clipping rate, and silence fraction.

The Markdown document summarizes the corpus rather than duplicating 10,890 long CSV rows. The
CSV remains the authoritative recording-level evidence and is protected by this manifest hash:

```text
2e59f4c0a7447516c09dd2fac6564087fe27aca54e809750c8853f3f79545e1a
```

### Accepted recordings by dataset

| Dataset | Accepted recordings | Experiment role |
|---|---:|---|
| CREMA-D | 7,435 | Development: train, validation, and in-domain test |
| RAVDESS | 1,055 | Development: train, validation, and in-domain test |
| TESS | 2,400 | External test only |
| **Total** | **10,890** | All accepted recordings |

### Accepted recordings by split

| Split | Recordings | Permitted use |
|---|---:|---|
| Train | 5,848 | Fit baselines, scaler, PCA, and HMM parameters |
| Validation | 1,318 | Fit probability calibration and the abstention threshold |
| In-domain test | 1,324 | Final held-out RAVDESS/CREMA-D evaluation only |
| External test | 2,400 | Final untouched TESS generalization evaluation only |
| **Total** | **10,890** | Frozen six-class corpus |

### Accepted recordings by canonical label

| Vocal-affect label | Recordings |
|---|---:|
| Angry | 1,862 |
| Disgust | 1,862 |
| Fearful | 1,862 |
| Happy | 1,861 |
| Neutral | 1,582 |
| Sad | 1,861 |
| **Total** | **10,890** |

### Discovery and exclusions

| Dataset | Discovered WAV recordings | Six-class candidates | Accepted |
|---|---:|---:|---:|
| CREMA-D | 7,442 | 7,442 | 7,435 |
| RAVDESS | 1,440 | 1,056 | 1,055 |
| TESS | 2,800 | 2,400 | 2,400 |

RAVDESS calm and surprised recordings and TESS pleasant-surprise recordings are outside the
declared six-class experiment. After selecting six-class candidates, eight recordings were
excluded by integrity screening:

| Exclusion reason | Recordings |
|---|---:|
| Conflicting identical audio labelled angry/fearful | 2 |
| Conflicting identical audio labelled disgust/sad | 2 |
| Conflicting identical audio labelled happy/neutral | 2 |
| Exact same-label duplicate | 1 |
| Nearly silent recording | 1 |
| **Total integrity exclusions** | **8** |

The complete exclusion-level evidence is stored in:

```text
data/manifests/exclusions.csv
```

## Leakage controls

- RAVDESS and CREMA-D speakers are separated across train, validation, and test partitions.
- TESS is permanently assigned to `external_test`.
- The scaler and PCA are fitted only on training frames.
- HMM and baseline parameters are fitted only on training recordings.
- Calibration and abstention selection use validation recordings only.
- Test and external-test recordings are reserved for final evaluation.
- Reverberation augmentation, when run later, is applied only to training recordings.

## Data-readiness evidence

The following preflight command completed successfully on 2026-09-29:

```bash
make data-ready
```

Verified cache coverage:

```text
train          5,848
validation     1,318
test           1,324
external_test  2,400
total         10,890
```

All accepted recordings have finite, hash-validated cached frame features governed by the same
manifest and feature contract.

## Primary training command

```bash
make train-all
```

The command runs three visible stages in order:

1. **Baseline training**
   - Aggregate every training recording into clip-level statistics.
   - Fit scaled logistic regression.
   - Fit a class-balanced random forest.
2. **Main HMM training**
   - Fit StandardScaler and PCA on training frames only.
   - Transform each recording while preserving frame order.
   - Fit one diagonal Gaussian HMM for each of the six vocal-affect classes.
3. **Validation calibration**
   - Score every validation recording with every class HMM.
   - Standardize the six HMM scores and fit multinomial logistic score calibration.
   - Measure validation ECE and Brier score.
   - Select an abstention threshold with the declared minimum coverage.

## Expected artifacts

| Artifact | Meaning |
|---|---|
| `models/baselines/logistic_regression.joblib` | Fitted linear baseline |
| `models/baselines/random_forest.joblib` | Fitted nonlinear baseline |
| `models/baselines/meta.json` | Baseline data/code/environment evidence |
| `models/reducer.joblib` | Fitted training-only scaler and PCA |
| `models/state_models.joblib` | Six fitted class-specific HMMs |
| `models/meta.json` | Main model data/code/environment evidence |
| `models/calibrator.joblib` | Fitted validation-only score calibrator |
| `models/calibration.json` | Calibration metrics and abstention threshold |

## Live training observations

| Stage | Status | Recorded observation |
|---|---|---|
| Data readiness | Complete | 10,890 of 10,890 accepted recordings have valid caches |
| Logistic-regression baseline | Complete | Fitted on 5,848 recordings represented by 161 aggregated features |
| Random-forest baseline | Complete | Fitted on the same 5,848-recording, 161-feature matrix |
| Training-only scaler and PCA | Complete | Fitted on 1,580,436 training frames and reduced to 8 dimensions |
| Six class-specific HMMs | Complete | Fitted neutral, happy, sad, angry, fearful, and disgust models |
| Validation calibration | Complete | Standardized logistic calibrator converged in 67 iterations on 1,318 validation recordings |

### HMM training recordings by class

| State ID | Vocal-affect class | Training recordings |
|---:|---|---:|
| 1 | Neutral | 815 |
| 3 | Happy | 1,006 |
| 4 | Sad | 1,006 |
| 5 | Angry | 1,007 |
| 6 | Fearful | 1,007 |
| 7 | Disgust | 1,007 |
| **Total** | **Six classes** | **5,848** |

### Validation-only calibration results

These are validation results used to fit and inspect calibration. They are **not final test
results**.

| Metric | Recorded value |
|---|---:|
| Validation recordings | 1,318 |
| Validation accuracy | 39.23% |
| Multiclass Brier score | 0.7229 |
| Expected calibration error | 0.0455 |
| Uncalibrated Brier reference | 0.8215 |
| Selected abstention threshold | 0.2358 |
| Coverage at selected threshold | 80.20% |
| Accuracy on retained recordings | 42.95% |

The first calibration attempt used unstandardized HMM scores and reached its 2,000-iteration
limit. That artifact was not accepted as final. The calibration workflow was corrected to fit a
`StandardScaler -> LogisticRegression` pipeline and to turn future convergence warnings into
hard failures. The corrected fit converged in 67 iterations and replaced the initial artifact.

### Generated artifact evidence

| Artifact | Result |
|---|---|
| `models/baselines/logistic_regression.joblib` | Created, approximately 12.7 KiB |
| `models/baselines/random_forest.joblib` | Created, approximately 87.5 MiB |
| `models/baselines/meta.json` | Created |
| `models/reducer.joblib` | Created, approximately 3.7 KiB |
| `models/state_models.joblib` | Created, approximately 9.4 KiB |
| `models/meta.json` | Created |
| `models/calibrator.joblib` | Created, approximately 1.8 KiB |
| `models/calibration.json` | Created |

The complete model directory occupies approximately 88 MiB. Model artifacts remain ignored by
Git as intended. The baseline and main-model metadata bind those artifacts to the manifest,
environment, and `e5f4c1b` code commit. The standardized-calibration correction is currently a
local code change and must be committed before the experiment is treated as release evidence.

## Post-training verification

The full repository quality workflow completed after the calibration correction:

```text
Ruff:   passed
Pytest: 142 passed
Patch:  git diff --check passed
```

The test warnings were dependency deprecations plus joblib's informational logical-core
fallback; no test failed.

A trained-pipeline smoke check loaded the saved PCA, six HMMs, and calibrator, then processed one
cached validation recording through `infer_features()`:

```text
clip_id          crema_d__1005_DFA_ANG_XX
true label       angry (state 5)
predicted label  angry (state 5)
confidence       0.812741
abstained        false
```

This single recording demonstrates that the serialized pipeline can be loaded and executed. It
is not an accuracy estimate; aggregate performance must come from the held-out evaluation.

## Held-out evaluation results

The final frozen-split evaluation command completed successfully:

```bash
make test-models
```

The machine-readable result is:

```text
experiments/six_class_v1/metrics_20260929-154227.json
```

The JSON is a generated, Git-ignored experiment artifact. The tables below preserve its key
measurements in the project documentation. Label order in all confusion matrices is:

```text
neutral, happy, sad, angry, fearful, disgust
```

### In-domain test: RAVDESS and CREMA-D

The in-domain test contains 1,324 recordings from 18 held-out speakers. A uniform six-class
guess would average 16.67% accuracy; the largest class represents 17.22% of this split.

| Model | Accuracy | Balanced accuracy | Macro-F1 | ECE | Brier | Coverage | Covered accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| Calibrated HMM | 38.97% | 39.45% | 37.42% | 0.0739 | 0.7376 | 74.92% | 42.54% |
| Logistic regression | **49.92%** | **50.19%** | **49.68%** | 0.1006 | 0.6618 | 100% | 49.92% |
| Random forest | 48.64% | 49.14% | 47.65% | 0.0896 | **0.6549** | 100% | 48.64% |

Interpretation:

- Logistic regression is the strongest in-domain classifier by accuracy, balanced accuracy,
  and macro-F1.
- Random forest is close and has the lowest in-domain Brier score.
- The HMM is clearly above chance but is substantially weaker than both non-temporal baselines.
- The validation-selected abstention rule removes about 25% of in-domain recordings and raises
  HMM accuracy from 38.97% overall to 42.54% on retained recordings.
- Calibration materially improves the HMM over its raw-score reference: uncalibrated accuracy
  was 26.96%, macro-F1 was 21.55%, ECE was 0.1022, and Brier score was 0.8139.

### In-domain class-level F1

| Class | Calibrated HMM | Logistic regression | Random forest |
|---|---:|---:|---:|
| Neutral | 41.95% | 49.88% | **52.44%** |
| Happy | 25.64% | **46.07%** | 41.43% |
| Sad | 50.11% | 50.26% | **54.85%** |
| Angry | 59.88% | **69.58%** | 64.76% |
| Fearful | 28.19% | **45.06%** | 34.27% |
| Disgust | 18.73% | 37.23% | **38.14%** |

Angry is the easiest in-domain class for all three models. Disgust is the weakest HMM class,
and happy/fearful are also difficult for the sequence model.

### In-domain HMM confusion matrix

Rows are true labels and columns are predictions.

| True / predicted | Neutral | Happy | Sad | Angry | Fearful | Disgust |
|---|---:|---:|---:|---:|---:|---:|
| Neutral | 99 | 16 | 25 | 0 | 11 | 33 |
| Happy | 33 | 50 | 11 | 53 | 61 | 20 |
| Sad | 60 | 7 | 114 | 3 | 27 | 17 |
| Angry | 8 | 39 | 1 | 156 | 15 | 9 |
| Fearful | 24 | 19 | 45 | 55 | 63 | 22 |
| Disgust | 64 | 31 | 31 | 26 | 42 | 34 |

### External test: untouched TESS

The external test contains 2,400 recordings from the two TESS speakers, with exactly 400
recordings per class. It measures cross-dataset generalization, not performance on new speakers
drawn from the same recording conditions.

| Model | Accuracy | Balanced accuracy | Macro-F1 | ECE | Brier | Coverage | Covered accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| Calibrated HMM | 26.88% | 26.88% | 22.94% | 0.1662 | 0.7960 | 100% | 26.88% |
| Logistic regression | 32.29% | 32.29% | 25.50% | 0.4954 | 1.1080 | 100% | 32.29% |
| Random forest | **32.54%** | **32.54%** | **30.80%** | **0.0280** | **0.7547** | 100% | **32.54%** |

Interpretation:

- All models degrade substantially on TESS, demonstrating strong cross-dataset domain shift.
- Random forest is the strongest external model, although 32.54% is still modest.
- Logistic regression retains similar accuracy but becomes extremely poorly calibrated on TESS;
  its 0.4954 ECE and 1.1080 Brier score indicate unreliable probabilities.
- Random forest has low external ECE, but its modest accuracy and Brier score mean that this
  must not be described as strong predictive performance.
- HMM calibration improves external Brier score over the raw-score reference (0.7960 versus
  0.8721) and improves accuracy (26.88% versus 23.42%), but external ECE becomes slightly worse
  (0.1662 versus 0.1594).
- The HMM abstention threshold retains every TESS recording, so the validation-derived threshold
  does not detect this external domain shift.

### External class-level F1

| Class | Calibrated HMM | Logistic regression | Random forest |
|---|---:|---:|---:|
| Neutral | 26.49% | **47.14%** | 8.80% |
| Happy | 29.76% | 23.05% | **30.42%** |
| Sad | 1.49% | 28.34% | **45.16%** |
| Angry | **43.08%** | 42.66% | 39.09% |
| Fearful | **36.82%** | 4.88% | 24.27% |
| Disgust | 0.00% | 6.94% | **37.07%** |

### External HMM confusion matrix

| True / predicted | Neutral | Happy | Sad | Angry | Fearful | Disgust |
|---|---:|---:|---:|---:|---:|---:|
| Neutral | 91 | 308 | 0 | 1 | 0 | 0 |
| Happy | 0 | 323 | 0 | 39 | 38 | 0 |
| Sad | 189 | 207 | 3 | 1 | 0 | 0 |
| Angry | 0 | 271 | 0 | 126 | 3 | 0 |
| Fearful | 0 | 295 | 0 | 3 | 102 | 0 |
| Disgust | 7 | 367 | 0 | 15 | 11 | 0 |

The HMM predicts `happy` for 1,771 of 2,400 TESS recordings (73.79%), predicts `sad` only three
times, and never predicts `disgust`. This class collapse is the clearest evidence that the
current HMM score distribution does not transfer reliably to TESS.

### Inference latency

| Split | HMM median | HMM p90 | HMM p95 |
|---|---:|---:|---:|
| In-domain test | 9.94 ms | 13.93 ms | 14.66 ms |
| External TESS | 8.29 ms | 10.14 ms | 10.72 ms |

The HMM is comfortably below the configured 300 ms inference budget for cached features. These
numbers measure model scoring only; they do not include microphone capture, audio decoding,
feature extraction, network calls, LLM generation, or TTS. Baseline timings were measured as
vectorized batches and are therefore not directly comparable with the per-recording HMM timing.

## Overall evaluation verdict

The current experiment is a valid baseline research result, but it does **not** support a claim
that the HMM outperforms simpler models. The opposite is measured:

- Logistic regression wins the in-domain comparison.
- Random forest wins the external TESS comparison.
- The HMM is fast and above chance, but loses accuracy and macro-F1 to both baselines.
- Validation calibration helps in-domain HMM predictions, but abstention and calibration do not
  transfer reliably to TESS.
- Cross-dataset robustness is the principal unresolved modeling problem.

This is still a useful ML-course conclusion: temporal modeling was implemented and tested, but
the selected three-state diagonal-Gaussian HMM, eight-dimensional PCA representation, and
current acoustic features do not beat strong clip-summary baselines.

## Evidence-based next experiments

Any improvement must be selected using training/validation data only. Do not tune against the
reported test or TESS numbers.

1. Save per-recording predictions and scores to support paired error analysis and confidence
   intervals.
2. Analyze score and feature distributions by dataset to quantify the RAVDESS/CREMA-D/TESS
   domain shift.
3. Add speaker-level acoustic normalization or cepstral mean/variance normalization within a
   newly versioned experiment.
4. Compare PCA dimensions and HMM state counts using validation performance only.
5. Consider delta and delta-delta MFCCs and explicit pitch/energy normalization.
6. Balance dataset influence during training because CREMA-D contributes 5,145 of 5,848
   training recordings while RAVDESS contributes 703.
7. Design out-of-distribution detection on a separate validation domain; do not retrofit the
   threshold using TESS.
8. Run the declared training-only reverberation ablation without changing the frozen test sets.

## Research-integrity statement

Timbre estimates patterns of vocal affect from acoustic speech features. Dataset labels are
acted or annotated categories, and a prediction does not prove a speaker's internal emotional
or mental state. Final results must identify the dataset split, model version, manifest hash,
and whether the number came from validation, in-domain test, or external testing.

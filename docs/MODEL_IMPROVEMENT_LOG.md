# Timbre Model Improvement Log

This is the live, reproducible record of work toward a high-accuracy Timbre classifier. It
records the target, evaluation rules, experiments, failures, and promotion decisions so that an
accuracy claim cannot be produced by speaker leakage, test-set tuning, or reporting training
performance.

## Target and non-negotiable rules

The requested target is at least **90% accuracy**. That is a target, not a guaranteed outcome.
The project will report the best result supported by held-out evidence.

A candidate can be called a 90% model only when all of the following are true:

1. Accuracy is at least 90% on a speaker-disjoint held-out set.
2. Macro-F1 is also at least 90%, so one dominant class cannot create the headline number.
3. No speaker, recording, duplicate audio, or augmented copy crosses the train/evaluation
   boundary.
4. Hyperparameters and feature choices are selected without consulting the final test labels.
5. Training and validation learning behavior shows no uncontrolled generalization gap.
6. The exact data manifest, code revision, configuration, seed, and artifacts are recorded.
7. Per-class metrics, confusion matrices, calibration, and external-domain behavior are reported
   beside overall accuracy.

Training accuracy alone is never an acceptance metric. Random clip splitting is prohibited
because the same speaker can otherwise appear on both sides and inflate performance.

## Evaluation protocol

The `six_class_v1` primary experiment is frozen. Its test and TESS results have already been
observed and must not be used to choose v2 features or hyperparameters.

Improvement work follows this order:

1. Use the existing training speakers for model fitting.
2. Use grouped, speaker-disjoint cross-validation inside the training partition for
   hyperparameter and model-family selection.
3. Use the existing validation speakers for calibration, abstention, and a final development
   check only after grouped selection.
4. Treat the existing test/TESS numbers as v1 historical diagnostics, not a tuning signal.
5. Before a final v2 claim, declare a newly sealed speaker/domain holdout or obtain a new
   external dataset that has not influenced development.

## Locked v1 baseline

Manifest SHA-256:

```text
2e59f4c0a7447516c09dd2fac6564087fe27aca54e809750c8853f3f79545e1a
```

| Model | In-domain accuracy | In-domain macro-F1 | TESS accuracy | TESS macro-F1 |
|---|---:|---:|---:|---:|
| Calibrated HMM | 38.97% | 37.42% | 26.88% | 22.94% |
| Logistic regression | 49.92% | 49.68% | 32.29% | 25.50% |
| Random forest | 48.64% | 47.65% | 32.54% | 30.80% |

The v1 comparison proves that the current HMM does not beat the clip-summary baselines and that
all three models have substantial cross-dataset domain shift.

## Overfitting and underfitting checks

Every new candidate will be examined using:

- Speaker-grouped cross-validation mean and fold spread.
- Training-versus-validation macro-F1 gap.
- Per-class precision, recall, and F1.
- Confusion matrices and predicted-class distribution.
- Accuracy and macro-F1 together.
- Calibration error and Brier score.
- Learning curves where the estimator exposes meaningful staged behavior.
- Duplicate/hash and split-leakage validation.
- External-domain diagnostics that are never used as the optimization objective.

Signs of overfitting include near-perfect training scores with materially worse grouped
validation, unstable folds, speaker-specific success, or external collapse. Signs of
underfitting include low training and validation performance together, class collapse, and weak
separation even on the development corpus.

## Planned improvement sequence

1. Complete the existing training-only RIRS reverberation ablation.
2. Persist per-recording scores/predictions for error and domain-shift analysis.
3. Add group-aware candidate evaluation on training speakers.
4. Add stronger leakage-safe acoustic representations, including delta features and explicit
   normalization, under a versioned feature contract.
5. Compare regularized linear, tree ensemble, kernel, and sequence models using grouped
   validation only.
6. Balance dataset influence so CREMA-D does not dominate RAVDESS.
7. Fit calibration only after choosing the model family and hyperparameters.
8. Freeze the winning v2 configuration before any final sealed evaluation.
9. Consider a local pretrained speech encoder only if classical improvements plateau and the
   dependency/model download can be made reproducible within available storage and compute.

## Experiment ledger

| ID | Change | Selection data | Result | Decision |
|---|---|---|---|---|
| v1-primary | Original cached features, PCA-8, three-state diagonal HMM | Frozen train/validation | HMM 38.97% in-domain; 26.88% TESS | Baseline only |
| v1-baselines | Aggregated features with logistic regression and random forest | Frozen train | Logistic 49.92% in-domain; RF 32.54% TESS | Baselines beat HMM |
| v1-rirs | Training-only RIRS reverberation | Frozen train/validation | 37.61% test accuracy; 25.33% TESS | Rejected: worse than v1 |
| v2-static-search | Nine classifiers on the 161-value static summary | Five-fold grouped train CV | Best: 51.13% accuracy / 50.51% macro-F1 | Insufficient; boosting overfit |
| v2-dynamic-search | Static + delta + acceleration + slope + normalized contour features (691 values) | Five-fold grouped train CV | Best: 54.27% accuracy / 53.83% macro-F1 | Keep representation; regularize further |

## Live work log

### 2026-09-29 — Accuracy target accepted with evidence constraints

- Recorded the requested 90% target.
- Refused to redefine the target using training accuracy or leaky random clip splits.
- Preserved the completed `six_class_v1` evaluation as the locked baseline.
- Confirmed that PyTorch, torchaudio, Transformers, XGBoost, and LightGBM are not installed in
  the current project environment.
- Started with the existing reproducible RIRS augmentation experiment before introducing a new
  feature contract or additional dependencies.

### 2026-09-29 — RIRS augmentation ablation completed

- Reused 5,848 validated training-only augmented feature caches and trained a separate model in
  `models/augmented`; augmented data never entered validation or test splits.
- Doubled training input from 5,848 to 11,696 sequences and from 1,580,436 to 3,160,872 frames.
- Evaluated with the same frozen protocol. In-domain HMM accuracy fell from 38.97% to 37.61%,
  while TESS accuracy fell from 26.88% to 25.33%.
- Rejected the augmentation configuration. It adds reverberation robustness but does not improve
  the emotion decision boundary for this corpus and model.
- Evidence: `experiments/six_class_v1_augmented/metrics_20260929-155613.json`.

### 2026-09-29 — Speaker-grouped static model screen

- Added a deterministic five-fold `StratifiedGroupKFold` screen over only the 5,848 training
  recordings and 79 training speakers.
- Compared three regularization strengths of logistic regression, random forest, two Extra Trees
  variants, two RBF-SVM strengths, and histogram gradient boosting.
- The best mean was histogram gradient boosting at 51.13% accuracy and 50.51% macro-F1.
- Its mean training macro-F1 was 95.61%, leaving a 45.10-point gap. Random forest and Extra Trees
  reached 100% training accuracy but validated below 50%, so they were explicitly rejected as
  overfit rather than promoted using their training scores.
- RBF-SVM with `C=1` was nearly tied at 51.09% accuracy / 50.43% macro-F1 with a smaller, though
  still substantial, 22.05-point macro-F1 gap.
- Evidence: `experiments/model_search_v2/summary.json`.

### 2026-09-29 — Dynamic, speaker-normalized representation

- Added a versioned 691-value clip representation while preserving the original 161-value
  baseline. It includes first differences, second differences, least-squares temporal slopes,
  clip duration, and four robustly normalized temporal segments.
- The robust within-utterance normalization reduces reliance on a speaker's absolute pitch and
  vocal-tract characteristics; the delta and segment features preserve emotion-related movement
  that the original distribution-only summary discarded.
- Added unit tests for fixed width, finite output, one-frame clips, invalid configuration, and
  model-search representation selection.
- Re-ran the same nine candidates with identical speaker-disjoint folds. The best mean improved
  to 54.27% accuracy / 53.83% macro-F1. RBF-SVM `C=10` reached 53.48% / 53.29%, and `C=1`
  reached 53.10% / 52.76%.
- The best boosting candidate still has a 41.86-point train/validation macro-F1 gap. The feature
  representation is retained because it improves grouped validation, but this boosting
  configuration is not considered deployment-ready.
- Evidence: `experiments/model_search_dynamic_v2/summary.json`.

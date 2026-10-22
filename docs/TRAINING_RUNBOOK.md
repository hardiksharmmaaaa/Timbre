# Six-class training and testing runbook

This runbook keeps data preparation, model training, calibration, and final testing as separate
gates. Run commands from the repository root.

## Prepared data contract

The canonical experiment uses:

- RAVDESS and CREMA-D for development.
- Six shared labels: neutral, happy, sad, angry, fearful, and disgust.
- Speaker-disjoint train, validation, and test partitions created independently per dataset.
- TESS only as untouched `external_test` data.
- 16 kHz mono working audio, 25 ms acoustic frames, 10 ms hops, and a 50 ms F0 window.

Generated data stays under ignored directories. The authoritative artifacts are:

```text
data/manifests/all_clips.csv
data/manifests/exclusions.csv
data/manifests/all_clips.sha256
data/manifests/summary.json
data/splits/six_class_v1.json
data/processed/<dataset>/<clip_id>.npz
```

## Today: prepare the immutable training inputs

```bash
make data-prepare WORKERS=4
make data-extract WORKERS=4
make data-ready
make ci
```

`data-prepare` is idempotent. It parses dataset-specific filenames, preserves CREMA-D voice-vote
agreement, hashes and decodes every candidate, records audio-quality measurements and exclusions,
rejects exact duplicates, freezes speaker splits, and records the manifest hash.

`data-extract` is also idempotent. A cache is reused only when its source-audio hash and complete
feature contract match. Interrupted writes are not published as valid `.npz` files.

## Tomorrow: training stage

```bash
make data-ready
make train-all
```

This command runs, in order:

1. Logistic-regression, random-forest, and cross-validated KNN/AdaBoost/SVM clip-summary baselines.
2. Train-only StandardScaler/PCA plus one diagonal Gaussian HMM per emotion.
3. Validation-only multinomial score calibration and abstention-threshold selection.

Artifacts are written under `models/` with the manifest hash, feature contract, seed, Git commit,
dependency versions, class counts, and dataset counts. Do not change data, splits, or feature
settings between these stages.

### KNN selection

`make train-baselines` (also run by `make train-all`) fits scikit-learn KNN on the
161 clip-summary features. It searches 32 combinations: `n_neighbors` in
`[1, 3, 5, 7, 11, 15, 21, 31]`, `weights` in `[uniform, distance]`, and Minkowski
`p` in `[1, 2]` (Manhattan or Euclidean distance).

Five seeded `StratifiedGroupKFold` folds use only the frozen training clips, with speakers
kept together. A `StandardScaler` inside the searched pipeline is fitted separately on each
fold's training subset. The highest **mean held-out CV accuracy** wins; exact ties use the
first candidate in scikit-learn's parameter-grid order. Neighbor counts larger than the
smallest fold's training set are excluded and recorded. Insufficient speakers cause an error.

The winner is refitted on all training clips and saved as `models/baselines/knn.joblib`.
`models/baselines/knn_cv.json` records the grid, every candidate's fold scores, means, standard
deviations and ranks, speaker assignments, selected parameters, and training provenance.
`models/baselines/meta.json` includes a compact selection summary. Validation, test, and TESS
clips never participate in this search or refit. CV accuracy is a selection score, not the
final test accuracy.

### AdaBoost selection

`make train-baselines` also searches 12 AdaBoost configurations on the same 161 clip-summary
features and the same five training-only speaker-disjoint folds as KNN. The grid is:

- `n_estimators`: 50 or 100.
- `learning_rate`: 0.1, 0.5, or 1.0.
- Decision-tree `max_depth`: 1 or 2.

The implementation wraps scikit-learn multiclass SAMME AdaBoost with seeded decision trees.
Tree learners use the unscaled clip statistics. Four CV fits can run concurrently. The highest
mean held-out CV accuracy wins; exact ties use parameter-grid order. The winner is refitted on
all training clips. Validation, final test, and TESS remain outside selection and fitting.

Artifacts are `models/baselines/adaboost.joblib` and `models/baselines/adaboost_cv.json`.
The report records every candidate's fold scores, mean, standard deviation and rank, fold
speakers, selected parameters, actual fitted tree count (boosting can stop early), seed,
manifest hash, feature contract, and environment. A selection summary also lives in
`models/baselines/meta.json`.

### SVM selection

`make train-baselines` searches 30 scikit-learn `SVC` configurations on the same 161 features
and five training-only speaker-disjoint folds. `StandardScaler` is fitted inside each fold.

- Linear kernel: `C` in `[0.1, 1, 10]`.
- RBF kernel: `C` in `[0.1, 1, 10, 100]`, `gamma` in `[scale, 0.001, 0.01]`.
- Both kernels: `class_weight` in `[None, balanced]`.

Four fits can run concurrently. The highest mean held-out CV accuracy wins, with parameter-grid
order breaking exact ties. Refit the selected pipeline on all training clips and save it as
`models/baselines/svm.joblib`. `svm_cv.json` records all fold scores, speaker assignments, ranks,
selected parameters, support-vector count, and provenance; `meta.json` includes the selection
summary. Validation, test, and TESS remain excluded from both selection and refitting.

SVC probability estimation is disabled. Evaluation uses the same `predict` decision rule as CV,
reports classification metrics and latency, and sets ECE, Brier score, and abstention threshold
to `null` with an explanation. All clips are classified (coverage 1); decision scores are never
treated as probabilities. This classifier is an experimental baseline, not the API's HMM pipeline.

## Testing stage

Only after training and calibration complete:

```bash
make test-models
```

The evaluation reports test and untouched TESS results separately. It records clip and speaker
counts, accuracy, balanced accuracy, macro precision/recall/F1, confusion matrices, ECE,
multiclass Brier score, abstention coverage/covered accuracy, and inference latency. Baselines are
evaluated on the same frozen clips when their artifacts exist.
The saved KNN, AdaBoost, and SVM models are included automatically, with separate `knn`,
`adaboost`, and `svm` metrics for `test` and `external_test` and baseline provenance in the report.
Evaluation never searches or refits them.

Do not use test or TESS results to choose features, HMM states, PCA dimensions, calibration, or
abstention thresholds. Any such choice requires a new versioned experiment and a newly declared
validation protocol.

## Training-only reverberation ablation

After the primary unaugmented model is frozen and tested, run the separately versioned ablation:

```bash
make augmentation-ablation WORKERS=4
```

This deterministically assigns one RIRS_NOISES simulated room response to each training clip,
creates a separate augmented feature cache, trains under `models/augmented/`, calibrates only on
the original validation clips, and evaluates on the exact same original test/TESS inputs. It does
not augment or overwrite validation, test, or external-test data.

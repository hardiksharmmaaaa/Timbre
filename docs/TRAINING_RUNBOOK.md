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

1. Logistic-regression and random-forest clip-summary baselines.
2. Train-only StandardScaler/PCA plus one diagonal Gaussian HMM per emotion.
3. Validation-only multinomial score calibration and abstention-threshold selection.

Artifacts are written under `models/` with the manifest hash, feature contract, seed, Git commit,
dependency versions, class counts, and dataset counts. Do not change data, splits, or feature
settings between these stages.

## Testing stage

Only after training and calibration complete:

```bash
make test-models
```

The evaluation reports test and untouched TESS results separately. It records clip and speaker
counts, accuracy, balanced accuracy, macro precision/recall/F1, confusion matrices, ECE,
multiclass Brier score, abstention coverage/covered accuracy, and inference latency. Baselines are
evaluated on the same frozen clips when their artifacts exist.

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

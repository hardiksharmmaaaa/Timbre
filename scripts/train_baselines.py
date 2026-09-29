"""Train logistic-regression and random-forest six-class baselines."""

from __future__ import annotations

import json
from collections import Counter

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from timbre.data.cache import load_cached_clips, manifest_sha256
from timbre.features.aggregation import aggregate_sequences
from timbre.features.extractor import feature_contract
from timbre.utils.common import get_logger, set_seed
from timbre.utils.config import ROOT, settings
from timbre.utils.repro import environment_metadata, git_commit

log = get_logger("baselines")


def main() -> None:
    set_seed(settings.seed)
    manifest_path = settings.manifest_dir / "all_clips.csv"
    clips = load_cached_clips(manifest_path, settings.processed_dir, {"train"})
    features = aggregate_sequences([clip.features for clip in clips])
    labels = np.array([clip.label for clip in clips])

    logistic = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=settings.seed,
                ),
            ),
        ]
    )
    forest = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced_subsample",
        random_state=settings.seed,
        n_jobs=-1,
    )
    log.info("training logistic regression on %s", features.shape)
    logistic.fit(features, labels)
    log.info("training random forest on %s", features.shape)
    forest.fit(features, labels)

    output_dir = settings.model_dir / "baselines"
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(logistic, output_dir / "logistic_regression.joblib")
    joblib.dump(forest, output_dir / "random_forest.joblib")
    metadata = {
        "schema_version": 1,
        "models": ["logistic_regression", "random_forest"],
        "summary_statistics": ["mean", "std", "median", "min", "max", "p25", "p75"],
        "feature_width": int(features.shape[1]),
        "training_clip_count": len(clips),
        "training_counts_by_dataset": dict(sorted(Counter(c.dataset for c in clips).items())),
        "manifest_sha256": manifest_sha256(manifest_path),
        "feature_contract": feature_contract(),
        "seed": settings.seed,
        "git_commit": git_commit(ROOT),
        "environment": environment_metadata(),
    }
    (output_dir / "meta.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    log.info("saved baseline artifacts to %s", output_dir)


if __name__ == "__main__":
    main()


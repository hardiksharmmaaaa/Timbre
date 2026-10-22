"""Train logistic-regression, random-forest, and cross-validated KNN/AdaBoost/SVM baselines."""

from __future__ import annotations

import json
from collections import Counter

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml.config import ROOT, environment_metadata, get_logger, git_commit, set_seed, settings
from ml.core.features.aggregation import aggregate_sequences
from ml.core.features.extractor import feature_contract
from ml.data.cache import load_cached_clips, manifest_sha256
from timbre.baselines import BASELINE_NAMES, fit_adaboost, fit_knn, fit_svm

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

    log.info("selecting KNN by five-fold speaker-disjoint CV accuracy (32 candidates)")
    knn, knn_report = fit_knn(clips, settings.seed)
    log.info("selected KNN %s; mean CV accuracy=%.4f",
             knn_report["best_params"], knn_report["best_mean_cv_accuracy"])

    log.info("selecting AdaBoost by five-fold speaker-disjoint CV accuracy (12 candidates)")
    adaboost, adaboost_report = fit_adaboost(clips, settings.seed, n_jobs=4)
    log.info("selected AdaBoost %s; mean CV accuracy=%.4f",
             adaboost_report["best_params"], adaboost_report["best_mean_cv_accuracy"])
    log.info("selecting SVM by five-fold speaker-disjoint CV accuracy (30 candidates)")
    svm, svm_report = fit_svm(clips, settings.seed, n_jobs=4)
    log.info("selected SVM %s; mean CV accuracy=%.4f",
             svm_report["best_params"], svm_report["best_mean_cv_accuracy"])
    selection_reports = {"knn": knn_report, "adaboost": adaboost_report, "svm": svm_report}

    output_dir = settings.model_dir / "baselines"
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(logistic, output_dir / "logistic_regression.joblib")
    joblib.dump(forest, output_dir / "random_forest.joblib")
    joblib.dump(knn, output_dir / "knn.joblib")
    joblib.dump(adaboost, output_dir / "adaboost.joblib")
    joblib.dump(svm, output_dir / "svm.joblib")
    metadata = {
        "schema_version": 2,
        "models": list(BASELINE_NAMES),
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
    for name, report in selection_reports.items():
        metadata[f"{name}_selection"] = {
            key: report[key] for key in (
                "selection_metric", "selection_split", "refit_split", "n_splits",
                "best_params", "best_mean_cv_accuracy",
            )
        }
        report.update({key: metadata[key] for key in (
            "manifest_sha256", "feature_contract", "git_commit", "environment",
            "training_counts_by_dataset", "feature_width", "summary_statistics",
        )})
        (output_dir / f"{name}_cv.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n"
        )
    (output_dir / "meta.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    log.info("saved baseline artifacts to %s", output_dir)


if __name__ == "__main__":
    main()

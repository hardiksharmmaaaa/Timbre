"""Speaker-grouped model screening on the frozen training partition only."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from timbre.data.cache import CachedClip, load_cached_clips, manifest_sha256
from timbre.features.aggregation import aggregate_sequences, aggregate_sequences_v2
from timbre.utils.common import get_logger, set_seed
from timbre.utils.config import settings

log = get_logger("model-search")


def candidate_models(seed: int) -> dict[str, object]:
    """Return the predeclared, leakage-safe candidate set."""
    return {
        "logistic_c0.1": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        C=0.1,
                        max_iter=3000,
                        class_weight="balanced",
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "logistic_c1": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        C=1.0,
                        max_iter=3000,
                        class_weight="balanced",
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "logistic_c10": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        C=10.0,
                        max_iter=3000,
                        class_weight="balanced",
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=600,
            class_weight="balanced_subsample",
            min_samples_leaf=1,
            max_features="sqrt",
            random_state=seed,
            n_jobs=-1,
        ),
        "extra_trees_leaf1": ExtraTreesClassifier(
            n_estimators=600,
            class_weight="balanced",
            min_samples_leaf=1,
            max_features="sqrt",
            random_state=seed,
            n_jobs=-1,
        ),
        "extra_trees_leaf2": ExtraTreesClassifier(
            n_estimators=600,
            class_weight="balanced",
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=seed,
            n_jobs=-1,
        ),
        "rbf_svc_c1": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", SVC(C=1.0, gamma="scale", class_weight="balanced")),
            ]
        ),
        "rbf_svc_c10": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", SVC(C=10.0, gamma="scale", class_weight="balanced")),
            ]
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            learning_rate=0.08,
            max_iter=300,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            class_weight="balanced",
            early_stopping=True,
            random_state=seed,
        ),
    }


def speaker_disjoint_folds(
    labels: np.ndarray,
    speakers: np.ndarray,
    n_splits: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create stratified folds and fail if a speaker or class leaks across a fold."""
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    indices = np.arange(labels.size)
    all_classes = set(labels.tolist())
    folds = list(splitter.split(indices, labels, speakers))
    for train_indices, validation_indices in folds:
        train_speakers = set(speakers[train_indices].tolist())
        validation_speakers = set(speakers[validation_indices].tolist())
        if train_speakers & validation_speakers:
            raise RuntimeError("speaker leakage detected in model-search folds")
        if set(labels[validation_indices].tolist()) != all_classes:
            raise RuntimeError("a model-search validation fold is missing a class")
    return folds


def score_candidate(
    name: str,
    estimator: object,
    features: np.ndarray,
    labels: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> dict:
    """Fit one estimator independently in every fold and report generalization gaps."""
    fold_results = []
    for fold_number, (train_indices, validation_indices) in enumerate(folds, start=1):
        model = clone(estimator)
        model.fit(features[train_indices], labels[train_indices])
        train_predictions = model.predict(features[train_indices])
        validation_predictions = model.predict(features[validation_indices])
        result = {
            "fold": fold_number,
            "train_accuracy": float(accuracy_score(labels[train_indices], train_predictions)),
            "train_macro_f1": float(
                f1_score(labels[train_indices], train_predictions, average="macro")
            ),
            "validation_accuracy": float(
                accuracy_score(labels[validation_indices], validation_predictions)
            ),
            "validation_macro_f1": float(
                f1_score(labels[validation_indices], validation_predictions, average="macro")
            ),
            "train_recordings": int(train_indices.size),
            "validation_recordings": int(validation_indices.size),
        }
        fold_results.append(result)
        log.info(
            "%s fold %d/%d: validation accuracy=%.4f macro_f1=%.4f",
            name,
            fold_number,
            len(folds),
            result["validation_accuracy"],
            result["validation_macro_f1"],
        )

    def mean(key: str) -> float:
        return float(np.mean([result[key] for result in fold_results]))

    def std(key: str) -> float:
        return float(np.std([result[key] for result in fold_results]))

    return {
        "name": name,
        "folds": fold_results,
        "mean_train_accuracy": mean("train_accuracy"),
        "mean_train_macro_f1": mean("train_macro_f1"),
        "mean_validation_accuracy": mean("validation_accuracy"),
        "std_validation_accuracy": std("validation_accuracy"),
        "mean_validation_macro_f1": mean("validation_macro_f1"),
        "std_validation_macro_f1": std("validation_macro_f1"),
        "macro_f1_generalization_gap": mean("train_macro_f1")
        - mean("validation_macro_f1"),
    }


def build_representation(clips: list[CachedClip], representation: str) -> np.ndarray:
    """Build one of the versioned clip representations used by model search."""
    sequences = [clip.features for clip in clips]
    if representation == "static_v1":
        return aggregate_sequences(sequences)
    if representation == "dynamic_v2":
        return aggregate_sequences_v2(sequences)
    raise ValueError(f"unknown representation: {representation}")


def run_search(
    clips: list[CachedClip],
    output_path: Path,
    n_splits: int = 5,
    representation: str = "static_v1",
) -> dict:
    """Run the predeclared candidate screen and write a deterministic JSON result."""
    if n_splits < 2:
        raise ValueError("model search requires at least two folds")
    set_seed(settings.seed)
    features = build_representation(clips, representation)
    labels = np.array([clip.label for clip in clips])
    speakers = np.array([clip.speaker_id for clip in clips])
    folds = speaker_disjoint_folds(labels, speakers, n_splits, settings.seed)

    results = []
    for name, estimator in candidate_models(settings.seed).items():
        log.info("screening %s", name)
        results.append(score_candidate(name, estimator, features, labels, folds))
    ranked = sorted(
        results,
        key=lambda result: (
            result["mean_validation_macro_f1"],
            result["mean_validation_accuracy"],
        ),
        reverse=True,
    )
    manifest_path = settings.manifest_dir / "all_clips.csv"
    document = {
        "schema_version": 1,
        "selection_scope": "frozen training partition only",
        "selection_metric": "mean speaker-grouped macro_f1",
        "representation": representation,
        "seed": settings.seed,
        "manifest_sha256": manifest_sha256(manifest_path),
        "recording_count": len(clips),
        "speaker_count": len(set(speakers.tolist())),
        "feature_width": int(features.shape[1]),
        "recordings_by_dataset": dict(sorted(Counter(clip.dataset for clip in clips).items())),
        "recordings_by_label": {
            str(key): value for key, value in sorted(Counter(labels.tolist()).items())
        },
        "fold_count": n_splits,
        "winner": ranked[0]["name"],
        "ranking": [result["name"] for result in ranked],
        "candidates": ranked,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    log.info("model search winner: %s; wrote %s", document["winner"], output_path)
    return document


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--representation",
        choices=("static_v1", "dynamic_v2"),
        default="static_v1",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=settings.experiment_dir / "model_search_v2" / "summary.json",
    )
    args = parser.parse_args()
    manifest_path = settings.manifest_dir / "all_clips.csv"
    clips = load_cached_clips(manifest_path, settings.processed_dir, {"train"})
    run_search(clips, args.output, args.folds, args.representation)


if __name__ == "__main__":
    main()

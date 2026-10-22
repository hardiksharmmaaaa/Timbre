"""Regularize and balance dynamic-v2 models using training speakers only."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ml.config import get_logger, set_seed, settings
from ml.data.cache import CachedClip, load_cached_clips, manifest_sha256
from ml.data.manifest import read_manifest
from ml.training.search_models import build_representation, speaker_disjoint_folds

log = get_logger("dynamic-model-tuning")


@dataclass(frozen=True)
class Candidate:
    name: str
    estimator: object
    weight_scheme: str = "uniform"


def tuning_candidates(seed: int) -> list[Candidate]:
    """Return the targeted regularization and weighting experiments."""

    def svc(c: float, gamma: str | float = "scale") -> Pipeline:
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", SVC(C=c, gamma=gamma, class_weight="balanced")),
            ]
        )

    def hgb(max_leaf_nodes: int) -> HistGradientBoostingClassifier:
        return HistGradientBoostingClassifier(
            learning_rate=0.06,
            max_iter=220,
            max_leaf_nodes=max_leaf_nodes,
            min_samples_leaf=60,
            l2_regularization=10.0,
            class_weight="balanced",
            early_stopping=True,
            random_state=seed,
        )

    return [
        Candidate("svc_c0.5", svc(0.5)),
        Candidate("svc_c2", svc(2.0)),
        Candidate("svc_c4", svc(4.0)),
        Candidate("svc_c1_agreement", svc(1.0), "agreement"),
        Candidate("svc_c1_dataset_agreement", svc(1.0), "dataset_agreement"),
        Candidate("hgb_leaf7", hgb(7)),
        Candidate("hgb_leaf7_agreement", hgb(7), "agreement"),
        Candidate("hgb_leaf15", hgb(15)),
        Candidate(
            "pca128_svc_c2",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "pca",
                        PCA(n_components=128, svd_solver="randomized", random_state=seed),
                    ),
                    ("classifier", SVC(C=2.0, gamma="scale", class_weight="balanced")),
                ]
            ),
        ),
        Candidate(
            "lda_shrinkage",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "classifier",
                        LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
                    ),
                ]
            ),
        ),
    ]


def build_sample_weights(
    clips: list[CachedClip], manifest_path: Path, scheme: str
) -> np.ndarray:
    """Create normalized label-reliability and/or dataset-balance weights."""
    if scheme not in {"uniform", "agreement", "dataset_agreement"}:
        raise ValueError(f"unknown weight scheme: {scheme}")
    rows = {row["clip_id"]: row for row in read_manifest(manifest_path)}
    weights = np.ones(len(clips), dtype=float)
    if scheme in {"agreement", "dataset_agreement"}:
        for index, clip in enumerate(clips):
            raw_agreement = rows[clip.clip_id]["annotation_agreement"]
            # Curated RAVDESS labels have no agreement value; retain full weight. CREMA-D
            # votes use their observed agreement with a floor so uncertain examples remain.
            weights[index] = max(0.25, float(raw_agreement)) if raw_agreement else 1.0
    if scheme == "dataset_agreement":
        counts = Counter(clip.dataset for clip in clips)
        for index, clip in enumerate(clips):
            weights[index] *= len(clips) / (len(counts) * counts[clip.dataset])
    return weights / np.mean(weights)


def _fit_with_weights(
    estimator: object,
    features: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
) -> object:
    """Fit either a pipeline or direct estimator with per-recording weights."""
    if np.allclose(weights, 1.0):
        return estimator.fit(features, labels)
    if isinstance(estimator, Pipeline):
        return estimator.fit(features, labels, classifier__sample_weight=weights)
    return estimator.fit(features, labels, sample_weight=weights)


def score_candidate(
    candidate: Candidate,
    features: np.ndarray,
    labels: np.ndarray,
    datasets: np.ndarray,
    weights: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> dict:
    """Return grouped-fold metrics, gaps, and out-of-fold dataset diagnostics."""
    fold_results = []
    oof_predictions = np.full(labels.shape, -1, dtype=int)
    for fold_number, (train_indices, validation_indices) in enumerate(folds, start=1):
        model = clone(candidate.estimator)
        _fit_with_weights(
            model,
            features[train_indices],
            labels[train_indices],
            weights[train_indices],
        )
        train_predictions = model.predict(features[train_indices])
        validation_predictions = model.predict(features[validation_indices])
        oof_predictions[validation_indices] = validation_predictions
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
        }
        fold_results.append(result)
        log.info(
            "%s fold %d/%d: accuracy=%.4f macro_f1=%.4f",
            candidate.name,
            fold_number,
            len(folds),
            result["validation_accuracy"],
            result["validation_macro_f1"],
        )

    if np.any(oof_predictions < 0):
        raise RuntimeError("out-of-fold predictions are incomplete")

    def mean(key: str) -> float:
        return float(np.mean([result[key] for result in fold_results]))

    per_dataset = {}
    for dataset in sorted(set(datasets.tolist())):
        mask = datasets == dataset
        per_dataset[dataset] = {
            "recordings": int(mask.sum()),
            "accuracy": float(accuracy_score(labels[mask], oof_predictions[mask])),
            "macro_f1": float(f1_score(labels[mask], oof_predictions[mask], average="macro")),
        }

    return {
        "name": candidate.name,
        "weight_scheme": candidate.weight_scheme,
        "folds": fold_results,
        "mean_train_accuracy": mean("train_accuracy"),
        "mean_train_macro_f1": mean("train_macro_f1"),
        "mean_validation_accuracy": mean("validation_accuracy"),
        "mean_validation_macro_f1": mean("validation_macro_f1"),
        "macro_f1_generalization_gap": mean("train_macro_f1")
        - mean("validation_macro_f1"),
        "oof_accuracy": float(accuracy_score(labels, oof_predictions)),
        "oof_macro_f1": float(f1_score(labels, oof_predictions, average="macro")),
        "oof_by_dataset": per_dataset,
    }


def select_candidates(names: list[str] | None, seed: int) -> list[Candidate]:
    """Select named candidates and reject typos instead of silently skipping them."""
    candidates = tuning_candidates(seed)
    if not names:
        return candidates
    by_name = {candidate.name: candidate for candidate in candidates}
    unknown = sorted(set(names) - set(by_name))
    if unknown:
        raise ValueError(f"unknown candidates: {', '.join(unknown)}")
    return [by_name[name] for name in names]


def run_tuning(
    clips: list[CachedClip],
    output_path: Path,
    n_splits: int = 3,
    candidate_names: list[str] | None = None,
) -> dict:
    """Run a coarse regularization screen without touching validation or test data."""
    set_seed(settings.seed)
    manifest_path = settings.manifest_dir / "all_clips.csv"
    features = build_representation(clips, "dynamic_v2")
    labels = np.array([clip.label for clip in clips])
    speakers = np.array([clip.speaker_id for clip in clips])
    datasets = np.array([clip.dataset for clip in clips])
    folds = speaker_disjoint_folds(labels, speakers, n_splits, settings.seed)
    weights_by_scheme = {
        scheme: build_sample_weights(clips, manifest_path, scheme)
        for scheme in ("uniform", "agreement", "dataset_agreement")
    }

    results = []
    selected_candidates = select_candidates(candidate_names, settings.seed)
    for candidate in selected_candidates:
        log.info("screening %s", candidate.name)
        results.append(
            score_candidate(
                candidate,
                features,
                labels,
                datasets,
                weights_by_scheme[candidate.weight_scheme],
                folds,
            )
        )
    ranked = sorted(
        results,
        key=lambda result: (
            result["mean_validation_macro_f1"],
            result["mean_validation_accuracy"],
        ),
        reverse=True,
    )
    document = {
        "schema_version": 1,
        "selection_scope": "frozen training partition only",
        "selection_metric": "mean speaker-grouped macro_f1",
        "screen_type": "coarse three-fold regularization and weighting screen",
        "seed": settings.seed,
        "manifest_sha256": manifest_sha256(manifest_path),
        "recording_count": len(clips),
        "speaker_count": len(set(speakers.tolist())),
        "representation": "dynamic_v2",
        "feature_width": int(features.shape[1]),
        "fold_count": n_splits,
        "winner": ranked[0]["name"],
        "ranking": [result["name"] for result in ranked],
        "candidates": ranked,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    log.info("tuning winner: %s; wrote %s", document["winner"], output_path)
    return document


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--candidate", action="append", dest="candidates")
    parser.add_argument(
        "--output",
        type=Path,
        default=settings.experiment_dir / "dynamic_tuning_v2" / "summary.json",
    )
    args = parser.parse_args()
    clips = load_cached_clips(
        settings.manifest_dir / "all_clips.csv", settings.processed_dir, {"train"}
    )
    run_tuning(clips, args.output, args.folds, args.candidates)


if __name__ == "__main__":
    main()

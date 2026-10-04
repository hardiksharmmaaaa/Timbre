"""Library-backed classical baselines and speaker-disjoint model selection."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from timbre.data.cache import CachedClip
from timbre.features.aggregation import aggregate_sequences

BASELINE_NAMES = ("logistic_regression", "random_forest", "knn", "adaboost", "svm")
KNN_NEIGHBOR_COUNTS = (1, 3, 5, 7, 11, 15, 21, 31)
ADABOOST_GRID = {
    "n_estimators": [50, 100],
    "learning_rate": [0.1, 0.5, 1.0],
    "estimator__max_depth": [1, 2],
}
SVM_GRID = [
    {
        "classifier__kernel": ["linear"],
        "classifier__C": [0.1, 1.0, 10.0],
        "classifier__class_weight": [None, "balanced"],
    },
    {
        "classifier__kernel": ["rbf"],
        "classifier__C": [0.1, 1.0, 10.0, 100.0],
        "classifier__gamma": ["scale", 0.001, 0.01],
        "classifier__class_weight": [None, "balanced"],
    },
]


def _training_folds(clips: list[CachedClip], seed: int, n_splits: int, name: str):
    if not clips or any(
        clip.split != "train" or clip.dataset not in {"RAVDESS", "CREMA-D"}
        for clip in clips
    ):
        raise ValueError(f"{name} selection requires development training clips only")
    clips = sorted(clips, key=lambda clip: clip.clip_id)
    features = aggregate_sequences([clip.features for clip in clips])
    labels = np.array([clip.label for clip in clips])
    groups = np.array([f"{clip.dataset}/{clip.speaker_id}" for clip in clips])
    if n_splits < 2 or np.unique(groups).size < n_splits:
        raise ValueError(f"{name} cross-validation needs at least n_splits distinct speakers")
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = list(splitter.split(features, labels, groups))
    classes = set(labels)
    for train, held_out in folds:
        if not len(held_out) or set(labels[train]) != classes:
            raise ValueError(f"each {name} fold needs held-out clips and every class in training")
    return features, labels, groups, folds


def fit_knn(
    clips: list[CachedClip],
    seed: int,
    n_splits: int = 5,
    neighbor_counts: tuple[int, ...] = KNN_NEIGHBOR_COUNTS,
) -> tuple[Pipeline, dict]:
    """Wrap sklearn grid search; select by mean CV accuracy and refit on training only."""
    features, labels, groups, folds = _training_folds(clips, seed, n_splits, "KNN")
    if not neighbor_counts or any(k < 1 for k in neighbor_counts):
        raise ValueError("neighbor counts must be positive")
    smallest_fold = min(len(train) for train, _ in folds)
    neighbors = sorted({k for k in neighbor_counts if k <= smallest_fold})
    if not neighbors:
        raise ValueError("no neighbor count fits the smallest CV training fold")
    grid = {
        "classifier__n_neighbors": neighbors,
        "classifier__weights": ["uniform", "distance"],
        "classifier__p": [1, 2],
    }
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", KNeighborsClassifier(algorithm="brute", metric="minkowski", n_jobs=1)),
    ])
    search = GridSearchCV(
        pipeline, grid, scoring="accuracy", cv=folds, refit=True,
        n_jobs=1, error_score="raise", return_train_score=False,
    )
    search.fit(features, labels)
    report = _cv_report(
        search, groups, folds, seed, "StandardScaler fitted inside each CV training fold"
    )
    report["excluded_neighbor_counts"] = sorted(set(neighbor_counts) - set(neighbors))
    return search.best_estimator_, report


def fit_adaboost(
    clips: list[CachedClip], seed: int, n_splits: int = 5,
    param_grid: dict | None = None, n_jobs: int = 1,
) -> tuple[AdaBoostClassifier, dict]:
    """Wrap sklearn SAMME AdaBoost; select by speaker-disjoint CV accuracy, then refit."""
    features, labels, groups, folds = _training_folds(clips, seed, n_splits, "AdaBoost")
    estimator = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(random_state=seed), random_state=seed,
    )
    # sklearn 1.3--1.5 defaulted to SAMME.R; newer releases implement SAMME only.
    if estimator.get_params().get("algorithm") == "SAMME.R":
        estimator.set_params(algorithm="SAMME")
    search = GridSearchCV(
        estimator, ADABOOST_GRID if param_grid is None else param_grid,
        scoring="accuracy", cv=folds, refit=True, n_jobs=n_jobs,
        pre_dispatch="n_jobs", error_score="raise", return_train_score=False,
    )
    search.fit(features, labels)
    report = _cv_report(search, groups, folds, seed, "none (decision-tree weak learners)")
    report["algorithm"] = "SAMME"
    report["n_jobs"] = n_jobs
    report["fitted_estimator_count"] = len(search.best_estimator_.estimators_)
    return search.best_estimator_, report


def fit_svm(
    clips: list[CachedClip], seed: int, n_splits: int = 5,
    param_grid: dict | list[dict] | None = None, n_jobs: int = 1,
) -> tuple[Pipeline, dict]:
    """Select sklearn SVC by speaker-disjoint CV accuracy with fold-local scaling."""
    features, labels, groups, folds = _training_folds(clips, seed, n_splits, "SVM")
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", SVC(random_state=seed)),
    ])
    search = GridSearchCV(
        pipeline, SVM_GRID if param_grid is None else param_grid,
        scoring="accuracy", cv=folds, refit=True, n_jobs=n_jobs,
        pre_dispatch="n_jobs", error_score="raise", return_train_score=False,
    )
    search.fit(features, labels)
    report = _cv_report(
        search, groups, folds, seed, "StandardScaler fitted inside each CV training fold"
    )
    report["n_jobs"] = n_jobs
    report["prediction_method"] = "predict (SVC class votes); probability estimation disabled"
    report["support_vector_count"] = int(search.best_estimator_["classifier"].n_support_.sum())
    return search.best_estimator_, report


def _cv_report(search, groups, folds, seed: int, preprocessing: str) -> dict:
    results = search.cv_results_
    n_splits = len(folds)
    return {
        "schema_version": 1,
        "selection_metric": "mean_cv_accuracy",
        "selection_split": "train",
        "refit_split": "train",
        "splitter": "StratifiedGroupKFold",
        "n_splits": n_splits,
        "seed": seed,
        "preprocessing": preprocessing,
        "tie_break": "first candidate in sklearn ParameterGrid order",
        "parameter_grid": search.param_grid,
        "best_params": search.best_params_,
        "best_mean_cv_accuracy": float(search.best_score_),
        "training_clip_count": len(groups),
        "training_speaker_count": int(np.unique(groups).size),
        "folds": [
            {
                "train_clip_count": len(train),
                "held_out_clip_count": len(held_out),
                "train_speakers": sorted(set(groups[train])),
                "held_out_speakers": sorted(set(groups[held_out])),
            }
            for train, held_out in folds
        ],
        "candidates": [
            {
                "params": params,
                "mean_cv_accuracy": float(results["mean_test_score"][index]),
                "std_cv_accuracy": float(results["std_test_score"][index]),
                "rank": int(results["rank_test_score"][index]),
                "fold_accuracies": [
                    float(results[f"split{fold}_test_score"][index])
                    for fold in range(n_splits)
                ],
            }
            for index, params in enumerate(results["params"])
        ],
    }

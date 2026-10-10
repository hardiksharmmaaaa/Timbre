"""Fit validation-only multiclass score calibration and an abstention threshold."""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from timbre.api.pipeline import PerceptionPipeline
from timbre.data.cache import load_cached_clips, manifest_sha256
from timbre.eval.metrics import expected_calibration_error, multiclass_brier_score
from timbre.eval.scoring import score_feature_sequences, softmax_probabilities
from timbre.utils.common import get_logger, set_seed
from timbre.utils.config import settings

log = get_logger("calibrate")


def fit_score_calibrator(scores: np.ndarray, labels: np.ndarray) -> Pipeline:
    """Fit a standardized score calibrator and reject unconverged artifacts."""
    calibrator = Pipeline(
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
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        calibrator.fit(scores, labels)
    return calibrator


def select_abstention_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    class_ids: list[int],
    minimum_coverage: float = 0.8,
) -> dict:
    """Choose the most accurate threshold that keeps at least the required validation coverage."""
    predicted = np.array(class_ids)[probabilities.argmax(axis=1)]
    confidence = probabilities.max(axis=1)
    candidates = np.unique(np.concatenate(([0.0], confidence)))
    choices = []
    for threshold in candidates:
        retained = confidence >= threshold
        coverage = float(np.mean(retained))
        if coverage < minimum_coverage or not retained.any():
            continue
        choices.append(
            {
                "threshold": float(threshold),
                "coverage": coverage,
                "accuracy": float(np.mean(predicted[retained] == y_true[retained])),
            }
        )
    if not choices:
        return {"threshold": 0.0, "coverage": 1.0, "accuracy": float(np.mean(predicted == y_true))}
    return max(choices, key=lambda item: (item["accuracy"], item["coverage"], item["threshold"]))


def calibrate_model(model_dir: Path) -> None:
    set_seed(settings.seed)
    manifest_path = settings.manifest_dir / "all_clips.csv"
    digest = manifest_sha256(manifest_path)
    model_metadata = json.loads((model_dir / "meta.json").read_text())
    if model_metadata.get("manifest_sha256") != digest:
        raise ValueError("trained model does not match the current canonical manifest")
    clips = load_cached_clips(manifest_path, settings.processed_dir, {"val"})
    pipeline = PerceptionPipeline.load(model_dir)
    scores = score_feature_sequences(pipeline, [clip.features for clip in clips])
    labels = np.array([clip.label for clip in clips])
    calibrator = fit_score_calibrator(scores, labels)
    probabilities = calibrator.predict_proba(scores)
    class_ids = [int(value) for value in calibrator.classes_]
    if class_ids != list(pipeline.state_ids):
        raise ValueError("validation data does not contain every trained model class")
    predicted = np.array(class_ids)[probabilities.argmax(axis=1)]
    confidence = probabilities.max(axis=1)
    correct = (predicted == labels).astype(float)
    uncalibrated = softmax_probabilities(scores)
    selection = select_abstention_threshold(labels, probabilities, class_ids)
    metadata = {
        "schema_version": 1,
        "method": "standardized_multinomial_logistic_score_calibration",
        "class_ids": class_ids,
        "validation_clip_count": len(clips),
        "manifest_sha256": digest,
        "abstention": selection,
        "validation_metrics": {
            "accuracy": float(np.mean(predicted == labels)),
            "ece": expected_calibration_error(correct, confidence),
            "brier": multiclass_brier_score(labels, probabilities, class_ids),
            "uncalibrated_brier": multiclass_brier_score(labels, uncalibrated, class_ids),
        },
    }
    joblib.dump(calibrator, model_dir / "calibrator.joblib")
    (model_dir / "calibration.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    log.info("saved calibration; threshold=%.4f", selection["threshold"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=settings.model_dir)
    args = parser.parse_args()
    calibrate_model(args.model_dir)


if __name__ == "__main__":
    main()

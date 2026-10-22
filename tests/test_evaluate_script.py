"""Exercise evaluation and abstention calculations on deterministic predictions."""

import numpy as np
import pytest

from ml.training.calibrate import fit_score_calibrator, select_abstention_threshold
from ml.training.evaluate import summarize_predictions


def test_evaluation_reports_declared_metrics_and_stable_class_order():
    class_ids = [1, 3, 4, 5, 6, 7]
    y_true = np.array([1, 3, 4, 5, 6, 7])
    probabilities = np.eye(6) * 0.9 + (np.ones((6, 6)) - np.eye(6)) * 0.02

    metrics = summarize_predictions(y_true, probabilities, class_ids)

    assert metrics["accuracy"] == 1.0
    assert metrics["balanced_accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["coverage"] == 1.0
    assert np.asarray(metrics["confusion_matrix"]).shape == (6, 6)


def test_evaluation_retains_model_classes_absent_from_test_labels():
    class_ids = [1, 3, 4, 5, 6, 7]
    probabilities = np.array([[0.1, 0.1, 0.1, 0.1, 0.1, 0.5]])

    metrics = summarize_predictions(np.array([1]), probabilities, class_ids)

    assert metrics["confusion_matrix"][0][5] == 1
    assert metrics["accuracy"] == 0.0


def test_evaluation_applies_validation_selected_abstention_threshold():
    class_ids = [1, 3]
    labels = np.array([1, 1, 3, 3, 3])
    probabilities = np.array(
        [
            [0.9, 0.1],
            [0.8, 0.2],
            [0.45, 0.55],
            [0.2, 0.8],
            [0.6, 0.4],
        ]
    )

    selection = select_abstention_threshold(labels, probabilities, class_ids, minimum_coverage=0.8)
    metrics = summarize_predictions(
        labels, probabilities, class_ids, selection["threshold"]
    )

    assert selection["coverage"] >= 0.8
    assert metrics["coverage"] == pytest.approx(selection["coverage"])
    assert metrics["covered_accuracy"] == pytest.approx(selection["accuracy"])


def test_score_calibrator_standardizes_differently_scaled_hmm_scores():
    labels = np.tile(np.array([1, 3, 4]), 20)
    scores = np.column_stack(
        [
            (labels == 1).astype(float) * 1e-3,
            (labels == 3).astype(float) * 1e3,
            (labels == 4).astype(float) * 1e6,
        ]
    )

    calibrator = fit_score_calibrator(scores, labels)
    probabilities = calibrator.predict_proba(scores)

    assert list(calibrator.named_steps) == ["scaler", "classifier"]
    assert calibrator.classes_.tolist() == [1, 3, 4]
    assert np.isfinite(probabilities).all()
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_evaluation_uses_explicit_class_predictions_and_their_confidence():
    metrics = summarize_predictions(
        np.array([1, 7]), np.array([[0.2, 0.8], [0.3, 0.7]]), [1, 7], 0.5,
        predicted=np.array([1, 7]),
    )
    assert metrics["accuracy"] == 1.0
    assert metrics["confusion_matrix"] == [[1, 0], [0, 1]]
    assert metrics["coverage"] == 0.5
    assert metrics["ece"] == pytest.approx(0.55)


def test_evaluation_without_probabilities_does_not_invent_confidence():
    metrics = summarize_predictions(
        np.array([1, 7]), None, [1, 7], predicted=np.array([1, 1]),
    )
    assert metrics["accuracy"] == metrics["covered_accuracy"] == 0.5
    assert metrics["confusion_matrix"] == [[1, 0], [1, 0]]
    assert metrics["ece"] is metrics["brier"] is metrics["abstention_threshold"] is None
    assert metrics["coverage"] == 1.0
    with pytest.raises(ValueError, match="predictions or probabilities"):
        summarize_predictions(np.array([1]), None, [1, 7])
    with pytest.raises(ValueError, match="confidence-based abstention"):
        summarize_predictions(np.array([1]), None, [1, 7], 0.5, predicted=np.array([1]))

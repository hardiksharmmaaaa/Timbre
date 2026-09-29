"""Tests for eval metrics (calibration / accuracy) on well-understood synthetic data."""

import numpy as np

from timbre.eval.metrics import (
    accuracy,
    balanced_accuracy,
    confusion_matrix,
    expected_calibration_error,
    multiclass_brier_score,
)


def test_balanced_accuracy_includes_classes_with_zero_recall():
    assert balanced_accuracy(np.array([0, 1]), np.array([0, 0]), 2) == 0.5


def test_balanced_accuracy_is_zero_when_every_prediction_is_wrong():
    assert balanced_accuracy(np.array([0, 1]), np.array([1, 0]), 2) == 0.0


def test_balanced_accuracy_ignores_classes_absent_from_ground_truth():
    assert balanced_accuracy(np.array([1, 1]), np.array([1, 0]), 3) == 0.5


def test_ece_includes_zero_confidence_predictions():
    assert expected_calibration_error(np.array([1]), np.array([0.0])) == 1.0


def test_accuracy_round_trip():
    y = np.array([0, 1, 2, 3])
    assert accuracy(y, np.array([0, 1, 2, 3])) == 1.0
    assert accuracy(y, np.array([1, 1, 2, 3])) == 0.75


def test_confusion_matrix_shape():
    y = np.array([0, 1, 1, 2])
    cm = confusion_matrix(y, np.array([0, 0, 1, 2]), n_classes=4)
    assert cm.shape == (4, 4)
    assert cm[1, 0] == 1  # one true=1 predicted=0


def test_perfectly_calibrated_model_has_low_ece():
    rng = np.random.default_rng(0)
    n = 2000
    # Predicted-class confidence == true accuracy within each bin ===> near-zero ECE.
    probs = rng.uniform(0.0, 1.0, n)
    correct = (rng.uniform(0.0, 1.0, n) < probs).astype(float)
    ece = expected_calibration_error(correct, probs, n_bins=10)
    assert ece < 0.05


def test_multiclass_brier_score_is_zero_for_perfect_probabilities():
    labels = np.array([1, 3])
    probabilities = np.array([[1.0, 0.0], [0.0, 1.0]])

    assert multiclass_brier_score(labels, probabilities, [1, 3]) == 0.0

"""Evaluation harness — hand-written scoring for the research report. Built by us.

Covers: accuracy, confusion matrix, calibration curve/ECE, and supports ablations
(±Bayesian, PCA vs LLE, HMM vs baselines). All deterministic and seeded.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from timbre.utils.config import settings


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float((y_true == y_pred).mean())


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    recall = []
    for c in range(n_classes):
        mask = y_true == c
        if mask.sum() > 0:
            recall.append((y_pred[mask] == c).mean())
    return float(np.mean(recall))


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def expected_calibration_error(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> float:
    """Does "72% confidence" correspond to ≈72% correctness? Computes bin-wise ECE.

    `probs` is the predicted probability of the predicted class for each sample.
    """
    y_true = np.asarray(y_true)
    probs = np.asarray(probs)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = ((probs >= lo) if i == 0 else (probs > lo)) & (probs <= hi)
        if mask.sum() == 0:
            continue
        acc = (y_true[mask] == 1).mean()  # binary-correct within bin
        conf = probs[mask].mean()
        ece += (mask.sum() / probs.size) * abs(acc - conf)
    return float(ece)


def run_and_record(run_name: str, metrics: dict) -> Path:
    """Write a metrics blob + manifest into experiments/<run_name>/, timestamped."""
    run_dir = settings.experiment_dir / f"{run_name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = run_dir / f"metrics_{stamp}.json"
    path.write_text(json.dumps({"run": run_name, **metrics}, indent=2, default=str))
    return path

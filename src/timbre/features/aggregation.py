"""Clip-level summary features used by classical non-temporal baselines."""

from __future__ import annotations

import numpy as np


def aggregate_sequence(features: np.ndarray) -> np.ndarray:
    """Summarize one frame sequence with stable per-column distribution statistics."""
    features = np.asarray(features)
    if features.ndim != 2 or features.shape[0] == 0 or not np.isfinite(features).all():
        raise ValueError("features must be a finite, non-empty 2D matrix")
    summaries = (
        np.mean(features, axis=0),
        np.std(features, axis=0),
        np.median(features, axis=0),
        np.min(features, axis=0),
        np.max(features, axis=0),
        np.quantile(features, 0.25, axis=0),
        np.quantile(features, 0.75, axis=0),
    )
    return np.concatenate(summaries)


def aggregate_sequences(sequences: list[np.ndarray]) -> np.ndarray:
    if not sequences:
        raise ValueError("at least one feature sequence is required")
    return np.vstack([aggregate_sequence(sequence) for sequence in sequences])


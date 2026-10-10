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


def _temporal_difference(features: np.ndarray) -> np.ndarray:
    """Return a length-preserving first difference for even very short clips."""
    if features.shape[0] == 1:
        return np.zeros_like(features)
    return np.diff(features, axis=0, prepend=features[[0]])


def _linear_slopes(features: np.ndarray) -> np.ndarray:
    """Return one least-squares temporal slope per feature column."""
    if features.shape[0] == 1:
        return np.zeros(features.shape[1], dtype=float)
    time = np.linspace(-1.0, 1.0, features.shape[0])
    denominator = float(np.dot(time, time))
    centered = features - np.mean(features, axis=0, keepdims=True)
    return np.dot(time, centered) / denominator


def aggregate_sequence_v2(features: np.ndarray, segment_count: int = 4) -> np.ndarray:
    """Summarize level, movement, acceleration, and coarse temporal shape.

    The original aggregation describes only the distribution of each frame feature.
    This representation keeps that baseline and adds first/second differences, slopes,
    and within-utterance-normalized segment statistics. The normalized contour is less
    sensitive to a speaker's absolute pitch and vocal-tract characteristics.
    """
    features = np.asarray(features, dtype=float)
    if features.ndim != 2 or features.shape[0] == 0 or not np.isfinite(features).all():
        raise ValueError("features must be a finite, non-empty 2D matrix")
    if segment_count < 2:
        raise ValueError("segment_count must be at least two")

    first_delta = _temporal_difference(features)
    second_delta = _temporal_difference(first_delta)
    median = np.median(features, axis=0, keepdims=True)
    scale = np.quantile(features, 0.75, axis=0, keepdims=True) - np.quantile(
        features, 0.25, axis=0, keepdims=True
    )
    scale = np.where(scale > 1e-8, scale, 1.0)
    normalized = (features - median) / scale

    # array_split also handles clips shorter than segment_count. Empty segments use the
    # nearest available frame so the output contract remains fixed-width and finite.
    segment_indices = np.array_split(np.arange(features.shape[0]), segment_count)
    segment_summaries = []
    for index, indices in enumerate(segment_indices):
        if indices.size == 0:
            nearest = min(index, features.shape[0] - 1)
            segment = normalized[[nearest]]
        else:
            segment = normalized[indices]
        segment_summaries.extend((np.mean(segment, axis=0), np.std(segment, axis=0)))

    return np.concatenate(
        (
            aggregate_sequence(features),
            aggregate_sequence(first_delta),
            aggregate_sequence(second_delta),
            _linear_slopes(features),
            *segment_summaries,
            np.array([np.log1p(features.shape[0])], dtype=float),
        )
    )


def aggregate_sequences_v2(sequences: list[np.ndarray], segment_count: int = 4) -> np.ndarray:
    """Apply :func:`aggregate_sequence_v2` to a non-empty sequence collection."""
    if not sequences:
        raise ValueError("at least one feature sequence is required")
    return np.vstack(
        [aggregate_sequence_v2(sequence, segment_count=segment_count) for sequence in sequences]
    )

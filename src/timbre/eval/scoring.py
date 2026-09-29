"""Shared HMM class-score and probability helpers for calibration and evaluation."""

from __future__ import annotations

import numpy as np

from timbre.hmm.decoder import score_viterbi_sequences


def score_feature_sequence(pipeline, features: np.ndarray) -> np.ndarray:
    reduced = pipeline.reducer.transform(features)
    return np.array(
        [
            score_viterbi_sequences(pipeline.state_models[state_id], [reduced])[0]
            for state_id in pipeline.state_ids
        ],
        dtype=float,
    )


def score_feature_sequences(pipeline, sequences: list[np.ndarray]) -> np.ndarray:
    if not sequences:
        raise ValueError("at least one feature sequence is required")
    return np.vstack([score_feature_sequence(pipeline, sequence) for sequence in sequences])


def softmax_probabilities(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 2 or scores.shape[1] == 0 or not np.isfinite(scores).all():
        raise ValueError("scores must be a finite two-dimensional matrix")
    shifted = scores - scores.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


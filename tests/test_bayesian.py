"""Tests for the Bayesian uncertainty layer (posterior properties)."""

import numpy as np

from timbre.bayesian.posterior import posterior_from_log_likelihoods
from timbre.core.types import EmotionState

STATES = [EmotionState.CALM, EmotionState.HAPPY, EmotionState.SAD]


def _lls():
    return {s: v for s, v in zip(STATES, [np.log(0.7), np.log(0.2), np.log(0.1)])}


def test_posterior_probabilities_sum_to_one():
    p = posterior_from_log_likelihoods(_lls(), n_frames=100, method="hybrid")
    assert abs(sum(p.probs.values()) - 1.0) < 1e-6


def test_most_likely_is_dominant_state():
    p = posterior_from_log_likelihoods(_lls(), n_frames=100, method="hybrid")
    assert p.most_likely() == EmotionState.CALM


def test_intervals_shrink_with_more_frames():
    p_few = posterior_from_log_likelihoods(_lls(), n_frames=5, method="hybrid")
    p_many = posterior_from_log_likelihoods(_lls(), n_frames=500, method="hybrid")
    assert p_many.intervals[EmotionState.CALM] < p_few.intervals[EmotionState.CALM]


def test_all_methods_return_valid_posterior():
    for method in ("hybrid", "vi", "mcmc"):
        p = posterior_from_log_likelihoods(_lls(), n_frames=50, method=method)
        assert abs(sum(p.probs.values()) - 1.0) < 1e-2
        assert p.most_likely() == EmotionState.CALM

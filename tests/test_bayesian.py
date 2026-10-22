"""Tests for the Bayesian uncertainty layer (posterior properties)."""

import numpy as np
import pytest

from ml.config import EmotionState
from ml.core.bayesian.posterior import posterior_from_log_likelihoods

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


@pytest.mark.parametrize("method", ["hybrid", "vi", "mcmc"])
@pytest.mark.parametrize("scores", [{}, {EmotionState.CALM: -np.inf},
                                   {EmotionState.CALM: np.nan}, {EmotionState.CALM: np.inf}])
def test_invalid_evidence_is_rejected_before_building_posterior(method, scores):
    with pytest.raises(ValueError, match="likelihood"):
        posterior_from_log_likelihoods(scores, n_frames=100, method=method)


@pytest.mark.parametrize("method", ["hybrid", "vi", "mcmc"])
def test_negative_infinity_is_valid_for_an_impossible_state(method):
    posterior = posterior_from_log_likelihoods(
        {EmotionState.CALM: 0.0, EmotionState.HAPPY: -np.inf}, n_frames=100, method=method
    )
    assert all(np.isfinite(value) for value in posterior.probs.values())
    assert posterior.most_likely() == EmotionState.CALM


def test_mcmc_surrogate_intervals_shrink_with_more_frames():
    few = posterior_from_log_likelihoods(_lls(), n_frames=5, method="mcmc")
    many = posterior_from_log_likelihoods(_lls(), n_frames=500, method="mcmc")
    for state in STATES:
        assert many.intervals[state] < few.intervals[state]


def test_mcmc_intervals_describe_probability_uncertainty():
    from scipy.stats import beta

    # Two uniform pseudo-observations plus a uniform prior yield Dirichlet(2, 2).
    # Each probability therefore has a Beta(2, 2) marginal distribution.
    posterior = posterior_from_log_likelihoods(
        {EmotionState.CALM: 0.0, EmotionState.HAPPY: 0.0}, n_frames=2, method="mcmc"
    )
    expected_half_width = beta.ppf(0.975, 2, 2) - 0.5
    for state in posterior.probs:
        assert posterior.probs[state] == pytest.approx(0.5, abs=0.02)
        assert posterior.intervals[state] == pytest.approx(expected_half_width, abs=0.02)


def test_mcmc_surrogate_is_deterministic_and_finite():
    first = posterior_from_log_likelihoods(_lls(), n_frames=50, method="mcmc")
    second = posterior_from_log_likelihoods(_lls(), n_frames=50, method="mcmc")
    assert first == second
    assert sum(first.probs.values()) == pytest.approx(1.0)
    assert all(np.isfinite(value) and 0 <= value <= 1 for value in first.probs.values())
    assert all(np.isfinite(value) and 0 <= value <= 1 for value in first.intervals.values())


@pytest.mark.parametrize("method", ["hybrid", "vi", "mcmc"])
@pytest.mark.parametrize("n_frames", [0, -1, 1.5, np.nan, np.inf, "5", True])
def test_invalid_frame_count_is_rejected(method, n_frames):
    with pytest.raises(ValueError, match="n_frames must be a positive integer"):
        posterior_from_log_likelihoods(_lls(), n_frames=n_frames, method=method)

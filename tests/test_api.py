"""Tests for API payload serialization without external service calls."""

import pytest

from timbre.api.main import posterior_from_payload
from timbre.core.types import EmotionState


def test_posterior_payload_uses_emotion_names():
    posterior = posterior_from_payload(
        {"calm": 0.7, "happy": 0.3},
        {"calm": 0.05, "happy": 0.08},
    )
    assert posterior.probs[EmotionState.CALM] == 0.7
    assert posterior.most_likely() == EmotionState.CALM


def test_posterior_payload_rejects_unknown_emotion():
    with pytest.raises(ValueError, match="unknown emotion"):
        posterior_from_payload({"stressed": 1.0}, {})


def test_posterior_payload_rejects_invalid_probability_mass():
    with pytest.raises(ValueError, match="sum to one"):
        posterior_from_payload({"calm": 0.4, "happy": 0.4}, {})

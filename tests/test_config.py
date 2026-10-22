"""Reject invalid runtime dimensions and timing before starting the app."""

import pytest
from pydantic import ValidationError

from ml.config import Settings


@pytest.mark.parametrize(
    "env_name",
    [
        "FRAME_MS",
        "STRIDE_MS",
        "N_MFCC",
        "SAMPLE_RATE",
        "F0_FRAME_MS",
        "F0_MIN_HZ",
        "F0_MAX_HZ",
        "N_COMPONENTS",
        "HMM_N_STATES",
        "HMM_N_STATES_EMISSION_GMM",
        "HMM_N_ITER",
        "INFERENCE_WINDOW_MS",
        "INFERENCE_LATENCY_BUDGET_MS",
    ],
)
def test_settings_reject_zero_dimensions_and_timing(monkeypatch, env_name):
    monkeypatch.setenv(env_name, "0")

    with pytest.raises(ValidationError, match="greater than 0"):
        Settings(_env_file=None)


def test_settings_reject_invalid_pitch_bounds(monkeypatch):
    monkeypatch.setenv("F0_MIN_HZ", "500")
    monkeypatch.setenv("F0_MAX_HZ", "50")

    with pytest.raises(ValidationError, match="F0_MIN_HZ"):
        Settings(_env_file=None)

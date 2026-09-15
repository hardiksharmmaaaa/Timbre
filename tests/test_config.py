"""Reject invalid runtime dimensions and timing before starting the app."""

import pytest
from pydantic import ValidationError

from timbre.utils.config import Settings


@pytest.mark.parametrize(
    "env_name",
    [
        "FRAME_MS",
        "STRIDE_MS",
        "N_MFCC",
        "SAMPLE_RATE",
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

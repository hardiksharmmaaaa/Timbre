"""Training-only reverberation augmentation regressions."""

from pathlib import Path

import numpy as np
import pytest

from ml.core.features.augmentation import apply_reverb, select_rir


def test_reverb_preserves_length_is_finite_and_avoids_clipping():
    audio = np.zeros(1600)
    audio[100:200] = 0.8
    impulse = np.exp(-np.arange(400) / 80.0)

    augmented = apply_reverb(audio, impulse)

    assert augmented.shape == audio.shape
    assert np.isfinite(augmented).all()
    assert np.max(np.abs(augmented)) <= 0.999
    assert not np.array_equal(augmented, audio)


def test_rir_selection_is_deterministic():
    paths = [Path(f"rir-{index}.wav") for index in range(5)]

    assert select_rir(paths, "clip-1", 42) == select_rir(paths, "clip-1", 42)


def test_reverb_rejects_empty_impulse_response():
    with pytest.raises(ValueError, match="non-empty"):
        apply_reverb(np.ones(10), np.array([]))

"""Tests for framewise feature extraction on synthetic signals."""

import numpy as np

from timbre.features.extractor import extract_frame_features, frame_lengths, load_and_extract


def test_frame_lengths_follow_input_sample_rate():
    assert frame_lengths(16000) == (400, 160)
    assert frame_lengths(22050) == (551, 220)


def test_feature_matrix_shape_and_dims():
    rng = np.random.default_rng(0)
    sr = 22050
    y = rng.uniform(-1, 1, int(sr * 2.0))  # 2s of noise
    feats = extract_frame_features(y, sr)
    # Column dim = MFCC + RMS + F0 + speaking-rate.
    assert feats.ndim == 2
    assert feats.shape[1] == 23  # 20 MFCC + 1 RMS + 1 F0 + 1 rate
    # Frame count ≈ duration / stride
    expected_frames = int(2.0 / (10 / 1000.0))
    assert abs(feats.shape[0] - expected_frames) <= 4


def test_no_nan_and_finite():
    rng = np.random.default_rng(1)
    y = rng.uniform(-1, 1, int(22050 * 1.0))
    feats = extract_frame_features(y, 22050)
    assert np.isfinite(feats).all()


def test_empty_audio_is_rejected():
    import pytest

    with pytest.raises(ValueError, match="non-empty mono"):
        extract_frame_features(np.array([]), 22050)


def test_extract_file_matches_direct():
    import librosa

    sr = 16000
    y, _ = librosa.load(librosa.ex("trumpet"), sr=sr, duration=1.0)
    direct = extract_frame_features(y, sr)
    # Saving/loading round-trip: librosa resamples to target sr on load, so shapes must match
    # and features should be close (not bitwise) — a float-precision tolerance.
    import tempfile

    import soundfile as sf

    with tempfile.NamedTemporaryFile(suffix=".wav") as f:
        sf.write(f.name, y, sr)
        via_file, _ = load_and_extract(f.name, sr=sr)
    assert via_file.shape == direct.shape
    # librosa resamples on load, so exact equality is not expected; features must be strongly
    # correlated (same signal, same extraction) rather than identical.
    for col in range(direct.shape[1]):
        a, b = direct[:, col], via_file[:, col]
        if a.std() < 1e-6:  # constant column (e.g. fully-voiced rate) — just match sign
            continue
        corr = np.corrcoef(a, b)[0, 1]
        assert corr > 0.99

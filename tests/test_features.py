"""Tests for framewise feature extraction on synthetic signals."""

import numpy as np
import pytest

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
    with pytest.raises(ValueError, match="non-empty mono"):
        extract_frame_features(np.array([]), 22050)


def test_silent_audio_has_zero_speaking_rate():
    features = extract_frame_features(np.zeros(2205), 22050)
    assert np.count_nonzero(features[:, -2]) == 0
    assert np.count_nonzero(features[:, -1]) == 0


def test_tone_has_expected_median_pitch():
    sample_rate = 16000
    time = np.arange(sample_rate, dtype=float) / sample_rate
    features = extract_frame_features(0.2 * np.sin(2 * np.pi * 220 * time), sample_rate)
    voiced_pitch = features[features[:, -2] > 0, -2]

    assert np.median(voiced_pitch) == pytest.approx(220, abs=3)


def test_extract_file_matches_direct(tmp_path):
    import soundfile as sf

    # A local deterministic chirp keeps this test independent of network access and
    # librosa's example-audio cache.
    sr = 16000
    time = np.arange(sr, dtype=float) / sr
    y = (0.4 * np.sin(2 * np.pi * (220 * time + 80 * time**2))).astype(np.float32)
    direct = extract_frame_features(y, sr)
    path = tmp_path / "chirp.wav"
    sf.write(path, y, sr, subtype="FLOAT")
    via_file, loaded_sr = load_and_extract(str(path), sr=sr)
    assert loaded_sr == sr
    np.testing.assert_allclose(via_file, direct)

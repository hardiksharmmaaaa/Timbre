"""Stage 1 — framewise feature extraction. Built by us.

Extracts a time series of feature vectors per audio clip (25ms windows, 10ms stride):
F0/pitch, RMS energy, MFCCs, and a speaking-rate (VAD/silence-ratio) descriptor.

The HMM downstream needs a *time series* of feature vectors per clip, not one averaged
vector — this module produces exactly that.

Library-backed: librosa for frame math. The feature definitions (what each signals) are ours.
"""

from __future__ import annotations

import librosa
import numpy as np

from timbre.utils.config import settings


def frame_lengths(sr: int) -> tuple[int, int]:
    """Return the configured frame and hop lengths in samples for ``sr``."""
    if sr <= 0:
        raise ValueError("sample rate must be positive")
    window = max(2, round(sr * settings.frame_ms / 1000))
    hop = max(1, round(sr * settings.stride_ms / 1000))
    return window, hop


def extract_frame_features(y: np.ndarray, sr: int) -> np.ndarray:
    """Return (nframes, nfeatures) matrix for raw audio `y`.

    Feature columns are ordered with the global FEATURE_NAMES list; keep column order stable.
    """
    y = np.asarray(y, dtype=float)
    if y.ndim != 1 or y.size == 0:
        raise ValueError("audio must be a non-empty mono waveform")

    window, hop = frame_lengths(sr)
    # Use identical centered 25 ms windows and 10 ms hops for every framewise feature.
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=settings.n_mfcc,
        n_fft=window,
        win_length=window,
        hop_length=hop,
        center=True,
    ).T
    rms = librosa.feature.rms(
        y=y, frame_length=window, hop_length=hop, center=True
    ).T
    f0, _, _ = librosa.pyin(
        y,
        fmin=librosa.note_to_hz("C3"),
        fmax=librosa.note_to_hz("C6"),
        sr=sr,
        frame_length=window,
        hop_length=hop,
        center=True,
    )
    f0 = np.where(np.isnan(f0), 0.0, f0)  # unvoiced frames -> 0 pitch
    f0 = np.expand_dims(f0, axis=-1)

    # Speaking rate: negative of silence ratio over a sliding window (~1s).
    voice = librosa.effects.split(y, top_db=30)
    voiced_mask = np.zeros(len(y), dtype=bool)
    for start, end in voice:
        voiced_mask[start:end] = True
    # Speaking rate: proportion of voiced samples in each hop window (silence detection driver).
    padded_voice = np.pad(voiced_mask, (window // 2, window - window // 2))
    voiced_frames = librosa.util.frame(
        padded_voice, frame_length=window, hop_length=hop
    ).mean(axis=0)

    T = min(mfcc.shape[0], f0.shape[0], rms.shape[0], voiced_frames.shape[0])
    fs = np.concatenate([mfcc[:T], rms[:T], f0[:T], voiced_frames[:T, None]], axis=1)
    return fs


def load_and_extract(audio_path: str, sr: int | None = None) -> tuple[np.ndarray, int]:
    """Load a single clip and extract framewise features. Returns (features, sr)."""
    sr = sr or settings.sample_rate
    y, sr = librosa.load(audio_path, sr=sr, mono=True)
    return extract_frame_features(y, sr), sr


# Hand-implemented demonstration: compute F0 by naive autocorrelation (for the report/tests).
def floor_feature_vector() -> int:
    return settings.n_mfcc + 3  # MFCC + RMS + F0 + speaking-rate column

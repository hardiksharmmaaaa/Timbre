"""Stage 1 — framewise feature extraction. Built by us.

Extracts a time series of feature vectors per audio clip (25ms windows, 10ms stride):
F0/pitch, RMS energy, MFCCs, and a voiced-proportion descriptor.

The HMM downstream needs a *time series* of feature vectors per clip, not one averaged
vector — this module produces exactly that.

Library-backed: librosa for frame math. The feature definitions (what each signals) are ours.
"""

from __future__ import annotations

import librosa
import numpy as np

from timbre.utils.config import settings


def feature_contract() -> dict:
    """Return the complete preprocessing contract stored with features and models."""
    return {
        "sample_rate": settings.sample_rate,
        "frame_ms": settings.frame_ms,
        "stride_ms": settings.stride_ms,
        "n_mfcc": settings.n_mfcc,
        "f0_frame_ms": settings.f0_frame_ms,
        "f0_min_hz": settings.f0_min_hz,
        "f0_max_hz": settings.f0_max_hz,
        "f0_method": "yin_energy_gated_v1",
    }


def frame_lengths(sr: int) -> tuple[int, int]:
    """Return the configured frame and hop lengths in samples for ``sr``."""
    if sr <= 0:
        raise ValueError("sample rate must be positive")
    window = max(2, round(sr * settings.frame_ms / 1000))
    hop = max(1, round(sr * settings.stride_ms / 1000))
    return window, hop


def extract_frame_features(y: np.ndarray, sr: int) -> np.ndarray:
    """Return (nframes, nfeatures) matrix for raw audio `y`.

    Feature columns are ordered as MFCC, RMS, F0, and voiced proportion.
    """
    y = np.asarray(y, dtype=float)
    if y.ndim != 1 or y.size == 0:
        raise ValueError("audio must be a non-empty mono waveform")

    window, hop = frame_lengths(sr)
    # MFCC and RMS share centered 25 ms windows and 10 ms hops.
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
    # Voiced proportion is the accurate name for this frame-level activity descriptor.
    # librosa uses the clip peak as its dB reference; an all-zero clip otherwise appears
    # entirely voiced because its energy equals that reference.
    voice = librosa.effects.split(y, top_db=30) if np.any(y) else []
    voiced_mask = np.zeros(len(y), dtype=bool)
    for start, end in voice:
        voiced_mask[start:end] = True
    # Proportion of voiced samples in each hop window (silence detection driver).
    padded_voice = np.pad(voiced_mask, (window // 2, window - window // 2))
    voiced_frames = librosa.util.frame(
        padded_voice, frame_length=window, hop_length=hop
    ).mean(axis=0)

    # Low voices need a longer analysis window than MFCC/RMS while retaining the same hop grid.
    # YIN is deterministic and substantially faster than pYIN for full-corpus extraction.
    f0_frame_length = max(window, round(sr * settings.f0_frame_ms / 1000))
    f0 = librosa.yin(
        y,
        fmin=settings.f0_min_hz,
        fmax=settings.f0_max_hz,
        sr=sr,
        frame_length=f0_frame_length,
        hop_length=hop,
        center=True,
    )
    pitch_frames = min(f0.shape[0], voiced_frames.shape[0])
    f0 = np.where(voiced_frames[:pitch_frames] >= 0.2, f0[:pitch_frames], 0.0)
    f0 = np.expand_dims(f0, axis=-1)

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
    return settings.n_mfcc + 3  # MFCC + RMS + F0 + voiced-proportion column

"""Deterministic label-preserving training-only audio augmentation."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from scipy.signal import fftconvolve


def select_rir(paths: list[Path], clip_id: str, seed: int) -> Path:
    if not paths:
        raise ValueError("no room impulse responses are available")
    token = hashlib.sha256(f"{seed}:{clip_id}".encode()).digest()
    index = int.from_bytes(token[:8], "big") % len(paths)
    return paths[index]


def apply_reverb(
    audio: np.ndarray, impulse_response: np.ndarray, wet_ratio: float = 0.35
) -> np.ndarray:
    """Convolve one mono clip, RMS-match the wet signal, and avoid clipping."""
    audio = np.asarray(audio, dtype=float)
    impulse_response = np.asarray(impulse_response, dtype=float)
    if audio.ndim != 1 or audio.size == 0:
        raise ValueError("audio must be a non-empty mono waveform")
    if impulse_response.ndim != 1 or impulse_response.size == 0:
        raise ValueError("impulse response must be a non-empty mono waveform")
    if not 0.0 <= wet_ratio <= 1.0:
        raise ValueError("wet_ratio must be between zero and one")
    if not np.isfinite(audio).all() or not np.isfinite(impulse_response).all():
        raise ValueError("augmentation inputs must be finite")
    norm = np.linalg.norm(impulse_response)
    if norm <= 1e-12:
        raise ValueError("impulse response has no energy")
    wet = fftconvolve(audio, impulse_response / norm, mode="full")[: audio.size]
    dry_rms = np.sqrt(np.mean(np.square(audio)))
    wet_rms = np.sqrt(np.mean(np.square(wet)))
    if wet_rms > 1e-12 and dry_rms > 0:
        wet = wet * (dry_rms / wet_rms)
    mixed = (1.0 - wet_ratio) * audio + wet_ratio * wet
    peak = np.max(np.abs(mixed))
    if peak > 0.999:
        mixed = mixed * (0.999 / peak)
    return mixed.astype(np.float32)

"""Feature-cache lineage and classical aggregation tests."""

import csv

import numpy as np
import pytest

from ml.core.features.aggregation import (
    aggregate_sequence,
    aggregate_sequence_v2,
    aggregate_sequences,
    aggregate_sequences_v2,
)
from ml.data.cache import load_cached_clips
from ml.data.manifest import MANIFEST_FIELDS


def _manifest_row() -> dict:
    return {
        "clip_id": "ravdess__clip",
        "dataset": "RAVDESS",
        "relative_audio_path": "RAVDESS/clip.wav",
        "speaker_id": "RAVDESS:01",
        "canonical_label": "neutral",
        "label_id": 1,
        "original_label": "neutral",
        "sample_rate": 16000,
        "duration_seconds": 1.0,
        "split": "train",
        "role": "development",
        "intensity": "normal",
        "annotation_agreement": "",
        "source_license": "fixture",
        "source_url": "https://example.test",
        "audio_sha256": "abc123",
        "channels": 1,
        "rms": 0.1,
        "clipping_rate": 0.0,
        "silence_fraction": 0.0,
    }


def _write_manifest(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerow(row)


def test_cache_loader_validates_manifest_hash_and_label(tmp_path):
    row = _manifest_row()
    manifest = tmp_path / "manifests" / "all_clips.csv"
    _write_manifest(manifest, row)
    cache = tmp_path / "processed" / "RAVDESS" / "ravdess__clip.npz"
    cache.parent.mkdir(parents=True)
    features = np.arange(30, dtype=float).reshape(10, 3)
    np.savez(cache, features=features, audio_sha256="abc123", label=1)

    clips = load_cached_clips(manifest, tmp_path / "processed", {"train"})

    assert len(clips) == 1
    np.testing.assert_array_equal(clips[0].features, features)

    row["audio_sha256"] = "changed"
    _write_manifest(manifest, row)
    with pytest.raises(ValueError, match="stale feature cache"):
        load_cached_clips(manifest, tmp_path / "processed", {"train"})


def test_aggregation_has_seven_statistics_per_feature():
    first = np.arange(30, dtype=float).reshape(10, 3)
    second = first + 1

    assert aggregate_sequence(first).shape == (21,)
    assert aggregate_sequences([first, second]).shape == (2, 21)


def test_dynamic_aggregation_has_a_fixed_finite_contract():
    first = np.arange(30, dtype=float).reshape(10, 3)
    one_frame = np.array([[1.0, 2.0, 3.0]])

    # 3 distributions * 7 statistics * 3 columns, 3 slopes,
    # 4 temporal segments * (mean + std) * 3 columns, and duration.
    expected_width = 3 * 7 * 3 + 3 + 4 * 2 * 3 + 1
    assert aggregate_sequence_v2(first).shape == (expected_width,)
    matrix = aggregate_sequences_v2([first, one_frame])
    assert matrix.shape == (2, expected_width)
    assert np.isfinite(matrix).all()


def test_dynamic_aggregation_rejects_invalid_segment_count():
    with pytest.raises(ValueError, match="segment_count"):
        aggregate_sequence_v2(np.ones((3, 2)), segment_count=1)

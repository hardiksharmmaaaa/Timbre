"""Canonical dataset parsing, screening, and split-contract tests."""

from __future__ import annotations

import csv
import json

import numpy as np
import pytest
import soundfile as sf

from timbre.core.types import EmotionState
from timbre.data.manifest import (
    ManifestRecord,
    parse_crema_clip,
    parse_ravdess_clip,
    parse_tess_clip,
    prepare_manifest,
    read_manifest,
    reject_exact_duplicates,
)


def _write_tone(path, frequency: float, sample_rate: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    time = np.arange(sample_rate // 4, dtype=float) / sample_rate
    audio = (0.2 * np.sin(2 * np.pi * frequency * time)).astype(np.float32)
    sf.write(path, audio, sample_rate)


def test_dataset_parsers_map_shared_labels_and_speakers(tmp_path):
    ravdess = parse_ravdess_clip(tmp_path / "03-01-06-02-01-01-17.wav")
    crema = parse_crema_clip(tmp_path / "1022_ITS_ANG_HI.wav", agreement=0.82)
    tess = parse_tess_clip(tmp_path / "YAF_wire_fear.wav")

    assert ravdess is not None
    assert ravdess.label == EmotionState.FEARFUL
    assert ravdess.speaker_id == "RAVDESS:17"
    assert crema.label == EmotionState.ANGRY
    assert crema.speaker_id == "CREMA-D:1022"
    assert crema.annotation_agreement == 0.82
    assert tess is not None
    assert tess.label == EmotionState.FEARFUL
    assert tess.role == "external_test"


def test_dataset_parsers_exclude_non_shared_labels(tmp_path):
    assert parse_ravdess_clip(tmp_path / "03-01-02-01-01-01-01.wav") is None
    assert parse_ravdess_clip(tmp_path / "03-01-08-01-01-01-01.wav") is None
    assert parse_tess_clip(tmp_path / "OAF_wire_ps.wav") is None


@pytest.mark.parametrize(
    "parser,name",
    [
        (parse_crema_clip, "bad.wav"),
        (parse_tess_clip, "unknown_wire_happy.wav"),
    ],
)
def test_dataset_parsers_reject_malformed_names(tmp_path, parser, name):
    with pytest.raises(ValueError):
        parser(tmp_path / name)


def test_prepare_manifest_screens_hashes_and_freezes_speaker_splits(tmp_path):
    raw = tmp_path / "raw"
    for actor in range(1, 4):
        _write_tone(
            raw / "RAVDESS" / f"Actor_{actor:02d}" / f"03-01-01-01-01-01-{actor:02d}.wav",
            180 + actor,
        )
    for offset, actor in enumerate(range(1001, 1004)):
        _write_tone(
            raw / "CREMA-D" / "AudioWAV" / f"{actor}_IEO_HAP_MD.wav",
            220 + offset,
        )
    ratings = raw / "CREMA-D" / "processedResults" / "summaryTable.csv"
    ratings.parent.mkdir(parents=True)
    with ratings.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["FileName", "VoiceLevel"])
        writer.writeheader()
        writer.writerow({"FileName": "1001_IEO_HAP_MD", "VoiceLevel": "75"})
    _write_tone(raw / "TESS" / "OAF_word_sad.wav", 260)
    _write_tone(raw / "TESS" / "YAF_word_sad.wav", 270)
    (raw / "CREMA-D" / "AudioWAV" / "1004_IEO_HAP_MD.wav").write_bytes(b"broken")

    manifests = tmp_path / "manifests"
    splits = tmp_path / "splits"
    summary = prepare_manifest(raw, manifests, splits, seed=42, workers=1)
    rows = read_manifest(manifests / "all_clips.csv")
    split_document = json.loads((splits / "six_class_v1.json").read_text())

    assert summary["accepted_clips"] == 8
    assert summary["excluded_clips"] == 1
    assert summary["accepted_by_split"]["external_test"] == 2
    assert {row["split"] for row in rows if row["dataset"] == "TESS"} == {"external_test"}
    assert next(row for row in rows if row["clip_id"].startswith("crema_d__1001"))[
        "annotation_agreement"
    ] == 0.75
    assert split_document["manifest_sha256"] == (manifests / "all_clips.sha256").read_text().strip()

    for dataset in ("RAVDESS", "CREMA-D"):
        speakers = split_document["speakers"][dataset]
        assert set(speakers["train"]).isdisjoint(speakers["val"])
        assert set(speakers["train"]).isdisjoint(speakers["test"])
        assert set(speakers["val"]).isdisjoint(speakers["test"])


def test_conflicting_label_duplicates_are_all_excluded():
    shared = {
        "dataset": "CREMA-D",
        "relative_audio_path": "clip.wav",
        "speaker_id": "CREMA-D:1001",
        "sample_rate": 16000,
        "duration_seconds": 1.0,
        "split": "",
        "role": "development",
        "intensity": "XX",
        "annotation_agreement": None,
        "source_license": "fixture",
        "source_url": "https://example.test",
        "audio_sha256": "same",
        "channels": 1,
        "rms": 0.1,
        "clipping_rate": 0.0,
        "silence_fraction": 0.0,
    }
    records = [
        ManifestRecord(
            clip_id="happy",
            canonical_label="happy",
            label_id=int(EmotionState.HAPPY),
            original_label="HAP",
            **shared,
        ),
        ManifestRecord(
            clip_id="neutral",
            canonical_label="neutral",
            label_id=int(EmotionState.NEUTRAL),
            original_label="NEU",
            **shared,
        ),
    ]

    rejected = reject_exact_duplicates(records)

    assert all(record.exclusion_reason.startswith("conflicting_labels") for record in rejected)

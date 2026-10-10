"""Tests for dynamic-model regularization and weighting."""

import csv

import numpy as np
import pytest

from scripts.tune_dynamic_model import (
    build_sample_weights,
    select_candidates,
    tuning_candidates,
)
from timbre.data.cache import CachedClip
from timbre.data.manifest import MANIFEST_FIELDS


def _clip(clip_id: str, dataset: str) -> CachedClip:
    return CachedClip(
        clip_id=clip_id,
        dataset=dataset,
        speaker_id=f"{dataset}:speaker",
        label=1,
        split="train",
        audio_sha256=f"{clip_id}-hash",
        features=np.ones((3, 2)),
    )


def _row(clip_id: str, dataset: str, agreement: str) -> dict:
    row = dict.fromkeys(MANIFEST_FIELDS, "")
    row.update(
        {
            "clip_id": clip_id,
            "dataset": dataset,
            "label_id": "1",
            "sample_rate": "16000",
            "duration_seconds": "1.0",
            "annotation_agreement": agreement,
        }
    )
    return row


def test_dataset_agreement_weights_preserve_examples_and_balance_domains(tmp_path):
    clips = [_clip("a", "large"), _clip("b", "large"), _clip("c", "small")]
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(
            [_row("a", "large", "0.1"), _row("b", "large", "0.8"), _row("c", "small", "")]
        )

    agreement = build_sample_weights(clips, manifest, "agreement")
    combined = build_sample_weights(clips, manifest, "dataset_agreement")

    assert agreement[0] > 0
    assert agreement[0] < agreement[1] < agreement[2]
    assert combined[2] > combined[1]
    assert np.mean(combined) == pytest.approx(1.0)


def test_tuning_candidates_cover_regularization_reduction_and_weighting():
    names = {candidate.name for candidate in tuning_candidates(seed=42)}

    assert {"svc_c0.5", "svc_c2", "svc_c4", "pca128_svc_c2", "lda_shrinkage"} <= names
    assert {candidate.weight_scheme for candidate in tuning_candidates(seed=42)} == {
        "uniform",
        "agreement",
        "dataset_agreement",
    }


def test_candidate_selection_preserves_order_and_rejects_unknown_names():
    selected = select_candidates(["svc_c2", "hgb_leaf7_agreement"], seed=42)

    assert [candidate.name for candidate in selected] == ["svc_c2", "hgb_leaf7_agreement"]
    with pytest.raises(ValueError, match="unknown candidates"):
        select_candidates(["not-a-model"], seed=42)

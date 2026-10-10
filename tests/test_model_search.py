"""Tests for speaker-grouped model screening."""

import numpy as np

from scripts.search_models import build_representation, candidate_models, speaker_disjoint_folds
from timbre.data.cache import CachedClip


def test_candidate_search_includes_linear_tree_kernel_and_boosting_models():
    candidates = candidate_models(seed=42)

    assert "logistic_c1" in candidates
    assert "random_forest" in candidates
    assert "extra_trees_leaf2" in candidates
    assert "rbf_svc_c10" in candidates
    assert "hist_gradient_boosting" in candidates


def test_model_search_folds_are_speaker_disjoint_and_contain_every_class():
    labels = np.tile(np.array([1, 3, 4]), 12)
    speakers = np.repeat([f"speaker-{index}" for index in range(12)], 3)

    folds = speaker_disjoint_folds(labels, speakers, n_splits=3, seed=42)

    assert len(folds) == 3
    for train_indices, validation_indices in folds:
        assert not set(speakers[train_indices]) & set(speakers[validation_indices])
        assert set(labels[validation_indices]) == {1, 3, 4}


def test_model_search_supports_static_and_dynamic_representations():
    clip = CachedClip(
        clip_id="clip",
        dataset="fixture",
        speaker_id="speaker",
        label=1,
        split="train",
        audio_sha256="fixture-hash",
        features=np.arange(30, dtype=float).reshape(10, 3),
    )

    assert build_representation([clip], "static_v1").shape == (1, 21)
    assert build_representation([clip], "dynamic_v2").shape[1] > 21

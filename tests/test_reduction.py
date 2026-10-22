"""Correctness tests for the reduction module (shape + no-leakage behavior)."""

import numpy as np

from ml.core.reduction.reducer import fit_reduction, reduce_sequences


def test_pca_reduces_dimensionality():
    rng = np.random.default_rng(0)
    frames = rng.normal(size=(500, 23))
    reducer = fit_reduction(frames, method="pca", n_components=8)
    assert reducer.transform(frames).shape == (500, 8)
    assert np.allclose(reducer.named_steps["scaler"].mean_, frames.mean(axis=0))


def test_scaler_is_fitted_only_on_supplied_training_frames():
    rng = np.random.default_rng(12)
    train = rng.normal(loc=2.0, scale=0.2, size=(100, 23))
    held_out = np.full((20, 23), 1000.0)
    reducer = fit_reduction(train, method="pca", n_components=2)
    assert np.allclose(reducer.named_steps["scaler"].mean_, train.mean(axis=0))
    assert not np.allclose(reducer.named_steps["scaler"].mean_, held_out.mean(axis=0))


def test_lle_reduces_dimensionality():
    rng = np.random.default_rng(0)
    frames = rng.normal(size=(300, 23))
    reducer = fit_reduction(frames, method="lle", n_components=8)
    out = reducer.transform(frames)
    assert out.shape == (300, 8)


def test_reduce_sequences_multiple_lengths():
    rng = np.random.default_rng(0)
    reducer = fit_reduction(rng.normal(size=(200, 23)), method="pca", n_components=6)
    seqs = [np.random.default_rng(i).normal(size=(L, 23)) for i, L in enumerate([40, 70, 25])]
    out = reduce_sequences(seqs, reducer)
    assert [s.shape[1] for s in out] == [6, 6, 6]
    assert [s.shape[0] for s in out] == [40, 70, 25]


def test_unknown_method_raises():
    import pytest

    with pytest.raises(ValueError):
        fit_reduction(np.zeros((10, 23)), method="nope")

"""Tests for serving the clip-statistic logistic baseline through PerceptionPipeline.

The runtime can serve either the per-emotion HMM (mode="hmm") or the logistic
regression over clip-level feature statistics (mode="clip"). These tests pin the
auto-detection, float, fallback, and abstention behavior of the clip mode.
"""

from __future__ import annotations

import json

import joblib
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from timbre.api import pipeline as pipeline_mod
from timbre.api.pipeline import PerceptionPipeline
from timbre.core.types import EmotionState
from timbre.features.extractor import feature_contract


CLIP_WIDTH = 161  # 7 summary statistics x 23 frame features


def build_model_dir(tmp_path, with_clip=True, classes=(1, 3), meta_width=None,
                    contract=None, baseline_contract=None):
    """Create a minimal model dir that from_trained() can load without a network."""
    rng = np.random.default_rng(0)
    contract = contract or feature_contract()
    (tmp_path / "meta.json").write_text(json.dumps({
        "state_ids": [2],
        "feature_contract": contract,
    }))
    # Dummy objects: never touched while serving the clip baseline.
    joblib.dump(object(), tmp_path / "reducer.joblib")
    joblib.dump({}, tmp_path / "state_models.joblib")
    if not with_clip:
        return
    # Uninformative data: the classifier cannot separate the two classes, so
    # predictions stay near chance magnitude instead of degenerating to 0/1,
    # which keeps the abstention tests meaningful.
    x = rng.normal(size=(120, CLIP_WIDTH))
    y = np.array([classes[0]] * 60 + [classes[1]] * 60)
    classifier = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=500)),
    ])
    classifier.fit(x, y)
    baseline_dir = tmp_path / "baselines"
    baseline_dir.mkdir()
    joblib.dump(classifier, baseline_dir / "logistic_regression.joblib")
    (baseline_dir / "meta.json").write_text(json.dumps({
        "feature_contract": baseline_contract or contract,
        "feature_width": CLIP_WIDTH if meta_width is None else meta_width,
        "models": ["logistic_regression"],
    }))


def infer_sine(pipe: PerceptionPipeline) -> object:
    # Features drawn from the same distribution the synthetic classifier was fitted
    # on, so the logistic score stays in the non-degenerate part of the sigmoid.
    features = np.random.default_rng(1).normal(size=(200, 23))
    return pipe.infer_features(features)


def test_from_trained_serves_clip_mode_when_baseline_present(tmp_path):
    build_model_dir(tmp_path)
    pipe = PerceptionPipeline.from_trained(tmp_path)
    assert pipe.mode == "clip"
    assert pipe.clip_classes == [1, 3]
    assert pipe.clip_abstention_threshold == pipeline_mod.settings.baseline_abstain_confidence


def test_clip_mode_infer_returns_point_posterior(tmp_path):
    build_model_dir(tmp_path)
    pipe = PerceptionPipeline.from_trained(tmp_path)
    posterior = infer_sine(pipe)
    assert set(posterior.probs.keys()) == {EmotionState(1), EmotionState(3)}
    assert abs(sum(posterior.probs.values()) - 1.0) < 1e-6
    assert 0.0 <= posterior.confidence <= 1.0
    # A point classifier defines no Bayesian credible interval.
    assert posterior.intervals == {}


def test_clip_mode_abstains_below_configured_threshold(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline_mod.settings, "baseline_abstain_confidence", 0.999)
    build_model_dir(tmp_path)
    pipe = PerceptionPipeline.from_trained(tmp_path)
    posterior = infer_sine(pipe)
    assert pipe.should_abstain(posterior) is True


def test_clip_mode_serves_prediction_above_threshold(tmp_path):
    build_model_dir(tmp_path)
    pipe = PerceptionPipeline.from_trained(tmp_path)
    posterior = infer_sine(pipe)
    assert not pipe.should_abstain(posterior)


def test_missing_baseline_keeps_hmm_mode(tmp_path):
    build_model_dir(tmp_path, with_clip=False)
    pipe = PerceptionPipeline.from_trained(tmp_path)
    assert pipe.mode == "hmm"
    assert pipe.clip_classifier is None


def test_tainted_baseline_falls_back_to_hmm(tmp_path, caplog):
    build_model_dir(tmp_path, meta_width=999)  # feature_width contradicts the classifier
    with caplog.at_level("WARNING"):
        pipe = PerceptionPipeline.from_trained(tmp_path)
    assert pipe.mode == "hmm"
    assert "serving HMM" in caplog.text


def test_invalid_baseline_classes_fall_back_to_hmm(tmp_path):
    build_model_dir(tmp_path, classes=(1, 99))
    pipe = PerceptionPipeline.from_trained(tmp_path)
    assert pipe.mode == "hmm"
    assert pipe.clip_classifier is None


def test_served_model_hmm_setting_disables_clip(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline_mod.settings, "served_model", "hmm")
    build_model_dir(tmp_path)
    pipe = PerceptionPipeline.from_trained(tmp_path)
    assert pipe.mode == "hmm"


def test_contract_mismatch_disables_clip(tmp_path, caplog):
    contract = dict(feature_contract())
    contract["n_mfcc"] = 13
    build_model_dir(tmp_path, baseline_contract=contract)
    with caplog.at_level("WARNING"):
        pipe = PerceptionPipeline.from_trained(tmp_path)
    assert pipe.mode == "hmm"
    assert "serving HMM" in caplog.text
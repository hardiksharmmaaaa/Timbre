"""Regression coverage for baseline selection, leakage boundaries, and saved artifacts."""

import json
from dataclasses import replace
from types import SimpleNamespace

import joblib
import numpy as np
import pytest
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import StandardScaler

from ml.core.features.aggregation import aggregate_sequences
from ml.data.cache import CachedClip
from ml.training import evaluate, train_baselines
from timbre.baselines import fit_adaboost, fit_knn, fit_svm


@pytest.fixture
def clips():
    return [
        CachedClip(
            clip_id=f"clip-{speaker:02d}-{label}", dataset="CREMA-D",
            speaker_id=f"CREMA-D:{speaker}", label=label, split="train",
            audio_sha256="fixture",
            features=np.array([[speaker * 10 + label, label * 100.0],
                               [speaker * 10 + label + 1, label * 100.0]]),
        )
        for speaker in range(6) for label in (1, 3, 7)
    ]


def test_knn_fits_scaler_only_on_fold_training_speakers(clips, monkeypatch, tmp_path):
    fitted_inputs = []
    original_fit = StandardScaler.fit

    def record_fit(self, X, y=None, **kwargs):
        fitted_inputs.append(X.copy())
        return original_fit(self, X, y, **kwargs)

    monkeypatch.setattr(StandardScaler, "fit", record_fit)
    model, report = fit_knn(clips, seed=42, n_splits=3, neighbor_counts=(1, 3))
    assert len(report["candidates"]) == 8
    best = max(report["candidates"], key=lambda candidate: candidate["mean_cv_accuracy"])
    assert report["best_params"] == best["params"]
    assert report["best_mean_cv_accuracy"] == best["mean_cv_accuracy"]
    held_out_speakers = []
    for fold_index, fold in enumerate(report["folds"]):
        assert set(fold["train_speakers"]).isdisjoint(fold["held_out_speakers"])
        held_out_speakers.extend(fold["held_out_speakers"])
        expected_first_column = {
            clip.features[:, 0].mean() for clip in clips
            if f"{clip.dataset}/{clip.speaker_id}" in fold["train_speakers"]
        }
        for X in fitted_inputs[fold_index:-1:3]:
            assert set(X[:, 0]) == expected_first_column
    assert len(held_out_speakers) == len(set(held_out_speakers)) == 6
    assert len(fitted_inputs) == 8 * 3 + 1
    assert fitted_inputs[-1].shape[0] == len(clips)
    assert model.named_steps["classifier"].n_samples_fit_ == len(clips)
    np.testing.assert_allclose(model.named_steps["scaler"].mean_, fitted_inputs[-1].mean(0))
    assert report["best_mean_cv_accuracy"] == pytest.approx(
        np.mean(best["fold_accuracies"])
    )
    path = tmp_path / "knn.joblib"
    joblib.dump(model, path)
    restored = joblib.load(path)
    np.testing.assert_array_equal(restored.classes_, [1, 3, 7])
    np.testing.assert_allclose(
        restored.predict_proba(fitted_inputs[-1]), model.predict_proba(fitted_inputs[-1])
    )
    _, repeated_report = fit_knn(clips, seed=42, n_splits=3, neighbor_counts=(1, 3))
    assert report == repeated_report
    json.dumps(report)  # All recorded scores and parameters must be serializable.


@pytest.mark.parametrize("split", ["val", "test", "external_test"])
@pytest.mark.parametrize("fit_model", [fit_knn, fit_adaboost, fit_svm])
def test_selection_rejects_nontraining_inputs(clips, split, fit_model):
    with pytest.raises(ValueError, match="training clips only"):
        fit_model([*clips, replace(clips[0], split=split)], seed=42)


@pytest.mark.parametrize("fit_model", [fit_knn, fit_adaboost, fit_svm])
def test_selection_rejects_tess_even_if_mislabeled_train(clips, fit_model):
    with pytest.raises(ValueError, match="training clips only"):
        fit_model([replace(clip, dataset="TESS") for clip in clips], seed=42)


def test_adaboost_cv_preserves_speakers_and_refits_winner(clips, monkeypatch, tmp_path):
    fitted_inputs = []
    original_fit = AdaBoostClassifier.fit

    def record_fit(self, X, y, **kwargs):
        fitted_inputs.append(X.copy())
        return original_fit(self, X, y, **kwargs)

    monkeypatch.setattr(AdaBoostClassifier, "fit", record_fit)
    grid = {"n_estimators": [4, 8], "learning_rate": [0.5, 1.0],
            "estimator__max_depth": [1, 2]}
    model, report = fit_adaboost(clips, seed=42, n_splits=3, param_grid=grid)
    assert len(report["candidates"]) == 8
    best = max(report["candidates"], key=lambda candidate: candidate["mean_cv_accuracy"])
    assert report["best_params"] == best["params"]
    assert report["best_mean_cv_accuracy"] == pytest.approx(np.mean(best["fold_accuracies"]))
    assert report["algorithm"] == "SAMME"
    assert model.get_params().get("algorithm", "SAMME") != "SAMME.R"
    assert model.random_state == model.estimator.random_state == 42
    for key, value in best["params"].items():
        assert model.get_params()[key] == value
    held_out_speakers = []
    for fold_index, fold in enumerate(report["folds"]):
        assert set(fold["train_speakers"]).isdisjoint(fold["held_out_speakers"])
        held_out_speakers.extend(fold["held_out_speakers"])
        expected = {
            clip.features[:, 0].mean() for clip in clips
            if f"{clip.dataset}/{clip.speaker_id}" in fold["train_speakers"]
        }
        for X in fitted_inputs[fold_index:-1:3]:
            assert set(X[:, 0]) == expected
    assert len(held_out_speakers) == len(set(held_out_speakers)) == 6
    assert len(fitted_inputs) == 8 * 3 + 1
    assert fitted_inputs[-1].shape[0] == len(clips)
    assert 1 <= report["fitted_estimator_count"] <= model.n_estimators
    path = tmp_path / "adaboost.joblib"
    joblib.dump(model, path)
    restored = joblib.load(path)
    np.testing.assert_array_equal(restored.classes_, [1, 3, 7])
    np.testing.assert_allclose(restored.predict_proba(fitted_inputs[-1]),
                               model.predict_proba(fitted_inputs[-1]))
    np.testing.assert_allclose(restored.predict_proba(fitted_inputs[-1]).sum(axis=1), 1.0)
    _, repeated = fit_adaboost(clips, seed=42, n_splits=3, param_grid=grid)
    assert repeated == report
    json.dumps(report)
    with pytest.raises(ValueError, match="distinct speakers"):
        fit_adaboost(clips[:3], seed=42)


def test_knn_rejects_insufficient_speakers_and_filters_oversized_k(clips):
    with pytest.raises(ValueError, match="distinct speakers"):
        fit_knn(clips[:3], seed=42)
    _, report = fit_knn(clips, seed=42, n_splits=3, neighbor_counts=(1, 100))
    assert report["excluded_neighbor_counts"] == [100]
    assert report["parameter_grid"]["classifier__n_neighbors"] == [1]
    with pytest.raises(ValueError, match="no neighbor count"):
        fit_knn(clips, seed=42, n_splits=3, neighbor_counts=(100,))


def test_svm_selects_best_accuracy_with_fold_local_scaling(clips, monkeypatch, tmp_path):
    fitted_inputs = []
    original_fit = StandardScaler.fit

    def record_fit(self, X, y=None, **kwargs):
        fitted_inputs.append(X.copy())
        return original_fit(self, X, y, **kwargs)

    monkeypatch.setattr(StandardScaler, "fit", record_fit)
    grid = [
        {"classifier__kernel": ["linear"], "classifier__C": [0.1, 1.0]},
        {"classifier__kernel": ["rbf"], "classifier__C": [0.1, 1.0],
         "classifier__gamma": ["scale", 1e6],
         "classifier__class_weight": [None, "balanced"]},
    ]
    model, report = fit_svm(clips, seed=42, n_splits=3, param_grid=grid)
    assert len(report["candidates"]) == 10
    best = max(report["candidates"], key=lambda candidate: candidate["mean_cv_accuracy"])
    assert report["best_params"] == best["params"]
    assert report["best_mean_cv_accuracy"] == pytest.approx(np.mean(best["fold_accuracies"]))
    assert len({candidate["mean_cv_accuracy"] for candidate in report["candidates"]}) > 1
    for key, value in best["params"].items():
        assert model.get_params()[key] == value
    held_out_speakers = []
    for fold_index, fold in enumerate(report["folds"]):
        assert set(fold["train_speakers"]).isdisjoint(fold["held_out_speakers"])
        held_out_speakers.extend(fold["held_out_speakers"])
        expected = {
            clip.features[:, 0].mean() for clip in clips
            if f"{clip.dataset}/{clip.speaker_id}" in fold["train_speakers"]
        }
        for X in fitted_inputs[fold_index:-1:3]:
            assert set(X[:, 0]) == expected
    assert len(held_out_speakers) == len(set(held_out_speakers)) == 6
    assert len(fitted_inputs) == 10 * 3 + 1
    assert fitted_inputs[-1].shape[0] == len(clips)
    np.testing.assert_allclose(model["scaler"].mean_, fitted_inputs[-1].mean(0))
    assert not hasattr(model, "predict_proba")
    assert report["support_vector_count"] == model["classifier"].support_.size
    path = tmp_path / "svm.joblib"
    joblib.dump(model, path)
    restored = joblib.load(path)
    np.testing.assert_array_equal(restored.classes_, [1, 3, 7])
    np.testing.assert_array_equal(restored.predict(fitted_inputs[-1]),
                                  model.predict(fitted_inputs[-1]))
    _, repeated = fit_svm(list(reversed(clips)), seed=42, n_splits=3, param_grid=grid)
    assert repeated == report
    json.dumps(report)
    with pytest.raises(ValueError, match="distinct speakers"):
        fit_svm(clips[:3], seed=42)


def test_baseline_training_and_evaluation_include_selected_models(clips, monkeypatch, tmp_path):
    settings = SimpleNamespace(
        seed=42, manifest_dir=tmp_path, processed_dir=tmp_path,
        model_dir=tmp_path / "models",
    )
    requested_splits = []

    def training_clips(manifest, processed, splits):
        requested_splits.append(splits)
        assert splits == {"train"}
        return clips

    monkeypatch.setattr(train_baselines, "settings", settings)
    monkeypatch.setattr(train_baselines, "load_cached_clips", training_clips)
    monkeypatch.setattr(train_baselines, "manifest_sha256", lambda path: "fixture-hash")
    train_baselines.main()
    assert requested_splits == [{"train"}]
    baseline_dir = settings.model_dir / "baselines"
    report = json.loads((baseline_dir / "knn_cv.json").read_text())
    meta = json.loads((baseline_dir / "meta.json").read_text())
    assert "knn" in meta["models"]
    assert report["manifest_sha256"] == "fixture-hash"
    assert meta["knn_selection"]["best_params"] == report["best_params"]
    assert (baseline_dir / "knn.joblib").exists()
    ada_report = json.loads((baseline_dir / "adaboost_cv.json").read_text())
    assert "adaboost" in meta["models"]
    assert len(ada_report["candidates"]) == 12
    assert ada_report["manifest_sha256"] == "fixture-hash"
    assert meta["adaboost_selection"]["best_params"] == ada_report["best_params"]
    assert (baseline_dir / "adaboost.joblib").exists()
    svm_report = json.loads((baseline_dir / "svm_cv.json").read_text())
    assert "svm" in meta["models"]
    assert len(svm_report["candidates"]) == 30
    assert svm_report["manifest_sha256"] == "fixture-hash"
    assert meta["svm_selection"]["best_params"] == svm_report["best_params"]
    svm = joblib.load(baseline_dir / "svm.joblib")
    expected_svm_accuracy = np.mean(
        svm.predict(aggregate_sequences([clip.features for clip in clips]))
        == np.array([clip.label for clip in clips])
    )

    (settings.model_dir / "meta.json").write_text(json.dumps({
        "manifest_sha256": "fixture-hash", "git_commit": "fixture",
    }))
    monkeypatch.setattr(evaluate, "settings", settings)
    monkeypatch.setattr(evaluate, "manifest_sha256", lambda path: "fixture-hash")
    monkeypatch.setattr(evaluate.PerceptionPipeline, "load",
                        lambda path: SimpleNamespace(state_ids=[1, 3, 7]))
    monkeypatch.setattr(evaluate, "evaluate_hmm", lambda *args: {"accuracy": 0.5})
    evaluated_splits = []

    def held_out_clips(manifest, processed, splits):
        evaluated_splits.append(splits)
        assert splits in ({"test"}, {"external_test"})
        split = next(iter(splits))
        dataset = "TESS" if split == "external_test" else "CREMA-D"
        return [replace(clip, split=split, dataset=dataset, speaker_id="unseen")
                for clip in clips]

    monkeypatch.setattr(evaluate, "load_cached_clips", held_out_clips)
    recorded = {}

    def record(name, output):
        recorded.update(output)
        return tmp_path / "metrics.json"

    monkeypatch.setattr(evaluate, "run_and_record", record)
    evaluate.evaluate_model(settings.model_dir, "knn-test")
    assert evaluated_splits == [{"test"}, {"external_test"}]
    assert recorded["baseline_metadata"]["knn_selection"] == meta["knn_selection"]
    assert recorded["baseline_metadata"]["adaboost_selection"] == meta["adaboost_selection"]
    assert recorded["baseline_metadata"]["svm_selection"] == meta["svm_selection"]
    for split in ("test", "external_test"):
        for name in ("knn", "adaboost", "svm"):
            metrics = recorded["splits"][split][name]
            assert metrics["clip_count"] == len(clips)
            assert metrics["class_ids"] == [1, 3, 7]
            assert np.asarray(metrics["confusion_matrix"]).shape == (3, 3)
            assert 0 <= metrics["accuracy"] <= 1
        svm_metrics = recorded["splits"][split]["svm"]
        assert svm_metrics["accuracy"] == expected_svm_accuracy
        assert svm_metrics["ece"] is None
        assert svm_metrics["brier"] is None
        assert svm_metrics["abstention_threshold"] is None
        assert svm_metrics["coverage"] == 1.0

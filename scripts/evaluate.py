"""Evaluate HMM and classical baselines on frozen in-domain and external-test splits."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix
from sklearn.metrics import precision_recall_fscore_support

from timbre.api.pipeline import PerceptionPipeline
from timbre.baselines import BASELINE_NAMES
from timbre.data.cache import CachedClip, load_cached_clips, manifest_sha256
from timbre.eval.metrics import (
    accuracy,
    balanced_accuracy,
    expected_calibration_error,
    multiclass_brier_score,
    run_and_record,
)
from timbre.eval.scoring import score_feature_sequences, softmax_probabilities
from timbre.features.aggregation import aggregate_sequences
from timbre.utils.common import get_logger, set_seed
from timbre.utils.config import settings

log = get_logger("eval")


def summarize_predictions(
    y_true: np.ndarray,
    probabilities: np.ndarray | None,
    class_ids: list[int],
    abstention_threshold: float = 0.0,
    *,
    predicted: np.ndarray | None = None,
) -> dict:
    """Return declared classification, calibration, and abstention metrics."""
    if predicted is None:
        if probabilities is None:
            raise ValueError("predictions or probabilities are required")
        predicted = np.array(class_ids)[probabilities.argmax(axis=1)]
    if probabilities is None:
        if abstention_threshold != 0.0:
            raise ValueError("probabilities are required for confidence-based abstention")
        confidence = None
        retained = np.ones(y_true.size, dtype=bool)
    else:
        columns = np.array([class_ids.index(int(label)) for label in predicted])
        confidence = probabilities[np.arange(y_true.size), columns]
        retained = confidence >= abstention_threshold
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        predicted,
        labels=class_ids,
        average=None,
        zero_division=0,
    )
    correct = (predicted == y_true).astype(float)
    return {
        "clip_count": int(y_true.size),
        "class_ids": class_ids,
        "accuracy": accuracy(y_true, predicted),
        "balanced_accuracy": balanced_accuracy(y_true, predicted, max(class_ids) + 1),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "per_class": {
            str(class_id): {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
            }
            for index, class_id in enumerate(class_ids)
        },
        "ece": expected_calibration_error(correct, confidence) if confidence is not None else None,
        "brier": (
            multiclass_brier_score(y_true, probabilities, class_ids)
            if probabilities is not None else None
        ),
        "confusion_matrix": sklearn_confusion_matrix(
            y_true, predicted, labels=class_ids
        ).tolist(),
        "abstention_threshold": abstention_threshold if probabilities is not None else None,
        "coverage": float(np.mean(retained)),
        "covered_accuracy": (
            float(np.mean(predicted[retained] == y_true[retained])) if retained.any() else None
        ),
    }


def _latency_summary(milliseconds: list[float]) -> dict:
    return {
        "median_ms": float(np.median(milliseconds)),
        "p90_ms": float(np.quantile(milliseconds, 0.90)),
        "p95_ms": float(np.quantile(milliseconds, 0.95)),
    }


def evaluate_hmm(
    pipeline: PerceptionPipeline, clips: list[CachedClip], class_ids: list[int]
) -> dict:
    sequences = [clip.features for clip in clips]
    labels = np.array([clip.label for clip in clips])
    score_rows = []
    latencies = []
    for index, sequence in enumerate(sequences, start=1):
        started = time.perf_counter()
        score_rows.append(score_feature_sequences(pipeline, [sequence])[0])
        latencies.append((time.perf_counter() - started) * 1000)
        if index % 100 == 0:
            log.info("scored %d/%d HMM clips", index, len(sequences))
    scores = np.vstack(score_rows)
    if pipeline.calibrator is None:
        probabilities = softmax_probabilities(scores)
    else:
        probabilities = pipeline.calibrator.predict_proba(scores)
    result = summarize_predictions(
        labels, probabilities, class_ids, pipeline.abstention_threshold
    )
    result["latency"] = _latency_summary(latencies)
    if pipeline.calibrator is not None:
        result["uncalibrated_reference"] = summarize_predictions(
            labels, softmax_probabilities(scores), class_ids
        )
    return result


def evaluate_baseline(model, clips: list[CachedClip]) -> dict:
    features = aggregate_sequences([clip.features for clip in clips])
    labels = np.array([clip.label for clip in clips])
    started = time.perf_counter()
    predicted = model.predict(features)
    probabilities = model.predict_proba(features) if hasattr(model, "predict_proba") else None
    elapsed_ms = (time.perf_counter() - started) * 1000
    class_ids = [int(value) for value in model.classes_]
    result = summarize_predictions(labels, probabilities, class_ids, predicted=predicted)
    if probabilities is None:
        result["probability_metrics_unavailable_reason"] = "model does not estimate probabilities"
    result["latency"] = {
        "batch_total_ms": elapsed_ms,
        "mean_per_clip_ms": elapsed_ms / len(clips),
    }
    return result


def _load_model_metadata(model_dir: Path, expected_manifest_hash: str) -> dict:
    metadata = json.loads((model_dir / "meta.json").read_text())
    if metadata.get("manifest_sha256") != expected_manifest_hash:
        raise ValueError("trained model does not match the current canonical manifest")
    return metadata


def evaluate_model(model_dir: Path, run_name: str) -> Path:
    set_seed(settings.seed)
    manifest_path = settings.manifest_dir / "all_clips.csv"
    digest = manifest_sha256(manifest_path)
    metadata = _load_model_metadata(model_dir, digest)
    pipeline = PerceptionPipeline.load(model_dir)
    class_ids = list(pipeline.state_ids)
    output = {
        "schema_version": 1,
        "manifest_sha256": digest,
        "model_git_commit": metadata.get("git_commit"),
        "model_artifact_bytes": sum(
            path.stat().st_size for path in model_dir.glob("*") if path.is_file()
        ),
        "splits": {},
    }

    baseline_dir = settings.model_dir / "baselines"
    baseline_models = {}
    baseline_metadata_path = baseline_dir / "meta.json"
    if baseline_metadata_path.exists():
        baseline_metadata = json.loads(baseline_metadata_path.read_text())
        if baseline_metadata.get("manifest_sha256") != digest:
            raise ValueError("baseline models do not match the current canonical manifest")
        output["baseline_metadata"] = baseline_metadata
    for name in BASELINE_NAMES:
        path = baseline_dir / f"{name}.joblib"
        if path.exists():
            baseline_models[name] = joblib.load(path)

    for split in ("test", "external_test"):
        clips = load_cached_clips(manifest_path, settings.processed_dir, {split})
        split_result = {
            "datasets": sorted({clip.dataset for clip in clips}),
            "speaker_count": len({clip.speaker_id for clip in clips}),
            "hmm": evaluate_hmm(pipeline, clips, class_ids),
        }
        for name, model in baseline_models.items():
            split_result[name] = evaluate_baseline(model, clips)
        output["splits"][split] = split_result

    path = run_and_record(run_name, output)
    log.info("wrote evaluation report to %s", path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=settings.model_dir)
    parser.add_argument("--run-name", default="six_class_v1")
    args = parser.parse_args()
    evaluate_model(args.model_dir, args.run_name)


if __name__ == "__main__":
    main()

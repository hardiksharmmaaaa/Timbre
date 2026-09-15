"""Evaluate the trained pipeline + baselines: accuracy, confusion matrix, calibration, ablations.

Command: python -m scripts.evaluate
Writes metrics + figures into experiments/<run>/.
"""

from __future__ import annotations

import numpy as np

from timbre.api.pipeline import PerceptionPipeline
from timbre.eval.metrics import (
    accuracy,
    balanced_accuracy,
    confusion_matrix,
    expected_calibration_error,
    run_and_record,
)
from timbre.utils.common import get_logger, set_seed
from timbre.utils.config import settings

log = get_logger("eval")


def load_eval_sequences(split_name: str):
    import json


    splits = json.loads((settings.splits_dir / "ravdess_splits.json").read_text())
    seqs, labels = [], []
    for clip_id, label in splits[split_name].items():
        npz = settings.processed_dir / "RAVDESS" / f"{clip_id}.npz"
        with np.load(npz) as d:
            seqs.append(d["features"])
        labels.append(int(label))
    return seqs, np.array(labels)


def main() -> None:
    set_seed(settings.seed)
    pipe = PerceptionPipeline.load(settings.model_dir)
    from timbre.hmm.decoder import score_viterbi_sequences

    seqs, y_true = load_eval_sequences("test")
    n_classes = max(int(y_true.max()), *pipe.state_ids) + 1
    pred_pairs = []
    conf_scores = []

    for i, s in enumerate(seqs):
        # run perception on the raw features (need matching feature dims at inference).
        reduced = pipe.reducer.transform(s)

        log_ll = {}
        for sid in pipe.state_ids:
            model = pipe.state_models[sid]
            log_ll[sid] = score_viterbi_sequences(model, [reduced])[0]
        posterior = PerceptionPipeline.bayes_reconstruct(log_ll, reduced.shape[0])
        pred_pairs.append(int(posterior.most_likely()))
        conf_scores.append(posterior.confidence)
        if i % 50 == 0:
            log.info("scored %d/%d", i, len(seqs))

    y_pred = np.array(pred_pairs)
    cm = confusion_matrix(y_true, y_pred, n_classes)
    # Calibration compares predicted-class confidence vs correctness (0/1 per sample).
    correct = (y_true == y_pred).astype(float)
    ece = expected_calibration_error(correct, np.array(conf_scores))

    metrics = {
        "accuracy": accuracy(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy(y_true, y_pred, n_classes),
        "ece": ece,
        "confusion_matrix": cm.tolist(),
    }
    log.info("accuracy=%.3f balanced=%.3f ECE=%.4f", metrics["accuracy"],
             metrics["balanced_accuracy"], metrics["ece"])
    run_and_record("ravdess_test", metrics)


if __name__ == "__main__":
    main()

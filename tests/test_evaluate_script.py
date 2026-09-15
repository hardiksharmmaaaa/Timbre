"""Exercise evaluation orchestration on deterministic cached-feature substitutes."""

from types import SimpleNamespace

import numpy as np
import pytest

from scripts import evaluate
from timbre.core.types import EmotionPosterior, EmotionState
from timbre.hmm import decoder


@pytest.mark.parametrize("label", [1, 8])
def test_evaluation_includes_model_classes_absent_from_test_labels(monkeypatch, label):
    recorded = {}
    pipeline = SimpleNamespace(
        reducer=SimpleNamespace(transform=lambda sequence: sequence),
        state_models={1: -10.0, 8: -1.0},
        state_ids=[1, 8],
    )
    monkeypatch.setattr(evaluate.PerceptionPipeline, "load", lambda path: pipeline)
    monkeypatch.setattr(
        evaluate, "load_eval_sequences", lambda split: ([np.ones((3, 2))], np.array([label]))
    )
    monkeypatch.setattr(decoder, "score_viterbi_sequences", lambda model, sequences: [model])
    monkeypatch.setattr(
        evaluate, "run_and_record", lambda name, metrics: recorded.update(metrics)
    )

    evaluate.main()

    assert recorded["confusion_matrix"][label][8] == 1
    assert recorded["accuracy"] == float(label == 8)


def test_evaluation_uses_same_posterior_class_for_prediction_and_confidence(monkeypatch):
    recorded = {}
    pipeline = SimpleNamespace(
        reducer=SimpleNamespace(transform=lambda sequence: sequence),
        state_models={1: -1.0, 8: -1.001},
        state_ids=[1, 8],
    )
    posterior = EmotionPosterior(
        probs={EmotionState.NEUTRAL: 0.4, EmotionState.SURPRISED: 0.6}
    )
    monkeypatch.setattr(evaluate.PerceptionPipeline, "load", lambda path: pipeline)
    monkeypatch.setattr(
        evaluate, "load_eval_sequences", lambda split: ([np.ones((3, 2))], np.array([8]))
    )
    monkeypatch.setattr(decoder, "score_viterbi_sequences", lambda model, sequences: [model])
    monkeypatch.setattr(
        evaluate.PerceptionPipeline, "bayes_reconstruct", lambda scores, frames: posterior
    )
    monkeypatch.setattr(
        evaluate, "run_and_record", lambda name, metrics: recorded.update(metrics)
    )

    evaluate.main()

    assert recorded["accuracy"] == 1.0
    assert recorded["confusion_matrix"][8][int(posterior.most_likely())] == 1
    assert recorded["ece"] == pytest.approx(1.0 - posterior.confidence)

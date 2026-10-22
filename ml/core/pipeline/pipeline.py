"""Perception pipeline wrapper: runs features -> reduction -> HMM -> Bayesian on one clip.

Composes the hand-written modules into a single `infer(y, sr) -> EmotionPosterior` call
suitable for live streaming. The pipeline is loaded once at startup and reused.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ml.config import EmotionPosterior, EmotionState, settings
from ml.core.bayesian.posterior import posterior_from_log_likelihoods
from ml.core.features.extractor import extract_frame_features
from ml.core.features.extractor import feature_contract as current_feature_contract
from ml.core.reduction.reducer import reduce_sequences

log = logging.getLogger(__name__)


@dataclass
class PerceptionPipeline:
    """Frozen, serializable composition of the four perception stages.

    Supports two serving modes:

    * ``"hmm"``  - per-emotion HMM over reduced frames + Bayesian posterior (v1);
    * ``"clip"`` - logistic regression over clip-level feature statistics (baseline).

    The clip mode skips reduction/HMM/Bayesian entirely and yields a point estimate
    (no credible interval), which is honest for a point classifier. The baseline scores
    better on the held-out in-domain set (49.9% accuracy vs 39.0% for the HMM).
    """

    reducer: object
    state_models: dict  # emotion-id -> fitted per-state HMM (for the N-classifier variant)
    state_ids: list[int]
    calibrator: object | None = None
    abstention_threshold: float = 0.0
    clip_classifier: object | None = None  # sklearn Pipeline: scaler -> logistic regression
    clip_classes: list[int] | None = None
    clip_abstention_threshold: float = 0.0
    mode: str = "hmm"  # "hmm" | "clip"

    @classmethod
    def from_trained(cls, model_dir: Path) -> PerceptionPipeline:
        """Reconstruct from serialized artifacts in `model_dir`."""
        import joblib

        reducer = joblib.load(model_dir / "reducer.joblib")
        state_models = joblib.load(model_dir / "state_models.joblib")
        meta = json.loads((model_dir / "meta.json").read_text())
        saved_contract = meta.get("feature_contract")
        if saved_contract is not None and saved_contract != current_feature_contract():
            raise ValueError("model feature contract does not match the serving configuration")
        calibrator = None
        threshold = 0.0
        calibrator_path = model_dir / "calibrator.joblib"
        calibration_path = model_dir / "calibration.json"
        if calibrator_path.exists() != calibration_path.exists():
            raise ValueError("calibration artifacts are incomplete")
        if calibrator_path.exists():
            calibrator = joblib.load(calibrator_path)
            calibration = json.loads(calibration_path.read_text())
            if calibration.get("class_ids") != meta["state_ids"]:
                raise ValueError("calibrator classes do not match HMM classes")
            threshold = float(calibration["abstention"]["threshold"])
        pipeline = cls(
            reducer=reducer,
            state_models=state_models,
            state_ids=meta["state_ids"],
            calibrator=calibrator,
            abstention_threshold=threshold,
        )
        pipeline._attach_clip_classifier(model_dir)
        return pipeline

    @classmethod
    def load(cls, model_dir: Path) -> PerceptionPipeline:
        return cls.from_trained(model_dir)

    @staticmethod
    def bayes_reconstruct(log_likelihoods: dict, n_frames: int,
                          method: str | None = None) -> EmotionPosterior:
        """Run the Bayesian layer on a dict of state -> log-likelihood (used in eval too)."""
        from ml.config import EmotionState

        ll_by_state = {EmotionState(sid): ll for sid, ll in log_likelihoods.items()}
        return posterior_from_log_likelihoods(ll_by_state, n_frames, method=method)

    def _attach_clip_classifier(self, model_dir: Path) -> None:
        """Serve the logistic clip baseline when present, compatible, and requested.

        Falls back (with a logged warning) to the HMM when the baseline artifacts are
        missing, tainted, or when `settings.served_model != "clip"`."""
        import joblib

        if settings.served_model != "clip":
            return
        baseline_dir = model_dir / "baselines"
        baseline_meta_path = baseline_dir / "meta.json"
        classifier_path = baseline_dir / "logistic_regression.joblib"
        if not baseline_meta_path.exists() or not classifier_path.exists():
            return  # no baseline artifacts: keep HMM mode
        try:
            metadata = json.loads(baseline_meta_path.read_text())
            if metadata.get("feature_contract") != current_feature_contract():
                raise ValueError("baseline feature contract mismatch")
            if "logistic_regression" not in metadata.get("models", []):
                raise ValueError("baseline manifest does not list logistic_regression")
            classifier = joblib.load(classifier_path)
            expected_width = metadata.get("feature_width")
            if expected_width is None or classifier.n_features_in_ != expected_width:
                raise ValueError("baseline feature width does not match the classifier")
            classes = [int(value) for value in classifier.classes_]
            valid_ids = {int(state.value) for state in EmotionState}
            if len(classes) < 2 or not set(classes) <= valid_ids:
                raise ValueError("baseline classes are not valid EmotionState ids")
            self.clip_classifier = classifier
            self.clip_classes = classes
            self.clip_abstention_threshold = settings.baseline_abstain_confidence
            self.mode = "clip"
            log.info("serving clip baseline (mode=clip) with classes %s", classes)
        except Exception as exc:  # noqa: BLE001 - corrupt baseline must not kill serving
            log.warning("clip baseline unavailable, serving HMM instead: %s", exc)

    def infer(self, y: np.ndarray, sr: int, method: str | None = None) -> EmotionPosterior:
        # Match the sample rate used by load_and_extract when building training features.
        if sr != settings.sample_rate:
            import librosa

            y = librosa.resample(y, orig_sr=sr, target_sr=settings.sample_rate)
            sr = settings.sample_rate
        # Stage 1: features
        feats = extract_frame_features(y, sr)  # (T, nf)
        return self.infer_features(feats, method=method)

    def infer_features(
        self, feats: np.ndarray, method: str | None = None
    ) -> EmotionPosterior:
        """Infer from cached/raw frame features using the trained scoring contract."""
        if self.mode == "clip" and self.clip_classifier is not None:
            return self._infer_clip_features(feats)
        # Stage 2: reduction
        reduced = reduce_sequences([feats], self.reducer)[0]  # (T, nc)
        n_frames = reduced.shape[0]

        # Stage 3: classify with length-normalized scores from our hand-written Viterbi.
        from ml.core.hmm.decoder import score_viterbi_sequences

        log_likelihoods = {}
        for sid in self.state_ids:
            model = self.state_models[sid]
            log_likelihoods[sid] = score_viterbi_sequences(model, [reduced])[0]

        # Stage 4: Bayesian posterior over states.
        from ml.config import EmotionState

        if self.calibrator is None:
            ll_by_state = {EmotionState(sid): ll for sid, ll in log_likelihoods.items()}
            return posterior_from_log_likelihoods(ll_by_state, n_frames, method=method)

        score_row = np.array([[log_likelihoods[sid] for sid in self.state_ids]])
        probability_row = self.calibrator.predict_proba(score_row)[0]
        classes = [int(value) for value in self.calibrator.classes_]
        if classes != self.state_ids:
            raise ValueError("calibrator classes do not match HMM classes")
        probabilities = {
            EmotionState(state_id): float(probability)
            for state_id, probability in zip(classes, probability_row)
        }
        intervals = {
            state: float(min(1.0, 1.96 * np.sqrt(probability * (1 - probability) / n_frames)))
            for state, probability in probabilities.items()
        }
        return EmotionPosterior(probs=probabilities, intervals=intervals)

    def _infer_clip_features(self, feats: np.ndarray) -> EmotionPosterior:
        """Clip-statistic point estimate: aggregate -> scale -> logistic regression."""
        from ml.core.features.aggregation import aggregate_sequence

        row = aggregate_sequence(feats).reshape(1, -1)
        probability_row = self.clip_classifier.predict_proba(row)[0]
        classes = [int(value) for value in self.clip_classifier.classes_]
        if self.clip_classes is not None and classes != self.clip_classes:
            raise ValueError("clip classifier classes changed after load")
        probabilities = {
            EmotionState(state_id): float(probability)
            for state_id, probability in zip(classes, probability_row)
        }
        # Point estimate: no Bayesian credible interval is defined for a clip classifier,
        # so intervals stay empty and the UI renders "—" for the interval estimate.
        return EmotionPosterior(probs=probabilities)

    def should_abstain(self, posterior: EmotionPosterior) -> bool:
        threshold = (
            self.clip_abstention_threshold
            if self.mode == "clip"
            else self.abstention_threshold
        )
        return threshold > 0 and posterior.confidence < threshold

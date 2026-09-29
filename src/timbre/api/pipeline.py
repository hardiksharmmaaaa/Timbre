"""Perception pipeline wrapper: runs features -> reduction -> HMM -> Bayesian on one clip.

Composes the hand-written modules into a single `infer(y, sr) -> EmotionPosterior` call
suitable for live streaming. The pipeline is loaded once at startup and reused.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from timbre.bayesian.posterior import posterior_from_log_likelihoods
from timbre.core.types import EmotionPosterior
from timbre.features.extractor import extract_frame_features
from timbre.features.extractor import feature_contract as current_feature_contract
from timbre.reduction.reducer import reduce_sequences
from timbre.utils.config import settings


@dataclass
class PerceptionPipeline:
    """Frozen, serializable composition of the four perception stages."""

    reducer: object
    state_models: dict  # emotion-id -> fitted per-state HMM (for the N-classifier variant)
    state_ids: list[int]
    calibrator: object | None = None
    abstention_threshold: float = 0.0

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
        return cls(
            reducer=reducer,
            state_models=state_models,
            state_ids=meta["state_ids"],
            calibrator=calibrator,
            abstention_threshold=threshold,
        )

    @classmethod
    def load(cls, model_dir: Path) -> PerceptionPipeline:
        return cls.from_trained(model_dir)

    @staticmethod
    def bayes_reconstruct(log_likelihoods: dict, n_frames: int,
                          method: str | None = None) -> EmotionPosterior:
        """Run the Bayesian layer on a dict of state -> log-likelihood (used in eval too)."""
        from timbre.core.types import EmotionState

        ll_by_state = {EmotionState(sid): ll for sid, ll in log_likelihoods.items()}
        return posterior_from_log_likelihoods(ll_by_state, n_frames, method=method)

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
        # Stage 2: reduction
        reduced = reduce_sequences([feats], self.reducer)[0]  # (T, nc)
        n_frames = reduced.shape[0]

        # Stage 3: classify with length-normalized scores from our hand-written Viterbi.
        from timbre.hmm.decoder import score_viterbi_sequences

        log_likelihoods = {}
        for sid in self.state_ids:
            model = self.state_models[sid]
            log_likelihoods[sid] = score_viterbi_sequences(model, [reduced])[0]

        # Stage 4: Bayesian posterior over states.
        from timbre.core.types import EmotionState

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

    def should_abstain(self, posterior: EmotionPosterior) -> bool:
        return self.abstention_threshold > 0 and posterior.confidence < self.abstention_threshold

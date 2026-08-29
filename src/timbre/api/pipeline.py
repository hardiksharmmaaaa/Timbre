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
from timbre.reduction.reducer import reduce_sequences


@dataclass
class PerceptionPipeline:
    """Frozen, serializable composition of the four perception stages."""

    reducer: object
    state_models: dict  # emotion-id -> fitted per-state HMM (for the N-classifier variant)
    state_ids: list[int]

    @classmethod
    def from_trained(cls, model_dir: Path) -> PerceptionPipeline:
        """Reconstruct from serialized artifacts in `model_dir`."""
        import joblib

        reducer = joblib.load(model_dir / "reducer.joblib")
        state_models = joblib.load(model_dir / "state_models.joblib")
        meta = json.loads((model_dir / "meta.json").read_text())
        return cls(reducer=reducer, state_models=state_models, state_ids=meta["state_ids"])

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
        # Stage 1: features
        feats = extract_frame_features(y, sr)  # (T, nf)
        # Stage 2: reduction
        reduced = reduce_sequences([feats], self.reducer)[0]  # (T, nc)
        n_frames = reduced.shape[0]

        # Stage 3: classify with length-normalized scores from our hand-written Viterbi.
        from timbre.hmm.decoder import score_viterbi_sequences

        log_likelihoods = {}
        for sid in self.state_ids:
            model = self.state_models[sid]
            log_likelihoods[sid] = score_viterbi_sequences([reduced], model)[0]

        # Stage 4: Bayesian posterior over states.
        from timbre.core.types import EmotionState

        ll_by_state = {EmotionState(sid): ll for sid, ll in log_likelihoods.items()}
        return posterior_from_log_likelihoods(ll_by_state, n_frames, method=method)

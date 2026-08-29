"""Shared types used across perception, glue, and API layers."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field

import numpy as np

EMOTION_NAMES = {
    1: "neutral",
    2: "calm",
    3: "happy",
    4: "sad",
    5: "angry",
    6: "fearful",
    7: "disgust",
    8: "surprised",
}


class EmotionState(enum.IntEnum):
    NEUTRAL = 1
    CALM = 2
    HAPPY = 3
    SAD = 4
    ANGRY = 5
    FEARFUL = 6
    DISGUST = 7
    SURPRISED = 8
    UNKNOWN = 0

    @property
    def name_lower(self) -> str:
        return EMOTION_NAMES.get(int(self), "unknown")


@dataclass
class Clip:
    """One labelled audio clip with its (optionally extracted) framewise features."""

    id: str
    path: str
    label: EmotionState
    features: np.ndarray | None = None  # (nframes, nfeatures) or None if not yet extracted
    sr: int = 22050


@dataclass
class EmotionPosterior:
    """Output of the Bayesian layer: probability mass + uncertainty per state."""

    probs: dict[EmotionState, float] = field(default_factory=dict)
    # credible interval half-width per state, e.g. {"calm": 0.04}
    intervals: dict[EmotionState, float] = field(default_factory=dict)

    def most_likely(self) -> EmotionState:
        if not self.probs:
            return EmotionState.UNKNOWN
        return max(self.probs, key=self.probs.get)

    @property
    def confidence(self) -> float:
        ml = self.most_likely()
        return self.probs.get(ml, 0.0)


@dataclass
class SessionLogPoint:
    """One mood-graph point: time + posterior."""

    turn: int
    timestamp: float
    posterior: EmotionPosterior

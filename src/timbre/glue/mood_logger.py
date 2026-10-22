"""Stage 8 — mood graph logger. Built by us (persistence layer for the demo payoff).

Logs one `SessionLogPoint` per conversation turn so the frontend can plot the emotion
trajectory + confidence over time. Persists to `data/mood_sessions/<session_id>.json`.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

from ml.config import EmotionPosterior, SessionLogPoint

SESSION_DIR = Path(__file__).resolve().parents[3] / "data" / "mood_sessions"


class MoodLogger:
    """Append-only log of EmotionPosterior per turn for a single session."""

    def __init__(self, session_id: str | None = None, out_dir: Path | None = None) -> None:
        self.session_id = session_id or uuid.uuid4().hex[:8]
        self._dir = out_dir or SESSION_DIR
        self._dir.mkdir(parents=True, exist_ok=True)
        self._points: list[dict] = []
        self._turn = 0

    def log(self, posterior: EmotionPosterior) -> SessionLogPoint:
        self._turn += 1
        point = SessionLogPoint(
            turn=self._turn,
            timestamp=time.time(),
            posterior=posterior,
        )
        rec = {
            "turn": point.turn,
            "timestamp": point.timestamp,
            "probs": {k.name_lower: v for k, v in point.posterior.probs.items()},
            "intervals": {k.name_lower: v for k, v in point.posterior.intervals.items()},
            "most_likely": point.posterior.most_likely().name_lower,
            "confidence": point.posterior.confidence,
        }
        self._points.append(rec)
        return point

    def save(self) -> Path:
        path = self._dir / f"{self.session_id}.json"
        path.write_text(json.dumps({"session_id": self.session_id, "points": self._points}))
        return path

    @property
    def points(self) -> list[dict]:
        return self._points

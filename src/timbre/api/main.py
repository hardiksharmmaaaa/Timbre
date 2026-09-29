"""FastAPI backend — serves the perception pipeline + LLM/TTS glue, and statically
serves the frontend. External-service calls live in glue/ and are labelled clearly.

Endpoints:
  GET  /api/health          -> model loaded? version?
  POST /api/inference       -> audio chunk in / EmotionPosterior out
  POST /api/chat            -> user text + posterior in / reply + tts out
  GET  /                    -> static frontend

When no trained models are present, inference falls back to a demo mock that generates
plausible posteriors so the full UI can be tested end-to-end without data/training.
"""

from __future__ import annotations

import io
import logging
import random
import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator
from starlette.concurrency import run_in_threadpool

from timbre.core.types import EmotionPosterior, EmotionState
from timbre.glue.companion import build_system_prompt, llm_reply
from timbre.utils.config import settings

app = FastAPI(title="Timbre", version="0.1.0")

# Perception model is loaded lazily at first inference (avoids slow startup).
_perception = None
_perception_lock = threading.Lock()
log = logging.getLogger(__name__)
_demo_counter = 0  # tracks inference calls in demo mode for cycling through emotions


class ChatRequest(BaseModel):
    """Stable frontend-to-chat contract using public emotion-name keys."""

    text: str = Field(min_length=1, max_length=4000)
    posterior: dict[str, float] = Field(default_factory=dict)
    intervals: dict[str, float] = Field(default_factory=dict)
    most_likely: str = "unknown"

    @field_validator("text")
    @classmethod
    def nonblank_text(cls, value: str) -> str:
        """Reject messages with no visible content."""
        value = value.strip()
        if not value:
            raise ValueError("text must contain a message")
        return value


def _emotion_key(name: str) -> EmotionState:
    by_name = {state.name_lower: state for state in EmotionState}
    try:
        return by_name[name.lower()]
    except KeyError as exc:
        raise ValueError(f"unknown emotion name: {name!r}") from exc


def posterior_from_payload(
    probs: dict[str, float], intervals: dict[str, float]
) -> EmotionPosterior:
    """Deserialize and validate the API's emotion-name representation."""
    posterior = EmotionPosterior(
        probs={_emotion_key(name): float(value) for name, value in probs.items()},
        intervals={_emotion_key(name): float(value) for name, value in intervals.items()},
    )
    if any(not 0.0 <= value <= 1.0 for value in posterior.probs.values()):
        raise ValueError("posterior probabilities must be between zero and one")
    if posterior.probs and not 0.999 <= sum(posterior.probs.values()) <= 1.001:
        raise ValueError("posterior probabilities must sum to one")
    if any(not 0.0 <= value <= 1.0 for value in posterior.intervals.values()):
        raise ValueError("posterior intervals must be between zero and one")
    if not posterior.intervals.keys() <= posterior.probs.keys():
        raise ValueError("posterior intervals must refer to a supplied probability")
    return posterior

# Demo reply templates keyed by emotion (used when no LLM key is set).
DEMO_REPLIES: dict[str, list[str]] = {
    "neutral": [
        "I'm here. What's on your mind?",
        "Take your time—what would you like to talk through?",
        "What would be useful to explore together?",
    ],
    "calm": [
        "I'll keep the pace steady. What would you like to explore?",
        "Where would you like to begin?",
        "We can take this one step at a time. What's next?",
    ],
    "happy": [
        "That sounds worth exploring—tell me more.",
        "What part of this feels most important to you?",
        "Let's build on that. What would you like to do next?",
    ],
    "sad": [
        "We can take this slowly. Would you like to say more?",
        "I'm here if you want to talk it through.",
        "Take your time—there's no need to rush your answer.",
    ],
    "angry": [
        "Let's slow this down and look at what happened.",
        "Would it help to walk through the situation one part at a time?",
        "What outcome would feel most useful right now?",
    ],
    "fearful": [
        "We can make this more concrete. What feels unclear right now?",
        "Let's take it one step at a time. Where should we start?",
        "Would a short list of options help?",
    ],
    "disgust": [
        "Would you like to unpack what happened?",
        "What part of the situation should we look at first?",
    ],
    "surprised": [
        "That sounds unexpected. What changed?",
        "Tell me more about what happened.",
    ],
    "unknown": [
        "I'm here. What would you like to talk about?",
        "Where would you like to begin?",
    ],
}


def _load_perception():
    global _perception
    with _perception_lock:
        if _perception is not None:
            return _perception
        meta = settings.model_dir / "meta.json"
        if not meta.exists():
            return None
        from timbre.api.pipeline import PerceptionPipeline

        try:
            _perception = PerceptionPipeline.load(settings.model_dir)
        except Exception as exc:  # trained artifacts may be missing or incompatible
            log.error("Unable to load perception model (%s)", type(exc).__name__)
            raise HTTPException(status_code=503, detail="Perception model is unavailable") from exc
        return _perception


def _mock_inference(n_frames: int = 60) -> dict:
    """Generate explicitly labelled random demo posteriors, independent of vocal affect."""
    global _demo_counter
    _demo_counter += 1
    emotions = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]
    # Bias toward a dominant emotion each cycle (rotation), with noise.
    primary_idx = _demo_counter % len(emotions)
    weights = [0.08] * len(emotions)
    weights[primary_idx] = 0.45
    weights[(primary_idx + 1) % len(emotions)] = 0.18
    w = np.array(weights) + np.random.default_rng(int(time.time() * 1000)).uniform(
        0, 0.05, len(emotions)
    )
    w /= w.sum()
    most_likely = emotions[int(w.argmax())]
    return {
        "posterior": {e: round(float(w[i]), 4) for i, e in enumerate(emotions)},
        "intervals": {e: round(0.03 + random.random() * 0.06, 3) for e in emotions},
        "most_likely": most_likely,
        "confidence": round(float(w.max()), 4),
        "_demo": True,
    }


@app.get("/api/health")
def health() -> dict:
    pipe = _load_perception()
    return {
        "service": "timbre",
        "version": app.version,
        "model_loaded": pipe is not None,
        "window_ms": settings.window_ms,
        "demo_mode": pipe is None,
    }


@app.post("/api/inference")
async def inference(file: UploadFile = File(...)) -> dict:
    """Accept an audio chunk (wav/mp3/ogg), run perception, return EmotionPosterior.

    Falls back to a demo mock when no trained models exist so the full UI can be
    demonstrated without downloading RAVDESS data or training."""
    raw = await file.read()
    return await run_in_threadpool(_infer_audio, raw)


def _infer_audio(raw: bytes) -> dict:
    """Decode and score audio away from the API event loop."""
    try:
        y, sr = sf.read(io.BytesIO(raw), dtype="float32")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"unreadable audio: {exc}") from exc

    if not y.size or not np.isfinite(y).all():
        raise HTTPException(status_code=400, detail="audio must contain finite, non-empty samples")
    if y.ndim == 2:
        y = y.mean(axis=1)

    pipe = _load_perception()
    if pipe is None:
        return _mock_inference()
    posterior = pipe.infer(y, sr)
    abstained = pipe.should_abstain(posterior) if hasattr(pipe, "should_abstain") else False

    return {
        "posterior": {k.name_lower: v for k, v in posterior.probs.items()},
        "intervals": {k.name_lower: v for k, v in posterior.intervals.items()},
        "most_likely": "unknown" if abstained else posterior.most_likely().name_lower,
        "confidence": posterior.confidence,
        "abstained": abstained,
    }


@app.post("/api/chat")
async def chat(payload: ChatRequest) -> dict:
    """Inject posterior into prompt, call LLM, return reply (+ optional TTS).

    Falls back to a canned demo reply when no LLM API key is configured."""
    try:
        posterior = posterior_from_payload(payload.posterior, payload.intervals)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    if not settings.openai_api_key and not settings.gemini_api_key:
        import time as _time

        rng = random.Random(int(_time.time()))
        pool = DEMO_REPLIES[posterior.most_likely().name_lower]
        reply = rng.choice(pool)
        return {"reply": reply, "_demo": True}

    system = build_system_prompt(posterior)
    # EXTERNAL SERVICE call.
    try:
        reply = await llm_reply(system, payload.text)
    except Exception as exc:  # noqa: BLE001 - convert provider failures into an API response
        raise HTTPException(status_code=502, detail="LLM request failed") from exc

    return {"reply": reply}


FRONTEND = Path(__file__).resolve().parents[3] / "frontend"

# Serve static frontend
app.mount("/static", StaticFiles(directory=str(FRONTEND)), name="static")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(str(FRONTEND / "index.html"))

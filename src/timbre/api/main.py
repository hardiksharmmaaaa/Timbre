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
import random
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from timbre.core.types import EMOTION_NAMES, EmotionPosterior, EmotionState
from timbre.glue.companion import build_system_prompt, llm_reply
from timbre.utils.config import settings

app = FastAPI(title="Timbre", version="0.1.0")

# Perception model is loaded lazily at first inference (avoids slow startup).
_perception = None
_demo_counter = 0  # tracks inference calls in demo mode for cycling through emotions


class ChatRequest(BaseModel):
    """Stable frontend-to-chat contract using public emotion-name keys."""

    text: str = Field(min_length=1, max_length=4000)
    posterior: dict[str, float] = Field(default_factory=dict)
    intervals: dict[str, float] = Field(default_factory=dict)
    most_likely: str = "unknown"


def _emotion_key(name: str) -> EmotionState:
    by_name = {emotion_name: EmotionState(sid) for sid, emotion_name in EMOTION_NAMES.items()}
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
    return posterior

# Demo reply templates keyed by emotion (used when no LLM key is set).
DEMO_REPLIES: dict[str, list[str]] = {
    "neutral": [
        "You sound composed right now. What's on your mind?",
        "You seem pretty relaxed. Anything you'd like to talk about?",
        "Your tone sounds steady — nice. What can I help with?",
    ],
    "calm": [
        "You're in a calm headspace — let's keep that going. What's next?",
        "Nice, calm energy. What would you like to explore?",
        "Good vibes — I'll keep it steady too.",
    ],
    "happy": [
        "You're sounding upbeat! Love the energy. What's making you happy?",
        "That positive tone is coming through. Let's build on it!",
        "Happiness detected — what's going well today?",
    ],
    "sad": [
        "I hear a heavier tone in your voice. Want to talk about what's weighing on you?",
        "Your voice sounds a bit down. I'm here if you want to share.",
        "I can tell things might be tough right now. Take your time.",
    ],
    "angry": [
        "Your voice sounds tense. Let's take a breath together — what's frustrating you?",
        "I'm picking up frustration. Want to walk me through what happened?",
        "Let's slow down. What's making you upset?",
    ],
    "fearful": [
        "There's some nervousness in your tone. Want to talk through what's worrying you?",
        "You sound a bit anxious. That's okay — let's figure this out together.",
        "I can hear the concern. What's causing it?",
    ],
    "disgust": [
        "Your tone suggests something rubbed you the wrong way. Want to unpack that?",
        "That sounds like it left a bad taste. What happened?",
    ],
    "surprised": [
        "Wow, that sounds unexpected! Tell me more.",
        "I hear surprise — what caught you off guard?",
    ],
    "unknown": [
        "I'm still getting a read on your voice. Say something and I'll tune in.",
        "Let me try to pick up your tone — keep talking.",
    ],
}


def _load_perception():
    global _perception
    if _perception is not None:
        return _perception
    meta = settings.model_dir / "meta.json"
    if not meta.exists():
        return None
    from timbre.api.pipeline import PerceptionPipeline

    _perception = PerceptionPipeline.load(settings.model_dir)
    return _perception


def _mock_inference(n_frames: int = 60) -> dict:
    """Generate a plausible mock posterior for demo mode. Cycles through emotions
    weighted toward the audio energy so the graph is interesting but coherent."""
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
    return {
        "service": "timbre",
        "version": app.version,
        "model_loaded": _perception is not None,
        "window_ms": settings.window_ms,
        "demo_mode": _perception is None,
    }


@app.post("/api/inference")
async def inference(file: UploadFile = File(...)) -> dict:
    """Accept an audio chunk (wav/mp3/ogg), run perception, return EmotionPosterior.

    Falls back to a demo mock when no trained models exist so the full UI can be
    demonstrated without downloading RAVDESS data or training."""
    raw = await file.read()

    pipe = _load_perception()

    if pipe is None:
        return _mock_inference()

    try:
        y, sr = sf.read(io.BytesIO(raw), dtype="float32")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"unreadable audio: {exc}") from exc

    posterior = pipe.infer(y, sr) if y.ndim == 1 else pipe.infer(y.mean(axis=1), sr)

    return {
        "posterior": {k.name_lower: v for k, v in posterior.probs.items()},
        "intervals": {k.name_lower: v for k, v in posterior.intervals.items()},
        "most_likely": posterior.most_likely().name_lower,
        "confidence": posterior.confidence,
    }


@app.post("/api/chat")
async def chat(payload: ChatRequest) -> dict:
    """Inject posterior into prompt, call LLM, return reply (+ optional TTS).

    Falls back to a canned demo reply when no LLM API key is configured."""
    if not settings.openai_api_key and not settings.gemini_api_key:
        import time as _time

        rng = random.Random(int(_time.time()))
        pool = DEMO_REPLIES.get(payload.most_likely, DEMO_REPLIES["unknown"])
        reply = rng.choice(pool)
        return {"reply": reply, "_demo": True}

    try:
        posterior = posterior_from_payload(payload.posterior, payload.intervals)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

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

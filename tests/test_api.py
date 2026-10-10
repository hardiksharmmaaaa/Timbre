"""Tests for API payload serialization without external service calls."""

import asyncio
import io
import threading
from types import SimpleNamespace

import httpx
import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from timbre.api import main
from timbre.api.main import posterior_from_payload
from timbre.core.types import EmotionPosterior, EmotionState


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(main.settings, "model_dir", tmp_path)
    monkeypatch.setattr(main.settings, "openai_api_key", "")
    monkeypatch.setattr(main.settings, "gemini_api_key", "")
    monkeypatch.setattr(main, "_perception", None)
    with TestClient(main.app, raise_server_exceptions=False) as client:
        yield client


def wav_bytes(samples, sr=22050):
    buffer = io.BytesIO()
    sf.write(buffer, np.asarray(samples, dtype=np.float32), sr, format="WAV", subtype="FLOAT")
    return buffer.getvalue()


def mock_posterior():
    return EmotionPosterior(probs={EmotionState.CALM: 1.0})


def test_posterior_payload_uses_emotion_names():
    posterior = posterior_from_payload(
        {"calm": 0.7, "happy": 0.3},
        {"calm": 0.05, "happy": 0.08},
    )
    assert posterior.probs[EmotionState.CALM] == 0.7
    assert posterior.most_likely() == EmotionState.CALM


def test_posterior_payload_rejects_unknown_emotion():
    with pytest.raises(ValueError, match="unknown emotion"):
        posterior_from_payload({"stressed": 1.0}, {})


def test_posterior_payload_rejects_invalid_probability_mass():
    with pytest.raises(ValueError, match="sum to one"):
        posterior_from_payload({"calm": 0.4, "happy": 0.4}, {})


@pytest.mark.parametrize("samples", [[], [float("nan")], [float("inf")]])
def test_invalid_audio_returns_client_error(client, monkeypatch, samples):
    def infer(y, sr):
        if not y.size or not np.isfinite(y).all():
            raise ValueError("invalid audio samples")
        return mock_posterior()

    monkeypatch.setattr(main, "_perception", SimpleNamespace(infer=infer))
    response = client.post("/api/inference", files={"file": ("test.wav", wav_bytes(samples))})
    assert response.status_code == 400


def test_demo_rejects_unreadable_audio(client):
    response = client.post("/api/inference", files={"file": ("test.wav", b"not audio")})
    assert response.status_code == 400


def test_valid_audio_demo_contract(client):
    response = client.post("/api/inference", files={"file": ("test.wav", wav_bytes([0.0] * 100))})
    assert response.status_code == 200
    result = response.json()
    assert result["_demo"] is True
    posterior = posterior_from_payload(result["posterior"], result["intervals"])
    assert posterior.most_likely().name_lower == result["most_likely"]


def test_silent_audio_abstains_without_calling_model(client, monkeypatch):
    def infer(y, sr):  # pragma: no cover - must never be reached for silent input
        raise AssertionError("silent audio must not reach the model")

    monkeypatch.setattr(main, "_perception", SimpleNamespace(infer=infer, should_abstain=lambda p: False))
    response = client.post(
        "/api/inference", files={"file": ("test.wav", wav_bytes([0.0] * 16000, sr=16000))}
    )
    assert response.status_code == 200
    result = response.json()
    assert result["silence_detected"] is True
    assert result["abstained"] is True
    assert result["most_likely"] == "unknown"
    assert result["confidence"] == 0.0
    assert set(result["posterior"]) == set(result["intervals"])
    assert all(value == 0.0 for value in result["posterior"].values())


def test_sounding_audio_still_uses_model(client, monkeypatch):
    calls = []

    def infer(y, sr):
        calls.append((y, sr))
        return mock_posterior()

    monkeypatch.setattr(main, "_perception", SimpleNamespace(infer=infer, should_abstain=lambda p: False))
    response = client.post(
        "/api/inference", files={"file": ("test.wav", wav_bytes([0.1] * 16000, sr=16000))}
    )
    assert response.status_code == 200
    assert len(calls) == 1
    result = response.json()
    assert "silence_detected" not in result
    assert result["most_likely"] == "calm"


def test_health_loads_available_model(client, monkeypatch):
    from timbre.api.pipeline import PerceptionPipeline

    (main.settings.model_dir / "meta.json").write_text("{}")
    monkeypatch.setattr(PerceptionPipeline, "load", lambda _: SimpleNamespace())
    result = client.get("/api/health").json()
    assert result["model_loaded"] is True
    assert result["demo_mode"] is False


def test_health_reports_served_model(client, monkeypatch):
    from timbre.api.pipeline import PerceptionPipeline

    (main.settings.model_dir / "meta.json").write_text("{}")
    monkeypatch.setattr(
        PerceptionPipeline, "load", lambda _: SimpleNamespace(mode="clip")
    )
    result = client.get("/api/health").json()
    assert result["served_model"] == "clip"
    assert result["model_loaded"] is True


def test_session_route_serves_dedicated_room(client):
    response = client.get("/session")
    assert response.status_code == 200
    assert 'id="sessionScreen"' in response.text
    assert 'id="endSessionBtn"' in response.text
    assert 'src="/static/app.js"' in response.text


def test_incomplete_model_returns_service_unavailable(client):
    (main.settings.model_dir / "meta.json").write_text("{}")
    response = client.post("/api/inference", files={"file": ("test.wav", wav_bytes([0.0] * 100))})
    assert response.status_code == 503
    assert "model" in response.json()["detail"].lower()


@pytest.mark.parametrize("payload", [
    {"text": "   "},
    {"text": "hello", "posterior": {"calm": 0.2}},
    {"text": "hello", "posterior": {"calm": 1.0}, "intervals": {"calm": -0.1}},
    {"text": "hello", "posterior": {"calm": 1.0}, "intervals": {"calm": "nan"}},
    {"text": "hello", "posterior": {"calm": 1.0}, "intervals": {"happy": 0.1}},
])
def test_chat_validates_payload_in_demo_mode(client, payload):
    response = client.post("/api/chat", json=payload)
    assert response.status_code == 422


def test_unknown_posterior_round_trips():
    posterior = posterior_from_payload({"unknown": 1.0}, {"unknown": 0.0})
    assert posterior.most_likely() == EmotionState.UNKNOWN


def test_inference_keeps_event_loop_responsive(client, monkeypatch):
    release = threading.Event()
    started = threading.Event()
    responsive = []

    def infer(y, sr):
        started.set()
        responsive.append(release.wait(timeout=1.0))
        return mock_posterior()

    monkeypatch.setattr(main, "_perception", SimpleNamespace(infer=infer))

    async def request_with_heartbeat():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=main.app), base_url="http://test"
        ) as session:
            request = asyncio.create_task(session.post(
                # Loud enough to pass the silence gate so infer() must run.
                "/api/inference", files={"file": ("test.wav", wav_bytes([0.1] * 100))}
            ))
            while not started.is_set():
                await asyncio.sleep(0.001)
            release.set()
            return await request

    response = asyncio.run(request_with_heartbeat())
    assert response.status_code == 200
    assert responsive == [True], "perception blocked other work on the API event loop"


@pytest.mark.parametrize("source_sr", [22050, 8000, 48000])
def test_trained_pipeline_inference_returns_posterior(client, monkeypatch, source_sr):
    from hmmlearn.hmm import GaussianHMM
    from sklearn.preprocessing import StandardScaler

    from timbre.api import pipeline
    from timbre.api.pipeline import PerceptionPipeline
    from timbre.features.extractor import extract_frame_features

    samples = 0.2 * np.sin(2 * np.pi * 220 * np.arange(4410) / 22050)
    features = extract_frame_features(samples, 22050)
    reducer = StandardScaler().fit(features)
    model = GaussianHMM(n_components=1, covariance_type="diag")
    model.startprob_ = np.array([1.0])
    model.transmat_ = np.array([[1.0]])
    model.means_ = np.zeros((1, features.shape[1]))
    model.covars_ = np.ones((1, features.shape[1]))
    monkeypatch.setattr(main, "_perception", PerceptionPipeline(reducer, {2: model}, [2]))
    feature_sample_rates = []

    def capture_features(y, sr):
        feature_sample_rates.append(sr)
        return extract_frame_features(y, sr)

    monkeypatch.setattr(pipeline, "extract_frame_features", capture_features)
    samples = 0.2 * np.sin(2 * np.pi * 220 * np.arange(source_sr // 5) / source_sr)

    response = client.post(
        "/api/inference", files={"file": ("test.wav", wav_bytes(samples, source_sr))}
    )
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["most_likely"] == "calm"
    assert result["posterior"] == {"calm": 1.0}
    assert "_demo" not in result
    assert feature_sample_rates == [main.settings.sample_rate]

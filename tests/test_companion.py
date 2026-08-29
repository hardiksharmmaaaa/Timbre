"""Tests for prioritized LLM glue without making external-service calls."""

import asyncio
from types import SimpleNamespace

import openai

from timbre.glue import companion
from timbre.utils.config import settings


class FakeResponses:
    def __init__(self, captured, error=None):
        self.captured = captured
        self.error = error

    async def create(self, **kwargs):
        self.captured.update(kwargs)
        if self.error:
            raise self.error
        return SimpleNamespace(output_text="An OpenAI reply")


class FakeOpenAIClient:
    def __init__(self, captured, error=None):
        self.responses = FakeResponses(captured, error)
        self.captured = captured

    async def close(self):
        self.captured["openai_closed"] = True


def test_llm_reply_prefers_openai(monkeypatch):
    captured = {}
    monkeypatch.setattr(settings, "llm_provider", "openai")
    monkeypatch.setattr(settings, "openai_api_key", "openai-test-key")
    monkeypatch.setattr(settings, "openai_model", "openai-test-model")
    monkeypatch.setattr(settings, "gemini_api_key", "gemini-test-key")
    monkeypatch.setattr(
        openai,
        "AsyncOpenAI",
        lambda api_key: FakeOpenAIClient(captured),
    )

    reply = asyncio.run(companion.llm_reply("Be kind", "Hello"))

    assert reply == "An OpenAI reply"
    assert captured["model"] == "openai-test-model"
    assert captured["instructions"] == "Be kind"
    assert captured["input"] == "Hello"
    assert captured["openai_closed"] is True


def test_llm_reply_falls_back_to_gemini(monkeypatch):
    fallback_calls = []

    async def fake_gemini(system_prompt, user_text):
        fallback_calls.append((system_prompt, user_text))
        return "A Gemini backup reply"

    monkeypatch.setattr(settings, "llm_provider", "openai")
    monkeypatch.setattr(settings, "openai_api_key", "openai-test-key")
    monkeypatch.setattr(settings, "gemini_api_key", "gemini-test-key")
    monkeypatch.setattr(
        openai,
        "AsyncOpenAI",
        lambda api_key: FakeOpenAIClient({}, RuntimeError("OpenAI unavailable")),
    )
    monkeypatch.setattr(companion, "_gemini_reply", fake_gemini)

    reply = asyncio.run(companion.llm_reply("Be kind", "Hello"))

    assert reply == "A Gemini backup reply"
    assert fallback_calls == [("Be kind", "Hello")]

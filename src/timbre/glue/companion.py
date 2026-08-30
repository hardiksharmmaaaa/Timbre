"""Stage 5-7 glue — OFF-THE-SHELF components. Marked EXTERNAL SERVICE.

These modules call third-party LLM / TTS APIs to adapt the reply based on the inferred
emotional posterior. This is glue, NOT the research contribution. Every external call is
labelled `# EXTERNAL SERVICE` so the transparency boundary is explicit.
"""

from __future__ import annotations

from timbre.core.types import EmotionPosterior

# EXTERNAL SERVICE: prompt templates.
TEMPLATE = (
    "The vocal-affect model currently assigns {emotion} the highest probability "
    "({confidence:.0%}). This is an uncertain acoustic estimate, not the user's verified "
    "internal state. Do not tell the user how they feel. Adjust your tone subtly: {guidance}"
)

# Mapping of the most-likely state to conversational guidance. This is our design choice,
# even though the LLM call itself is external.
GUIDANCE = {
    "stressed": "Respond gently, slow down, ask a caring follow-up before changing topic.",
    "happy": "Match the upbeat energy, keep it warm.",
    "sad": "Be soft, patient, and supportive.",
    "angry": "Stay calm, validate, avoid escalating.",
    "calm": "Keep it steady and clear.",
    "fearful": "Reassure, be clear, reduce ambiguity.",
    "disgust": "Stay neutral and open.",
    "surprised": "Acknowledge the surprise, offer context.",
}


def build_system_prompt(posterior: EmotionPosterior) -> str:
    """Inject the emotion posterior into the LLM system prompt (stage 5)."""
    state = posterior.most_likely()
    emotion = state.name_lower
    guidance = GUIDANCE.get(emotion, "Keep it natural.")
    return TEMPLATE.format(emotion=emotion, confidence=posterior.confidence, guidance=guidance)


async def llm_reply(system_prompt: str, user_text: str) -> str:
    """EXTERNAL SERVICE: use OpenAI first, with optional Gemini fallback."""
    from timbre.utils.config import settings

    if settings.llm_provider.lower() != "openai":
        raise ValueError("LLM_PROVIDER must be 'openai'; Gemini is fallback-only")

    if settings.openai_api_key:
        try:
            return await _openai_reply(system_prompt, user_text)
        except Exception:  # noqa: BLE001 - fall back only when Gemini is configured
            if not settings.gemini_api_key:
                raise

    if settings.gemini_api_key:
        return await _gemini_reply(system_prompt, user_text)
    raise ValueError("OPENAI_API_KEY is not configured")


async def _openai_reply(system_prompt: str, user_text: str) -> str:
    """EXTERNAL SERVICE: generate a reply through OpenAI's Responses API."""
    from openai import AsyncOpenAI

    from timbre.utils.config import settings

    client = AsyncOpenAI(api_key=settings.openai_api_key)
    try:
        response = await client.responses.create(
            model=settings.openai_model,
            instructions=system_prompt,
            input=user_text,
        )
        return response.output_text or ""
    finally:
        await client.close()


async def _gemini_reply(system_prompt: str, user_text: str) -> str:
    """EXTERNAL SERVICE: generate a backup reply through Gemini."""
    from google import genai
    from google.genai import types

    from timbre.utils.config import settings

    # EXTERNAL SERVICE: Google's Gemini Developer API.
    client = genai.Client(api_key=settings.gemini_api_key)
    async_client = client.aio
    try:
        response = await async_client.models.generate_content(
            model=settings.gemini_model,
            contents=user_text,
            config=types.GenerateContentConfig(system_instruction=system_prompt),
        )
        return response.text or ""
    finally:
        await async_client.aclose()


def tts_speak(text: str, emotion: str = "calm") -> bytes:
    """EXTERNAL SERVICE: call the configured TTS API; optionally modulate rate by emotion.

    `emotion` influences speaking rate (slower under stress) — human conversational mirroring.
    Returns raw audio bytes (MP3/WAV) to stream to the client.
    """
    import io

    import requests

    from timbre.utils.config import settings

    # Simplistic rate mirroring: stressed -> slower. This is our glue-level design decision.
    rate = 0.95 if emotion in {"stressed", "sad", "fearful"} else 1.0

    # EXTERNAL SERVICE: generic HTTP TTS. Swap provider URL/params as needed.
    resp = requests.post(
        settings.tts_provider,
        json={"text": text, "voice": settings.tts_voice_id, "rate": rate},
        headers={"Authorization": f"Bearer {settings.tts_api_key}"},
        timeout=10,
    )
    resp.raise_for_status()
    return io.BytesIO(resp.content).read()

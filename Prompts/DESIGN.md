# Timbre Design Approach

## Purpose

Timbre is a research prototype that estimates **vocal affect from audio features** and uses the
uncertain estimate to adapt a companion response. It does not infer a person's internal emotion,
mental health, truthfulness, or performance quality.

## Architecture

```text
Browser microphone
  -> PCM WAV upload
  -> FastAPI (`src/timbre/api`)
  -> perception pipeline (`ml/core`)
  -> posterior + uncertainty
  -> optional LLM/TTS glue (`src/timbre/glue`)
  -> session UI and mood chart
```

The product separates the public landing page (`/`) from the live session room (`/session`).
The session UI owns microphone permission, recording, inference display, chat, ending, and reset.

## Engineering Decisions

- **Separation of concerns:** ML is isolated from web and provider code so it is testable without network access.
- **Leakage-safe research:** dataset splits are speaker-disjoint; TESS is an external held-out evaluation set; feature transforms and model selection are fitted using allowed training data.
- **Uncertainty first:** the UI receives posterior probabilities and can surface uncertainty rather than presenting a single label as fact.
- **Privacy-aware interaction:** the microphone is requested only when the user joins and browser tracks are stopped when a session ends or fails.
- **Graceful degradation:** the API reports clear errors for invalid audio or missing model artifacts; the companion can remain usable in a local demo mode when no provider key exists.

## Quality Gate

For every change: inspect the end-to-end path, make the smallest compatible edit, add or update a
deterministic regression test, then run `make ci`. Run data or model commands only when the task
needs them; do not overwrite generated research artifacts casually.

For detailed interfaces, implementation locations, and data contracts, see `docs/DESIGN.md`.

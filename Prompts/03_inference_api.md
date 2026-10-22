# Prompt 3 — Harden Audio Inference

```text
You are a senior Python/FastAPI engineer. Improve this Timbre inference/API behavior:
[describe the endpoint issue or reliability requirement].

Before editing, trace the whole request path: frontend upload construction, /api/inference or
/api/chat in src/timbre/api/main.py, configuration, the invoked pipeline, and tests/test_api.py.
Explain the current failure mode using the code; do not guess.

Keep these contracts intact:
- /api/inference accepts browser-produced PCM WAV audio and returns posterior/uncertainty data.
- Invalid, empty, unreadable, or silent input returns a clear client error without a server crash.
- Missing/incompatible model artifacts return a clear service-unavailable response.
- Model work must not turn the async API into an unresponsive handler.
- No API key, raw recording, or provider response is logged or committed.
- The response must describe an uncertain vocal-affect estimate, not clinical or emotional truth.

Make the smallest compatible change. Add deterministic API tests for the reported failure and at
least one neighboring guard case. Run the focused tests and make ci. Finish with endpoint behavior
before/after, changed files, test evidence, and any operational caveat.
```

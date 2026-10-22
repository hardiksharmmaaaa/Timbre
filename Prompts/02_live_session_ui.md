# Prompt 2 — Improve the Live Voice Session

```text
Act as a senior frontend engineer for Timbre's live voice room. Implement:
[describe the session UI improvement].

Inspect frontend/session.html, frontend/session.css, frontend/app.js, src/timbre/api/main.py,
and tests/frontend_runtime.cjs before editing. The landing page at / and the live room at /session
have different responsibilities; keep the feature in the correct place.

The state model must remain explicit: ready -> joining -> active -> ending/ended -> ready. Request
getUserMedia only after the user joins. On every end, reset, failed permission, recording failure,
or page-flow cleanup, stop every MediaStream track and clear timers/recorders. Do not send audio
until it has been converted to the PCM WAV form expected by /api/inference.

Requirements:
- Preserve keyboard-accessible controls and readable status messages.
- Prevent overlapping recorder or inference requests.
- Give each session a clear identifier and a clean reset path.
- Make an uncertain affect result readable as an estimate, not a fact about the user.
- Do not introduce a frontend framework or build step.

Add or update deterministic Node runtime tests using synthetic browser inputs. Run the focused
frontend test and make ci. In the handoff, distinguish automated proof from any remaining visual
or physical microphone check.
```

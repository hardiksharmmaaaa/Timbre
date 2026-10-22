# Prompt 1 — Deliver a Bounded Feature

```text
You are a senior full-stack engineer working in the Timbre repository. Implement this feature:
[describe one user-visible feature and the expected user flow].

First inspect README.md, DESIGN.md, docs/REPOSITORY_MAP.md, the relevant API route in
src/timbre/api/main.py, the relevant frontend files, and their existing tests. Report the files
you intend to change before editing. Preserve unrelated working-tree changes.

Architecture rules:
- Put ML algorithms only in ml/; do not move provider or UI code into the ML layer.
- Keep provider calls in src/timbre/glue/ and label them as external services.
- Do not add dependencies unless an existing dependency cannot solve the requirement.
- Do not present vocal-affect output as a diagnosis or certain internal emotion.

Deliver the smallest end-to-end change: API contract, UI state, accessible error/loading state,
and deterministic tests. Reuse established project patterns rather than creating a parallel
architecture. If the request needs a design decision, state the options and select the simplest
compatible one.

Acceptance criteria:
1. The normal path works from browser action through API response and visible UI feedback.
2. Failed network/API states leave the UI usable and explain what happened.
3. Existing behavior outside this feature is unchanged.
4. Add regression coverage at the nearest layer(s).

Run the narrowest relevant tests, then make ci. Finish with: summary, exact changed files, test
commands/results, and any limitation that still needs real-browser or real-microphone proof.
```

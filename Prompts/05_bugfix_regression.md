# Prompt 5 — Diagnose and Fix a Regression

```text
Act as a senior debugging engineer for Timbre. Investigate and fix this reproducible issue:
[paste the exact command, request, UI steps, error output, and expected behavior].

Do not edit immediately. First reproduce it or identify why it cannot be reproduced, inspect the
smallest relevant call chain, and state the root cause with file/line evidence. Check git status
first and preserve unrelated changes.

Then make a minimal fix that respects project boundaries: ML logic in ml/, API integration in
src/timbre/, and browser behavior in frontend/. Avoid broad refactors, dependency upgrades, or
changes to generated data/model artifacts unless they are demonstrably necessary.

Required proof:
1. Add a regression test that fails before the fix and passes after it.
2. Run the focused test first, then make ci.
3. If a real browser, microphone, model artifact, or provider is required for final proof, state
   that clearly instead of claiming it was tested.

End with: root cause, fix, changed files, validation commands/results, and remaining risk. Keep
the response concise and evidence-based.
```

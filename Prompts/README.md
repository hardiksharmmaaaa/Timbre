# Timbre Prompt Pack

These are presentation-ready examples of how a senior developer can direct an AI coding assistant
to make safe, efficient changes to Timbre. They are reusable task briefs, not a claim that every
line of the project was generated from these exact words.

Each prompt supplies four things that prevent vague AI output: project context, non-negotiable
constraints, concrete acceptance criteria, and required verification. Start with the closest card,
replace the bracketed request, and keep the evidence from the final checks.

| Prompt | Best for |
| --- | --- |
| `01_feature_delivery.md` | Delivering a bounded end-to-end feature |
| `02_live_session_ui.md` | Improving the microphone/session experience |
| `03_inference_api.md` | Hardening audio inference and API behavior |
| `04_ml_experiment.md` | Adding a leakage-safe ML experiment or baseline |
| `05_bugfix_regression.md` | Diagnosing and fixing a reproducible defect |

## Prompting Approach

1. Ask the AI to inspect before changing code.
2. Name the authoritative modules and boundaries.
3. State safety and research-integrity constraints explicitly.
4. Require focused tests and commands as proof.
5. Ask for a concise handoff: changed files, behavior, tests, and remaining limitations.

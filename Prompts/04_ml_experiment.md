# Prompt 4 — Add a Leakage-Safe ML Experiment

```text
You are a senior ML engineer working on Timbre. Add or evaluate this experiment:
[name the baseline, feature variation, or research question].

Read docs/TRAINING_RUNBOOK.md, docs/PROFESSOR_ML_CODE_GUIDE.md, ml/data/prepare.py,
ml/data/verify.py, ml/training/train_baselines.py, ml/training/evaluate.py, and relevant tests
before proposing code. Identify which split each operation may access.

Research constraints are mandatory:
- Use the frozen manifest and speaker-disjoint development partitions.
- Fit feature transforms, scalers, dimensionality reduction, tuning, calibration, and abstention
  thresholds only with their permitted training/validation data.
- Never choose a model or hyperparameter using in-domain test or external TESS results.
- Keep TESS external-only and report domain gaps honestly.
- Store enough configuration, seed, split hash, and candidate-score provenance to reproduce the
  result.
- Do not call a library model or external provider a hand-built ML contribution.

Implement the experiment through the existing Makefile/training conventions. Add focused tests
for split isolation, preprocessing fit scope, output schema, or selection logic as appropriate.
Do not invent metrics: if data/artifacts are unavailable, provide exact commands needed to produce
them and label results as unrun.

Run static/unit checks that do not require downloading data. Finish with the hypothesis, changed
files, reproducibility command, tests run, measured-versus-unrun evidence, and limitations.
```

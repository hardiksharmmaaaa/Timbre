# Timbre — Research Report (in progress)

This directory holds the write-up. Scaffolding notes below; expand per milestone.

## Framing
- **Contributions (built by us):** framewise feature extraction (F0, RMS, MFCCs, speaking
  rate), PCA/LLE reduction, HMM + hand-written Viterbi decoder, Bayesian uncertainty layer
  (MCMC/VI), evaluation/calibration harness.
- **Glue (off-the-shelf, not claimed):** LLM API, TTS API (see `src/timbre/glue/`).
- **Research framing:** "adapted a classical sequence-modeling technique (HMM + Viterbi) to a
  new observation type (acoustic emotion features)."

## Experiments to run
- Baseline: logistic regression & random forest on hand-crafted features (Week 3-6).
- HMM+Bayesian vs. baselines on continuous speech → temporal context wins.
- Calibration curve: does 72% confidence ≈ 72% correct? (ECE).
- Ablations: PCA vs. LLE, ±Bayesian, ±Kalman smoothing.
- Cross-dataset: train RAVDESS → test CREMA-D. Personal/native-language transfer test —
  report the accuracy drop honestly (that's a finding, not a failure).

## Known caveats (document, don't hide)
1. Per-emotion HMMs are N sequence classifiers whose hidden states represent within-emotion
   acoustic regimes; compare normalized Viterbi and forward scores explicitly.
2. RAVDESS is acted + English; cross-lingual degradation expected.
3. Real-time MCMC is expensive → VI primary at inference, MCMC offline.
4. Target 65–75%; never chase ~100%.

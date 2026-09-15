# Timbre — Research Report (in progress)

This directory holds the write-up. Scaffolding notes below; expand per milestone.

## Framing
- **Contributions (built by us):** framewise feature extraction (F0, RMS, MFCCs, speaking
  rate), PCA/LLE reduction, HMM + hand-written Viterbi decoder, Bayesian uncertainty layer
  (softmax, responsibility, and Dirichlet surrogates), evaluation/calibration harness.
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
3. Uncertainty is currently a surrogate over emotion scores, not a posterior over the
   trained HMM parameters. `vi` uses iterative responsibilities and heuristic widths;
   `hybrid` uses tempered softmax and heuristic widths. Neither has demonstrated calibrated
   coverage on held-out speech.
4. Target 65–75%; never chase ~100%.

## Dirichlet uncertainty surrogate

The historical `method="mcmc"` name remains supported for compatibility. Its implementation
draws independent probability vectors from a Dirichlet distribution; it does not perform
Gibbs sampling, a Markov chain, or full HMM posterior inference.

For K emotion classes, normalize the class log scores with softmax to obtain weights `p`.
The surrogate uses a uniform Dirichlet prior plus weighted pseudo-observations:
`alpha[k] = 1 + n_frames * p[k]`. Its total concentration is `K + n_frames`, so increasing
the frame count concentrates the surrogate. The pseudo-observations are a modeling choice,
not independently labelled emotional observations: adjacent acoustic frames are correlated.

Using the configured seed, draw 4,000 probability vectors. Report their mean probabilities
and the larger distance from each mean to its 2.5th or 97.5th percentile as a symmetric
uncertainty half-width. These widths describe variation in surrogate probability draws,
not the Monte Carlo standard error of a categorical frequency estimate. The old code used
the latter and also cancelled frame count out of its concentration, causing identical widths
for short and long clips. Regression tests check frame-dependent concentration and compare
a two-class case with its analytic Beta marginal.

These changes correct the stated surrogate computation. They provide no measured improvement
in classification accuracy, empirical calibration, or coverage; those require evaluation on
held-out speakers and datasets.

# Timbre Machine Learning Q&A

## Professor-style practice guide for the codebase

This question bank covers only the machine-learning domain of Timbre: problem framing, audio
data, feature engineering, dimensionality reduction, baselines, HMMs, Viterbi decoding,
probability calibration, uncertainty, abstention, evaluation, augmentation, reproducibility,
and research limitations. It intentionally excludes frontend design, browser behavior, HTTP
routing, LLM prompting, and other non-ML application topics.

The answers describe the repository and local artifacts as inspected on **29 September 2026**.
Whenever the code, data manifest, or model artifacts change, re-check any answer containing a
number.

## How to use this guide

1. Hide the answer and speak for 60–90 seconds.
2. Draw the relevant tensor shapes or equation on paper.
3. Name the source file that implements the idea.
4. Separate what is implemented, what is library-backed, and what remains unverified.
5. For defense questions, state a limitation before proposing an improvement.

## Learning goals

After practicing, you should be able to:

- explain the complete path from a waveform to a calibrated vocal-affect distribution;
- derive and trace the hand-written Viterbi algorithm;
- justify the data split, features, PCA, HMMs, baselines, calibration, and abstention design;
- interpret accuracy, balanced accuracy, F1, ECE, Brier score, coverage, and latency correctly;
- identify leakage risks, statistical assumptions, and domain-shift limitations; and
- distinguish measured evidence from plans, mocks, and unsupported claims.

## Current ML evidence snapshot

These values are useful for a viva, but they are not all final test results.

| Item | Current evidence |
| --- | --- |
| Canonical task | Six-class vocal-affect classification |
| Classes | neutral, happy, sad, angry, fearful, disgust |
| Accepted clips | 10,890 |
| Training / validation / test / external test | 5,848 / 1,318 / 1,324 / 2,400 |
| Development datasets | RAVDESS and CREMA-D |
| External dataset | TESS |
| Frame feature width | 23 = 20 MFCC + RMS + F0 + voiced proportion |
| PCA output width | 8 |
| Current PCA explained-variance sum | approximately 0.6267 on training frames |
| Main classifier | Six class-conditional, three-state diagonal-Gaussian HMMs |
| Class score | Length-normalized hand-written Viterbi best-path score |
| Validation calibrator | StandardScaler + multinomial logistic regression |
| Validation accuracy | approximately 0.3923 |
| Validation ECE / Brier | approximately 0.0455 / 0.7229 |
| Uncalibrated validation Brier | approximately 0.8215 |
| Abstention threshold | approximately 0.2358 |
| Validation coverage / covered accuracy | approximately 0.8020 / 0.4295 |
| Final test evidence | Not yet recorded: `experiments/` is empty |

The validation values were measured on the same validation partition used to fit the
calibrator and choose the threshold. They are useful diagnostics, but they are optimistic and
must not be reported as final test performance.

---

## Part 1 — Problem definition and ML boundaries

### Q1. What machine-learning problem does Timbre solve?

**Answer:** Timbre performs supervised sequence classification of short speech recordings. It
maps a variable-length waveform to a probability distribution over six vocal-affect labels:
neutral, happy, sad, angry, fearful, and disgust. The observation is acoustic behavior in the
voice, not the speaker's private mental state. The main implementation is classical ML rather
than deep learning. See `README.md`, `src/timbre/core/types.py`, and
`src/timbre/data/manifest.py`.

### Q2. Why should we say “vocal affect” rather than “true emotion”?

**Answer:** The datasets contain acted or annotated categories, and the model sees only acoustic
signals. It has no direct access to a person's internal state, context, intent, culture, or
self-report. A prediction therefore means “this recording acoustically resembles examples
labeled angry,” not “this person is truly angry.” This wording prevents an invalid psychological
or clinical claim.

### Q3. What is the end-to-end ML pipeline?

**Answer:** The implemented path is:

```text
waveform
  -> mono/resample to 16 kHz
  -> framewise MFCC + RMS + F0 + voiced proportion
  -> training-fitted StandardScaler
  -> training-fitted PCA
  -> score the sequence under six class-specific HMMs
  -> length-normalized Viterbi score per class
  -> validation-fitted score calibrator
  -> probability distribution
  -> validation-selected abstention decision
```

The core files are `features/extractor.py`, `reduction/reducer.py`, `hmm/decoder.py`,
`eval/scoring.py`, `scripts/calibrate.py`, and `api/pipeline.py` under `src/timbre/`.

### Q4. Is Timbre a deep-learning system?

**Answer:** No. The main model uses hand-crafted acoustic features, standardization, PCA, and
Gaussian HMMs. Logistic regression and random forest are classical baselines. Librosa,
scikit-learn, SciPy, and hmmlearn supply standard algorithms, but there is no trained neural
network in the primary perception path.

### Q5. What is the main research question?

**Answer:** The central question is whether preserving the temporal sequence of acoustic frames
with an HMM provides useful vocal-affect classification beyond fixed-length, non-temporal
summary features. That is why logistic regression and random forest baselines summarize each
clip, while the HMM consumes the ordered frame sequence.

### Q6. What are the project's original ML contributions?

**Answer:** The project contribution is the experiment design and integration: the feature
contract, dataset harmonization, leakage guards, class-conditional sequence-model design,
hand-written Viterbi decoder, consistent scoring contract, calibration/abstention workflow,
evaluation harness, and training-only augmentation protocol. The project should not claim that
MFCC, PCA, Gaussian HMM training, logistic regression, or random forest were invented here.

### Q7. What is hand-written, and what is library-backed?

**Answer:** `viterbi_decode()` is hand-written dynamic programming. Dataset parsing, screening,
cache validation, score normalization, metric integration, and abstention selection are also
project code. Librosa computes MFCC, RMS, YIN F0, loading, and resampling. Scikit-learn supplies
StandardScaler, PCA/LLE, logistic regression, and random forest. hmmlearn fits HMM parameters by
Baum–Welch. SciPy performs FFT convolution for reverberation.

### Q8. Is this a discriminative or generative classifier?

**Answer:** The class-specific HMM layer is generative: each HMM models how a feature sequence
could be generated conditional on one class. Classification compares one score from every class
model. However, the fitted logistic score calibrator is discriminative because it learns a
mapping from the six HMM scores to class probabilities. The complete pipeline is therefore a
hybrid of generative sequence models and discriminative calibration.

### Q9. What does “one HMM per emotion” really mean?

**Answer:** It means six separate sequence models are trained. Within an angry HMM, for example,
the three hidden states are latent acoustic regimes, not emotion labels. At inference, the same
sequence is scored by all six HMMs and the highest calibrated evidence determines the class.
This is equivalent to a bank of class-conditional sequence classifiers.

### Q10. Why is `unknown` not a seventh training class?

**Answer:** “Unknown” has no coherent acoustic distribution in the present datasets. It may mean
low confidence, silence, corrupt audio, out-of-domain speech, or a near tie. The repository
therefore treats it as an abstention outcome rather than a supervised class. The six HMMs use
state IDs 1, 3, 4, 5, 6, and 7; ID 0 is reserved for unknown.

---

## Part 2 — Datasets and label design

### Q11. Which datasets are used, and what roles do they have?

**Answer:** RAVDESS and CREMA-D are development datasets used for train, validation, and
in-domain test partitions. TESS is locked to `external_test`. The canonical manifest currently
contains 1,055 accepted RAVDESS clips, 7,435 CREMA-D clips, and 2,400 TESS clips. Dataset roles
are declared in `src/timbre/data/manifest.py`.

### Q12. Why pool RAVDESS and CREMA-D?

**Answer:** Pooling increases the number of speakers, recordings, recording conditions, and
vocal realizations available for development. RAVDESS alone has only 24 actors and a narrow
lexical structure. CREMA-D adds 91 actors and more clips. Pooling does not remove domain bias,
but it is more defensible than training on one small acted corpus.

### Q13. Why is TESS reserved for external testing?

**Answer:** TESS differs from the development corpora in speakers, collection process, lexical
material, and acoustics. Reserving it evaluates domain transfer rather than memorization of the
development distribution. It has only two speakers, so it is also unsuitable as the sole basis
for a meaningful speaker-disjoint train/validation/test study.

### Q14. Why are there six classes instead of all eight RAVDESS classes?

**Answer:** Calm and surprised are not shared consistently across RAVDESS, CREMA-D, and TESS.
Keeping them would make class identity correlated with dataset identity. A model could learn
recording-domain cues instead of vocal affect. The shared six-class label space supports a
cleaner pooled experiment.

### Q15. What are the numeric class IDs, and why are they non-contiguous?

**Answer:** The six classes retain their original `EmotionState` IDs: neutral 1, happy 3, sad 4,
angry 5, fearful 6, and disgust 7. ID 2 is calm and ID 8 is surprised, which are excluded from
the pooled task. Non-contiguous IDs are safe because scoring and evaluation preserve an explicit
ordered `class_ids` list rather than assuming labels are `0..K-1`.

### Q16. How does RAVDESS encode labels and speakers?

**Answer:** Its seven-part filename encodes modality, vocal channel, emotion, intensity,
statement, repetition, and actor. For `03-01-05-01-02-01-16.wav`, field three is angry and field
seven is actor 16. The parser validates every field, selects audio-only speech `(03, 01)`, and
namespaces the speaker as `RAVDESS:16`.

### Q17. How does CREMA-D encode labels and speakers?

**Answer:** A name such as `1022_ITS_ANG_XX.wav` contains actor, sentence, emotion code, and
intensity. The parser maps `ANG`, `DIS`, `FEA`, `HAP`, `NEU`, and `SAD` to canonical classes,
validates the four fields, and stores the speaker as `CREMA-D:1022`.

### Q18. How does TESS encode labels and speakers?

**Answer:** A filename such as `YAF_wire_happy.wav` identifies the speaker in the first token and
the label in the final token. `OAF` and `YAF` are the two speakers. The `ps` suffix means pleasant
surprise and is excluded. All accepted TESS clips receive the `external_test` split.

### Q19. Why namespace speaker IDs by dataset?

**Answer:** Raw identifiers can collide across corpora. `RAVDESS:01` and `CREMA-D:1001` make the
speaker's source explicit and prevent accidental merging. Namespacing also makes split and
speaker-count audits unambiguous.

### Q20. What is annotation agreement in this codebase?

**Answer:** For CREMA-D, the manifest optionally loads a normalized `VoiceLevel` value from the
dataset's summary table. It is metadata describing voice-vote agreement or strength, not a
feature currently supplied to the model. Preserving it enables future label-quality filtering
or weighted experiments without silently changing today's label definition.

### Q21. What type of label noise should we expect?

**Answer:** Acted intent may not match perceived affect; raters may disagree; categorical labels
compress continuous affect; and the same acoustic cue can occur across classes. Dataset-specific
recording characteristics can also become shortcuts. These are reasons to preserve agreement
metadata, use external evaluation, inspect confusion matrices, and avoid claiming ground truth
about internal emotion.

### Q22. Is class imbalance present?

**Answer:** Yes, especially for neutral. Across all accepted clips, neutral has 1,582 examples,
while most other classes have about 1,861–1,862. In the training split, neutral has 815 clips and
the other classes have about 1,006–1,007. This motivates balanced accuracy and class weighting
for the baseline and calibration classifiers.

### Q23. Why not use demographic variables as features?

**Answer:** Age, gender, or other demographic metadata could create unfair shortcuts and would
not describe the acoustic event itself. They may be appropriate for carefully governed subgroup
evaluation, but not as prediction inputs in this experiment. Their inclusion would also worsen
privacy and deployment risks.

### Q24. What is the greatest dataset limitation?

**Answer:** The corpora are acted, English-language, and recorded under curated conditions.
Performance may fall for spontaneous speech, Arabic or multilingual speakers, UAE accents,
different microphones, noise, code-switching, or subtle affect. External TESS testing measures
one domain shift, but it is not a substitute for a consented target-population evaluation.

---

## Part 3 — Manifest, screening, splitting, and feature cache

### Q25. What is the canonical manifest?

**Answer:** `data/manifests/all_clips.csv` is the recording-level experiment contract. Each row
stores the clip ID, dataset, audio path, speaker, canonical and source label, sample rate,
duration, split, role, intensity, agreement, license, source URL, audio hash, channel count,
RMS, clipping rate, and silence fraction. Training and evaluation select data through this
shared contract rather than rediscovering files independently.

### Q26. Why hash each audio file?

**Answer:** SHA-256 identifies the exact bytes used. It detects changed or replaced source audio,
supports duplicate detection, and lets the feature-cache loader reject a cache produced from an
older recording. It is provenance evidence, not a claim that SHA-256 improves the model.

### Q27. What quality screening is performed?

**Answer:** `screen_clip()` decodes the file, checks for empty or non-finite audio, converts
channels to a mono analysis signal, and measures duration, RMS, clipping rate, and silence
fraction. It rejects clips shorter than 0.2 seconds, RMS below `1e-4`, silence fraction at least
0.98, or clipping rate above 0.05. Unreadable files receive an explicit exclusion reason.

### Q28. How is silence fraction computed during screening?

**Answer:** The mono waveform is divided into non-overlapping 25 ms frames, padding the final
frame if necessary. A frame is considered silent when its RMS is below `1e-4`. Silence fraction
is the proportion of frames meeting that condition. This is a simple quality heuristic, not a
learned voice-activity detector.

### Q29. How are exact duplicates handled?

**Answer:** Accepted candidates are grouped by audio SHA-256. If identical audio has conflicting
labels, every conflicting copy is excluded. If labels agree, one deterministic copy is retained
and later copies are marked duplicates. The current manifest excluded six conflicting-label
copies and one same-label duplicate.

### Q30. Why are conflicting duplicates especially dangerous?

**Answer:** They make the same input map to different targets, creating irreducible supervision
conflict. If copies land in different splits, they also leak the exact waveform into evaluation.
Rejecting all conflicting copies is safer than arbitrarily choosing one label.

### Q31. What does speaker-disjoint splitting prevent?

**Answer:** It prevents the same speaker's vocal identity from appearing in both training and
evaluation. Random clip splitting would let the model exploit speaker timbre, microphone, or
recording-session cues. Speaker-disjoint evaluation better tests whether affect-related patterns
transfer to unseen speakers.

### Q32. How are the development splits created?

**Answer:** RAVDESS and CREMA-D speakers are partitioned independently using a deterministic
dataset-specific seed derived from the global seed plus the dataset name. Approximately 15% of
speakers go to validation and 15% to test, with at least one speaker in every partition and at
least one left for training. Clips inherit their speaker's split.

### Q33. What are the current speaker counts per split?

**Answer:** CREMA-D has 63 train, 14 validation, and 14 test speakers. RAVDESS has 16 train, 4
validation, and 4 test speakers. TESS has 2 external-test speakers. The development partitions
are disjoint within each dataset.

### Q34. Why split independently within each development dataset?

**Answer:** Independent partitions guarantee that each development corpus contributes speakers
to train, validation, and test. A single pooled speaker shuffle could, by chance, place most of a
small corpus in one split and confound dataset mix with split identity.

### Q35. What is frozen in `six_class_v1.json`?

**Answer:** It stores the experiment version, seed, splitting strategy, manifest SHA-256,
speaker groups, and exact clip IDs for each partition. Freezing this document prevents a later
run from quietly evaluating on a different sample.

### Q36. What is the current split size?

**Answer:** There are 5,848 training clips, 1,318 validation clips, 1,324 in-domain test clips,
and 2,400 external TESS clips, totaling 10,890 accepted recordings.

### Q37. Why must splitting happen before feature extraction or augmentation?

**Answer:** Split assignment must be based on the original recording and speaker identity.
Otherwise derivatives of one recording could be assigned independently and cross partitions.
The current workflow freezes the manifest and split first, extracts features for those rows, and
permits augmentation only for rows already marked `train`.

### Q38. What does each feature-cache file contain?

**Answer:** Each compressed NPZ stores the frame-feature matrix, working sample rate, label,
clip ID, dataset, split, audio SHA-256, and serialized feature contract. Its path is
`data/processed/<dataset>/<clip_id>.npz`.

### Q39. How does extraction remain idempotent?

**Answer:** Before recomputation, `cache_matches()` checks that the cache's audio hash and feature
contract match the current manifest row and settings. Valid caches are reused. New caches are
written to a temporary path and atomically renamed so an interrupted write is not published as a
complete artifact.

### Q40. What does `load_cached_clips()` validate?

**Answer:** It selects manifest rows by split and optional dataset, verifies that every expected
cache exists, loads the features, and checks audio hash, label, two-dimensional non-empty shape,
and finite values. It returns immutable `CachedClip` records.

### Q41. Is there any remaining cache-contract weakness?

**Answer:** Yes. Extraction reuse checks the serialized feature contract, but
`load_cached_clips()` itself does not compare that field with the current contract. Therefore an
old cache could pass `data-ready` if the audio hash and label still match, unless extraction was
rerun first. A stronger loader would validate `feature_contract` and probably the cached sample
rate too.

### Q42. Why keep raw audio, processed features, models, and experiments out of Git?

**Answer:** They are large, generated, sometimes license-restricted, and potentially sensitive.
The reproducible code and metadata belong in version control; the run record should identify the
manifest hash and environment needed to reproduce generated artifacts.

---

## Part 4 — Signal processing and feature extraction

### Q43. What preprocessing occurs before feature extraction?

**Answer:** File-based extraction uses Librosa to decode audio, mix it to mono, and resample it to
the configured 16 kHz. Live inference also resamples any other input rate to 16 kHz before using
the same feature extractor. This train/serve consistency is essential.

### Q44. Why use a common sample rate?

**Answer:** The source corpora use different rates. A shared 16 kHz rate makes frame sizes,
frequency content, and learned parameters comparable across clips. Sixteen kilohertz preserves
speech frequencies up to 8 kHz, which is adequate for many speech tasks and lowers computation.

### Q45. What are the frame and hop sizes in samples?

**Answer:** At 16 kHz, a 25 ms window contains `16000 × 0.025 = 400` samples, and a 10 ms hop is
`16000 × 0.010 = 160` samples. The F0 analysis window is 50 ms, or 800 samples, but it uses the
same 160-sample hop grid.

### Q46. Why analyze short overlapping frames?

**Answer:** Speech is approximately stationary only over short intervals. A 25 ms frame is short
enough for local spectral analysis, while a 10 ms stride provides dense temporal tracking.
Overlap reduces the chance that rapid changes fall only at hard frame boundaries.

### Q47. What is the feature vector for one frame?

**Answer:** It contains 20 MFCC coefficients, one RMS energy value, one F0 estimate, and one
voiced-proportion value, for a total width of 23. `extract_frame_features()` returns shape
`(T, 23)`, where `T` varies with utterance duration.

### Q48. What do MFCCs represent?

**Answer:** MFCCs compactly describe the short-term spectral envelope. Conceptually, the frame is
mapped to a spectrum, energies are pooled on a perceptually motivated Mel scale, log energies are
taken, and a cosine transform decorrelates them. They capture vocal-tract and timbral structure
useful for speech acoustics, though they are not inherently emotion-specific.

### Q49. Does this project implement MFCCs from scratch?

**Answer:** No. `librosa.feature.mfcc()` computes them. The project chooses their number,
windowing contract, alignment, and role in the model. A correct defense distinguishes selecting
and integrating a feature from implementing its numerical algorithm.

### Q50. What is RMS energy?

**Answer:** For samples `x_1 ... x_N`, RMS is
`sqrt((1/N) × sum(x_i^2))`. It summarizes signal magnitude in a frame. Energy can help separate
forceful from quiet vocal delivery, but it is also affected by microphone gain and distance, so
it must not be interpreted as affect in isolation.

### Q51. What is F0?

**Answer:** Fundamental frequency is the estimated repetition rate of quasi-periodic vocal-fold
vibration, perceived roughly as pitch. Timbre estimates F0 with Librosa's deterministic YIN
algorithm between 50 and 500 Hz. Changes in level and contour may carry affective information,
but speaker physiology also strongly influences F0.

### Q52. Why use a longer window for F0 than for MFCCs?

**Answer:** Low pitch periods require more time samples. At 50 Hz, one period lasts 20 ms; a
25 ms window contains barely more than one period. A 50 ms window provides about 2.5 periods,
making periodicity estimation more stable while retaining the 10 ms output stride.

### Q53. How is pitch gated on unvoiced frames?

**Answer:** The extractor first identifies non-silent intervals with `librosa.effects.split()`.
For each aligned frame it computes the proportion of samples inside those intervals. If that
proportion is below 0.2, the YIN F0 estimate is replaced with zero. This prevents arbitrary
pitch estimates in mostly silent frames.

### Q54. What is voiced proportion here?

**Answer:** It is the fraction of samples in a frame that fall within Librosa's detected active
intervals. It ranges from zero to one. The name is deliberately cautious: it is driven by an
energy-based split, not a phonetic voiced/unvoiced classifier.

### Q55. How does the extractor handle an all-zero waveform?

**Answer:** It bypasses `librosa.effects.split()` and uses no active intervals. The voiced mask is
all false, so voiced proportion and gated F0 are both zero. Tests verify this behavior.

### Q56. Why are feature streams truncated to a common `T`?

**Answer:** MFCC, RMS, F0, and framed voiced-proportion calculations can differ slightly in frame
count because of window length and centering. The code takes the minimum length and concatenates
only aligned rows, guaranteeing a rectangular `(T, 23)` matrix.

### Q57. Why store a feature contract?

**Answer:** A model trained with 20 MFCCs, 16 kHz audio, and a 50 ms pitch window is not generally
compatible with different preprocessing. The contract records all these choices and the F0
method. It is saved with caches and model metadata and checked when the trained pipeline loads.

### Q58. What input validation does the feature extractor perform?

**Answer:** It requires a non-empty, one-dimensional waveform. The configuration requires
positive time and dimension values and ensures `f0_min_hz < f0_max_hz < sample_rate/2`. Cached
features and augmented features must also be finite and non-empty.

### Q59. What information do these features miss?

**Answer:** They omit lexical content, detailed prosodic contours such as explicit deltas,
spectral flux, voice quality measures such as jitter or shimmer, speaker normalization, and
pretrained speech representations. More features are not automatically better; each addition
should be validated by an ablation and leakage-safe evaluation.

### Q60. Why keep a sequence instead of averaging immediately?

**Answer:** Averaging removes order. Two clips can have the same mean pitch and energy but very
different trajectories. The HMM is intended to capture transitions among latent acoustic
regimes, so it needs the ordered `(T, 23)` sequence.

---

## Part 5 — Baselines, scaling, PCA, and LLE

### Q61. Why do the baseline models require aggregation?

**Answer:** Standard tabular logistic regression and random forest expect one fixed-width vector
per sample, while clips contain different numbers of frames. `aggregate_sequence()` converts
each feature column to seven statistics: mean, standard deviation, median, minimum, maximum,
25th percentile, and 75th percentile.

### Q62. What is the baseline feature width?

**Answer:** There are 23 frame features and 7 statistics per feature, so the clip-level vector
has `23 × 7 = 161` values. The fitted baseline metadata confirms `feature_width: 161`.

### Q63. What temporal information does aggregation discard?

**Answer:** It discards order, duration-dependent transition patterns, local trajectories, and
which acoustic regimes followed which. It preserves only marginal per-feature distributions.
That makes it a suitable test of whether the HMM's temporal structure adds value.

### Q64. Why use logistic regression as a baseline?

**Answer:** It is a simple linear discriminative classifier with interpretable capacity. If it
matches or beats the HMM, temporal modeling may not justify its complexity, or the chosen HMM
may be poorly specified. The code scales the 161 inputs and uses balanced class weights.

### Q65. Why use random forest as another baseline?

**Answer:** A random forest captures nonlinear interactions without using frame order. Comparing
it with logistic regression separates linear from nonlinear tabular capacity; comparing both
with the HMM isolates the value of sequence modeling more clearly.

### Q66. Why are the baseline class weights balanced?

**Answer:** Neutral has fewer training examples. `class_weight="balanced"` for logistic regression
and `balanced_subsample` for the forest reduce the incentive to optimize mainly for larger
classes. Class weighting does not replace reporting per-class and balanced metrics.

### Q67. Why standardize raw frame features before PCA?

**Answer:** MFCCs, RMS, F0 in hertz, and proportions have different units and scales. Without
standardization, high-variance numeric scales could dominate the principal components.
StandardScaler computes `z = (x - training_mean) / training_std` for every feature column.

### Q68. Why must the scaler be fitted only on training frames?

**Answer:** The scaler's means and variances are learned statistics. Fitting them on validation
or test data leaks information about the evaluation distribution into preprocessing. The code
concatenates only training sequences before fitting the scaler/PCA pipeline.

### Q69. What does PCA do here?

**Answer:** PCA finds orthogonal directions of maximum variance in standardized training frames
and projects 23-dimensional observations to 8 dimensions. It reduces computation and can remove
correlated or noisy directions. It is unsupervised, so high variance does not guarantee high
class discrimination.

### Q70. How much variance does the current PCA preserve?

**Answer:** The fitted artifact's eight components explain approximately 62.67% of standardized
training-frame variance. This describes reconstruction variance, not classification accuracy.
The right number of components should be selected through validation experiments, not by test
results.

### Q71. Why preserve sequence boundaries after PCA?

**Answer:** HMM training needs to know where one utterance ends and the next begins. The reducer
transforms each `(T_i, 23)` clip to `(T_i, 8)` independently after being fitted globally.
Concatenating clips without passing their lengths would create false transitions between the end
of one speaker's clip and the beginning of another.

### Q72. How are sequence boundaries passed to hmmlearn?

**Answer:** Per-class reduced sequences are concatenated into one matrix `X`, and a parallel
`lengths` list records each clip's frame count. `GaussianHMM.fit(X, lengths)` uses those lengths
to reset the sequence at every clip boundary.

### Q73. What is LLE, and is it the current main reducer?

**Answer:** Locally Linear Embedding is a nonlinear manifold-learning method that tries to
preserve local neighborhood relationships. The reducer supports it as an experimental option,
but the current trained model uses PCA. LLE can be more expensive and less stable for large
frame collections, and any comparison must use the same train-only protocol.

### Q74. Why is PCA a more practical default than LLE here?

**Answer:** PCA is deterministic under this configuration, fast on many frames, easy to apply to
new observations, and interpretable through explained variance. LLE's neighborhood graph is more
computationally demanding and its out-of-sample behavior can be sensitive to data geometry and
hyperparameters.

### Q75. Does PCA guarantee better classification?

**Answer:** No. PCA preserves variance, not label information. A low-variance direction could be
highly discriminative, while a high-variance direction could mainly encode speaker or dataset.
Only a validation-controlled ablation can show whether PCA helps this task.

---

## Part 6 — HMM model and training

### Q76. What are the components of an HMM?

**Answer:** An HMM has hidden states `z_t`, observations `x_t`, start probabilities
`P(z_1)`, transition probabilities `P(z_t | z_{t-1})`, and emission distributions
`P(x_t | z_t)`. The hidden state sequence is unobserved; only the frame features are observed.

### Q77. What do hidden states mean in Timbre?

**Answer:** Each class HMM has three latent acoustic regimes. They are learned statistical
clusters with transition structure; the code does not assign semantic names such as “rising
pitch” or “high energy.” Any semantic interpretation would require post-hoc evidence.

### Q78. What is the probability modeled by one class HMM?

**Answer:** For a hidden path `z_1...z_T` and observations `x_1...x_T`, the joint model is:

```text
P(z_1) P(x_1 | z_1) × product from t=2..T of
P(z_t | z_{t-1}) P(x_t | z_t)
```

The class score in Timbre uses the probability of the single best hidden path, expressed in log
space and divided by `T`.

### Q79. Why use one class-conditional HMM per vocal-affect label?

**Answer:** It provides a direct generative classification rule: ask which class model best
explains the complete sequence. It also permits class-specific transition and emission patterns.
The tradeoff is more parameters and no parameter sharing across emotions.

### Q80. What type of emissions are used?

**Answer:** Each hidden state has an 8-dimensional Gaussian emission after PCA. The configured
`covariance_type="diag"` assumes zero covariance between PCA coordinates within a state, so each
state learns a mean vector and one variance per dimension.

### Q81. Why might diagonal covariance be reasonable?

**Answer:** PCA makes dimensions uncorrelated globally, which makes a diagonal approximation more
plausible and reduces parameters. With limited data, full covariance can be unstable. However,
global PCA decorrelation does not guarantee independence inside every class and hidden state.

### Q82. How many HMMs and hidden states exist in the current model?

**Answer:** There are six HMMs, one for each pooled class. Each has three hidden states, so there
are 18 class-specific latent regimes in total. They do not form one shared 18-state model.

### Q83. How are HMM parameters learned?

**Answer:** `hmmlearn.GaussianHMM.fit()` performs expectation-maximization through the
Baum–Welch algorithm. It iteratively estimates expected state responsibilities and transitions,
then updates start, transition, mean, and variance parameters to improve data likelihood.

### Q84. Is Baum–Welch guaranteed to find the global optimum?

**Answer:** No. HMM likelihood is non-convex, so EM can converge to a local optimum and depends on
initialization. The fixed seed improves reproducibility but does not prove optimality. Multiple
seeds and validation comparison would provide stronger evidence.

### Q85. What do the current convergence artifacts show?

**Answer:** Each loaded HMM monitor reports convergence, but angry, fearful, and disgust reached
the configured 100-iteration limit. Reaching the cap should trigger inspection of the likelihood
trace; a boolean monitor alone is not enough to claim stable convergence.

### Q86. What are the key HMM hyperparameters?

**Answer:** The current configuration uses three hidden states, diagonal covariance, at most 100
EM iterations, and seed 42. The input dimension is eight after PCA. State count, covariance
structure, PCA width, and scoring rule should be selected on validation data only.

### Q87. Is `hmm_n_states_emission_gmm` used?

**Answer:** No. The setting exists, but `train_hmm()` constructs `GaussianHMM`, not `GMMHMM`, so
each hidden state has one Gaussian emission. The unused setting should not be described as an
implemented mixture model.

### Q88. Does raw HMM classification use class priors?

**Answer:** No explicit class prior `P(class)` is added to the length-normalized class scores.
The raw comparison is therefore similar to assuming equal priors. The later discriminative
calibrator can learn class-specific offsets, but that is not the same as a declared generative
prior model.

### Q89. What is the difference between hidden state and output class?

**Answer:** Output class is one of six vocal-affect categories. Hidden state is a latent regime
inside one class HMM at one time step. Confusing them leads to the false statement that a
three-state HMM predicts three emotions.

### Q90. What would a single shared HMM alternative look like?

**Answer:** A shared model might contain latent states used across all classes, with a separate
classifier over sequence representations or class-labeled state structure. It could share
statistical strength but would require a new, explicit formulation. The current repository's
planned “single HMM” discussion is an ablation idea, not implemented evidence.

---

## Part 7 — Hand-written Viterbi decoding

### Q91. What problem does Viterbi solve?

**Answer:** Given a fitted HMM and an observation sequence, Viterbi finds the single most probable
hidden-state path. It uses dynamic programming rather than enumerating all `N^T` possible paths,
where `N` is the number of states and `T` is the number of frames.

### Q92. What is the Viterbi recurrence in this code?

**Answer:** In log space:

```text
delta[0, j] = log_start[j] + log_emission[0, j]

delta[t, j] = max_i(
    delta[t-1, i] + log_transition[i, j]
) + log_emission[t, j]
```

`psi[t, j]` stores the maximizing predecessor `i`. The final state is the argmax of the last
`delta` row, followed by backtracking through `psi`.

### Q93. What do `delta` and `psi` mean?

**Answer:** `delta[t, j]` is the best log probability of any partial path that ends in state `j`
at time `t`. `psi[t, j]` is the previous state on that best partial path. `delta` provides the
optimal score; `psi` makes it possible to reconstruct the full path.

### Q94. Why compute in log space?

**Answer:** Multiplying many probabilities smaller than one quickly underflows toward zero.
Taking logs converts products to sums and preserves numerical resolution. The maximization is
unchanged because logarithm is monotonic.

### Q95. How are impossible starts and transitions represented?

**Answer:** A zero probability becomes negative infinity under `log(0)`. Adding negative infinity
keeps the path impossible, even if later emissions are strong. Regression tests confirm that the
decoder never selects a forbidden start or transition.

### Q96. What is the Viterbi algorithm's complexity?

**Answer:** Time complexity is `O(TN^2)` because each of `T` time steps and `N` destination states
examines `N` predecessors. The implementation stores `delta` and `psi` for all time steps, so
space is `O(TN)`. With three states per HMM, scoring is small relative to feature extraction.

### Q97. How is the hand-written decoder validated?

**Answer:** `tests/test_viterbi.py` builds a tiny two-state HMM and brute-forces all paths. It
asserts that the decoder returns the same best path and score. Separate tests cover one frame,
empty observations, length normalization, impossible starts, and impossible transitions.

### Q98. What is the difference between Viterbi score and forward likelihood?

**Answer:** Viterbi keeps only the best hidden path:
`max_z P(X, z)`. The forward algorithm sums over every hidden path:
`sum_z P(X, z) = P(X)`. Therefore the Viterbi score is a best-path score, not the full sequence
log-likelihood. A forward-score ablation is discussed but not implemented in the main evaluator.

### Q99. Why divide the Viterbi score by the number of frames?

**Answer:** Raw log probabilities become more negative as a sequence gets longer because more
terms are added. Dividing by `T` creates an average best-path log score per frame, making clips of
different duration more comparable.

### Q100. Does length normalization remove every duration effect?

**Answer:** No. The start probability contributes once, boundary effects differ for short clips,
and duration can affect which transitions are possible. It is a useful normalization, not a
formal duration model.

### Q101. Why does `viterbi_decode()` call a private hmmlearn method?

**Answer:** It needs state-by-state emission log densities, obtained from
`model._compute_log_likelihood(obs)`. This makes the hand-written dynamic program possible but
depends on a private library API that could change. A robust improvement would implement the
diagonal-Gaussian log-density locally or pin and test the dependency version carefully.

### Q102. How does a clip become a six-dimensional score vector?

**Answer:** The clip's frame features are transformed by the saved scaler/PCA. The reduced
sequence is passed to `score_viterbi_sequences()` once per class HMM. The six normalized scores
are returned in the explicit `state_ids` order.

---

## Part 8 — Probabilities, calibration, uncertainty, and abstention

### Q103. Why are HMM scores not automatically probabilities over classes?

**Answer:** They are unbounded, model-dependent average log scores. Their relative scales can be
misaligned, and the Viterbi approximation is not a normalized class posterior. A softmax makes
numbers sum to one but does not guarantee that 70% confidence corresponds to 70% accuracy.

### Q104. What is numerically stable softmax?

**Answer:** For scores `s_k`, the code computes:

```text
p_k = exp(s_k - max(s)) / sum_j exp(s_j - max(s))
```

Subtracting the maximum leaves the result unchanged but prevents overflow. This is the
uncalibrated reference in evaluation.

### Q105. What calibrator is fitted?

**Answer:** A scikit-learn pipeline first standardizes the six HMM scores and then fits
multinomial logistic regression with balanced class weights. It is trained on validation clips
only and saved as `models/calibrator.joblib`.

### Q106. Why standardize the HMM scores before logistic calibration?

**Answer:** Class score columns may have different means and scales. Standardization improves
optimization and prevents one column's numeric scale from dominating solely due to units.
`tests/test_evaluate_script.py` includes a deliberately mismatched-scale example.

### Q107. What does calibrated probability mean?

**Answer:** Ideally, among predictions assigned approximately 0.7 confidence, about 70% should be
correct. Calibration concerns reliability of probability values, not just whether the top class
is correct. A model can have good accuracy and poor calibration, or the reverse.

### Q108. Why is calibration fitted on validation rather than test data?

**Answer:** Calibration learns parameters and abstention learns a threshold, so both are model
selection steps. Using test data would leak final outcomes into the model. The test and external
test partitions must remain untouched until the pipeline is frozen.

### Q109. What is the current calibration evidence?

**Answer:** On the 1,318 validation clips used to fit it, the calibrator records about 39.23%
accuracy, 0.0455 ECE, and 0.7229 multiclass Brier score, compared with 0.8215 uncalibrated Brier.
These are in-sample validation diagnostics, not final test results.

### Q110. What is ECE?

**Answer:** Expected Calibration Error bins predictions by top-class confidence. For each bin it
computes the absolute difference between mean confidence and empirical accuracy, then takes a
sample-count-weighted average:

```text
ECE = sum_b (n_b / N) × |accuracy_b - confidence_b|
```

Lower is better, but the value depends on binning and sample size.

### Q111. What is the multiclass Brier score?

**Answer:** It is the mean squared distance between the predicted probability vector and the
one-hot true label:

```text
Brier = mean_i sum_k (p_i,k - y_i,k)^2
```

Lower is better, and zero is perfect. It evaluates the full distribution, unlike top-class
accuracy.

### Q112. How is the abstention threshold selected?

**Answer:** The code considers zero and every unique validation confidence as candidates. It
retains predictions with confidence at least the threshold, rejects candidates below 80%
coverage, and selects the option with greatest retained accuracy, then coverage, then threshold.

### Q113. What are coverage and covered accuracy?

**Answer:** Coverage is the fraction of inputs for which the model returns a class rather than
abstaining. Covered accuracy is accuracy only among those retained inputs. Raising the threshold
usually lowers coverage and may raise covered accuracy; this is a selective-prediction tradeoff.

### Q114. What is the current abstention operating point?

**Answer:** The validation-selected threshold is about 0.2358. On that same validation set,
coverage is about 80.20% and covered accuracy about 42.95%. These values must be re-measured
unchanged on test and external-test data before making a generalization claim.

### Q115. How is abstention applied during inference?

**Answer:** `PerceptionPipeline.should_abstain()` returns true when a positive stored threshold
exists and the maximum posterior probability is below it. The application can then expose
`unknown` while preserving the underlying distribution and confidence for analysis.

### Q116. Is low maximum probability a complete unknown detector?

**Answer:** No. Neural and classical classifiers can be confidently wrong out of distribution.
The current rule has no explicit OOD model, quality gate at inference, entropy/margin criterion,
or distance-to-training detector. It is a useful selective-classification baseline, not a
complete safety solution.

### Q117. What happens if no fitted calibrator is present?

**Answer:** The inference pipeline calls `posterior_from_log_likelihoods()` and uses the selected
`hybrid`, `vi`, or historical `mcmc` surrogate. These methods normalize scores and attach
heuristic or surrogate intervals. They are fallbacks, weaker than held-out empirical calibration.

### Q118. Is the `vi` method true Bayesian variational inference over HMM parameters?

**Answer:** No. It repeatedly updates a categorical responsibility vector using the class scores
and a small log-responsibility term. It does not define and optimize a full evidence lower bound
for uncertain HMM parameters. The code correctly calls it a surrogate.

### Q119. Is the `mcmc` method actually MCMC?

**Answer:** No. The name remains for compatibility, but it draws 4,000 independent samples from a
Dirichlet distribution. There is no Markov chain, transition kernel, burn-in, Gibbs sampler, or
convergence diagnostic. It must be called a Dirichlet uncertainty surrogate.

### Q120. What is the Dirichlet surrogate?

**Answer:** It first softmax-normalizes the class scores to weights `p_k`, then defines
`alpha_k = 1 + n_frames × p_k`. It samples class-probability vectors from
`Dirichlet(alpha)`. More frames increase total concentration and narrow the sampled
distribution, but the frames are not independent class labels, so the intervals are not
empirically validated confidence guarantees.

### Q121. How are Dirichlet interval widths reported?

**Answer:** The code computes the sample mean and the 2.5th and 97.5th percentiles for each class.
It reports the larger distance from the mean to either percentile as a symmetric half-width.
The resulting interval is a summary of the chosen surrogate distribution.

### Q122. What intervals are returned by the fitted-calibrator path?

**Answer:** For probability `p` and `n_frames = T`, the code uses
`1.96 × sqrt(p(1-p)/T)`, capped at one. This resembles a binomial normal-approximation standard
error, but adjacent frames are correlated and are not independent class trials. It is therefore
a heuristic display interval, not a valid calibrated confidence interval.

### Q123. What is the biggest statistical limitation of the calibration workflow?

**Answer:** The same validation data fits the logistic calibrator, selects the abstention
threshold, and reports validation metrics. This can produce optimistic diagnostics. Stronger
designs include cross-validated calibration, a separate calibration and threshold-selection
partition, or nested validation, followed by one untouched test evaluation.

---

## Part 9 — Evaluation and experiment interpretation

### Q124. What final evaluation partitions are defined?

**Answer:** `test` contains unseen RAVDESS and CREMA-D speakers for in-domain evaluation.
`external_test` contains all accepted TESS clips for cross-dataset evaluation. Results must be
reported separately because they answer different questions.

### Q125. What does accuracy measure?

**Answer:** Accuracy is the fraction of clips whose top predicted class equals the label. It is
easy to understand but can hide poor minority-class performance.

### Q126. What does balanced accuracy measure?

**Answer:** Balanced accuracy is the average recall across classes present in the ground truth.
Each class contributes equally regardless of sample count. It is valuable here because neutral
has fewer samples.

### Q127. What are precision, recall, and F1 for a class?

**Answer:** Precision is `TP/(TP+FP)`, measuring how often predictions of that class are correct.
Recall is `TP/(TP+FN)`, measuring how many true examples are found. F1 is their harmonic mean,
`2PR/(P+R)`. Macro scores average class scores equally.

### Q128. How should a confusion matrix be read?

**Answer:** In this evaluator, rows are true classes and columns are predicted classes because
that is scikit-learn's convention. Diagonal cells are correct. A large off-diagonal cell reveals
a specific confusion, such as fearful recordings predicted as angry.

### Q129. Why report per-class metrics as well as macro metrics?

**Answer:** A single average can hide systematic failure. Per-class precision and recall show
which affects are confused or underdetected. This is especially important when labels and sample
sizes are not equally reliable.

### Q130. Why compare calibration metrics as well as classification metrics?

**Answer:** The downstream abstention decision depends on confidence quality. Two models with the
same accuracy may behave very differently if one is overconfident. ECE and Brier score assess
probability reliability, while accuracy and F1 assess decisions.

### Q131. How are baselines evaluated fairly?

**Answer:** They load the same frozen test clips and cached frame features. Only their
representation differs: aggregation followed by the fitted classifier. The manifest hash in
baseline metadata must match the current manifest before evaluation.

### Q132. What artifact checks occur before evaluation?

**Answer:** The evaluator compares the main and baseline model metadata's manifest SHA-256 with
the current canonical manifest. `PerceptionPipeline.load()` also checks the saved feature
contract and calibration class ordering. These checks prevent some stale-artifact comparisons.

### Q133. What latency does `evaluate_hmm()` actually measure?

**Answer:** It measures feature-sequence scoring after cached features already exist: reducer
transform plus six HMM scores. It does not include audio decoding, resampling, or feature
extraction. Therefore it is not end-to-end live inference latency.

### Q134. Are baseline and HMM latency numbers directly comparable?

**Answer:** Not perfectly. HMM latency is timed clip by clip, while baseline prediction is timed
as one batch and divided by clip count. Batching can improve throughput. A fair latency study
would use the same batch size, hardware warm-up, timing boundary, and repeated trials.

### Q135. Is the configured 300 ms latency budget enforced?

**Answer:** No. The setting exists, but the model pipeline does not reject or flag an inference
that exceeds it. Evaluation records scoring latency, yet the budget is not an automated gate.

### Q136. What does the current empty `experiments/` directory imply?

**Answer:** The final evaluator has not written a preserved in-domain and external-test report.
Consequently, no final accuracy, macro F1, test calibration, external generalization, or test
latency claim is currently supported, even though model and calibration artifacts exist.

### Q137. Why must test results not be used to choose hyperparameters?

**Answer:** Repeatedly adapting the model after seeing test results makes the test set part of
development, biasing the estimate upward. Choices such as PCA width, HMM state count, feature
set, calibration method, and threshold belong to validation. A changed choice requires a new
versioned experiment and an untouched final test.

### Q138. What would demonstrate that temporal modeling helps?

**Answer:** On the same frozen splits, the HMM should improve relevant metrics over both
aggregated logistic regression and random forest, ideally across in-domain and external tests.
Confidence intervals or repeated-seed analysis would strengthen the comparison. One favorable
metric on one split is not enough.

### Q139. What if random forest beats the HMM?

**Answer:** Report it honestly. It may mean the summary distribution contains most useful signal,
the HMM assumptions are poor, sequence order adds little, or HMM hyperparameters were not well
selected. The result would refine the research conclusion rather than invalidate the project.

### Q140. What if in-domain performance is much higher than TESS performance?

**Answer:** That is evidence of domain shift. Likely causes include speakers, lexical content,
recording equipment, acting style, and label realization. The correct response is to quantify
the drop, inspect class confusions, and avoid claiming deployment generalization.

---

## Part 10 — Augmentation, reproducibility, and inference consistency

### Q141. What augmentation is implemented?

**Answer:** A deterministic training-only reverberation ablation convolves each waveform with a
simulated room impulse response from RIRS_NOISES. It RMS-matches the wet signal, mixes 35% wet
and 65% dry, prevents clipping, extracts features, and writes a separate augmented cache.

### Q142. How is the room impulse response selected reproducibly?

**Answer:** The code hashes the global seed and clip ID, converts part of the digest to an integer,
and uses modulo indexing into the sorted RIR list. The same clip, seed, and RIR inventory produce
the same selection.

### Q143. Why RMS-match the reverberated signal?

**Answer:** Convolution can change overall energy. Matching wet RMS to dry RMS reduces the chance
that augmentation teaches the model a trivial loudness shift instead of robustness to room
response. It does not make the augmented signal perceptually identical in loudness.

### Q144. Why keep augmented features in a separate directory?

**Answer:** It preserves the unaugmented primary experiment, avoids overwriting original caches,
and makes the ablation explicit. The augmented model is trained under `models/augmented/` while
validation and both test partitions remain original.

### Q145. What leakage would occur if validation audio were augmented?

**Answer:** If derivatives of validation or test clips entered training, the model would see the
same underlying speech content and speaker during fitting. Even evaluating only the original
version would no longer be independent.

### Q146. Has the augmentation ablation been run?

**Answer:** The code and command are implemented, but the inspected repository does not contain a
`models/augmented/` result or a versioned experiment report. It must be described as implemented
but unrun.

### Q147. How is reproducibility supported?

**Answer:** The repository records the seed, manifest hash, feature contract, split document,
Git commit, dependency versions, training counts, reduction method, PCA width, HMM settings, and
scoring rule. Deterministic tests cover key algorithms. Reproducibility is improved, though exact
bitwise results can still vary across platforms and library implementations.

### Q148. What does seed 42 control?

**Answer:** It controls speaker shuffling, PCA/HMM and classifier randomness where supported,
random forest construction, Dirichlet surrogate draws, and deterministic RIR selection. It does
not eliminate every possible source of platform or parallelism variation.

### Q149. How does inference preserve the training contract?

**Answer:** `PerceptionPipeline.from_trained()` loads the saved reducer and HMMs, compares model
metadata's feature contract with current settings, loads compatible calibration artifacts, and
preserves the saved class order. `infer()` resamples to 16 kHz and calls the same feature and
Viterbi implementations used offline.

### Q150. What happens if calibration artifacts are incomplete?

**Answer:** If exactly one of `calibrator.joblib` or `calibration.json` exists, model loading
fails. If both exist, the saved calibration class IDs must match the model's state IDs. This
prevents silently mixing unrelated artifacts.

---

## Part 11 — Critical defense questions

### Q151. What are the three strongest parts of the ML design?

**Answer:** First, speaker-disjoint and external partitions reduce identity leakage. Second, a
single manifest/hash/feature contract connects data, training, calibration, evaluation, and
serving. Third, the hand-written Viterbi decoder is verified against brute force, giving strong
algorithmic evidence rather than only integration evidence.

### Q152. What are the three biggest ML weaknesses?

**Answer:** Final test results are not yet preserved; uncertainty intervals are heuristic rather
than empirically validated; and the acted English datasets do not establish real conversational
or UAE-domain validity. Additional weaknesses include validation reuse, no explicit OOD detector,
and an unrun augmentation ablation.

### Q153. Is 39.23% validation accuracy good?

**Answer:** It is above six-class uniform chance of about 16.67%, but “good” depends on baselines,
class distribution, label reliability, and external performance. Because it is measured on the
calibrator's fitting set, it is not an unbiased final estimate. The correct answer is to avoid a
quality claim until frozen test comparisons exist.

### Q154. Does low validation ECE prove trustworthy confidence?

**Answer:** No. The ECE was computed on the same validation data used to fit calibration, depends
on ten confidence bins, and says nothing by itself about TESS or deployment data. Reliability
must be checked on untouched splits with calibration plots and sufficient bin counts.

### Q155. Could the model learn dataset identity instead of affect?

**Answer:** Yes. Microphone response, sample rate history, room acoustics, lexical content, and
acting style can identify a dataset. Pooling only shared labels reduces one obvious shortcut but
does not remove domain cues. Dataset-stratified analysis and cross-dataset evaluation are needed.

### Q156. Could the model learn speaker identity?

**Answer:** It may encode speaker characteristics during training, but speaker-disjoint
validation and test prevent direct identity memorization from inflating evaluation on those
partitions. With only a few held-out speakers in RAVDESS and two in TESS, uncertainty around
speaker generalization still matters.

### Q157. Why might F0 cause fairness problems?

**Answer:** Baseline pitch varies with anatomy, age, sex-related physiology, speaking style, and
culture. A model may associate demographic pitch ranges with labels. Speaker normalization,
subgroup evaluation, and broader data can help, but each changes the research protocol and must
be validated.

### Q158. Why might acted labels not transfer to natural speech?

**Answer:** Actors may exaggerate prototypical acoustic cues, and fixed phrases reduce lexical and
contextual diversity. Natural emotion can be subtle, mixed, masked, or culturally expressed.
High acted-corpus accuracy may therefore overstate real-world utility.

### Q159. What is a better uncertainty study?

**Answer:** Preserve an untouched test set, generate calibrated probabilities, assess ECE and
Brier with uncertainty estimates, draw reliability diagrams, measure selective risk across a
coverage curve, and evaluate OOD or quality-shift datasets. If intervals are desired, define a
statistically justified target and measure empirical coverage.

### Q160. What is a better hyperparameter-selection protocol?

**Answer:** Predefine a grid for PCA dimensions, HMM state counts, covariance types, feature
ablations, and scoring rules. Use grouped cross-validation by speaker inside development data or
nested validation. Freeze one configuration, refit using permitted development data, then run
the in-domain and external test exactly once.

### Q161. What ablations should be prioritized?

**Answer:** Compare HMM versus both baselines, PCA versus no PCA or alternative widths, Viterbi
versus forward score, features with and without F0/energy/voiced proportion, different HMM state
counts, calibrated versus uncalibrated probabilities, and unaugmented versus training-only
reverberation. Each ablation should change one declared factor.

### Q162. How could the class-conditional generative model be improved?

**Answer:** Possibilities include explicit class priors, state-count tuning, covariance
regularization, mixture emissions, left-to-right versus ergodic transition constraints, forward
likelihood scoring, duration-aware models, or a discriminative sequence baseline. Improvements
must be justified by validation, not assumed.

### Q163. How could data validity be improved?

**Answer:** Collect consented target-domain speech across accents, languages, devices, rooms, and
speakers; use multiple perceptual raters and self-report where appropriate; preserve ambiguity;
pre-register splits; and hold a real deployment-like corpus completely outside development.

### Q164. What code change would most improve artifact safety?

**Answer:** Make the cache loader verify the serialized feature contract and cached sample rate,
not just audio hash and label. Also save a full run configuration and fail when HMM convergence
or expected class coverage is inadequate.

### Q165. What experiment change would most improve evidence quality?

**Answer:** Run and preserve the frozen in-domain and external evaluations without further
tuning, then compare HMM and baselines with per-class results and calibration. Today the largest
evidence gap is not another algorithm; it is the absence of final test artifacts.

---

## Part 12 — Whiteboard and numerical practice

### Q166. A two-second clip is analyzed every 10 ms. Roughly how many frames do you expect?

**Answer:** Approximately `2 / 0.01 = 200` hop positions, with a small difference possible because
centered framing pads boundaries. The tests accept a value within a few frames of 200.

### Q167. If a clip has 150 frames and its raw Viterbi log score is -900, what class score is used?

**Answer:** `-900 / 150 = -6.0` as the average best-path log score per frame.

### Q168. If 80 of 100 predictions remain after abstention and 60 retained predictions are correct, what are coverage and covered accuracy?

**Answer:** Coverage is `80/100 = 0.80`. Covered accuracy is `60/80 = 0.75`. Overall accuracy is a
separate quantity and cannot be reconstructed without knowing rejected predictions.

### Q169. If a class has TP=30, FP=10, and FN=20, what are precision, recall, and F1?

**Answer:** Precision is `30/(30+10)=0.75`. Recall is `30/(30+20)=0.60`. F1 is
`2×0.75×0.60/(0.75+0.60) ≈ 0.667`.

### Q170. Why does subtracting the maximum not change softmax probabilities?

**Answer:** Multiplying every numerator and denominator term by the same positive constant leaves
the ratio unchanged:
`exp(s_k-m)/sum_j exp(s_j-m) = exp(s_k)/sum_j exp(s_j)`.

### Q171. How many aggregate baseline values come from four frame features if seven statistics are used?

**Answer:** `4 × 7 = 28`. For Timbre's 23 features, it is `23 × 7 = 161`.

### Q172. If all six class probabilities are equal, what is each probability?

**Answer:** `1/6 ≈ 0.1667`. A confidence near this value indicates no class preference under a
six-class normalized output, though calibrated models need not produce exact uniformity on
ambiguous data.

### Q173. What should happen if a transition probability is zero?

**Answer:** Its log is negative infinity, so every path using that transition remains impossible.
The decoder must not replace it with a small finite epsilon unless that smoothing choice is
explicitly part of the model.

### Q174. What shape enters one HMM for a 120-frame clip?

**Answer:** Raw extraction produces `(120, 23)`. The saved scaler/PCA transforms it to `(120, 8)`,
which is the observation matrix passed to each three-state HMM.

### Q175. What are the main learned array shapes in one current HMM?

**Answer:** Start probabilities have shape `(3,)`; the transition matrix is `(3, 3)`; means are
`(3, 8)`; and conceptually diagonal variances provide one value for each of 3 states × 8
dimensions. hmmlearn may expose the diagonal covariance through an expanded array representation.

---

## Part 13 — Rapid-fire viva answers

### Q176. Why not randomly split frames?

**Answer:** Frames from the same recording and speaker are strongly correlated; doing so would
create severe leakage and destroy sequence boundaries.

### Q177. Why not randomly split clips without grouping by speaker?

**Answer:** The same speaker could appear in training and test, letting the model exploit speaker
identity rather than transfer to unseen speakers.

### Q178. Why not tune on TESS?

**Answer:** TESS is the external generalization test. Tuning on it converts it into development
data and removes the independent domain-shift estimate.

### Q179. Why not call Viterbi output a likelihood?

**Answer:** It is the probability of the best path, while the true observation likelihood sums
over all paths.

### Q180. Why not call low ECE high accuracy?

**Answer:** ECE measures confidence reliability, not classification correctness. A consistently
uncertain weak classifier can be calibrated.

### Q181. Why not call the intervals credible intervals without qualification?

**Answer:** They come from heuristics or a score-derived surrogate, not a validated posterior over
model parameters and data-generating uncertainty.

### Q182. Why is a manifest hash useful?

**Answer:** It binds models and results to the exact accepted rows and split assignments used in
the experiment.

### Q183. Why save the Git commit and package versions?

**Answer:** Code and dependency changes can alter parsing, features, fitting, serialization, and
numerics even when hyperparameter names look unchanged.

### Q184. Why is a fixed seed not enough for reproducibility?

**Answer:** Library versions, numerical backends, platform, parallelism, file order, and changed
data can still alter results.

### Q185. What result may currently be claimed?

**Answer:** The data pipeline, fitted main/baseline/calibration artifacts, algorithmic tests, and
validation diagnostics exist. Final in-domain and external-test performance may not yet be
claimed because no evaluation report is preserved.

### Q186. What is the safest one-sentence project description?

**Answer:** “Timbre is a classical speech sequence-classification study that estimates a
calibrated distribution over six vocal-affect labels from framewise acoustic features using
training-only PCA and class-specific Gaussian HMMs scored by a hand-written Viterbi decoder.”

---

## Five-minute professor demonstration script

1. **Problem:** “We estimate vocal affect, not a person's true emotion.”
2. **Data:** “RAVDESS and CREMA-D are speaker-disjoint development data; TESS is untouched
   external testing.”
3. **Features:** “Every 10 ms we compute a 23-value vector: 20 MFCCs, RMS, F0, and voiced
   proportion.”
4. **Reduction:** “StandardScaler and eight-component PCA are fitted only on training frames.”
5. **Model:** “We fit one three-state diagonal-Gaussian HMM per class with hmmlearn.”
6. **Original algorithm:** “Our log-space Viterbi decoder computes the best latent path and is
   verified against brute-force enumeration.”
7. **Classification:** “We divide each best-path score by frame count and compare six scores.”
8. **Confidence:** “A validation-fitted multinomial calibrator maps scores to probabilities and a
   validation-selected threshold permits abstention.”
9. **Evaluation:** “We compare against logistic regression and random forest on identical frozen
   splits using accuracy, balanced accuracy, macro F1, ECE, Brier, coverage, and latency.”
10. **Integrity:** “The final test run is not yet preserved, so I will not claim final accuracy.”

## Source-reading order

1. `src/timbre/core/types.py`
2. `src/timbre/utils/config.py`
3. `src/timbre/data/manifest.py`
4. `src/timbre/data/cache.py`
5. `src/timbre/features/extractor.py`
6. `src/timbre/features/aggregation.py`
7. `src/timbre/reduction/reducer.py`
8. `src/timbre/hmm/decoder.py`
9. `scripts/train_baselines.py`
10. `scripts/train.py`
11. `src/timbre/eval/scoring.py`
12. `scripts/calibrate.py`
13. `src/timbre/bayesian/posterior.py`
14. `src/timbre/eval/metrics.py`
15. `scripts/evaluate.py`
16. `src/timbre/features/augmentation.py`
17. `scripts/extract_augmented_features.py`
18. `src/timbre/api/pipeline.py` — ML inference orchestration only

## Final claim checklist

- [ ] I say vocal affect, not true emotion or mental state.
- [ ] I call the project classical ML, not deep learning.
- [ ] I distinguish six output classes from three hidden states per HMM.
- [ ] I distinguish Viterbi best-path score from forward likelihood.
- [ ] I explain why scaling and PCA are fitted only on training frames.
- [ ] I explain speaker-disjoint validation/test and external TESS evaluation.
- [ ] I distinguish calibration from accuracy.
- [ ] I call `mcmc` a Dirichlet surrogate, not genuine MCMC.
- [ ] I label all reported validation numbers as validation diagnostics.
- [ ] I do not claim final test performance until `make test-models` produces an inspected
  experiment artifact.

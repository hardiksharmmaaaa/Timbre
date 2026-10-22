# Timbre Model, Data, and Training Plan

**Status:** Team planning document
**Last reviewed:** 24 September 2026
**Scope:** Speech-emotion perception, local voice-assistant components, datasets, training,
evaluation, and implementation milestones

## 1. Purpose

Timbre is a voice-emotion-aware companion. It analyses the acoustic character of a user's
speech, estimates a probability distribution over vocal-affect classes, and uses that estimate
to adjust the tone of a conversational response.

The system must not claim to know a person's true internal emotional state. Its output should
be described as an uncertain estimate of **vocal affect**, for example:

> The voice is most acoustically consistent with sadness, with low confidence.

The project has two distinct parts:

1. **Research contribution:** feature extraction, dimensionality reduction, HMM training,
   hand-written Viterbi decoding, uncertainty estimation, and evaluation.
2. **Product glue:** voice activity detection, speech-to-text, the conversational language
   model, text-to-speech, the API, and the frontend.

Keeping this boundary clear is important for both research integrity and the final report.

## 2. Recommended End-to-End Architecture

```text
Microphone audio
  -> voice activity detection / utterance segmentation
  -> shared audio preprocessing
  -> two parallel paths:
       A. vocal-affect perception
          -> MFCC + RMS + F0 + voiced proportion
          -> StandardScaler
          -> PCA
          -> one Gaussian HMM per emotion
          -> hand-written Viterbi scoring
          -> calibrated probabilities and uncertainty

       B. speech recognition
          -> Whisper
          -> transcript

  -> transcript + cautious vocal-affect context
  -> local or external LLM
  -> response text
  -> local or external TTS
  -> audio reply + mood graph
```

The classical HMM pipeline remains the main submitted model. Modern pretrained models are
used as application components or clearly labelled experimental baselines; they do not replace
the project's original ML contribution.

## 3. Recommended Open Models and Libraries

| Role | Recommended component | Training required | Intended use |
|---|---|---:|---|
| Voice activity detection | [Silero VAD](https://github.com/snakers4/silero-vad) | No | Detect speech and utterance boundaries |
| Main affect classifier | Our feature/PCA/Gaussian-HMM pipeline | Yes | Primary research model |
| Classical baselines | Scikit-learn logistic regression and random forest | Yes | Establish whether temporal modelling helps |
| Deep affect baseline | [Microsoft WavLM Base+](https://github.com/microsoft/unilm/tree/master/wavlm) | Classification head first | Strong modern comparison |
| Ready-made affect check | [SpeechBrain wav2vec2-IEMOCAP](https://huggingface.co/speechbrain/emotion-recognition-wav2vec2-IEMOCAP) | No initially | Four-class integration sanity check |
| Speech-to-text | [OpenAI Whisper](https://github.com/openai/whisper) | No | Turn microphone speech into chat text |
| Local conversation | [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) | No | Offline conversational response generation |
| Local English TTS | [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) | No | Lightweight local speech synthesis |
| Multilingual CPU TTS alternative | [Piper](https://github.com/OHF-Voice/piper1-gpl) | No | Fast local TTS where a suitable voice exists |

### 3.1 Primary perception model

The primary model should remain:

```text
framewise acoustic features
  -> train-only standardisation
  -> PCA
  -> one Gaussian HMM per emotion
  -> length-normalised Viterbi score per HMM
  -> calibrated class probabilities
```

`hmmlearn` may estimate the Gaussian emission and transition parameters. The Viterbi decoder
must remain hand-written and must not call a decoder library.

### 3.2 Required classical baselines

Train at least these baselines on clip-level summaries of the same acoustic features:

- Multinomial logistic regression
- Random forest

Useful summary statistics include the mean, standard deviation, median, minimum, maximum, and
selected percentiles of each feature over the utterance. The comparison determines whether the
HMM's temporal modelling provides measurable value.

### 3.3 WavLM experiment

Use WavLM as a modern comparison, not the main claimed contribution:

```text
16 kHz waveform
  -> frozen WavLM Base+ encoder
  -> temporal mean and standard-deviation pooling
  -> linear layer or small MLP
  -> emotion probabilities
```

Begin by freezing WavLM and training only the classification head. Fine-tune the upper encoder
layers only if suitable GPU resources and time are available. Record the pretrained checkpoint,
library version, seed, and all fine-tuning choices in the experiment manifest.

The SpeechBrain IEMOCAP checkpoint is useful only as a ready-made four-class check because it
predicts anger, happiness, sadness, and neutrality rather than Timbre's full label set.

### 3.4 Voice assistant components

- **VAD:** Silero VAD can replace the current energy-only utterance-boundary logic. It is
  MIT-licensed and can run locally.
- **STT:** use `base.en` for a small English demo, `small.en` when accuracy is more important,
  or multilingual `small` for Arabic/English and other multilingual use. Transcribe complete
  utterances rather than unrelated two-second fragments.
- **LLM:** use a 4-bit-quantized Qwen3-4B through a local runner when offline operation is a
  requirement. Retain the existing external provider as an optional configuration.
- **TTS:** use Kokoro-82M for lightweight English output. Piper can be evaluated for languages
  or voices not covered adequately by Kokoro, but Piper's current GPL-3.0 licensing implications
  must be reviewed before distribution.

No emotion-labelled data is needed to train Whisper, Qwen, or Kokoro for the first complete
version. They are pretrained glue components.

## 4. Emotion Label Strategy

### 4.1 Recommended main label set

The primary cross-dataset experiment should use six shared labels:

```text
neutral, happy, sad, angry, fearful, disgust
```

These classes are shared by RAVDESS, CREMA-D, and TESS. Use one canonical mapping internally,
regardless of spelling differences such as `fear` versus `fearful`.

| Canonical ID | Canonical label | RAVDESS | CREMA-D | TESS |
|---:|---|---|---|---|
| 1 | neutral | neutral | NEU | neutral |
| 2 | happy | happy | HAP | happy |
| 3 | sad | sad | SAD | sad |
| 4 | angry | angry | ANG | angry |
| 5 | fearful | fearful | FEA | fear |
| 6 | disgust | disgust | DIS | disgust |

### 4.2 Why eight classes should not be the main pooled experiment

RAVDESS contains `calm` and `surprised`, TESS contains `pleasant surprise` but not `calm`, and
CREMA-D contains neither. If all data is pooled into an eight-class model, `calm` becomes strongly
associated with RAVDESS recording characteristics. The model may learn which dataset produced a
clip instead of learning vocal affect.

Keep the existing eight-class RAVDESS-only experiment as an ablation:

```text
neutral, calm, happy, sad, angry, fearful, disgust, surprised
```

Do not add `stressed` as a prediction label without a properly licensed, speech-based stress
dataset. Stress is not interchangeable with fear or anger.

### 4.3 Unknown and abstention behaviour

`unknown` should not be trained as an ordinary ninth emotion. It should be an abstention result
used when, for example:

- speech is too short or mostly silent;
- audio quality is below an agreed threshold;
- the calibrated maximum probability is too low;
- the input is far outside the training distribution;
- multiple classes are nearly tied.

The confidence threshold must be selected on validation data, not chosen after inspecting the
test results.

## 5. Dataset Plan

### 5.1 RAVDESS

**Role:** initial implementation, eight-class experiment, and part of six-class development.

- 1,440 audio-only speech clips
- 24 professional actors
- North American English
- Eight emotion classes
- Two fixed spoken statements
- Approximately 215 MB for the audio-speech archive
- Official source: [RAVDESS on Zenodo](https://zenodo.org/records/1188976)
- License: CC BY-NC-SA 4.0; separate permission is required for commercial use

The six-class subset contains 1,056 clips after removing calm and surprise. RAVDESS is clean,
small, acted, and lexically narrow. It is useful for development but insufficient evidence of
real conversational performance.

### 5.2 CREMA-D

**Role:** main six-class training corpus.

- 7,442 clips
- 91 actors
- Six emotions: anger, disgust, fear, happiness, neutral, and sadness
- 12 spoken sentences
- Multiple intended intensity levels
- Includes crowd-sourced perceptual ratings
- Official source: [CREMA-D repository](https://github.com/CheyneyComputerScience/CREMA-D)
- License: Open Database License for the database and Database Contents License for contents

Use the `AudioWAV` directory. The repository uses Git LFS; downloading the ordinary GitHub ZIP
can produce small LFS pointer files rather than the audio. Follow the official clone instructions
or implement a verified dataset-specific retrieval process.

Where possible, retain the crowd-rating information, not just the intended filename label. It
can support an experiment that excludes low-agreement recordings or represents label ambiguity.

### 5.3 TESS

**Role:** external-domain evaluation and a supplementary experiment.

- 2,800 clips
- Two actresses
- Seven emotions
- 200 target words in the carrier phrase “Say the word ...”
- Official source: [Toronto Emotional Speech Set](https://doi.org/10.5683/SP2/E8H2MF)

The six shared classes provide 2,400 clips after excluding pleasant surprise. Because TESS has
only two speakers, it is not suitable by itself for a meaningful three-way speaker-independent
train/validation/test split. Preserve it primarily as an external test corpus.

### 5.4 IEMOCAP

**Role:** optional conversational and four-class evaluation.

- Approximately 12 hours of audiovisual sessions
- 10 actors in five dyadic sessions
- Scripted and improvised interactions
- Includes transcripts
- Includes categorical emotion and dimensional valence/activation/dominance labels
- Access: [official USC IEMOCAP site](https://sail.usc.edu/iemocap/)

IEMOCAP normally requires an access request. A common evaluation protocol uses angry, happy or
excited, sad, and neutral. Any label merging must be declared explicitly.

### 5.5 MSP-Podcast

**Role:** long-term naturalistic model and dimensional-affect research.

- Naturalistic podcast speech
- Version 2.0 reports 264,705 speaking turns and approximately 409 hours
- Categorical emotions plus activation, dominance, and valence
- Speaker-independent official partitions
- Information and access: [official MSP-Podcast page](https://lab-msp.com/MSP/MSP-Podcast.html)

This corpus is the strongest long-term route toward real-world speech, but it is much larger than
needed for the first complete system. Access and redistribution depend on the agreement for the
specific released version; the team must save and review those terms before downloading.

### 5.6 Noise and reverberation data

**OpenSLR RIRS_NOISES**

- Real and simulated room impulse responses plus noise
- 16 kHz, 16-bit audio
- Apache-2.0
- Source: [OpenSLR SLR28](https://us.openslr.org/28/)

**MUSAN**

- Music, speech, and environmental noise
- Useful for background-noise and competing-speech augmentation
- Source: [OpenSLR SLR17](https://www.openslr.org/17/)
- Contains source-level licensing metadata that must be preserved

Apply augmentation to training audio only. Never produce augmented versions of validation or
test clips and then allow their originals into training.

### 5.7 Personal and native-language evaluation set

Create a small, consented external test set that resembles the application environment:

- At least 5–10 speakers for an initial study; more is preferable
- Different accents, genders, microphones, rooms, and noise conditions
- Fixed statements plus spontaneous responses lasting roughly 5–15 seconds
- Ideally 20–30 examples per target class across the collection
- Anonymous speaker IDs rather than names in manifests
- Explicit consent, retention, and deletion rules
- Self-reported valence/arousal plus perceived vocal-affect ratings where feasible
- Ideally at least three independent perceptual raters per clip

Keep this corpus out of training initially. It is the final test of whether laboratory results
transfer to the actual use case. Repeatedly tuning against it would turn it into development data.

## 6. Dataset Quantities for the First Main Experiment

Before validation and quality filtering:

| Corpus | Six-class clips |
|---|---:|
| RAVDESS | 1,056 |
| CREMA-D | 7,442 |
| **Development total** | **8,498** |
| TESS external test | 2,400 |

The exact retained total must be recorded after corrupt, silent, duplicate, or invalid clips are
removed. Never silently change the reported dataset size.

## 7. Data Manifest and Storage Contract

Every processed clip should have a manifest row containing at least:

```text
clip_id
dataset
relative_audio_path
speaker_id
canonical_label
original_label
sample_rate
duration_seconds
split
intensity_if_available
annotation_agreement_if_available
source_license
source_url
audio_sha256
```

Recommended storage layout:

```text
data/
  raw/
    RAVDESS/
    CREMA-D/
    TESS/
    IEMOCAP/             # optional and access-controlled
    augmentation/
  manifests/
    all_clips.csv
    exclusions.csv
  processed/
    RAVDESS/*.npz
    CREMA-D/*.npz
    TESS/*.npz
  splits/
    six_class_v1.json
    ravdess_eight_class_v1.json
```

Raw audio, processed data, model weights, and experiments remain gitignored. Manifests may be
committed only when they contain no restricted paths, personal data, or license-prohibited
metadata.

### 7.1 Local download inventory

The first-release datasets were downloaded from their official sources on 12 September 2026.
This table records the reproducibility metadata only; all archives and extracted audio remain
under `data/raw/` and must not be committed.

| Dataset | Intended use | Local contents | Verification |
|---|---|---:|---|
| RAVDESS | Development and eight-class ablation | 1,440 WAV files, 24 actors | Zenodo archive MD5 `bc696df654c87fed845eb13823edef8a` |
| CREMA-D | Main six-class training corpus | 7,442 WAV files, 91 actors, ratings metadata | Git commit `1658cd342dff90010aa843eaeebd53610a08b1dc`; Git LFS object check passed |
| TESS | Untouched external-domain test | 2,800 WAV files, 2 speakers | Downloaded archive SHA-256 `cf33eed0cb23671af0e7c607eb6cf89685ef6469eb7c51987b2d32cbe081b0b9` |
| RIRS_NOISES | Training-only noise and reverberation augmentation | 61,260 WAV files | Downloaded archive SHA-256 `3b50cfde915b3984738169b4beb341e9f6b8062ae4c2076146c5db71c2c05dc7` |

Local storage layout:

```text
data/raw/RAVDESS/
data/raw/CREMA-D/AudioWAV/
data/raw/CREMA-D/processedResults/
data/raw/TESS/
data/raw/augmentation/RIRS_NOISES/
```

Before audio-quality exclusions, the downloaded corpora provide the planned 8,498 six-class
development clips: 1,056 from RAVDESS after excluding calm and surprise, plus all 7,442 CREMA-D
clips. TESS contributes 2,400 six-class external-test clips after excluding pleasant surprise.
TESS must not be used for training, threshold selection, or hyperparameter tuning. RIRS_NOISES
may be applied only to training partitions after the speaker-disjoint split is frozen.

MUSAN was not downloaded because it is optional. IEMOCAP and MSP-Podcast were not downloaded
because they are optional and access-controlled. A personal/native-language evaluation set must
be collected separately under the consent and retention rules in Section 5.7.

The canonical preparation run on 24 September 2026 screened 10,898 six-class candidates and
retained 10,890. It excluded six files involved in three conflicting-label exact-audio pairs,
one same-label duplicate, and one nearly silent clip. The frozen split contains 5,848 training,
1,318 validation, 1,324 in-domain test, and 2,400 TESS external-test clips. Its manifest SHA-256
is `2e59f4c0a7447516c09dd2fac6564087fe27aca54e809750c8853f3f79545e1a`.

Dataset-specific parsing, integrity screening, per-dataset speaker splitting, canonical feature
caching, baseline/HMM training commands, validation calibration, and test/TESS evaluation are
implemented. Model artifacts and performance results do not exist until those commands are run.

## 8. Preprocessing Plan

### 8.1 Shared audio preparation

For cross-dataset consistency:

1. Decode the original file without overwriting it.
2. Convert the working waveform to mono.
3. Resample to 16 kHz for shared compatibility with WavLM and Whisper.
4. Preserve floating-point amplitude without peak-normalising every clip by default.
5. Record duration, clipping rate, RMS level, and silence proportion.
6. Reject unreadable, empty, severely clipped, or nearly silent files with a documented reason.
7. Cache extracted features once and reuse them for all classical experiments.

### 8.2 Classical acoustic features

The initial frame configuration remains:

- Window: 25 ms
- Hop: 10 ms
- MFCCs: start with 13, then ablate 20
- RMS energy
- Fundamental frequency (F0)
- Voiced proportion

F0 uses deterministic YIN over a configurable 50--500 Hz search range. A separate 50 ms pitch
window provides enough context for lower voices while retaining the shared 10 ms hop grid.

The activity feature is named voiced proportion. It is not syllables or words per second; a
genuine speaking-rate estimator remains a possible later experiment.

Consider adding clearly documented feature ablations:

- MFCC delta and delta-delta
- Spectral centroid and rolloff
- Zero-crossing rate
- Jitter and shimmer, if estimated reliably
- Clip duration and pause statistics

Do not add features solely because they are available. Each addition should have an ablation or
a defensible acoustic motivation.

### 8.3 Training-only augmentation

Candidate augmentations include:

- Background noise at approximately 5–25 dB SNR
- Room reverberation
- Microphone gain changes
- Mild codec or compression simulation
- Small speed perturbations

Use pitch shifts and aggressive time stretching cautiously because pitch and timing are direct
emotion cues. An augmentation should preserve the label it is supposed to teach.

## 9. Split Strategy and Leakage Prevention

All splits must be speaker-disjoint. A speaker must appear in exactly one of train, validation,
or test for a given experiment.

For the first combined experiment:

- Split RAVDESS actors independently into train/validation/test groups.
- Split CREMA-D actors independently into train/validation/test groups.
- Combine corresponding partitions after splitting within each dataset.
- Keep TESS untouched as an external test.
- Keep the personal/native-language corpus untouched as a second external test.

This prevents a split from containing only one dataset and avoids direct speaker leakage.

Additional leakage rules:

- Fit scalers, PCA, LLE, feature selection, and calibration on training/validation data only.
- Never split at frame level.
- Keep every original clip and all of its augmented variants in one partition.
- If duplicate or near-duplicate utterances exist, group them before splitting.
- Do not use test results to choose HMM states, PCA dimensions, thresholds, or features.
- Save the final split manifest and its hash before the final evaluation.

## 10. Training Plan

### Phase A — data foundation

1. Add dataset-specific download/access instructions.
2. Implement separate RAVDESS, CREMA-D, and TESS parsers.
3. Produce a canonical six-class manifest.
4. Run audio integrity and duplicate checks.
5. Create deterministic speaker-disjoint splits.
6. Extract and cache acoustic features.

**Exit criterion:** counts, speakers, classes, exclusions, split overlap checks, and manifest
hashes are reported automatically.

### Phase B — classical baselines

1. Aggregate frame features into clip-level statistics.
2. Train multinomial logistic regression.
3. Train random forest.
4. Select hyperparameters using validation data only.
5. Save metrics and confusion matrices.

**Exit criterion:** reproducible baseline results exist for the same split used by the HMM.

### Phase C — primary HMM model

1. Fit `StandardScaler` on concatenated training frames only.
2. Fit PCA on scaled training frames only.
3. Compare a small, declared set of PCA dimensions.
4. Train one diagonal-covariance Gaussian HMM per emotion.
5. Compare a small set of hidden-state counts, such as 3, 5, and 7.
6. Score each clip using length-normalised hand-written Viterbi scores.
7. Compare Viterbi scores with forward log-likelihood as an ablation.

**Exit criterion:** the hand-written Viterbi decoder matches brute force on tiny test cases and
the end-to-end model beats or meaningfully differs from at least one classical baseline.

### Phase D — probability calibration and uncertainty

Raw softmax values over HMM scores are not automatically calibrated probabilities. The current
heuristic intervals must not be presented as statistically validated credible intervals.

Recommended first approach:

1. Collect per-class HMM scores on the validation set.
2. Fit temperature scaling or multinomial logistic calibration.
3. Measure expected calibration error and multiclass Brier score.
4. Define abstention thresholds on validation data.
5. Use bootstrap intervals for reported aggregate metrics.

If the Bayesian layer is retained, formally state its likelihood, priors, approximation, and
meaning of each returned interval. Otherwise call the output an uncertainty estimate or
confidence band rather than a Bayesian credible interval.

**Exit criterion:** probabilities are demonstrably better calibrated than uncalibrated scores,
or the report honestly states that calibration did not improve.

### Phase E — WavLM baseline

1. Load 16 kHz waveforms using the same speaker-disjoint manifests.
2. Extract frozen WavLM Base+ representations.
3. Train a small classification head.
4. Calibrate it using the same validation protocol.
5. Compare accuracy, macro-F1, calibration, cross-dataset performance, compute, and latency.

**Exit criterion:** the comparison clearly separates pretrained components from original work.

### Phase F — complete local voice loop

1. Use Silero VAD to create complete utterances.
2. Run emotion inference and Whisper transcription in parallel.
3. Pass the transcript and cautious affect summary to the LLM.
4. Generate a reply with Qwen3-4B locally or the configured external provider.
5. Synthesize the response with Kokoro or the configured TTS provider.
6. Return text, audio, affect probabilities, and latency measurements to the frontend.

**Exit criterion:** a user can speak, receive a text and audio response, and view an uncertainty-
aware mood graph without typing.

## 11. Evaluation Plan

Report at least:

- Accuracy
- Balanced accuracy
- Macro precision, recall, and F1
- Per-class precision and recall
- Confusion matrix
- Expected calibration error (ECE)
- Multiclass Brier score
- Coverage and accuracy when abstention is enabled
- Median, p90, and p95 inference latency
- Model artifact size and peak memory where practical

Evaluate in three settings:

1. **In-domain:** unseen RAVDESS/CREMA-D speakers from the same corpora.
2. **Cross-dataset:** TESS, without retraining or threshold changes.
3. **Use-case transfer:** consented personal/native-language recordings.

Required ablations:

- Logistic regression versus random forest versus HMM
- PCA versus LLE, if LLE remains supported
- Different HMM hidden-state counts
- Viterbi versus forward likelihood
- With and without training-only noise/reverb augmentation
- Uncalibrated versus calibrated probabilities
- Classical HMM versus frozen WavLM baseline
- Optional eight-class RAVDESS-only versus six-class shared-label experiment

All results must include the number of clips and speakers evaluated. Avoid reporting a single
headline accuracy without its split and class distribution.

## 12. Current Repository State and Gaps

The six-class data and experiment contract is now implemented:

1. Dataset-specific RAVDESS, CREMA-D, and TESS parsers map shared labels and speaker IDs.
2. Preparation writes a hashed canonical manifest, explicit exclusions, quality measurements,
   CREMA-D voice-vote agreement, and deterministic per-dataset speaker splits.
3. TESS is locked to `external_test`; it cannot enter training or validation through the
   canonical commands.
4. Feature caches retain source hashes and the complete 16 kHz extraction contract.
5. Baseline, pooled HMM, validation calibration, abstention, and test/TESS evaluation commands
   consume the same frozen manifest and reject incompatible artifacts.
6. F0 uses a configurable 50--500 Hz range with a separate 50 ms analysis window. The final
   feature remains voiced proportion rather than a claim of words or syllables per second.

Important remaining gaps are:

1. Full baseline/HMM training, calibration, and evaluation still need to be executed; implemented
   commands are not evidence of model quality.
2. A deterministic RIRS training-only reverberation ablation is implemented but remains unrun;
   broader noise/SNR augmentation remains future work.
3. The browser records emotion chunks, but chat text is still typed; STT is missing.
4. The TTS function is a generic HTTP placeholder rather than a complete provider integration.
5. Heuristic uncertainty intervals are not validated credible intervals; only the fitted class
   probabilities can be evaluated for calibration after training.
6. `ml/data/download.py` remains a ZIP helper and is not the supported CREMA-D Git LFS
   acquisition path.

These gaps must remain visible until their exit criteria have actually passed.

## 13. Reproducibility Requirements

Each experiment must save:

```text
experiment name and timestamp
Git commit hash, when available
random seed
dataset versions and source URLs
dataset license identifiers
raw file or archive hashes
manifest and split hashes
included and excluded clip counts
feature configuration
reducer configuration
model hyperparameters
calibration configuration
dependency versions
hardware information
metrics and generated figures
```

Use the fixed workflow:

```text
download/access -> validate -> manifest -> split -> extract -> train -> calibrate -> evaluate
```

Feature extraction should be cached. Training and evaluation must reuse the cached `.npz` files
instead of independently regenerating features.

## 14. Ethics, Privacy, and Product Language

- Obtain clear consent for every newly recorded voice.
- Define how recordings can be deleted and how long they are retained.
- Avoid storing raw personal recordings when derived features are sufficient.
- Never use affect predictions for diagnosis, employment, discipline, surveillance, or other
  consequential decisions.
- Do not tell users that the system knows how they feel.
- Do not interpret low confidence as a meaningful emotional state.
- Report accent, language, gender, age, dataset, and microphone limitations honestly.
- Make local processing available where practical because voice recordings are sensitive data.

Suitable UI language:

> Vocal-affect estimate — uncertain and based only on acoustic patterns.

Unsuitable UI language:

> You are angry.

## 15. Recommended Immediate Scope

The most achievable first complete stack is:

```text
Silero VAD
+ six-class PCA/HMM/hand-written-Viterbi perception model
+ Whisper base.en or multilingual small
+ Qwen3-4B
+ Kokoro-82M
```

The first data release should use:

```text
Development: RAVDESS six-class subset + CREMA-D
External test: TESS six-class subset
Use-case test: consented personal/native-language recordings
Augmentation: OpenSLR RIRS_NOISES, with MUSAN as optional additional noise
```

This scope is large enough to create a functional and defensible project while remaining small
enough for a student team to implement, train, evaluate, and explain.

## 16. Team Decision Checklist

Before implementation begins, the team should record decisions for the following:

- [ ] Confirm six classes as the primary experiment.
- [ ] Confirm eight-class RAVDESS as an ablation rather than the pooled target.
- [ ] Confirm academic-only versus possible future commercial use.
- [ ] Review and save the exact license terms for every downloaded dataset and model.
- [ ] Choose English-only or multilingual STT.
- [ ] Choose local-only, API-only, or configurable LLM/TTS operation.
- [ ] Confirm available CPU, GPU, memory, and storage resources.
- [ ] Approve the personal-recording consent and retention procedure.
- [ ] Freeze the speaker-disjoint split before final experiments.
- [ ] Agree on primary metrics: macro-F1, balanced accuracy, and ECE.
- [ ] Agree on the maximum perception latency and full voice-loop latency.
- [ ] Decide whether the Bayesian layer will be formalised or renamed as calibrated uncertainty.

## 17. Source Index

- [Timbre engineering design](../DESIGN.md)
- [RAVDESS official record](https://zenodo.org/records/1188976)
- [CREMA-D official repository](https://github.com/CheyneyComputerScience/CREMA-D)
- [TESS official record](https://doi.org/10.5683/SP2/E8H2MF)
- [IEMOCAP official site](https://sail.usc.edu/iemocap/)
- [MSP-Podcast official site](https://lab-msp.com/MSP/MSP-Podcast.html)
- [OpenSLR room impulse response and noise data](https://us.openslr.org/28/)
- [MUSAN on OpenSLR](https://www.openslr.org/17/)
- [Silero VAD](https://github.com/snakers4/silero-vad)
- [WavLM](https://github.com/microsoft/unilm/tree/master/wavlm)
- [SpeechBrain IEMOCAP emotion model](https://huggingface.co/speechbrain/emotion-recognition-wav2vec2-IEMOCAP)
- [Whisper](https://github.com/openai/whisper)
- [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B)
- [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M)
- [Piper](https://github.com/OHF-Voice/piper1-gpl)

# Timbre Dataset Guide

## Purpose

This document explains the data used by Timbre from first principles: what each dataset
contains, which file types are present, how labels are encoded, and how the files move through
the training pipeline.

Timbre estimates **vocal affect** from short speech recordings. A single labelled example is
approximately:

```text
audio recording + speaker ID + intended or perceived vocal-affect label
```

The source datasets contain short utterances, not complete conversations. The planned product
will make one prediction per speaking turn and retain the sequence of predictions as a
conversation timeline.

## Current Local Inventory

The following raw files were verified locally on 20 September 2026:

| Dataset | Role | WAV files | Speakers | Local location |
| --- | --- | ---: | ---: | --- |
| RAVDESS | Development and eight-class ablation | 1,440 | 24 | `data/raw/RAVDESS/` |
| CREMA-D | Main six-class development corpus | 7,442 | 91 | `data/raw/CREMA-D/AudioWAV/` |
| TESS | Untouched external evaluation | 2,800 | 2 | `data/raw/TESS/` |
| RIRS_NOISES | Training-only noise and reverberation augmentation | 61,260 | Not applicable | `data/raw/augmentation/RIRS_NOISES/` |

CREMA-D was retrieved through Git LFS and its local Git LFS object check passes. Raw audio,
processed features, model artifacts, and experiment outputs are ignored by Git and must remain
untracked.

## What Is a WAV File?

A `.wav` file is a container for a sequence of audio measurements. Important properties are:

- **Sample rate:** the number of audio samples stored per second.
- **Bit depth:** the precision of each sample.
- **Channels:** mono contains one channel; stereo contains two.
- **Duration:** the length of the recording.

Representative local files have these formats:

| Dataset | Local audio format |
| --- | --- |
| RAVDESS | 48,000 Hz, 16-bit PCM, mono WAV |
| CREMA-D | 16,000 Hz, 16-bit PCM, mono WAV |
| TESS | 24,414 Hz, 16-bit PCM, mono WAV |

The corpora use different sample rates. Timbre must decode them without modifying the originals,
convert them to mono, and resample working waveforms to one shared rate before extracting
features. The main data plan recommends 16 kHz. The chosen rate must remain identical during
feature extraction, training, evaluation, and live inference.

## Primary Six-Class Label Set

The main cross-dataset experiment uses six labels shared by RAVDESS, CREMA-D, and TESS:

| Canonical label | RAVDESS code | CREMA-D code | TESS filename label |
| --- | --- | --- | --- |
| `neutral` | `01` | `NEU` | `neutral` |
| `happy` | `03` | `HAP` | `happy` |
| `sad` | `04` | `SAD` | `sad` |
| `angry` | `05` | `ANG` | `angry` |
| `fearful` | `06` | `FEA` | `fear` |
| `disgust` | `07` | `DIS` | `disgust` |

RAVDESS `calm` and `surprised`, and TESS `pleasant surprise`, are not part of the primary pooled
experiment because those labels are not shared consistently across all three corpora.

`unknown` is not a training class. It is an abstention result for short, silent, low-quality,
out-of-distribution, tied, or low-confidence input.

## RAVDESS

### Contents

RAVDESS contains clean, acted emotional speech from 24 professional actors. The downloaded
audio-only speech subset contains two fixed statements, two repetitions, normal or strong
intensity, and eight labels.

| RAVDESS code | Label | Local clips |
| ---: | --- | ---: |
| `01` | Neutral | 96 |
| `02` | Calm | 192 |
| `03` | Happy | 192 |
| `04` | Sad | 192 |
| `05` | Angry | 192 |
| `06` | Fearful | 192 |
| `07` | Disgust | 192 |
| `08` | Surprised | 192 |
| | **Total** | **1,440** |

Removing calm and surprise leaves **1,056 six-class clips** for the main experiment.

### Directory Structure

```text
data/raw/RAVDESS/
├── Actor_01/
├── Actor_02/
├── ...
└── Actor_24/
```

### Filename Encoding

Example:

```text
03-01-05-01-02-01-16.wav
```

| Position | Example | Meaning |
| ---: | --- | --- |
| 1 | `03` | Audio-only modality |
| 2 | `01` | Speech rather than song |
| 3 | `05` | Angry emotion label |
| 4 | `01` | Normal intensity |
| 5 | `02` | Second fixed statement |
| 6 | `01` | First repetition |
| 7 | `16` | Actor 16 |

Timbre validates all seven filename fields. It does not depend on a particular archive folder
name to discover RAVDESS speech.

### Use and Limitations

RAVDESS is useful for initial implementation because its labels and speakers are easy to parse,
and its recordings are clean. It is small, acted, English-language, and lexically narrow, so it
cannot establish real conversational or UAE-market performance by itself.

## CREMA-D

### Contents

CREMA-D is the main six-class development corpus. It contains 7,442 recordings from 91 actors,
12 fixed sentences, multiple intended intensities, demographic metadata, and crowd-sourced
perceptual ratings.

| CREMA-D code | Label | Local clips |
| --- | --- | ---: |
| `ANG` | Angry | 1,271 |
| `DIS` | Disgust | 1,271 |
| `FEA` | Fearful | 1,271 |
| `HAP` | Happy | 1,271 |
| `NEU` | Neutral | 1,087 |
| `SAD` | Sad | 1,271 |
| | **Total** | **7,442** |

### Directory Structure and File Types

```text
data/raw/CREMA-D/
├── AudioWAV/                         # 7,442 audio files
├── VideoDemographics.csv             # actor demographics
├── SentenceFilenames.csv             # stimulus and filename mapping
├── finishedEmoResponses.csv          # annotation responses
├── finishedResponses.csv             # annotation responses
├── finishedResponsesWithRepeatWithPractice.csv
├── processedResults/
│   ├── summaryTable.csv              # summarized perceived labels/intensities
│   └── tabulatedVotes.csv            # detailed crowd votes and agreement
├── README.md
└── LICENSE.txt
```

The `.csv` files are comma-separated tables. The `.R` files in the source repository are the
dataset authors' scripts for processing response data; they are not inputs to the Timbre model.

### Filename Encoding

Example:

```text
1022_ITS_ANG_XX.wav
```

| Field | Example | Meaning |
| --- | --- | --- |
| Actor | `1022` | Speaker ID |
| Sentence | `ITS` | One of the fixed sentence codes |
| Emotion | `ANG` | Angry intended label |
| Intensity | `XX` | Default or unspecified intensity code |

Other intensity codes describe low, medium, or high intended intensity.

### Ratings Metadata

CREMA-D distinguishes the emotion an actor was instructed to perform from the emotion perceived
by human raters. Its processed tables include:

- Per-emotion vote counts.
- The majority perceived emotion.
- Number of responses.
- Agreement between raters.
- Voice, face, and multimodal intensity ratings.

This allows Timbre to record label ambiguity, evaluate intended versus perceived labels, and
optionally exclude low-agreement clips. Demographic fields must not be used as prediction
features; they may support responsible subgroup evaluation where licensing and sample sizes
permit it.

### Use and Limitations

All 7,442 recordings are eligible for the planned six-class development pool before quality
screening. CREMA-D provides substantially more speakers than RAVDESS, but it is still acted,
English-language speech rather than natural UAE customer conversations.

## TESS

### Contents

TESS contains 2,800 recordings from two speakers. Each speaker produces 200 target words using
a carrier phrase such as "Say the word ..." across seven emotions.

| TESS label | Meaning | Local clips |
| --- | --- | ---: |
| `angry` | Angry | 400 |
| `disgust` | Disgust | 400 |
| `fear` | Fearful | 400 |
| `happy` | Happy | 400 |
| `neutral` | Neutral | 400 |
| `ps` | Pleasant surprise | 400 |
| `sad` | Sad | 400 |
| | **Total** | **2,800** |

Removing pleasant surprise leaves **2,400 six-class clips**.

### Directory Structure

TESS is stored as a flat collection of WAV files:

```text
data/raw/TESS/
├── OAF_fat_disgust.wav
├── YAF_wire_happy.wav
└── ...
```

### Filename Encoding

Example:

```text
YAF_wire_happy.wav
```

| Field | Meaning |
| --- | --- |
| `YAF` | Younger adult female speaker |
| `wire` | Target word |
| `happy` | Vocal-affect label |

`OAF` identifies the older adult female speaker.

### Use and Limitations

TESS must remain an **untouched external test set**. It must not be used for training,
hyperparameter selection, probability calibration, or threshold selection. Its purpose is to
measure whether a model trained on RAVDESS and CREMA-D transfers to a separate recording corpus.

Because it contains only two speakers, it is not suitable as the primary training corpus or as
an independent three-way speaker-disjoint development split.

## RIRS_NOISES

### Contents

RIRS_NOISES is an augmentation resource, not an emotion dataset. Its 61,260 WAV files include:

- Point-source noises.
- Real room impulse responses.
- Simulated room impulse responses.
- Small, medium, and large room configurations.

Simplified layout:

```text
data/raw/augmentation/RIRS_NOISES/RIRS_NOISES/
├── pointsource_noises/
├── real_rirs_isotropic_noises/
└── simulated_rirs/
    ├── smallroom/
    ├── mediumroom/
    └── largeroom/
```

A room impulse response describes how a room changes sound through echoes and reverberation. It
can be convolved with clean speech to simulate a different room. Noise recordings can be mixed
with speech at controlled signal-to-noise ratios.

### Use and Leakage Rule

RIRS_NOISES may create noisy and reverberant variants of **training recordings only**. An
original clip and all of its augmented variants must remain in the same partition. Validation,
test, and external-test recordings must never receive training augmentation.

## Development and Evaluation Allocation

Before integrity screening:

| Allocation | Corpus | Six-class clips |
| --- | --- | ---: |
| Development | RAVDESS | 1,056 |
| Development | CREMA-D | 7,442 |
| **Development total** | | **8,498** |
| External test | TESS | 2,400 |

The retained totals must be recalculated after corrupt, silent, clipped, duplicate, invalid, or
low-agreement clips are excluded. Exclusions must always include a documented reason.

## Required Canonical Manifest

The raw datasets use different filename rules. Timbre therefore needs one canonical manifest in
which every row describes one clip using the same schema:

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

Example:

```csv
clip_id,dataset,speaker_id,canonical_label,sample_rate,split
rav_0001,RAVDESS,16,angry,48000,train
crema_0001,CREMA-D,1022,angry,16000,train
tess_0001,TESS,YAF,happy,24414,external_test
```

The canonical manifest does not exist yet. Dataset-specific parsers and manifest generation are
the next data-engineering milestone.

## Raw, Processed, Split, Model, and Experiment Files

The complete file lifecycle is:

```text
raw WAV and CSV files
  -> validated canonical manifest
  -> speaker-disjoint split JSON
  -> cached feature NPZ files
  -> trained Joblib artifacts
  -> experiment JSON and figures
```

### Raw Files

Original `.wav`, `.csv`, license, and dataset metadata files under `data/raw/`. They are never
modified and never committed.

### Manifest Files

Planned `.csv` files under `data/manifests/`, including the retained corpus and exclusions with
documented reasons.

### Split Files

`.json` files under `data/splits/` containing deterministic train, validation, test, and external
test clip IDs plus their speaker groups and seed.

### Processed Feature Files

Each retained WAV file becomes one cached `.npz` file under `data/processed/<dataset>/`. An NPZ
is a compressed NumPy archive. The current extractor stores:

```text
features  # frame count x feature count matrix
sr        # processing sample rate
label     # integer emotion label
```

With the current configuration, each frame has 23 values:

```text
20 MFCC values + RMS energy + F0/pitch + voiced proportion
```

Frames use a 25 ms window and a 10 ms hop. Processed features should be extracted once and then
reused for all classical experiments.

### Model Files

Training currently writes:

```text
models/reducer.joblib
models/state_models.joblib
models/meta.json
```

The reducer contains train-only scaling and PCA. The state-model file contains one Gaussian HMM
per emotion class. Final metadata must also record feature settings, dataset versions, manifest
and split hashes, source-code commit, dependency versions, and hyperparameters.

### Experiment Files

Evaluation outputs belong under `experiments/<run>/` and must record metrics, confusion matrices,
configuration, model provenance, clip and speaker counts, and latency results.

## Speaker-Disjoint Splitting

Splits must happen by speaker, never by frame or random clip alone. Every speaker must appear in
exactly one of train, validation, or test for a given experiment.

The combined split process is:

1. Split RAVDESS actors into train, validation, and test groups.
2. Split CREMA-D actors independently into train, validation, and test groups.
3. Combine the corresponding partitions.
4. Keep all TESS clips in `external_test`.
5. Keep every augmented version beside its original training clip.
6. Freeze and hash the split before final evaluation.

This prevents the model from seeing a speaker during training and then appearing accurate merely
because it recognizes the same voice during testing.

## Current Implementation Status

As of 20 September 2026:

- The raw datasets are present.
- RAVDESS filename validation and discovery are implemented.
- CREMA-D and TESS canonical parsers are not implemented.
- The shared manifest does not exist.
- Dataset-wide audio integrity screening is not implemented.
- The multi-dataset speaker-disjoint split is not frozen.
- `data/processed/`, `data/splits/`, `models/`, and `experiments/` contain no generated outputs.
- The executable scripts currently support only the older RAVDESS eight-class route.

The six-class pooled model must not be described as trained until the multi-dataset data
foundation, model artifacts, and evaluation results actually exist.

## Known Dataset Limitations

The current emotion-labelled datasets are primarily acted English speech. They do not establish
performance for:

- Emirati or Gulf Arabic.
- Arabic-English code-switching.
- Common UAE expatriate English accents.
- Natural hotel, airline, banking, telecom, or government-service conversations.
- Real customer-service microphones and background conditions.
- A person's true internal emotional state.

They are appropriate for developing and evaluating the first research pipeline. UAE-market
claims require a separately consented, representative, locally evaluated corpus that initially
remains outside training.

## Next Data Milestone

The next implementation batch should:

1. Add separate RAVDESS, CREMA-D, and TESS parsers.
2. Map all source labels to the six canonical labels.
3. Generate the canonical manifest and exclusions table.
4. Screen audio integrity and duplicates.
5. Freeze speaker-disjoint development splits and untouched external tests.
6. Extract and cache features from the retained clips.

Further architectural and research decisions are documented in
[`docs/MODEL_AND_DATA_PLAN.md`](docs/MODEL_AND_DATA_PLAN.md).

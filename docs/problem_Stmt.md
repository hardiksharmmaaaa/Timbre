# Timbre Problem Statement and UAE Pitch

## Project Summary

Timbre is a voice-aware companion and research prototype that estimates **vocal affect** from
short speech recordings. It analyses acoustic patterns such as pitch, energy, spectral shape,
and voiced proportion, then returns a probability distribution over vocal-affect classes.

Timbre does not claim to know a person's true feelings, intentions, mental health, or honesty.
Its output describes which learned acoustic pattern a speaking turn most closely resembles.

## One-Sentence Problem Statement

> Given a short speech utterance from an unseen speaker, estimate which supported vocal-affect
> pattern it most closely resembles, report calibrated uncertainty, and abstain when the signal
> is insufficient or outside the model's reliable scope.

## The Problem

Text transcripts describe **what** a person said, but they omit much of **how** it was said.
The same words may be delivered with different pitch, energy, rhythm, pauses, and voice quality:

```text
"I'm completely fine." spoken calmly
"I'm completely fine!" spoken loudly with a tense voice
```

Customer-service systems and conversational assistants commonly use the transcript alone. This
can miss acoustic changes that may help a human understand whether a conversation needs more
care, clarification, or escalation. At the same time, emotion-recognition systems can easily
overclaim by presenting an uncertain acoustic classification as a fact about a person.

Timbre addresses both problems by producing cautious, turn-level vocal-affect estimates with
confidence information and explicit limitations.

## Research Problem Statement

The main research question is:

> Can a classical sequence model built from acoustic features, train-only dimensionality
> reduction, Gaussian HMMs, and a hand-written Viterbi decoder distinguish six vocal-affect
> classes for speakers who were not present during training?

Supporting questions include:

1. Does temporal HMM modelling perform better than clip-level logistic-regression and
   random-forest baselines?
2. How well does a model trained on RAVDESS and CREMA-D transfer to untouched TESS recordings?
3. Which classes are most frequently confused?
4. Are reported probabilities calibrated, or is the model overconfident?
5. When should the system abstain and return `unknown`?
6. How do background noise and reverberation affect performance and latency?

## Primary Prediction Task

### Input

A complete short speech utterance, normally one speaking turn ending in a pause.

### Output

A probability distribution over six supported labels:

```text
neutral, happy, sad, angry, fearful, disgust
```

Example:

```json
{
  "neutral": 0.08,
  "happy": 0.05,
  "sad": 0.62,
  "angry": 0.07,
  "fearful": 0.12,
  "disgust": 0.06,
  "most_likely": "sad",
  "confidence": 0.62
}
```

A suitable user-facing statement is:

> This speaking turn is most acoustically consistent with sadness, with moderate confidence.

An unsuitable statement is:

> You are sad.

## Prediction Unit: Utterances, Not Entire Conversations

Timbre should make one prediction for each utterance or speaking turn, not one permanent label
for an entire conversation.

```text
Turn 1 -> sad, 62%
Turn 2 -> neutral, 51%
Turn 3 -> happy, 68%
Turn 4 -> unknown, low confidence
```

In the planned flow, voice activity detection identifies when speech starts and when a meaningful
pause ends the utterance. If someone speaks several grammatical sentences without pausing, those
sentences may form one utterance. If they pause between short phrases, each phrase may become a
separate prediction.

The conversation view preserves these turn-level results as a timeline. It may lightly smooth the
visual trend, but it must not turn the timeline into a claim about the person's lasting emotional
state.

## Technical Approach

```text
microphone audio
  -> voice activity detection and utterance segmentation
  -> mono waveform at the training sample rate
  -> 25 ms frames with a 10 ms hop
  -> MFCC + RMS energy + F0/pitch + voiced proportion
  -> train-only StandardScaler
  -> train-only PCA
  -> one diagonal-covariance Gaussian HMM per vocal-affect class
  -> length-normalized score from the hand-written Viterbi decoder
  -> calibrated class probabilities
  -> confidence, abstention, and turn-level timeline
```

Each emotion-specific HMM models acoustic regimes within that class. At inference time, the same
utterance is scored by all six HMMs. The highest calibrated score becomes the leading vocal-affect
estimate when it passes quality and confidence thresholds.

Speech-to-text is a separate path. It supplies the words to the conversational model, while
Timbre's perception path supplies cautious acoustic context. The transcript and vocal signal may
inform a response, but neither should independently determine a consequential decision.

## Development and Evaluation Data

The planned primary experiment uses:

```text
Development: RAVDESS six-class subset + CREMA-D
External test: TESS six-class subset
Training-only augmentation: RIRS_NOISES
```

Before audio-quality filtering, development contains 8,498 six-class clips. TESS contributes
2,400 six-class external-test clips. Speakers must be disjoint across the development train,
validation, and test partitions. TESS remains untouched until final external evaluation.

Dataset details are documented in [`data.md`](data.md).

## Current Implementation Versus Planned Product

### Implemented

- Frame-level MFCC, RMS energy, F0, and voiced-proportion extraction.
- Train-only standardisation and PCA utilities.
- Gaussian HMM training through `hmmlearn`.
- A hand-written log-space Viterbi decoder with brute-force regression tests.
- Posterior-shaped vocal-affect responses and uncertainty surrogates.
- FastAPI audio inference endpoint and browser microphone capture.
- A fixed-window demonstration interface and mood graph.
- Canonical RAVDESS/CREMA-D/TESS parsing, integrity screening, frozen speaker splits, and caches.
- Pooled baseline/HMM training, validation calibration, test/TESS evaluation, and a separate
  training-only reverberation-ablation workflow.

### Planned or Incomplete

- Fitted pooled six-class baseline/HMM artifacts and measured results.
- Executed validation calibration and empirically measured abstention behaviour.
- Executed training-only augmentation comparison.
- Complete utterance segmentation through voice activity detection.
- Speech-to-text and complete spoken-response wiring.
- UAE Arabic, code-switching, accent, and real service-environment evaluation.
- Production privacy controls, retention policies, monitoring, and local deployment packaging.

No final six-class model performance should be claimed until these items are implemented and
evaluated.

## Success Criteria

The research model succeeds when it:

1. Uses deterministic, speaker-disjoint splits without train/test leakage.
2. Reports macro-F1, balanced accuracy, per-class precision and recall, and confusion matrices.
3. Compares HMM performance with logistic regression and random forest on the same split.
4. Reports ECE and multiclass Brier score before and after calibration.
5. Defines abstention thresholds using validation data only.
6. Evaluates once on untouched TESS without tuning against it.
7. Reports results by clip count, speaker count, dataset, and relevant subgroups.
8. Measures median, p90, and p95 perception latency.
9. Records dataset, split, feature, model, dependency, and code provenance.
10. Uses cautious product language that distinguishes vocal affect from internal emotion.

# UAE Market Positioning

## Recommended Beachhead

The strongest initial UAE use case is a **privacy-conscious vocal coaching and escalation
assistant for hospitality and customer-service teams**.

The first deployment should be a controlled training and quality-assurance pilot using consented
role-play recordings. This is safer and easier to validate than immediately monitoring live
customers or automatically scoring employees.

Potential pilot organisations include:

- Hotel groups and hospitality academies.
- Airline and airport customer-service training teams.
- Government customer-happiness centres.
- Bank and telecom contact-centre learning teams.
- Universities teaching communication and service skills.

## One-Line Pitch

> Timbre helps UAE service teams recognise meaningful changes in vocal tone during customer
> conversations, so trainers and supervisors can identify coaching opportunities and interactions
> that may need human attention.

## Thirty-Second Pitch

> UAE service teams communicate with customers across many languages, accents, and cultural
> backgrounds. Important vocal cues can be missed, while managers cannot manually review every
> interaction. Timbre analyses how each speaking turn sounds, not just the words, and produces
> cautious, explainable vocal-affect signals. It helps trainers and supervisors find conversations
> that may need coaching or human attention, while keeping people in control and supporting
> privacy-conscious local deployment.

## Customer Problem

Service leaders need to maintain interaction quality across many staff members, languages,
channels, and customer situations. Transcript-only tools may miss acoustic changes, while manual
review cannot cover every interaction. Existing emotion tools may also provide opaque scores that
are difficult to trust across accents and recording conditions.

Timbre's proposed role is not to replace human judgment. It is to:

- Surface the relevant speaking turn and timestamp.
- Show a cautious acoustic estimate and confidence.
- Help a trainer prioritise review and coaching.
- Identify changes over the course of an interaction.
- Allow uncertain inputs to return `unknown` rather than forcing a label.

## Market Progression

### Stage 1: Consented Training and Coaching

An employee completes a simulated customer conversation. Timbre shows turn-level changes in
vocal delivery, and a qualified trainer reviews the recording and provides feedback. Timbre is an
additional coaching signal, not an automatic employee-performance score.

### Stage 2: Post-Call Quality Assurance

After local validation, Timbre analyses completed customer-service calls and highlights sections
for human review. It does not automatically penalise staff or make decisions about customers.

### Stage 3: Live Escalation Assistance

An advanced version monitors an active conversation and alerts a supervisor when vocal tension
appears to increase across several turns. Live use requires strong multilingual validation,
consent, secure deployment, careful false-alert thresholds, and mandatory human oversight.

## Differentiation

Emotion detection alone is not a defensible market position. Timbre should differentiate through:

- **Acoustic signal plus transcript:** combine how something sounds with what was said.
- **UAE-local validation:** evaluate Arabic, English, code-switching, accents, and realistic noise.
- **Explainability:** show evidence, confidence, and relevant turns instead of an opaque label.
- **Privacy:** support local or on-premises processing, limited retention, and raw-audio deletion.
- **Human oversight:** prioritise review rather than automatically judging a person.
- **Research transparency:** publish performance and failure modes by language, accent, noise, and
  recording condition.
- **Abstention:** return `unknown` when the model lacks sufficient evidence.

The UAE-local capabilities above are product requirements and positioning goals. They are not
current proof points.

## Pilot Proposal

A responsible first pilot would:

1. Partner with one hospitality, university, or customer-service training team.
2. Record consented role-play conversations with anonymous speaker identifiers.
3. Include English, Arabic, code-switching, representative accents, microphones, and rooms.
4. Obtain independent perceived-vocal-affect ratings instead of relying only on acted intent.
5. Keep the initial local corpus outside model training to measure transfer honestly.
6. Compare Timbre flags with trained human reviewers.
7. Measure macro-F1, calibration, false-alert rate, latency, and reviewer usefulness.
8. Define deletion, retention, access-control, and withdrawal procedures before collection.

## Responsible Positioning

Use these terms:

- Vocal-affect estimate.
- Acoustic pattern.
- Speaking-turn signal.
- Low, moderate, or high confidence.
- Human-review assistance.
- Uncertain or unknown.

Avoid these claims:

- "Timbre knows how you feel."
- "Timbre reads emotions accurately."
- "Timbre detects lies."
- "Timbre diagnoses stress, depression, or another health condition."
- "Timbre objectively scores employee attitude."

## Prohibited or Unsuitable Uses

Timbre must not be used as the deciding signal for:

- Mental-health diagnosis or treatment.
- Lie detection or criminal suspicion.
- Hiring, promotion, discipline, or termination.
- Credit, insurance, pricing, or eligibility decisions.
- Immigration, policing, or security profiling.
- Surveillance without informed, lawful consent.

## UAE Privacy and AI Considerations

Voice recordings and derived inferences require a documented lawful basis, clear purpose,
appropriate consent where required, data minimisation, access controls, retention limits, and
deletion procedures. Deployments must be reviewed against the rules that apply to the entity,
sector, and jurisdiction, including free-zone regimes where relevant.

The product should follow the UAE's responsible-AI principles of fairness, accountability,
transparency, explainability, robustness, safety, human-centred values, and privacy preservation.
Representative local evaluation is necessary before commercial claims are made.

## Market Evidence and Reference Links

- [UAE Ministry of Economy and Tourism: 2024 hotel-sector results](https://www.moet.gov.ae/en/-/h.e.-bin-touq-uae-tourism-generates-aed-45-billion-in-hotel-revenues-in-2024-as-efforts-to-elevate-the-country-s-status-as-the-best-tourism-brand-continue)
- [UAE AI adoption guidance for government services](https://u.ae/-/media/AI-publications/AI-Adoption-Guideline-in-Government-Services-Eng.pdf)
- [UAE AI Ethics Principles and Guidelines](https://ai.gov.ae/wp-content/uploads/2023/05/MOCAI-AI-Ethics-EN.pdf)
- [UAE government overview of data-protection laws](https://u.ae/en/about-the-uae/digital-uae/data/data-protection-laws.)
- [TDRA Digital Customer and Digital Government Service Policy](https://dgov.tdra.gov.ae/en/publications/digital-customer-and-digital-government-service-policy)

## Project References

- [`README.md`](README.md)
- [`DESIGN.md`](DESIGN.md)
- [`MODEL_AND_DATA_PLAN.md`](MODEL_AND_DATA_PLAN.md)
- [`tracker.md`](tracker.md)

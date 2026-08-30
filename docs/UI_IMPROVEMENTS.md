# Timbre landing page — 9/10 improvement brief

## Objective

Move Timbre from an expressive research-demo aesthetic to a credible, distinctive corporate
product experience. The page should still feel voice-native, but clarity, trust, and restraint
must lead the visual system.

## Design principles

1. **Explain the product in one screen.** State the category, input, output, and intended use
   before introducing the metaphor.
2. **One signature motion system.** The spatial voice signal is the hero animation. Supporting
   motion should clarify hierarchy instead of competing with it.
3. **Evidence before superlatives.** Show implementation boundaries, evaluation practices, and
   research caveats. Do not publish placeholder accuracy or latency claims.
4. **Corporate structure, not corporate blandness.** Use clear navigation, proof points, use
   cases, safety information, and documentation paths without losing Timbre's identity.
5. **Uncertainty is a feature.** Confidence and intervals remain visible, and the copy must never
   equate vocal affect with a person's true internal state.

## Improvements and acceptance criteria

### 1. Positioning and navigation

- Add Product, Method, Use cases, Safety, and Research navigation paths.
- Describe Timbre as an uncertainty-aware vocal-affect system above the fold.
- Use one primary CTA (`Try live demo`) and one secondary CTA (`View methodology`).
- Add a compact prototype label so visitors understand product maturity immediately.

**Accepted when:** a first-time visitor can answer what Timbre does, how it works at a high
level, and what it must not be used for without scrolling.

### 2. Corporate trust layer

- Surface verifiable facts: framewise acoustic features, train-only preprocessing,
  speaker-independent splits, hand-written Viterbi, posterior intervals, and explicit external
  LLM separation.
- Add use cases and exclusions.
- Link the page to the repository's architecture and research report.
- Avoid unmeasured performance numbers.

**Accepted when:** technical and non-technical reviewers can distinguish implemented research,
external-service glue, planned evaluation, and known limitations.

### 3. Visual hierarchy

- Keep oversized type in the hero only.
- Reduce section-heading scale and repeated gradient treatments.
- Standardize cards, borders, radii, labels, and button states.
- Use orange as the primary accent; violet appears only in signal visualization depth.
- Increase body-copy contrast while keeping secondary metadata quiet.

**Accepted when:** the eye moves through headline → explanation → evidence → action without
multiple elements competing at the same visual weight.

### 4. Components

- Add a proof strip for key technical boundaries.
- Add a single pipeline overview before detailed stages.
- Add use-case cards with suitability labels.
- Add a research-integrity panel and clearer footer navigation.
- Preserve all IDs required by `frontend/app.js`.

**Accepted when:** the page reads as one reusable component system rather than a sequence of
unrelated visual moments.

### 5. Motion and performance

- Retain the hero's Three.js signal as the signature animation.
- Remove cursor-following glow, magnetic movement, decorative floating loops, and closing-canvas
  animation.
- Use short entrance transitions with small travel distances.
- Pause WebGL rendering while the page is hidden or the hero is offscreen.
- Respect `prefers-reduced-motion` and provide a visually complete static state.

**Accepted when:** motion makes the voice concept clearer, and the page remains calm and usable
on mobile or lower-powered hardware.

### 6. Live demo

- Keep the microphone and current signal as the primary focus.
- Label demo/model state clearly.
- Treat the chart and conversation as secondary inspection tools.
- Keep confidence and interval language visible.

**Accepted when:** visitors understand where to begin and never mistake mock inference for a
trained result.

## Definition of done

- Responsive at desktop, tablet, and mobile breakpoints.
- Keyboard focus states are visible.
- Reduced-motion mode is supported.
- JavaScript syntax checks pass.
- Required frontend IDs remain present.
- `make ci` passes.
- No fabricated research, performance, privacy, or deployment claims are introduced.

## Campaign layer — attention without false spectacle

The corporate foundation remains the source of truth, but the hero can behave like a campaign:

- Use a kinetic headline that cycles through acoustic dimensions such as tone, rhythm, energy,
  and doubt.
- Replace the generic particle sphere with a distinctive 3D voice-knot object that deforms like
  a living signal.
- Add an instrument-style Voice AI scanner to make the product legible before the live demo.
- Label simulated preview values explicitly; only `/api/inference` results may be presented as
  live model output.
- Use a high-contrast signal ticker and scroll telemetry to create urgency.
- Concentrate the aggressive motion above the fold so research, safety, and product information
  remain readable.

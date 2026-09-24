# Hyperframes Composition Brief: Timbre

## Objective
Create a short launch-style brag video for Timbre that demonstrates its real visitor flow while keeping the product’s uncertainty and safety boundary explicit.

## Output
- Composition directory: `brag-output/composition/`
- Rendered video: `brag-output/brag.mp4`
- Format: landscape — 1920x1080
- Duration: 20 seconds

## Source Material
- Project root: `/Users/hardiksharma/Timbre`
- Primary files read: `frontend/index.html`, `frontend/style.css`, `frontend/app.js`, `frontend/favicon.svg`, `src/timbre/api/main.py`, `README.md`
- Product name: Timbre
- Tagline / strongest claim: Voice, in context — uncertainty remains visible.
- Key UI or visual moment to recreate: The live-demo product shell moving from “Ready to listen” to a probability dashboard and considered companion reply.
- Copy that must appear verbatim:
  - `Voice carries tone.`
  - `Listening to the shape of your voice`
  - `A voice is a signal, not a confession.`
  - `Vocal affect ≠ internal state`

## Creative Direction
- Tone preset: polished
- Creative direction: clean, professional research-product film
- Interpretation: Use restrained typography, generous holds, deliberate UI animation, and a premium dark palette. Confidence should come from clarity, not hype.
- Angle: Make vocal context visible as a measured, inspectable probability signal, then show how that context can support a more considered response.
- Hook: An orange signal draws behind “Voice carries tone.”
- Outro / punchline: “A voice is a signal, not a confession.” followed by “Timbre — Voice, in context.”
- Avoid:
  - Generic SaaS language
  - Abstract filler visuals
  - Unrelated visual redesign
  - Any implication that vocal affect reveals a person’s true internal emotional state
  - Presenting simulated demo values as trained-model output

## Visual Identity
- Background: `#050505`
- Text: `#f2f0ec`
- Accent: `#ff5c1a`
- Supporting colors: `#0b0b0c`, `#101011`, `#9b9995`, `#7b61ff`, `#c9ff63`
- Display font: local Manrope if available; otherwise a locally shipped geometric sans with comparable proportions
- Body font: local Manrope if available; DM Mono for labels and instrumentation
- Visual references from the project: four-bar waveform wordmark, orange signal glow, clipped-corner scanner geometry, circular probability ring, service-status dot, black/paper contrast

## Storyboard
Use the storyboard in `brag-output/brag-plan.md` as the creative contract.

Scene summary:
1. Voice carries tone — 4.4s — waveform mark, headline, and signal terms.
2. Begin the session — 5.0s — actual Ready to listen shell, cursor click, listening state.
3. Uncertainty visible — 6.0s — probability dashboard, explicitly simulated values, adapted reply.
4. The boundary — 4.6s — safety line and final Timbre lockup.

## Audio
- Audio role: sparse professional accents over a steady, clean bed
- Audio arc: quiet forward motion, controlled lift at the dashboard, clean final resonance.
- Music: `happy-beats-business-moves-vol-12-by-ende-dot-app.mp3`
- Music treatment: static low-volume bed at 0.24 for the 20s composition; no narration and no ducking.
- Music cue guidance: bundled preset `.agents/skills/brag/assets/music/cues/happy-beats-business-moves-vol-12-by-ende-dot-app.music-cues.json`; major reveal near 8.74s, probability sequence at 9.83s/10.37s/10.93s, final lockup near 17.47s.
- Audio-reactive treatment: subtle; extract precomputed audio bands and map bass/overall energy to orange glow opacity/scale and dashboard depth by only 2–5%.
- Audio-coupled moments:
  - Start control — simulated cursor click and single ripple
  - Dashboard — soft reveal hit
  - Probability rows — restrained beat-grid sequence
  - Final logo — resonant accent
- SFX selection guidance: use only low/medium high-frequency-risk files from the provided analysis; keep repeated UI cues very quiet.
- SFX analysis guidance: `.agents/skills/brag/assets/sfx/sfx-analysis.md`
- Exact SFX choice: Hyperframes may tune timestamps and volume against the implemented animation. Initial candidates are `interface/click_003.ogg`, `impact/impactSoft_medium_001.ogg`, and `impact/impactBell_heavy_000.ogg`.
- Audio files: copy selected music and SFX into `brag-output/composition/assets/`.

## Hyperframes Instructions
Use the current Hyperframes domain skills: `hyperframes-core`, `hyperframes-animation`, `hyperframes-creative`, `hyperframes-keyframes`, and `hyperframes-cli`. `/brag` owns this workflow; do not enter the generic Hyperframes intent interview or generic launch-video workflow.

Requirements:
- Recreate the live product shell and probability dashboard from the source project; do not substitute generic feature cards.
- Keep every text line readable in a 1920x1080 final render.
- Keep the total duration at 20 seconds.
- Use local assets for music, SFX, fonts, and runtime dependencies.
- Include no narration.
- Add a subtle, precomputed audio-reactive treatment to the existing orange glow/dashboard presence; no waveform/equalizer graphics tied to music.
- Beat-lock the dashboard reveal near 8.74s and final logo near 17.47s; mark both in source comments.
- Align probability fills near 9.83s, 10.37s, and 10.93s, then hold the complete set for readability; mark the sequence in source comments.
- Use tasteful, sparse SFX matched to the click, dashboard reveal, and final logo.
- Add a motion sidecar covering hook appearance, click-before-dashboard order, in-frame dashboard, and ongoing ambient motion.
- Run `npx hyperframes check --snapshots` before preview and fix every error.

/* ==========================================================================
   Timbre — emulation orchestration
   Mic capture -> POST audio -> render perception (emotion ring, graph, voice journal)
   ========================================================================== */

const API = "";

/* ---- Emotion theme mapping (icon + accent per state) ---- */
const THEMES = {
  neutral:  { icon: "◌", color: "#a2a8b8" },
  calm:     { icon: "☾", color: "#7c9cff" },
  happy:    { icon: "☀", color: "#fde68a" },
  sad:      { icon: "☁", color: "#7c9cff" },
  angry:    { icon: "⚡", color: "#ff6b6b" },
  fearful:  { icon: "◉", color: "#ffb454" },
  disgust:  { icon: "✕", color: "#b7a3ff" },
  surprised:{ icon: "!",  color: "#4ade80" },
  unknown:  { icon: "◌", color: "#a2a8b8" },
};
const EMOTIONS = Object.keys(THEMES).filter((e) => e !== "unknown");

/* ---- Element refs ---- */
const statusDot = document.getElementById("statusDot");
const statusText = document.getElementById("statusText");
const statusDetail = document.getElementById("statusDetail");
const micBtn = document.getElementById("micBtn");
const micState = document.getElementById("micState");
const micHint = document.getElementById("micHint");
const micBadge = document.getElementById("micBadge");
const micIcon = document.getElementById("micIcon");
const liveBadge = document.getElementById("liveBadge");
const emotionLabel = document.getElementById("emotionLabel");
const emotionIcon = document.getElementById("emotionIcon");
const confidencePct = document.getElementById("confidencePct");
const emotionTop3 = document.getElementById("emotionTop3");
const ringFg = document.getElementById("ringFg");
const intervalEl = document.getElementById("intervalEl");
const turnEl = document.getElementById("turnEl");
const windowEl = document.getElementById("windowEl");
const journalState = document.getElementById("journalState");
const journalPrompt = document.getElementById("journalPrompt");
const promptText = document.getElementById("promptText");
const nextPromptBtn = document.getElementById("nextPromptBtn");
const journalRecordBtn = document.getElementById("journalRecordBtn");
const journalTimer = document.getElementById("journalTimer");
const journalEntriesEl = document.getElementById("journalEntries");
const sessionScreen = document.getElementById("sessionScreen");
const sessionContent = document.getElementById("sessionContent");
const startSessionBtn = document.getElementById("startSessionBtn");
const joinSessionBtn = document.getElementById("joinSessionBtn");
const sessionIdEl = document.getElementById("sessionId");
const controlSessionId = document.getElementById("controlSessionId");
const sessionTimer = document.getElementById("sessionTimer");
const micControlLabel = document.getElementById("micControlLabel");
const endSessionBtn = document.getElementById("endSessionBtn");
const leaveTopBtn = document.getElementById("leaveTopBtn");
const sessionEnded = document.getElementById("sessionEnded");
const endedSessionId = document.getElementById("endedSessionId");
const endedDuration = document.getElementById("endedDuration");
const endedWindows = document.getElementById("endedWindows");
const startNewSessionBtn = document.getElementById("startNewSessionBtn");

/* ---- State ---- */
let recorder = null;
let micStream = null;
let segmentTimer = null;
let lastPosterior = {};
let turn = 0;
let isListening = false;
let isStartingMic = false;
let isJournalRecording = false;
let audioContext = null;
let inferenceSequence = 0;
let lastRenderedInference = 0;
let sessionStarted = false;
let roomCode = "";
let sessionStartedAt = 0;
let sessionTimerHandle = null;
let sessionDurationSeconds = 0;

/* ---- Voice journal state ---- */
let journalRecorder = null;
let journalStream = null;
let journalChunks = [];
let journalRecordHandle = null;
let journalRecordSeconds = 0;
let activePlayback = null;
let promptKey = "";
let promptIndex = 0;

/* ---- Voice Activity Detection ---- */
let vadAnalyser = null;
let vadTimeData = null;
let vadByteTimeData = null;
let vadCheckInterval = null;
const VAD_THRESHOLD = 0.015; // Time-domain RMS above this counts as voice (0.0-1.0)
const VAD_MIN_SPEECH_FRAMES = 3; // Consecutive voiced checks (100ms each) before recording starts
const VAD_SILENCE_FRAMES = 10; // Consecutive silent checks before a recording stops
const VAD_MAX_SEGMENT_MS = 8000; // Hard cap so a long monologue cannot produce one giant upload
let vadSpeechFrames = 0;
let vadSilenceFrames = 0;
let isRecordingSpeech = false;
let vadRecorder = null;

function initVAD(stream) {
  try {
    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    if (!AudioContextClass || !stream) return false;
    audioContext ??= new AudioContextClass();
    vadAnalyser = audioContext.createAnalyser();
    vadAnalyser.fftSize = 2048;
    vadTimeData = typeof vadAnalyser.getFloatTimeDomainData === "function"
      ? new Float32Array(vadAnalyser.fftSize)
      : null;
    vadByteTimeData = vadTimeData ? null : new Uint8Array(vadAnalyser.fftSize);
    audioContext.createMediaStreamSource(stream).connect(vadAnalyser);
    return true;
  } catch (_) {
    vadAnalyser = null;
    return false;
  }
}

function getVADLevel() {
  if (!vadAnalyser) return 0;
  // Time-domain RMS, directly comparable to the backend's silence threshold.
  if (vadTimeData) {
    vadAnalyser.getFloatTimeDomainData(vadTimeData);
    let sum = 0;
    for (let i = 0; i < vadTimeData.length; i += 1) sum += vadTimeData[i] * vadTimeData[i];
    return Math.sqrt(sum / vadTimeData.length);
  }
  vadAnalyser.getByteTimeDomainData(vadByteTimeData);
  let sum = 0;
  for (let i = 0; i < vadByteTimeData.length; i += 1) {
    const v = (vadByteTimeData[i] - 128) / 128;
    sum += v * v;
  }
  return Math.sqrt(sum / vadByteTimeData.length);
}

function startVADMonitoring() {
  if (vadCheckInterval) return;
  vadSpeechFrames = 0;
  vadSilenceFrames = 0;
  isRecordingSpeech = false;

  vadCheckInterval = setInterval(() => {
    const level = getVADLevel();
    const hasVoice = level > VAD_THRESHOLD;
    
    if (hasVoice) {
      vadSpeechFrames++;
      vadSilenceFrames = 0;
      
      if (!isRecordingSpeech && vadSpeechFrames >= VAD_MIN_SPEECH_FRAMES) {
        // Start recording - we have enough speech frames
        isRecordingSpeech = true;
        startVADRecording();
      }
    } else {
      vadSilenceFrames++;
      vadSpeechFrames = 0;
      
      if (isRecordingSpeech && vadSilenceFrames >= VAD_SILENCE_FRAMES) {
        // Stop recording - silence detected
        stopVADRecording();
      }
    }
  }, 100); // Check every 100ms
}

function stopVADMonitoring() {
  if (vadCheckInterval) {
    clearInterval(vadCheckInterval);
    vadCheckInterval = null;
  }
  if (vadRecorder && vadRecorder.state === "recording") {
    vadRecorder.stop();
  }
}

function startVADRecording() {
  if (!micStream || vadRecorder) return;
  // Keep chunks local to this recorder so a queued stop event from a previous
  // segment can never mix audio into the next one.
  const activeRecorder = new MediaRecorder(micStream);
  const chunks = [];
  vadRecorder = activeRecorder;
  activeRecorder.ondataavailable = (e) => { if (e.data.size) chunks.push(e.data); };
  activeRecorder.onstop = () => {
    if (vadRecorder === activeRecorder) vadRecorder = null;
    if (chunks.length) {
      const blob = new Blob(chunks, { type: activeRecorder.mimeType || "audio/webm" });
      sendAudio(blob);
    }
  };
  activeRecorder.onerror = () => {
    if (vadRecorder === activeRecorder) vadRecorder = null;
    isRecordingSpeech = false;
  };
  activeRecorder.start(100); // Collect data every 100ms
  segmentTimer = setTimeout(() => stopVADRecording(), VAD_MAX_SEGMENT_MS);
}

function stopVADRecording() {
  isRecordingSpeech = false;
  clearTimeout(segmentTimer);
  segmentTimer = null;
  if (vadRecorder && vadRecorder.state === "recording") {
    vadRecorder.stop();
  }
}

function createRoomCode() {
  const suffix = Math.random().toString(36).slice(2, 6).toUpperCase().padEnd(4, "0");
  return `TMBR-${suffix}`;
}

function setRoomCode(code = createRoomCode()) {
  roomCode = code;
  sessionIdEl.textContent = code;
  controlSessionId.textContent = code;
}

function formatDuration(totalSeconds) {
  const minutes = Math.floor(totalSeconds / 60).toString().padStart(2, "0");
  const seconds = (totalSeconds % 60).toString().padStart(2, "0");
  return `${minutes}:${seconds}`;
}

function updateSessionTimer() {
  sessionDurationSeconds = Math.max(0, Math.floor((Date.now() - sessionStartedAt) / 1000));
  sessionTimer.textContent = formatDuration(sessionDurationSeconds);
  sessionTimer.setAttribute("datetime", `PT${sessionDurationSeconds}S`);
}

function startSessionTimer() {
  sessionStartedAt = Date.now();
  sessionDurationSeconds = 0;
  updateSessionTimer();
  sessionTimerHandle = setInterval(updateSessionTimer, 1000);
}

function stopSessionTimer() {
  if (sessionStartedAt) updateSessionTimer();
  if (sessionTimerHandle !== null && typeof clearInterval === "function") {
    clearInterval(sessionTimerHandle);
  }
  sessionTimerHandle = null;
}

/* ---- Mood graph ---- */
const colors = ["#f2f0ec", "#9d8cff", "#ff7a3d", "#6e8fff", "#ff4d4d", "#c987ff", "#d98265", "#ffd166"];
const chartData = { labels: [], datasets: [] };
const datasets = {};
let chart = null;

function ensureChart() {
  if (chart) return;
  if (typeof Chart === "undefined") {
    document.getElementById("moodChart").setAttribute("aria-label", "Graph unavailable. Current signal is shown alongside it.");
    return;
  }
  chart = new Chart(document.getElementById("moodChart"), {
    type: "line",
    data: chartData,
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      animation: { duration: 500, easing: "easeOutQuart" },
      plugins: {
        legend: { labels: { color: "#888a82", usePointStyle: true, pointStyle: "circle", boxWidth: 6, padding: 14, font: { family: "DM Mono", size: 9 } } },
        tooltip: {
          backgroundColor: "rgba(12,13,11,0.94)",
          borderColor: "rgba(255,92,26,0.22)",
          borderWidth: 1,
          titleColor: "#eeeade",
          bodyColor: "#9d9f96",
          padding: 10,
          cornerRadius: 8,
        },
      },
      scales: {
        x: { grid: { color: "rgba(238,234,222,0.05)" }, ticks: { color: "#686a63", font: { family: "DM Mono", size: 9 } }, title: { display: false } },
        y: {
          min: 0, max: 1,
          grid: { color: "rgba(238,234,222,0.05)" },
          ticks: { color: "#686a63", font: { family: "DM Mono", size: 9 }, callback: (v) => v },
          title: { display: false },
        },
      },
    },
  });
}

/* ---- Service status ---- */
function setStatus(on) {
  statusDot.classList.toggle("live", on);
  statusText.textContent = on ? "Online" : "Offline";
  windowEl.textContent = `${Math.round((lastWindowMs || 2000) / 1000)}s`;
}
async function pingHealth() {
  try {
    const r = await fetch(`${API}/api/health`);
    if (r.ok) {
      const d = await r.json();
      setStatus(true);
      statusDetail.textContent = d.demo_mode
        ? `v${d.version} · demo mode (mock inference)`
        : `v${d.version} · model loaded`;
      if (d.window_ms) {
        lastWindowMs = d.window_ms;
        windowEl.textContent = `${Math.round(d.window_ms / 1000)}s`;
      }
    } else {
      setStatus(false);
      const error = await r.json().catch(() => null);
      statusDetail.textContent = typeof error?.detail === "string"
        ? error.detail
        : `backend unavailable (HTTP ${r.status})`;
    }
  } catch (_) {
    setStatus(false);
    statusDetail.textContent = "cannot reach backend";
  }
}

/* ---- Perception render ---- */
let lastWindowMs = 2000;
function renderPosterior(data) {
  lastPosterior = data;
  
  // No voice detected: show an idle state and do not count the window.
  if (data.silence_detected) {
    emotionLabel.textContent = "Listening...";
    emotionIcon.textContent = "◌";
    emotionIcon.style.color = "";
    emotionLabel.style.color = "";
    confidencePct.textContent = "—";
    ringFg.style.strokeDashoffset = 327;
    ringFg.style.stroke = "";
    intervalEl.textContent = "± —";
    emotionTop3.textContent = "—";
    emotionTop3.classList.remove("is-low-confidence");
    // Don't increment turn for silence
    return;
  }

  const probs = data.posterior || {};
  const most = data.most_likely || "unknown";
  const theme = THEMES[most] || THEMES.unknown;

  emotionLabel.textContent = most;
  emotionIcon.textContent = theme.icon;
  emotionIcon.style.color = theme.color;
  emotionLabel.style.color = theme.color;
  offerPrompt(most);

  const p = probs[most] ?? 0;
  confidencePct.textContent = `${(p * 100).toFixed(0)}%`;
  ringFg.style.strokeDashoffset = 327 * (1 - p);
  ringFg.style.stroke = theme.color;

  // Show the top runner-up emotions too, so the full posterior is visible
  // even when the same emotion keeps winning the argmax.
  const ranked = EMOTIONS
    .map((e) => ({ name: e, prob: probs[e] ?? 0 }))
    .sort((a, b) => b.prob - a.prob)
    .filter((e) => e.prob > 0)
    .slice(0, 3);
  emotionTop3.textContent = ranked.length
    ? ranked.map((e) => `${e.name} ${(e.prob * 100).toFixed(0)}%`).join("  ·  ")
    : "—";
  emotionTop3.classList.toggle("is-low-confidence", (data.confidence ?? p) < 0.35);

  const iv = (data.intervals && data.intervals[most]) || 0;
  intervalEl.textContent = `± ${(iv * 100).toFixed(1)}%`;

  turn += 1;
  turnEl.textContent = String(turn);

  // Graph
  chartData.labels.push(String(turn));
  EMOTIONS.forEach((e) => {
    if (!datasets[e]) {
      const c = colors[EMOTIONS.indexOf(e)];
      datasets[e] = {
        label: e,
        data: [],
        borderColor: c,
        backgroundColor: c + "0d",
        pointRadius: 1.5,
        pointHoverRadius: 3,
        borderWidth: 1.6,
        fill: false,
        tension: 0.35,
        spanGaps: true,
      };
      chartData.datasets.push(datasets[e]);
    }
    datasets[e].data.push(probs[e] ?? 0);
  });
  ensureChart();
  chart?.update();
}

/* ---- Audio ---- */
async function recordingToWav(blob) {
  // MediaRecorder defaults to WebM or MP4, which the server's WAV decoder cannot read.
  const AudioContextClass = window.AudioContext || window.webkitAudioContext;
  if (!AudioContextClass) throw new Error("This browser cannot convert microphone audio.");
  audioContext ??= new AudioContextClass();
  const audio = await audioContext.decodeAudioData(await blob.arrayBuffer());
  const buffer = new ArrayBuffer(44 + audio.length * 2);
  const view = new DataView(buffer);
  const writeText = (offset, text) => {
    for (let i = 0; i < text.length; i += 1) view.setUint8(offset + i, text.charCodeAt(i));
  };
  writeText(0, "RIFF");
  view.setUint32(4, buffer.byteLength - 8, true);
  writeText(8, "WAVE");
  writeText(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); // Linear PCM.
  view.setUint16(22, 1, true); // Mono, matching feature extraction.
  view.setUint32(24, audio.sampleRate, true);
  view.setUint32(28, audio.sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeText(36, "data");
  view.setUint32(40, audio.length * 2, true);
  const channels = Array.from({ length: audio.numberOfChannels }, (_, i) => audio.getChannelData(i));
  for (let i = 0; i < audio.length; i += 1) {
    const average = channels.reduce((sum, channel) => sum + channel[i], 0) / channels.length;
    const sample = Math.max(-1, Math.min(1, average));
    view.setInt16(44 + i * 2, Math.round(sample * (sample < 0 ? 32768 : 32767)), true);
  }
  return new Blob([buffer], { type: "audio/wav" });
}

async function sendAudio(blob) {
  const sequence = ++inferenceSequence;
  try {
    const fd = new FormData();
    fd.append("file", await recordingToWav(blob), "chunk.wav");
    const r = await fetch(`${API}/api/inference`, { method: "POST", body: fd });
    if (!r.ok) throw new Error(await r.text());
    const data = await r.json();
    if (sequence < lastRenderedInference) return;
    renderPosterior(data);
    lastRenderedInference = sequence;
    if (!isJournalRecording) journalState.textContent = data.silence_detected ? "LISTENING" : "ANALYSED";
  } catch (e) {
    if (sequence >= lastRenderedInference) micHint.textContent = `inference error: ${e.message}`;
  }
}

function recordSegment() {
  if (!isListening || !micStream) return;
  const activeRecorder = new MediaRecorder(micStream);
  const chunks = [];
  recorder = activeRecorder;
  activeRecorder.ondataavailable = (e) => { if (e.data.size) chunks.push(e.data); };
  activeRecorder.onstop = () => {
    // A previous session's queued stop event must not restart the current session.
    if (recorder !== activeRecorder) return;
    if (chunks.length) {
      const blob = new Blob(chunks, { type: activeRecorder.mimeType || "audio/webm" });
      sendAudio(blob);
    }
    recorder = null;
    if (isListening) {
      try {
        recordSegment();
      } catch (error) {
        stopListening(`Microphone recording failed: ${error.message}`);
      }
    }
  };
  activeRecorder.onerror = (event) => {
    if (recorder === activeRecorder) {
      stopListening(`Microphone recording failed: ${event.error?.message || "please try again"}`);
    }
  };
  activeRecorder.start();
  segmentTimer = setTimeout(() => {
    if (activeRecorder.state === "recording") activeRecorder.stop();
  }, lastWindowMs);
}

function stopListening(hint = "Session paused. Tap to resume listening.") {
  isListening = false;
  clearTimeout(segmentTimer);
  segmentTimer = null;
  stopVADMonitoring();
  const activeRecorder = recorder;
  recorder = null;
  if (activeRecorder?.state === "recording") activeRecorder.stop();
  micStream?.getTracks().forEach((track) => track.stop());
  micStream = null;
  micBtn.classList.remove("listening");
  sessionContent.classList.remove("is-listening");
  micBtn.setAttribute("aria-label", "Start listening");
  micBadge.classList.remove("active");
  micIcon.style.color = "";
  liveBadge.textContent = "● PAUSED";
  liveBadge.classList.remove("live-badge");
  micState.textContent = "Ready to listen";
  micState.classList.remove("listening");
  micHint.textContent = hint;
  micControlLabel.textContent = "Mic on";
}

async function toggleMic() {
  if (isStartingMic) return;
  if (isListening) {
    stopListening();
    return;
  }
  isStartingMic = true;
  micBtn.disabled = true;
  try {
    if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === "undefined") {
      throw new Error("Microphone recording requires a supported browser on HTTPS or localhost.");
    }
    micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    isListening = true;
    if (initVAD(micStream)) {
      startVADMonitoring();
    } else {
      // VAD unavailable (e.g. no AnalyserNode): fall back to continuous segments.
      // The backend still gates silent audio via RMS, so no predictions leak in.
      recordSegment();
    }
    micBtn.classList.add("listening");
    sessionContent.classList.add("is-listening");
    micBtn.setAttribute("aria-label", "Pause listening");
    micBadge.classList.add("active");
    micIcon.style.color = "#06231f";
    liveBadge.textContent = "● LIVE";
    liveBadge.classList.add("live-badge");
    micState.textContent = "Listening to the shape of your voice";
    micState.classList.add("listening");
    micHint.textContent = `Speak naturally. Timbre analyses speech segments.`;
    micControlLabel.textContent = "Mute";
    windowEl.textContent = `${Math.round(lastWindowMs / 1000)}s`;
  } catch (e) {
    stopListening(`Could not start microphone: ${e.message}`);
  } finally {
    isStartingMic = false;
    micBtn.disabled = false;
  }
}

/* ---- Session transition ---- */
function startSession() {
  if (sessionStarted) return;
  sessionStarted = true;
  startSessionBtn.disabled = true;
  joinSessionBtn.disabled = true;
  journalRecordBtn.disabled = false;
  journalState.textContent = "READY";
  leaveTopBtn.hidden = false;
  startSessionTimer();
  sessionScreen.classList.add("is-hiding");
  setTimeout(() => {
    sessionScreen.style.display = "none";
    sessionContent.style.display = "block";
    sessionContent.classList.add("is-revealing");
    ensureChart();
  }, 450);
  return toggleMic();
}

function endSession() {
  if (!sessionStarted) return;
  stopSessionTimer();
  stopJournalRecording();
  stopListening("Session ended.");
  sessionStarted = false;
  journalRecordBtn.disabled = true;
  sessionContent.classList.remove("is-revealing", "is-listening");
  sessionContent.style.display = "none";
  leaveTopBtn.hidden = true;
  endedSessionId.textContent = roomCode;
  endedDuration.textContent = formatDuration(sessionDurationSeconds);
  endedWindows.textContent = String(turn);
  sessionEnded.style.display = "grid";
}

function startNewSession() {
  sessionEnded.style.display = "none";
  sessionScreen.style.display = "grid";
  sessionScreen.classList.remove("is-hiding");
  startSessionBtn.disabled = false;
  joinSessionBtn.disabled = false;
  sessionTimer.textContent = "00:00";
  sessionTimer.setAttribute("datetime", "PT0S");
  turn = 0;
  turnEl.textContent = "0";
  lastPosterior = {};
  emotionLabel.textContent = "Awaiting voice";
  emotionLabel.style.color = "";
  emotionIcon.textContent = "◌";
  emotionIcon.style.color = "";
  confidencePct.textContent = "—";
  intervalEl.textContent = "± —";
  ringFg.style.strokeDashoffset = 327;
  chartData.labels.length = 0;
  Object.values(datasets).forEach((dataset) => { dataset.data.length = 0; });
  chart?.update();
  journalEntriesEl.replaceChildren();
  journalState.textContent = "READY";
  journalTimer.hidden = true;
  journalRecordBtn.disabled = true;
  promptKey = "";
  promptIndex = 0;
  setPrompt(IDLE_PROMPT);
  setRoomCode();
}

/* ---- Voice journal ---- */
const IDLE_PROMPT = "Record a short reflection: what brought you to this session, and what do you hope to notice in your voice?";
const JOURNAL_PROMPTS = {
  neutral: [
    "What is steady around you right now? Describe the room, the moment, the pause.",
    "Nothing is pulling at you — what would you like to notice more carefully?",
  ],
  calm: [
    "What is contributing to this calm? Speak slowly and let it settle.",
    "Name three small things that are going well while this ease lasts.",
  ],
  happy: [
    "What is bringing this lightness forward? Stay with it for a moment.",
    "Who would you tell about this right now, and what would you say?",
  ],
  sad: [
    "What weight is present? You do not have to fix it — just describe it.",
    "What would kindness toward yourself sound like out loud today?",
  ],
  angry: [
    "What is pushing against you? Say it plainly, without softening it.",
    "Where do you feel this tension in your body as you speak?",
  ],
  fearful: [
    "What uncertainty is here? Name it as specifically as you can.",
    "What is within your control in the next hour?",
  ],
  disgust: [
    "What sits poorly with you at this moment, and what does it violate?",
    "What boundary is asking to be stated out loud?",
  ],
  surprised: [
    "What caught you off guard, and what did it reveal?",
    "How has your reading of the situation changed in the last minute?",
  ],
};
let currentPromptText = IDLE_PROMPT;

function promptListFor(key) {
  return (key && JOURNAL_PROMPTS[key]) || [IDLE_PROMPT];
}

function setPrompt(text) {
  currentPromptText = text;
  if (promptText) promptText.textContent = text;
}

/* Offer a prompt tied to the latest vocal signal; only swaps when the signal changes. */
function offerPrompt(emotion) {
  if (emotion === promptKey) return;
  promptKey = emotion;
  promptIndex = 0;
  setPrompt(promptListFor(emotion)[0]);
}

/* Manual refresh: cycle prompts within the current signal's list. */
function cyclePrompt() {
  const list = promptListFor(promptKey);
  promptIndex = (promptIndex + 1) % list.length;
  setPrompt(list[promptIndex]);
}

/* ---- Journal recording ---- */
async function toggleJournalRecording() {
  if (isJournalRecording) {
    stopJournalRecording();
    return;
  }
  if (!sessionStarted || journalRecordBtn.disabled) return;
  journalRecordBtn.disabled = true;
  try {
    if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === "undefined") {
      throw new Error("Voice notes require a supported browser on HTTPS or localhost.");
    }
    journalStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const recorder = new MediaRecorder(journalStream);
    journalRecorder = recorder;
    journalChunks = [];
    const mimeType = recorder.mimeType || "audio/webm";
    recorder.ondataavailable = (e) => { if (e.data.size) journalChunks.push(e.data); };
    recorder.onstop = () => {
      journalRecorder = null;
      journalStream?.getTracks().forEach((track) => track.stop());
      journalStream = null;
      if (journalChunks.length) {
        const blob = new Blob(journalChunks, { type: mimeType });
        journalChunks = [];
        addJournalEntry(blob, Math.max(1, journalRecordSeconds));
      } else {
        journalState.textContent = "READY";
      }
    };
    recorder.start(200);
    isJournalRecording = true;
    journalRecordBtn.classList.add("recording");
    journalRecordBtn.setAttribute("aria-label", "Stop reflection recording");
    journalState.textContent = "RECORDING";
    journalTimer.hidden = false;
    journalRecordSeconds = 0;
    journalTimer.textContent = "00:00";
    journalRecordHandle = setInterval(() => {
      journalRecordSeconds += 1;
      journalTimer.textContent = formatDuration(journalRecordSeconds);
    }, 1000);
  } catch (e) {
    journalStream?.getTracks().forEach((track) => track.stop());
    journalStream = null;
    journalState.textContent = "NO MIC";
    promptText.textContent = `Could not start recording: ${e.message}`;
  } finally {
    journalRecordBtn.disabled = !sessionStarted;
  }
}

function stopJournalRecording() {
  if (!isJournalRecording) return;
  isJournalRecording = false;
  clearInterval(journalRecordHandle);
  journalRecordHandle = null;
  journalTimer.hidden = true;
  journalRecordBtn.classList.remove("recording");
  journalRecordBtn.setAttribute("aria-label", "Record reflection");
  if (journalRecorder && journalRecorder.state === "recording") {
    journalRecorder.stop();
  }
  journalRecordBtn.disabled = !sessionStarted;
}

/* ---- Journal entries ---- */
function syntheticPeaks(seed, bars = 44) {
  let s = (Math.floor(seed) % 2147483647) || 7;
  return Array.from({ length: bars }, (_, i) => {
    s = (s * 16807) % 2147483647;
    const noise = s / 2147483647;
    return 0.14 + (noise * 0.62) * (0.55 + 0.45 * Math.abs(Math.sin(i / 3.1)));
  });
}

function peaksFromBuffer(audio) {
  const data = audio.getChannelData(0);
  const bars = 44;
  const step = Math.max(1, Math.floor(data.length / bars));
  const peaks = [];
  for (let i = 0; i < bars; i += 1) {
    let peak = 0;
    for (let j = 0; j < step; j += 1) {
      peak = Math.max(peak, Math.abs(data[i * step + j] || 0));
    }
    peaks.push(Math.max(0.08, Math.min(1, peak)));
  }
  return peaks;
}

function drawEntryWaveform(canvas, peaks) {
  const ctx = canvas?.getContext?.("2d");
  if (!ctx) return;
  const width = 320;
  const height = 28;
  canvas.width = width;
  canvas.height = height;
  const gap = 2;
  const barWidth = Math.max(1.5, (width - gap * (peaks.length - 1)) / peaks.length);
  ctx.fillStyle = "rgba(255, 92, 26, 0.72)";
  peaks.forEach((peak, i) => {
    const barHeight = Math.max(2, peak * height);
    ctx.fillRect(i * (barWidth + gap), (height - barHeight) / 2, barWidth, barHeight);
  });
}

function stopJournalPlayback() {
  if (!activePlayback) return;
  const { audio, entry, button } = activePlayback;
  activePlayback = null;
  try { audio.pause(); } catch (_) { /* already stopped */ }
  try { URL.revokeObjectURL(audio.src); } catch (_) { /* no-op */ }
  entry.classList.remove("playing");
  button.classList.remove("playing");
  button.textContent = "▶";
}

function toggleJournalPlayback(entry, button) {
  if (activePlayback?.entry === entry) {
    stopJournalPlayback();
    return;
  }
  stopJournalPlayback();
  if (typeof Audio === "undefined" || !entry.blob) return;
  const audio = new Audio(URL.createObjectURL(entry.blob));
  activePlayback = { audio, entry, button };
  entry.classList.add("playing");
  button.classList.add("playing");
  button.textContent = "❚❚";
  audio.onended = stopJournalPlayback;
  const played = audio.play();
  if (played?.catch) played.catch(() => stopJournalPlayback());
}

async function addJournalEntry(blob, durationSeconds) {
  const entry = document.createElement("div");
  entry.className = "journal-entry";
  entry.blob = blob;

  const meta = document.createElement("div");
  meta.className = "entry-meta";
  const time = document.createElement("span");
  time.className = "entry-time";
  time.textContent = formatDuration(durationSeconds);
  const button = document.createElement("button");
  button.type = "button";
  button.className = "entry-play";
  button.textContent = "▶";
  button.setAttribute("aria-label", "Play reflection");
  button.addEventListener("click", () => toggleJournalPlayback(entry, button));
  meta.appendChild(time);
  meta.appendChild(button);

  const content = document.createElement("div");
  content.className = "entry-content";
  const label = document.createElement("div");
  label.className = "entry-prompt";
  const signal = (lastPosterior.most_likely || "reflection").toUpperCase();
  label.textContent = `${signal} · ${currentPromptText}`;
  const wave = document.createElement("div");
  wave.className = "entry-waveform";
  const canvas = document.createElement("canvas");
  wave.appendChild(canvas);
  content.appendChild(label);
  content.appendChild(wave);

  entry.appendChild(meta);
  entry.appendChild(content);
  journalEntriesEl.appendChild(entry);
  drawEntryWaveform(canvas, syntheticPeaks(blob.size + durationSeconds));

  try {
    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    if (AudioContextClass) {
      audioContext ??= new AudioContextClass();
      const buffer = await audioContext.decodeAudioData(await blob.arrayBuffer());
      drawEntryWaveform(canvas, peaksFromBuffer(buffer));
    }
  } catch (_) {
    /* synthetic waveform already drawn */
  }

  journalState.textContent = "SAVED";
  journalEntriesEl.scrollTop = journalEntriesEl.scrollHeight;
}

/* ---- Wire up ---- */
startSessionBtn.addEventListener("click", startSession);
joinSessionBtn.addEventListener("click", startSession);
micBtn.addEventListener("click", toggleMic);
endSessionBtn.addEventListener("click", endSession);
leaveTopBtn.addEventListener("click", endSession);
startNewSessionBtn.addEventListener("click", startNewSession);
journalRecordBtn.addEventListener("click", toggleJournalRecording);
nextPromptBtn.addEventListener("click", cyclePrompt);
journalRecordBtn.disabled = true;
setRoomCode();
setPrompt(IDLE_PROMPT);
pingHealth();
setInterval(pingHealth, 5000);
window.addEventListener?.("beforeunload", () => {
  if (isListening) stopListening("Session closed.");
  stopSessionTimer();
});

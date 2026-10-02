/* ==========================================================================
   Timbre — emulation orchestration
   Mic capture -> POST audio -> render perception (emotion ring, graph, chat)
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
const ringFg = document.getElementById("ringFg");
const intervalEl = document.getElementById("intervalEl");
const turnEl = document.getElementById("turnEl");
const windowEl = document.getElementById("windowEl");
const logEl = document.getElementById("log");
let chatEmpty = document.getElementById("chatEmpty");
const chatForm = document.getElementById("chatForm");
const userText = document.getElementById("userText");
const sendBtn = document.getElementById("sendBtn");
const chatState = document.getElementById("chatState");
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
let isSendingChat = false;
let audioContext = null;
let inferenceSequence = 0;
let lastRenderedInference = 0;
let sessionStarted = false;
let roomCode = "";
let sessionStartedAt = 0;
let sessionTimerHandle = null;
let sessionDurationSeconds = 0;

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
  const probs = data.posterior || {};
  const most = data.most_likely || "unknown";
  const theme = THEMES[most] || THEMES.unknown;

  emotionLabel.textContent = most;
  emotionIcon.textContent = theme.icon;
  emotionIcon.style.color = theme.color;
  emotionLabel.style.color = theme.color;

  const p = probs[most] ?? 0;
  confidencePct.textContent = `${(p * 100).toFixed(0)}%`;
  ringFg.style.strokeDashoffset = 327 * (1 - p);
  ringFg.style.stroke = theme.color;

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
    if (!isSendingChat) chatState.textContent = "analysed";
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
    recordSegment();
    micBtn.classList.add("listening");
    sessionContent.classList.add("is-listening");
    micBtn.setAttribute("aria-label", "Pause listening");
    micBadge.classList.add("active");
    micIcon.style.color = "#06231f";
    liveBadge.textContent = "● LIVE";
    liveBadge.classList.add("live-badge");
    micState.textContent = "Listening to the shape of your voice";
    micState.classList.add("listening");
    micHint.textContent = `Speak naturally. Timbre samples a ${Math.round(lastWindowMs / 1000)}-second window.`;
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
  leaveTopBtn.hidden = false;
  startSessionTimer();
  sessionScreen.classList.add("is-hiding");
  setTimeout(() => {
    sessionScreen.style.display = "none";
    sessionContent.style.display = "block";
    sessionContent.classList.add("is-revealing");
  }, 450);
  return toggleMic();
}

function endSession() {
  if (!sessionStarted) return;
  stopSessionTimer();
  stopListening("Session ended.");
  sessionStarted = false;
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
  logEl.replaceChildren();
  chatEmpty = document.createElement("div");
  chatEmpty.className = "chat-empty";
  chatEmpty.id = "chatEmpty";
  const emptyIcon = document.createElement("span");
  emptyIcon.textContent = "◎";
  const emptyCopy = document.createElement("p");
  emptyCopy.textContent = "Speak, inspect the signal, then write what is on your mind.";
  chatEmpty.appendChild(emptyIcon);
  chatEmpty.appendChild(emptyCopy);
  logEl.appendChild(chatEmpty);
  userText.value = "";
  sendBtn.disabled = true;
  chatState.textContent = "idle";
  setRoomCode();
}

/* ---- Chat ---- */
function addMsg(sender, text) {
  chatEmpty?.remove();
  const row = document.createElement("div");
  row.className = `msg ${sender}`;
  const avatar = document.createElement("span");
  avatar.className = "msg-avatar";
  avatar.textContent = sender === "user" ? "Y" : "T";
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;
  row.appendChild(avatar);
  row.appendChild(bubble);
  logEl.appendChild(row);
  logEl.scrollTop = logEl.scrollHeight;
}

async function sendChat(e) {
  e.preventDefault();
  const text = userText.value.trim();
  if (!text || isSendingChat) return;
  isSendingChat = true;
  addMsg("user", text);
  userText.value = "";
  sendBtn.disabled = true;
  chatState.textContent = "thinking…";
  try {
    const r = await fetch(`${API}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, posterior: lastPosterior.posterior, intervals: lastPosterior.intervals, most_likely: lastPosterior.most_likely }),
    });
    const data = await r.json();
    if (!r.ok) {
      throw new Error(typeof data.detail === "string" ? data.detail : `Request failed (${r.status})`);
    }
    addMsg("bot", data.reply || "No reply was returned. Please try again.");
  } catch (err) {
    addMsg("bot", `error: ${err.message}`);
  } finally {
    isSendingChat = false;
    sendBtn.disabled = !userText.value.trim();
    chatState.textContent = "idle";
  }
}

/* ---- Wire up ---- */
startSessionBtn.addEventListener("click", startSession);
joinSessionBtn.addEventListener("click", startSession);
micBtn.addEventListener("click", toggleMic);
endSessionBtn.addEventListener("click", endSession);
leaveTopBtn.addEventListener("click", endSession);
startNewSessionBtn.addEventListener("click", startNewSession);
chatForm.addEventListener("submit", sendChat);
userText.addEventListener("input", () => { sendBtn.disabled = isSendingChat || !userText.value.trim(); });
sendBtn.disabled = true;
setRoomCode();
pingHealth();
setInterval(pingHealth, 5000);
window.addEventListener?.("beforeunload", () => {
  if (isListening) stopListening("Session closed.");
  stopSessionTimer();
});

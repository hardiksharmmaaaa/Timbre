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
const chatEmpty = document.getElementById("chatEmpty");
const chatForm = document.getElementById("chatForm");
const userText = document.getElementById("userText");
const sendBtn = document.getElementById("sendBtn");
const chatState = document.getElementById("chatState");

/* ---- State ---- */
let recorder = null;
let micStream = null;
let segmentTimer = null;
let lastPosterior = {};
let turn = 0;
let isListening = false;
let live = false;

/* ---- Mood graph ---- */
const colors = ["#f2f0ec", "#9d8cff", "#ff7a3d", "#6e8fff", "#ff4d4d", "#c987ff", "#d98265", "#ffd166"];
const chartData = { labels: [], datasets: [] };
const datasets = {};
let chart = null;

function ensureChart() {
  if (chart) return;
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
  live = on;
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
    } else setStatus(false);
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
  chatState.textContent = "analysing";

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
  chart.update();
}

/* ---- Audio ---- */
async function sendAudio(blob) {
  const fd = new FormData();
  fd.append("file", blob, "chunk.webm");
  try {
    const r = await fetch(`${API}/api/inference`, { method: "POST", body: fd });
    if (!r.ok) throw new Error(await r.text());
    renderPosterior(await r.json());
    chatState.textContent = "analysed";
  } catch (e) {
    micHint.textContent = `inference error: ${e.message}`;
  }
}

function recordSegment() {
  if (!isListening || !micStream) return;
  const activeRecorder = new MediaRecorder(micStream);
  const chunks = [];
  recorder = activeRecorder;
  activeRecorder.ondataavailable = (e) => { if (e.data.size) chunks.push(e.data); };
  activeRecorder.onstop = () => {
    if (chunks.length) {
      const blob = new Blob(chunks, { type: activeRecorder.mimeType || "audio/webm" });
      sendAudio(blob);
    }
    if (recorder === activeRecorder) recorder = null;
    if (isListening) recordSegment();
  };
  activeRecorder.start();
  segmentTimer = setTimeout(() => {
    if (activeRecorder.state === "recording") activeRecorder.stop();
  }, lastWindowMs);
}

async function toggleMic() {
  if (isListening) {
    isListening = false;
    clearTimeout(segmentTimer);
    if (recorder?.state === "recording") recorder.stop();
    recorder = null;
    micStream?.getTracks().forEach((track) => track.stop());
    micStream = null;
    micBtn.classList.remove("listening");
    micBadge.classList.remove("active");
    micIcon.style.color = "";
    liveBadge.textContent = "● PAUSED";
    liveBadge.classList.remove("live-badge");
    micState.textContent = "Ready to listen";
    micState.classList.remove("listening");
    micHint.textContent = "Session paused. Tap to resume listening.";
    setStatus(live); // keep service status as-is (health-based)
    return;
  }
  try {
    micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    isListening = true;
    recordSegment();
    micBtn.classList.add("listening");
    micBadge.classList.add("active");
    micIcon.style.color = "#06231f";
    liveBadge.textContent = "● LIVE";
    liveBadge.classList.add("live-badge");
    micState.textContent = "Listening to the shape of your voice";
    micState.classList.add("listening");
    micHint.textContent = `Speak naturally. Timbre samples a ${Math.round(lastWindowMs / 1000)}-second window.`;
    windowEl.textContent = `${Math.round(lastWindowMs / 1000)}s`;
  } catch (e) {
    micHint.textContent = `Mic access denied: ${e.message}`;
  }
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
  if (!text) return;
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
    addMsg("bot", data.reply || "(no reply — LLM API key not configured)");
    chatState.textContent = "idle";
  } catch (err) {
    addMsg("bot", `error: ${err.message}`);
    chatState.textContent = "idle";
  }
}

/* ---- Wire up ---- */
micBtn.addEventListener("click", toggleMic);
chatForm.addEventListener("submit", sendChat);
userText.addEventListener("input", () => { sendBtn.disabled = !userText.value.trim(); });
sendBtn.disabled = true;
pingHealth();
setInterval(pingHealth, 5000);

/* Exercise the shipped browser scripts with deterministic DOM/media boundaries. */
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const { test } = require("node:test");

const frontend = path.join(__dirname, "..", "frontend");

function deferred() {
  let resolve;
  const promise = new Promise((done) => { resolve = done; });
  return { promise, resolve };
}

function element() {
  const classes = new Set();
  return {
    textContent: "", value: "", disabled: false, style: {}, children: [], handlers: {},
    classList: {
      add: (name) => classes.add(name), remove: (name) => classes.delete(name),
      toggle: (name, on) => on ? classes.add(name) : classes.delete(name),
    },
    addEventListener(name, handler) { this.handlers[name] = handler; },
    setAttribute(name, value) { this[name] = value; },
    appendChild(child) { this.children.push(child); },
    remove() {},
    getContext() { return null; },
  };
}

function loadApp(options = {}) {
  const html = fs.readFileSync(path.join(frontend, "index.html"), "utf8");
  const elements = Object.fromEntries([...html.matchAll(/id="([^"]+)"/g)]
    .map((match) => [match[1], element()]));
  const timers = new Map();
  const recorders = [];
  let timerId = 0;
  class Recorder {
    constructor(stream) {
      this.stream = stream;
      this.state = "inactive";
      this.mimeType = "audio/webm";
      recorders.push(this);
      if (options.recorderFailure) throw new Error("Recorder unavailable");
    }
    start() { this.state = "recording"; }
    stop() { this.state = "inactive"; }
  }
  const streams = [];
  const getUserMedia = options.getUserMedia || (async () => {
    const track = { stopped: false, stop() { this.stopped = true; } };
    const stream = { getTracks: () => [track] };
    streams.push(stream);
    return stream;
  });
  const fetch = async (url, args) => {
    if (url === "/api/health") {
      return { ok: true, json: async () => ({ version: "test", demo_mode: true }) };
    }
    return options.fetch(url, args);
  };
  const context = vm.createContext({
    document: {
      getElementById: (id) => elements[id], createElement: element,
      querySelector: () => null,
    },
    navigator: { mediaDevices: { getUserMedia } }, MediaRecorder: Recorder,
    fetch, Blob, FormData,
    window: { AudioContext: class {
      async decodeAudioData() {
        return {
          length: 4, numberOfChannels: 1, sampleRate: 16000,
          getChannelData: () => new Float32Array([1, -1, 0, 0.5]),
        };
      }
    } },
    setTimeout: (callback) => { timers.set(++timerId, callback); return timerId; },
    clearTimeout: (id) => timers.delete(id), setInterval() {},
    ...(options.chart === false ? {} : { Chart: class { update() {} } }),
  });
  vm.runInContext(fs.readFileSync(path.join(frontend, "app.js"), "utf8"), context);
  return { elements, context, recorders, streams, timers };
}

function submit(app, text) {
  app.elements.userText.value = text;
  app.elements.userText.handlers.input();
  return app.elements.chatForm.handlers.submit({ preventDefault() {} });
}

test("microphone permission requests cannot overlap on repeated clicks", async () => {
  const permission = deferred();
  let requests = 0;
  const app = loadApp({ getUserMedia: () => { requests += 1; return permission.promise; } });
  const first = app.elements.micBtn.handlers.click();
  const second = app.elements.micBtn.handlers.click();
  permission.resolve({ getTracks: () => [] });
  await Promise.all([first, second]);
  assert.equal(requests, 1);
  assert.equal(app.recorders.length, 1);
});

test("Start voice session requests the microphone as promised by the page", async () => {
  const app = loadApp();
  await app.elements.startSessionBtn.handlers.click();
  assert.equal(app.streams.length, 1);
  assert.equal(app.recorders[0].state, "recording");
});

test("recorder startup failure releases the microphone and resets listening", async () => {
  const app = loadApp({ recorderFailure: true });
  await app.elements.micBtn.handlers.click();
  assert.equal(app.streams[0].getTracks()[0].stopped, true);
  assert.equal(vm.runInContext("isListening", app.context), false);
});

test("a stopped recorder cannot restart capture after the microphone resumes", async () => {
  const app = loadApp();
  await app.elements.micBtn.handlers.click();
  const previous = app.recorders[0];
  await app.elements.micBtn.handlers.click();
  await app.elements.micBtn.handlers.click();
  previous.onstop();
  assert.equal(app.recorders.length, 2);
  assert.equal(app.recorders[1].state, "recording");
});

test("chat displays an HTTP failure instead of claiming the API key is absent", async () => {
  const app = loadApp({ fetch: async () => ({
    ok: false, status: 503,
    json: async () => ({ detail: "Companion temporarily unavailable" }),
  }) });
  await submit(app, "hello");
  const last = app.elements.log.children.at(-1).children.at(-1).textContent;
  assert.match(last, /Companion temporarily unavailable/);
  assert.doesNotMatch(last, /API key not configured/);
});

test("typing and submitting during a pending reply cannot send duplicate chat requests", async () => {
  const response = deferred();
  let requests = 0;
  const app = loadApp({ fetch: () => { requests += 1; return response.promise; } });
  const first = submit(app, "hello");
  app.elements.userText.value = "next message";
  app.elements.userText.handlers.input();
  const disabledWhileSending = app.elements.sendBtn.disabled;
  const duplicate = app.elements.chatForm.handlers.submit({ preventDefault() {} });
  response.resolve({ ok: true, json: async () => ({ reply: "hello back" }) });
  await Promise.all([first, duplicate]);
  assert.equal(disabledWhileSending, true);
  assert.equal(requests, 1);
  assert.equal(app.elements.userText.value, "next message");
  assert.equal(app.elements.sendBtn.disabled, false);
});

test("a blocked chart CDN does not turn successful inference into an error", async () => {
  const app = loadApp({ chart: false, fetch: async () => ({
    ok: true,
    json: async () => ({ posterior: { calm: 1 }, most_likely: "calm", intervals: {} }),
  }) });
  await vm.runInContext("sendAudio(new Blob(['audio']))", app.context);
  assert.equal(app.elements.emotionLabel.textContent, "calm");
  assert.equal(app.elements.chatState.textContent, "analysed");
  assert.doesNotMatch(app.elements.micHint.textContent, /error/);
});

test("recorded browser audio is uploaded as PCM WAV for the backend decoder", async () => {
  let uploaded;
  const app = loadApp({ fetch: async (url, args) => {
    uploaded = args.body.get("file");
    return {
      ok: true,
      json: async () => ({ posterior: { calm: 1 }, most_likely: "calm", intervals: {} }),
    };
  } });
  await vm.runInContext("sendAudio(new Blob(['compressed recording']))", app.context);
  assert.equal(uploaded.name, "chunk.wav");
  assert.equal(uploaded.type, "audio/wav");
  const bytes = Buffer.from(await uploaded.arrayBuffer());
  assert.equal(bytes.toString("ascii", 0, 4), "RIFF");
  assert.equal(bytes.toString("ascii", 8, 12), "WAVE");
  assert.equal(bytes.readUInt32LE(24), 16000);
  assert.equal(bytes.readUInt16LE(22), 1);
  assert.equal(bytes.readUInt16LE(34), 16);
  assert.equal(bytes.readInt16LE(44), 32767);
  assert.equal(bytes.readInt16LE(46), -32768);
});

test("the landing script tolerates unavailable canvas contexts", () => {
  const window = {
    matchMedia: () => ({ matches: true }), setInterval() {}, addEventListener() {},
  };
  const context = vm.createContext({
    window, devicePixelRatio: 1, requestAnimationFrame() {},
    document: { getElementById: () => element(), querySelector: () => null },
    IntersectionObserver: class { observe() {} },
  });
  assert.doesNotThrow(() => {
    vm.runInContext(fs.readFileSync(path.join(frontend, "landing.js"), "utf8"), context);
  });
});

test("late inference responses cannot overwrite a more recent vocal signal", async () => {
  const responses = [deferred(), deferred()];
  let requests = 0;
  const app = loadApp({ fetch: () => responses[requests++].promise });
  const earlier = vm.runInContext("sendAudio(new Blob(['earlier']))", app.context);
  await new Promise((resolve) => setImmediate(resolve));
  const later = vm.runInContext("sendAudio(new Blob(['later']))", app.context);
  responses[1].resolve({
    ok: true, json: async () => ({ posterior: { calm: 1 }, most_likely: "calm" }),
  });
  await later;
  responses[0].resolve({
    ok: true, json: async () => ({ posterior: { sad: 1 }, most_likely: "sad" }),
  });
  await earlier;
  assert.equal(app.elements.emotionLabel.textContent, "calm");
});

test("unavailable WebGL does not crash the landing page", () => {
  const context = vm.createContext({
    window: {
      matchMedia: () => ({ matches: false }), setInterval() {}, addEventListener() {},
      THREE: {
        Scene: class {}, PerspectiveCamera: class { position = {}; },
        WebGLRenderer: class { constructor() { throw new Error("WebGL unavailable"); } },
      },
    },
    innerWidth: 800, innerHeight: 600, devicePixelRatio: 1,
    document: {
      getElementById: () => element(), createElement: element, querySelector: () => null,
    },
  });
  assert.doesNotThrow(() => {
    vm.runInContext(fs.readFileSync(path.join(frontend, "landing.js"), "utf8"), context);
  });
});

test("health failures replace stale model details with the current service error", async () => {
  const app = loadApp();
  await vm.runInContext("pingHealth()", app.context);
  app.context.fetch = async () => ({
    ok: false, status: 503,
    json: async () => ({ detail: "Perception model is unavailable" }),
  });
  await vm.runInContext("pingHealth()", app.context);
  assert.equal(app.elements.statusText.textContent, "Offline");
  assert.equal(app.elements.statusDetail.textContent, "Perception model is unavailable");
});

test("a non-JSON health failure still replaces stale model details", async () => {
  const app = loadApp();
  await vm.runInContext("pingHealth()", app.context);
  app.context.fetch = async () => ({
    ok: false, status: 503, json: async () => { throw new SyntaxError("not JSON"); },
  });
  await vm.runInContext("pingHealth()", app.context);
  assert.equal(app.elements.statusText.textContent, "Offline");
  assert.match(app.elements.statusDetail.textContent, /unavailable.*503/i);
  assert.doesNotMatch(app.elements.statusDetail.textContent, /model loaded|demo mode/i);
});

test("demo mode remains visibly labeled after successful mock inference", async () => {
  const app = loadApp({ fetch: async () => ({
    ok: true,
    json: async () => ({ posterior: { calm: 1 }, most_likely: "calm", _demo: true }),
  }) });
  await vm.runInContext("pingHealth()", app.context);
  await vm.runInContext("sendAudio(new Blob(['audio']))", app.context);
  assert.equal(app.elements.emotionLabel.textContent, "calm");
  assert.match(app.elements.statusDetail.textContent, /demo mode \(mock inference\)/);
});

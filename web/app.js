/**
 * app.js — Avin Web Client
 *
 * Core responsibilities:
 *   1. Settings: load/save backend URL + API key to localStorage.
 *   2. Recording pipeline:
 *        getUserMedia → AudioContext → ScriptProcessorNode (downsample to 16 kHz,
 *        convert float32 → int16) → WebSocket binary frames → backend.
 *   3. WebSocket message handler: render transcript / note / action / error cards.
 *   4. Action confirmation: Confirm / Dismiss buttons call PATCH /api/v1/actions/{id}.
 *   5. History: GET /api/v1/notes and render as cards.
 *
 * --- Why ScriptProcessorNode instead of AudioWorklet? ---
 * AudioWorklet is the modern API, but it requires a separate JS module file served
 * over HTTPS (or localhost). Since this is a zero-build static app and the user
 * might open it as a local file:// URL during development, we use ScriptProcessorNode
 * (deprecated but universally supported) as the MVP choice. A comment marks the spot
 * where an AudioWorklet upgrade would go. bufferSize 4096 keeps CPU comfortable on
 * mobile while keeping latency under ~100 ms at 44.1 kHz.
 *
 * --- 16 kHz / mono / int16 downsampling ---
 * The browser's AudioContext typically runs at 44 100 Hz or 48 000 Hz (device-
 * dependent). The backend expects 16 000 Hz mono int16 PCM. We downsample by:
 *   1. Collapsing stereo channels to mono via averaging (left + right) / 2.
 *   2. Applying a naive decimation: pick every N-th sample where
 *      N = Math.round(inputSampleRate / 16000). This simple approach is sufficient
 *      for speech (cut-off ~6 kHz) given that the browser already applies a low-pass
 *      filter to prevent aliasing when the source sample rate is much higher.
 *   3. Clamping the float32 [-1, 1] values to int16 range [-32768, 32767] by
 *      multiplying by 32767 and rounding, then packing into an Int16Array which
 *      is sent as a binary WebSocket frame.
 *
 * --- Auth ---
 * REST calls include the header "X-API-Key: <key>" when an API key is configured.
 * WebSocket opens with the query param "?api_key=<key>" per the backend's WS auth
 * spec (browsers cannot set custom headers on WebSocket upgrades).
 * When no key is configured both mechanisms are omitted and the backend runs in
 * auth-disabled mode (its default for local dev).
 *
 * --- Guardrail: no auto-confirm ---
 * Action cards with needs_confirmation=true ALWAYS show Confirm + Dismiss buttons.
 * The Confirm button calls PATCH /api/v1/actions/{id} with {"status":"confirmed"}.
 * This module never calls that endpoint automatically. The user must tap.
 */

"use strict";

// ─── Constants ────────────────────────────────────────────────────────────────

const LS_KEY_BACKEND_URL = "avin_backend_url";
const LS_KEY_API_KEY     = "avin_api_key";
const TARGET_SAMPLE_RATE = 16000;   // Hz — backend expects this
const SCRIPT_PROC_BUFFER = 4096;    // ScriptProcessorNode buffer size (samples)

// ─── State ────────────────────────────────────────────────────────────────────

let audioCtx         = null;   // AudioContext
let mediaStream      = null;   // getUserMedia stream
let scriptProcessor  = null;   // ScriptProcessorNode
let sourceNode       = null;   // MediaStreamAudioSourceNode
let ws               = null;   // WebSocket
let isRecording      = false;
let lastLevelUpdate  = 0;      // throttle for the mic-level meter

// ─── DOM refs (populated on DOMContentLoaded) ─────────────────────────────────

let btnRecord, statusLine, feed, btnHistory;
let settingsPanel, btnSettingsToggle;
let inputBackendUrl, inputApiKey, btnSaveSettings;

// ─── Settings helpers ─────────────────────────────────────────────────────────

function loadSettings() {
  inputBackendUrl.value = localStorage.getItem(LS_KEY_BACKEND_URL) || "";
  inputApiKey.value     = localStorage.getItem(LS_KEY_API_KEY)     || "";
}

function saveSettings() {
  const url = inputBackendUrl.value.trim().replace(/\/$/, ""); // strip trailing /
  const key = inputApiKey.value.trim();
  localStorage.setItem(LS_KEY_BACKEND_URL, url);
  localStorage.setItem(LS_KEY_API_KEY,     key);
  setStatus("Settings saved.", "");
  settingsPanel.classList.remove("open");
}

function getBackendUrl() {
  return (localStorage.getItem(LS_KEY_BACKEND_URL) || "").replace(/\/$/, "");
}

function getApiKey() {
  return localStorage.getItem(LS_KEY_API_KEY) || "";
}

/** Build the WebSocket URL from the REST backend URL.
 *  https://foo.run.app  → wss://foo.run.app
 *  http://localhost:8000 → ws://localhost:8000
 */
function toWsUrl(httpUrl) {
  return httpUrl.replace(/^https:\/\//, "wss://").replace(/^http:\/\//, "ws://");
}

/** Return headers object with X-API-Key when a key is configured. */
function authHeaders() {
  const key = getApiKey();
  return key ? { "X-API-Key": key } : {};
}

// ─── Status line ──────────────────────────────────────────────────────────────

/**
 * @param {string} msg      - Message to display.
 * @param {string} [cls=""] - CSS class: "error" | "active" | "warning" | "".
 */
function setStatus(msg, cls = "") {
  statusLine.textContent  = msg;
  statusLine.className    = "status-line " + cls;
}

// ─── Feed / card rendering ────────────────────────────────────────────────────

function clearEmptyNotice() {
  const empty = feed.querySelector(".feed-empty");
  if (empty) empty.remove();
}

/**
 * Prepend a card to the results feed.
 * @param {string} type       - Card type: "transcript" | "note" | "action" | "error-card" | "history"
 * @param {string} typeLabel  - Human-readable label in the header pill.
 * @param {Node|string} body  - Card body content (HTML string or DOM node).
 * @returns {HTMLElement} The created card element.
 */
function prependCard(type, typeLabel, body) {
  clearEmptyNotice();

  const card = document.createElement("div");
  card.className = `card ${type}`;

  const typeEl = document.createElement("div");
  typeEl.className = "card-type";
  typeEl.textContent = typeLabel;
  card.appendChild(typeEl);

  const bodyEl = document.createElement("div");
  bodyEl.className = "card-body";
  if (typeof body === "string") {
    bodyEl.textContent = body;
  } else {
    bodyEl.appendChild(body);
  }
  card.appendChild(bodyEl);

  feed.insertBefore(card, feed.firstChild);
  return card;
}

/** Render a transcript message. */
function renderTranscript(msg) {
  prependCard("transcript", "Transcript", msg.text || "(empty)");
}

/** Render a note message. */
function renderNote(msg) {
  const frag = document.createDocumentFragment();
  const p = document.createElement("p");
  p.textContent = msg.summary || "(no summary)";
  frag.appendChild(p);
  if (msg.note_id) {
    const meta = document.createElement("div");
    meta.className = "card-meta";
    meta.textContent = `note_id: ${msg.note_id}`;
    frag.appendChild(meta);
  }
  prependCard("note", "Note", frag);
}

/**
 * Render an action message.
 * For needs_confirmation=true: show Confirm + Dismiss buttons.
 * GUARDRAIL: Never auto-confirm. User must tap.
 */
function renderAction(msg) {
  const frag = document.createDocumentFragment();

  const intentEl = document.createElement("p");
  intentEl.textContent = msg.intent || "unknown intent";
  frag.appendChild(intentEl);

  // Show structured details as JSON snippet
  if (msg.details && Object.keys(msg.details).length > 0) {
    const details = document.createElement("div");
    details.className = "details";
    details.textContent = JSON.stringify(msg.details, null, 2);
    frag.appendChild(details);
  }

  const card = prependCard("action", "Action", frag);

  if (msg.needs_confirmation) {
    const btnRow = document.createElement("div");
    btnRow.className = "action-btns";

    const btnConfirm = document.createElement("button");
    btnConfirm.className = "btn-confirm";
    btnConfirm.textContent = "Confirm";

    const btnDismiss = document.createElement("button");
    btnDismiss.className = "btn-dismiss";
    btnDismiss.textContent = "Dismiss";

    btnRow.appendChild(btnConfirm);
    btnRow.appendChild(btnDismiss);
    card.appendChild(btnRow);

    // Confirm: PATCH /api/v1/actions/{id} with {"status":"confirmed"}
    btnConfirm.addEventListener("click", () =>
      handleActionDecision(msg.action_id, "confirmed", btnRow, card)
    );

    // Dismiss: PATCH /api/v1/actions/{id} with {"status":"dismissed"}
    btnDismiss.addEventListener("click", () =>
      handleActionDecision(msg.action_id, "dismissed", btnRow, card)
    );
  }
}

/** Render an error message prominently. */
function renderError(msg) {
  prependCard("error-card", "Error", msg.message || "Unknown error");
  setStatus("Error received from server.", "error");
}

/**
 * Handle Confirm / Dismiss tap on an action card.
 * Calls PATCH /api/v1/actions/{action_id} with the chosen status.
 * @param {string} actionId
 * @param {"confirmed"|"dismissed"} status
 * @param {HTMLElement} btnRow  - The button row to replace with the badge.
 * @param {HTMLElement} card    - The card element (for error display).
 */
async function handleActionDecision(actionId, status, btnRow, card) {
  const backendUrl = getBackendUrl();
  if (!backendUrl) {
    showInlineError(card, "Backend URL not configured in Settings.");
    return;
  }

  // Disable buttons immediately to prevent double-tap
  btnRow.querySelectorAll("button").forEach(b => { b.disabled = true; });

  try {
    const resp = await fetch(`${backendUrl}/api/v1/actions/${actionId}`, {
      method: "PATCH",
      headers: {
        "Content-Type": "application/json",
        ...authHeaders(),
      },
      body: JSON.stringify({ status }),
    });

    if (!resp.ok) {
      const errText = await resp.text();
      throw new Error(`HTTP ${resp.status}: ${errText}`);
    }

    // Replace the button row with a status badge
    const badge = document.createElement("span");
    badge.className = `action-status-badge ${status}`;
    badge.textContent = status === "confirmed" ? "Confirmed" : "Dismissed";
    btnRow.replaceWith(badge);

  } catch (err) {
    showInlineError(card, `Failed to update action: ${err.message}`);
    // Re-enable buttons so the user can retry
    btnRow.querySelectorAll("button").forEach(b => { b.disabled = false; });
  }
}

/** Show a small inline error message inside a card. */
function showInlineError(card, message) {
  // Remove any existing inline error first
  const existing = card.querySelector(".inline-error");
  if (existing) existing.remove();

  const el = document.createElement("div");
  el.className = "inline-error";
  el.style.cssText = "color: var(--danger); font-size: 0.82rem; margin-top: 8px;";
  el.textContent = message;
  card.appendChild(el);
}

// ─── WebSocket message router ─────────────────────────────────────────────────

/** Dispatch an incoming JSON message frame to the correct renderer. */
function handleWsMessage(event) {
  let msg;
  try {
    msg = JSON.parse(event.data);
  } catch {
    console.error("Non-JSON WebSocket message:", event.data);
    return;
  }

  switch (msg.type) {
    case "transcript": renderTranscript(msg); break;
    case "note":       renderNote(msg);       break;
    case "action":     renderAction(msg);     break;
    case "error":      renderError(msg);      break;
    default:
      console.warn("Unknown message type:", msg.type, msg);
  }
}

// ─── Audio pipeline ───────────────────────────────────────────────────────────

/**
 * Downsample a mono Float32Array from inputSampleRate to TARGET_SAMPLE_RATE
 * and pack the result as an Int16Array.
 *
 * Algorithm: naive decimation — pick every N-th sample where
 *   N = inputSampleRate / TARGET_SAMPLE_RATE
 * A simple linear interpolation blends the nearest two samples when N is
 * non-integer, giving slightly better quality than floor/round alone.
 *
 * @param {Float32Array} inputBuffer
 * @param {number} inputSampleRate
 * @returns {Int16Array}
 */
function downsampleToInt16(inputBuffer, inputSampleRate) {
  if (inputSampleRate === TARGET_SAMPLE_RATE) {
    // No resampling needed — just convert float32 → int16
    return float32ToInt16(inputBuffer);
  }

  const ratio   = inputSampleRate / TARGET_SAMPLE_RATE;
  const outLen  = Math.floor(inputBuffer.length / ratio);
  const out     = new Int16Array(outLen);

  for (let i = 0; i < outLen; i++) {
    const srcPos  = i * ratio;
    const srcIdx  = Math.floor(srcPos);
    const frac    = srcPos - srcIdx;

    const s0 = inputBuffer[srcIdx]     ?? 0;
    const s1 = inputBuffer[srcIdx + 1] ?? 0;

    // Linear interpolation between adjacent samples
    const sample = s0 + frac * (s1 - s0);

    // Clamp and convert to int16
    out[i] = Math.max(-32768, Math.min(32767, Math.round(sample * 32767)));
  }

  return out;
}

/**
 * Convert a Float32Array to Int16Array without resampling.
 * Values are clamped to [-32768, 32767].
 * @param {Float32Array} buffer
 * @returns {Int16Array}
 */
function float32ToInt16(buffer) {
  const out = new Int16Array(buffer.length);
  for (let i = 0; i < buffer.length; i++) {
    out[i] = Math.max(-32768, Math.min(32767, Math.round(buffer[i] * 32767)));
  }
  return out;
}

/**
 * Start recording: open mic → connect AudioContext → open WebSocket.
 * Uses ScriptProcessorNode for broad browser compatibility (see module header).
 *
 * TODO (AudioWorklet upgrade path): Replace ScriptProcessorNode with an
 *   AudioWorklet that loads a separate "processor.js" worklet module. This
 *   would remove the main-thread audio callback, improving performance on
 *   low-end devices. The downsampling math is identical.
 */
async function startRecording() {
  const backendUrl = getBackendUrl();
  if (!backendUrl) {
    setStatus("Configure Backend URL in Settings first.", "error");
    return;
  }

  setStatus("Requesting microphone access…", "warning");

  // 1. Get microphone stream
  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
  } catch (err) {
    const msg = err.name === "NotAllowedError"
      ? "Microphone permission denied. Allow access in browser settings."
      : `Microphone error: ${err.message}`;
    setStatus(msg, "error");
    return;
  }

  // 2. Open AudioContext
  // We let the browser choose its native sample rate and downsample in JS.
  audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  // iOS Safari starts the AudioContext in a "suspended" state — it must be
  // resumed within the user-gesture (the tap that called startRecording) or the
  // ScriptProcessorNode callback never fires and only silence is streamed.
  if (audioCtx.state === "suspended") {
    try {
      await audioCtx.resume();
    } catch (err) {
      console.warn("AudioContext.resume() failed:", err);
    }
  }
  const inputSampleRate = audioCtx.sampleRate; // typically 44100 or 48000
  console.log(`AudioContext sample rate: ${inputSampleRate} Hz, state: ${audioCtx.state}`);

  // 3. Build API key query param for WebSocket URL (browsers can't set headers on WS)
  const apiKey  = getApiKey();
  const wsBase  = toWsUrl(backendUrl);
  const wsUrl   = apiKey
    ? `${wsBase}/api/v1/stream?api_key=${encodeURIComponent(apiKey)}`
    : `${wsBase}/api/v1/stream`;

  setStatus("Connecting to backend…", "warning");

  // 4. Open WebSocket before starting audio capture so no frames are lost
  try {
    ws = new WebSocket(wsUrl);
    ws.binaryType = "arraybuffer";
  } catch (err) {
    setStatus(`WebSocket error: ${err.message}`, "error");
    stopRecording(false);
    return;
  }

  ws.onopen = () => {
    setStatus("🔴 Listening…", "active");
    isRecording = true;
    btnRecord.classList.add("recording");
  };

  ws.onmessage = handleWsMessage;

  ws.onerror = (event) => {
    console.error("WebSocket error:", event);
    setStatus("WebSocket error — check backend URL and connectivity.", "error");
  };

  ws.onclose = (event) => {
    console.log(`WebSocket closed: code=${event.code} reason=${event.reason}`);
    if (isRecording) {
      // Unexpected close — the user didn't press Stop
      setStatus(`Connection closed (code ${event.code}).`, "warning");
      stopRecording(false); // clean up audio pipeline without sending stop frame
    }
  };

  // 5. Wire up ScriptProcessorNode to capture and downsample audio
  // UPGRADE NOTE: Replace with AudioWorklet here when upgrading beyond MVP.
  sourceNode      = audioCtx.createMediaStreamSource(mediaStream);
  scriptProcessor = audioCtx.createScriptProcessor(SCRIPT_PROC_BUFFER, 1, 1);
  // channelCount=1 forces mono; if the browser gives stereo, average channels below.

  scriptProcessor.onaudioprocess = (event) => {
    if (!isRecording || ws?.readyState !== WebSocket.OPEN) return;

    // Grab the input data — channel 0 is mono (we requested mono above, but
    // some browsers give stereo regardless; ScriptProcessorNode exposes all
    // input channels on the inputBuffer, so we average them for safety).
    const inputBuffer   = event.inputBuffer;
    const numChannels   = inputBuffer.numberOfChannels;

    let monoBuffer;
    if (numChannels === 1) {
      monoBuffer = inputBuffer.getChannelData(0).slice(); // copy Float32Array
    } else {
      // Average all channels into mono
      monoBuffer = new Float32Array(inputBuffer.length);
      for (let ch = 0; ch < numChannels; ch++) {
        const chData = inputBuffer.getChannelData(ch);
        for (let i = 0; i < inputBuffer.length; i++) {
          monoBuffer[i] += chData[i];
        }
      }
      for (let i = 0; i < monoBuffer.length; i++) {
        monoBuffer[i] /= numChannels;
      }
    }

    // Live mic-level meter (throttled ~6/sec) so the user can SEE that audio is
    // actually being captured — flat bars mean the mic/AudioContext isn't working.
    const now = performance.now();
    if (now - lastLevelUpdate > 150) {
      lastLevelUpdate = now;
      let sumSq = 0;
      for (let i = 0; i < monoBuffer.length; i++) sumSq += monoBuffer[i] * monoBuffer[i];
      const rms = Math.sqrt(sumSq / monoBuffer.length);
      const bars = Math.max(0, Math.min(10, Math.round(rms * 300)));
      setStatus(`🔴 Listening… [${"█".repeat(bars)}${"·".repeat(10 - bars)}]`, "active");
    }

    // Downsample to 16 kHz and convert to int16 PCM
    const pcm16 = downsampleToInt16(monoBuffer, inputSampleRate);

    // Send as binary WebSocket frame
    ws.send(pcm16.buffer);
  };

  // Connect: mic → processor → (silent) destination
  // The destination node is needed to keep Chrome's audio engine active;
  // we don't actually play anything back.
  sourceNode.connect(scriptProcessor);
  scriptProcessor.connect(audioCtx.destination);
}

/**
 * Stop recording.
 * @param {boolean} [sendStop=true] - Whether to send the control/stop frame.
 *   False when called after an unexpected WebSocket close.
 */
function stopRecording(sendStop = true) {
  isRecording = false;
  btnRecord.classList.remove("recording");

  // Disconnect audio pipeline
  if (scriptProcessor) {
    scriptProcessor.disconnect();
    scriptProcessor.onaudioprocess = null;
    scriptProcessor = null;
  }
  if (sourceNode) {
    sourceNode.disconnect();
    sourceNode = null;
  }
  if (mediaStream) {
    mediaStream.getTracks().forEach(t => t.stop());
    mediaStream = null;
  }
  if (audioCtx) {
    audioCtx.close().catch(() => {});
    audioCtx = null;
  }

  // Send stop frame and close WebSocket
  if (ws) {
    if (sendStop && ws.readyState === WebSocket.OPEN) {
      try {
        ws.send(JSON.stringify({ type: "control", action: "stop" }));
      } catch (err) {
        console.warn("Could not send stop frame:", err);
      }
    }
    // Close gracefully; the server will flush and close from its side too.
    if (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING) {
      ws.close(1000, "Client stopped recording");
    }
    ws = null;
  }

  setStatus("Stopped.", "");
}

// ─── History ──────────────────────────────────────────────────────────────────

/** Fetch recent notes and render them as history cards. */
async function loadHistory() {
  const backendUrl = getBackendUrl();
  if (!backendUrl) {
    setStatus("Configure Backend URL in Settings first.", "error");
    return;
  }

  setStatus("Loading history…", "warning");

  try {
    const resp = await fetch(`${backendUrl}/api/v1/notes?limit=20`, {
      headers: authHeaders(),
    });

    if (resp.status === 401) {
      throw new Error("Unauthorised — check your API key in Settings.");
    }
    if (!resp.ok) {
      throw new Error(`HTTP ${resp.status}`);
    }

    const data = await resp.json();
    const notes = data.notes || [];

    setStatus("", "");

    if (notes.length === 0) {
      clearEmptyNotice();
      const el = document.createElement("div");
      el.className = "feed-empty";
      el.textContent = "No notes yet.";
      feed.insertBefore(el, feed.firstChild);
      return;
    }

    // Render a divider before history entries
    clearEmptyNotice();
    const divider = document.createElement("div");
    divider.className = "card history";
    divider.innerHTML = `<div class="card-type">History — ${notes.length} notes</div>`;
    feed.insertBefore(divider, feed.firstChild);

    notes.forEach(note => {
      const frag = document.createDocumentFragment();

      const summary = document.createElement("p");
      summary.textContent = note.summary || "(no summary)";
      frag.appendChild(summary);

      if (note.created_at) {
        const meta = document.createElement("div");
        meta.className = "card-meta";
        meta.textContent = new Date(note.created_at).toLocaleString();
        frag.appendChild(meta);
      }

      prependCard("history", "Note", frag);
    });

  } catch (err) {
    setStatus(`History error: ${err.message}`, "error");
  }
}

// ─── Init ─────────────────────────────────────────────────────────────────────

document.addEventListener("DOMContentLoaded", () => {
  // Bind DOM refs
  btnRecord         = document.getElementById("btn-record");
  statusLine        = document.getElementById("status-line");
  feed              = document.getElementById("feed");
  btnHistory        = document.getElementById("btn-history");
  settingsPanel     = document.getElementById("settings-panel");
  btnSettingsToggle = document.getElementById("btn-settings-toggle");
  inputBackendUrl   = document.getElementById("input-backend-url");
  inputApiKey       = document.getElementById("input-api-key");
  btnSaveSettings   = document.getElementById("btn-save-settings");

  // Load persisted settings into the form
  loadSettings();

  // Settings toggle
  btnSettingsToggle.addEventListener("click", () => {
    settingsPanel.classList.toggle("open");
  });

  // Save settings
  btnSaveSettings.addEventListener("click", saveSettings);

  // Allow Enter key in the settings inputs to save
  [inputBackendUrl, inputApiKey].forEach(el => {
    el.addEventListener("keydown", e => {
      if (e.key === "Enter") saveSettings();
    });
  });

  // Record / Stop button toggle
  btnRecord.addEventListener("click", () => {
    if (isRecording) {
      stopRecording(true);
    } else {
      startRecording();
    }
  });

  // History button
  btnHistory.addEventListener("click", loadHistory);

  // Initial status message
  const backend = getBackendUrl();
  if (!backend) {
    setStatus("Open Settings and enter your Backend URL.", "warning");
  } else {
    setStatus("Ready. Tap the mic to start.", "");
  }
});

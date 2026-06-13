# Avin Web Client

A mobile-first, framework-free (vanilla HTML/CSS/JS) web app for the Avin assistant.
Open it on your phone, stream your microphone to the Python backend, and see
transcripts, notes, and actions arrive in real time — confirming or dismissing any
action that needs approval.

It is a **static** site: deploy it to Cloudflare Pages and point it at your
separately-hosted backend (Cloud Run) via the in-app **Settings**.

## Files

| File | Purpose |
|---|---|
| `index.html` | Markup + element IDs the app binds to |
| `app.js` | Settings, audio capture/downsampling, WebSocket streaming, rendering, action confirm |
| `styles.css` | Mobile-first styling (dark/light via `prefers-color-scheme`) |

## How it talks to the backend

- **Settings** (stored in `localStorage`, never committed): **Backend URL**
  (e.g. `https://avin-xxxx.run.app`) and an optional **API key**.
- **Streaming** — opens `WS {backend}/api/v1/stream`:
  - Captures mic audio via `getUserMedia` + Web Audio API.
  - Downsamples the browser's native rate (usually 44.1/48 kHz) to **16 kHz mono
    int16 PCM** and sends it as **binary** WebSocket frames.
  - On **Stop**, sends a text control frame `{"type":"control","action":"stop"}`.
  - Renders incoming JSON text frames by `type`: `transcript`, `note`, `action`, `error`.
- **Action confirmation** — action cards with `needs_confirmation: true` show
  **Confirm** / **Dismiss**, which `PATCH {backend}/api/v1/actions/{id}` with
  `{"status":"confirmed"|"dismissed"}`. Nothing is ever auto-confirmed — you tap.
- **History** — `GET {backend}/api/v1/notes?limit=20`.
- **Auth** — when the backend has API keys configured, REST calls send
  `X-API-Key: <key>` and the WebSocket appends `?api_key=<key>` (browsers can't set
  custom headers on a WebSocket upgrade). With no key configured the backend runs
  auth-disabled (its local-dev default).

> **Audio note:** the downsampler uses linear-interpolation decimation — fine for
> speech (the backend's Silero VAD + Gemini handle the rest). An `AudioWorklet`
> upgrade path is marked in `app.js` (currently `ScriptProcessorNode` for maximum
> browser compatibility with zero build tooling).

## Run locally

```bash
cd web
python -m http.server 5173
# then open http://localhost:5173 (or http://<your-LAN-ip>:5173 on your phone)
```

Mic capture requires a **secure context**: `http://localhost` works, but accessing
by LAN IP over plain HTTP will block the mic on most browsers — use the Cloudflare
Pages URL (HTTPS) or a local HTTPS tunnel for real phone testing.

## Deploy to Cloudflare Pages

1. Push this repo to GitHub (already done).
2. Cloudflare dashboard → **Workers & Pages** → **Create** → **Pages** → connect the repo.
3. Build settings:
   - **Framework preset:** None
   - **Build command:** *(leave empty — it's static)*
   - **Build output directory:** `web`
4. Deploy. Open the `*.pages.dev` URL on your phone, open **Settings**, and set
   the **Backend URL** (see below), then tap the mic.

> **Backend URL = the Cloudflare Worker, not Cloud Run directly.** For security
> level B the app talks to the `../worker` proxy (`https://avin-proxy.<you>.workers.dev`),
> which injects the shared `X-Proxy-Secret` so only Cloudflare can reach the backend.
> See [`../worker/README.md`](../worker/README.md). (You *can* point straight at the
> Cloud Run URL for quick local testing when the backend has no proxy secret set.)
>
> **CORS** is handled both by the Worker and by the backend's CORS middleware
> (`AVIN_CORS_ORIGINS`, default `*`).

# Avin Proxy Worker

A Cloudflare Worker that sits between the phone web app and the Cloud Run backend:

```
phone browser → Cloudflare Pages (web/) → THIS Worker → Cloud Run backend
```

**Why it exists (security level B):** the Cloud Run backend is deployed publicly
(a browser-facing API has to be), but it rejects any request lacking the shared
`X-Proxy-Secret` header. Only this Worker holds that secret (a Cloudflare-encrypted
secret, never sent to the browser), so in practice **only Cloudflare can reach the
backend** — not the public internet.

It proxies both REST and the audio WebSocket, injects the secret on every
forwarded request, and passes through `X-API-Key` / `?api_key=` so the backend's
own per-user auth still works as an optional second factor.

## Deploy

Prereqs: a [Cloudflare account] and `npm i -g wrangler`, then `wrangler login`.

1. **Set the backend URL** in `wrangler.toml` → `BACKEND_URL` to your Cloud Run
   service URL. Set `ALLOWED_ORIGIN` to your Pages URL (or leave `"*"` while testing).

2. **Set the shared secret** (must equal the backend's `AVIN_API_PROXY_SECRET`):
   ```bash
   cd worker
   wrangler secret put PROXY_SECRET
   # paste the same secret you set on Cloud Run
   ```

3. **Deploy:**
   ```bash
   wrangler deploy
   ```
   Wrangler prints the Worker URL (e.g. `https://avin-proxy.<you>.workers.dev`).

4. **Point the web app at the Worker:** open the Pages app → Settings → set
   **Backend URL** to the Worker URL (NOT the Cloud Run URL). The app now talks
   only to Cloudflare; the backend only accepts traffic carrying the secret.

## Secret rotation

Generate a fresh secret, update both sides, and redeploy:
```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"   # new value
# backend:  gcloud run services update avin --set-env-vars AVIN_API_PROXY_SECRET=<new>
# worker:   wrangler secret put PROXY_SECRET   (paste <new>) && wrangler deploy
```

> Local dev: if the backend runs with `AVIN_API_PROXY_SECRET` unset, the check is
> disabled and you can hit the backend directly without the Worker.

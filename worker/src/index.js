/**
 * Avin proxy Worker — the trusted front door to the Cloud Run backend.
 *
 * Security model (level B): the backend is deployed publicly (Cloud Run with
 * --allow-unauthenticated, required for a browser-facing API) but REJECTS any
 * request that doesn't carry the shared `X-Proxy-Secret` header.  Only THIS
 * Worker knows the secret (stored as a Cloudflare-encrypted secret, never sent
 * to the browser), so in practice only Cloudflare can reach the backend.
 *
 * Flow:  phone browser  →  this Worker  →  Cloud Run backend
 *
 * - Injects `X-Proxy-Secret: <env.PROXY_SECRET>` on every forwarded request.
 * - Proxies both REST (fetch) and the audio WebSocket (Upgrade) — the secret is
 *   added to the upstream WS handshake even though browsers can't set WS headers.
 * - Passes through `X-API-Key` / `?api_key=` untouched so the backend's own
 *   per-user API-key auth still applies (optional second factor).
 * - Answers CORS preflight so the Cloudflare Pages frontend can call it.
 *
 * Config (wrangler.toml [vars] + secrets):
 *   BACKEND_URL     (var)    e.g. https://avin-xxxx.run.app
 *   ALLOWED_ORIGIN  (var)    CORS origin to allow, e.g. https://avin.pages.dev or "*"
 *   PROXY_SECRET    (secret) set via: wrangler secret put PROXY_SECRET
 */

function corsHeaders(env) {
  return {
    "Access-Control-Allow-Origin": env.ALLOWED_ORIGIN || "*",
    "Access-Control-Allow-Methods": "GET,POST,PATCH,OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type, X-API-Key",
    "Access-Control-Max-Age": "86400",
  };
}

export default {
  async fetch(request, env) {
    if (!env.BACKEND_URL) {
      return new Response("Worker misconfigured: BACKEND_URL is not set.", { status: 500 });
    }

    // Rewrite the URL onto the backend host, preserving path + query (?api_key=…).
    const incoming = new URL(request.url);
    const backend = new URL(env.BACKEND_URL);
    incoming.protocol = backend.protocol;
    incoming.host = backend.host;
    incoming.port = backend.port;
    const targetUrl = incoming.toString();

    // CORS preflight — answer at the edge (no secret needed; no body).
    if (request.method === "OPTIONS") {
      return new Response(null, { status: 204, headers: corsHeaders(env) });
    }

    // WebSocket upgrade — forward with the secret injected on the handshake.
    if ((request.headers.get("Upgrade") || "").toLowerCase() === "websocket") {
      // Clone the original request (preserves the WebSocket upgrade) onto the
      // backend URL, then inject the secret.  Reconstructing the request from
      // scratch breaks the upgrade handshake (results in a 502), so we clone.
      const wsReq = new Request(targetUrl, request);
      wsReq.headers.set("X-Proxy-Secret", env.PROXY_SECRET || "");
      return fetch(wsReq);
    }

    // Regular REST request.
    const headers = new Headers(request.headers);
    headers.set("X-Proxy-Secret", env.PROXY_SECRET || "");

    const upstream = await fetch(
      new Request(targetUrl, {
        method: request.method,
        headers,
        body: request.body,
        redirect: "manual",
      }),
    );

    // Copy the response and attach CORS headers for the browser.
    const response = new Response(upstream.body, upstream);
    const ch = corsHeaders(env);
    for (const [key, value] of Object.entries(ch)) {
      response.headers.set(key, value);
    }
    return response;
  },
};

# Know Your Website — Static Deployment (GitHub Pages)

> Branch: `deploy-pages` — fully static build, no backend required.
> Main branch (`main`) contains the full FastAPI + React version with server-side scanning.

Live at: **[aman-24052001.github.io/Know-Your-Website](https://aman-24052001.github.io/Know-Your-Website/)**

---

## How It Works

All 7 security audit modules run entirely in the browser using public APIs and direct fetch calls — no backend server required.

| Module | Implementation |
|---|---|
| DNS Resolution | Cloudflare DNS-over-HTTPS (`cloudflare-dns.com/dns-query`) |
| SSL / TLS | Qualys SSL Labs API (`api.ssllabs.com`) with HTTPS fallback |
| Security Headers | securityheaders.com API with allorigins proxy fallback |
| Exposed Endpoints | Direct `HEAD` requests with baseline false-positive filtering |
| Robots & Sitemap | Direct `fetch` to `/robots.txt` and `/sitemap.xml` |
| Open Redirects | Redirect parameter probing with manual redirect mode |
| Cookie Security | allorigins proxy to inspect `Set-Cookie` response headers |

The Anthropic API key is entered in the UI, stored in `localStorage` (browser only), and sent directly to `api.anthropic.com` — it never touches any server.

---

## Differences from `main` Branch

| Feature | `main` (FastAPI) | `deploy-pages` (Static) |
|---|---|---|
| Infrastructure | FastAPI backend + React frontend | React only |
| DNS | Raw socket (accurate) | DNS-over-HTTPS |
| SSL | Direct TLS inspection | Qualys SSL Labs API |
| Headers | Direct HTTP fetch | Proxy via allorigins |
| Deployment | Requires server (Render, Railway, etc.) | GitHub Pages (free, zero config) |
| CORS limitations | None | Some sites may block browser requests |

---

## Deployment

This branch deploys automatically via GitHub Actions on every push.

The workflow (`.github/workflows/deploy.yml`) does:
1. `npm install`
2. `npm run build` (outputs to `dist/`)
3. Deploys `dist/` to GitHub Pages

---

## Local Development

```bash
npm install
npm run dev
```

Open [http://localhost:5173](http://localhost:5173).

---

## API Key

The Anthropic API key is optional. Without it, all 7 audit modules still run and produce findings — only the LLM synthesis summary at the end is skipped.

To add a key:
1. Click **Key** in the left panel
2. Enter your `sk-ant-...` key
3. Click **Save Key**

The key is stored in `localStorage` and persists across sessions. It is sent directly to `api.anthropic.com` from your browser and is never logged or stored anywhere else.

Get a key at [console.anthropic.com](https://console.anthropic.com).

---

## Known Limitations

- Some sites block cross-origin `fetch` requests (CORS). In those cases, the tool attempts fallback methods via proxy.
- SSL Labs analysis can take 30–60 seconds for uncached domains.
- Cookie inspection via proxy may not capture all cookies if the proxy strips headers.
- Browser `fetch` cannot inspect raw TLS handshake details — SSL data comes from the Qualys API.

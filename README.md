# Know Your Website

Autonomous web security audit agent. Enter a URL — the agent runs 7 inspection modules concurrently, streams live findings to a terminal-style dashboard, and synthesises everything into a prioritised security brief via LLM.

Built with FastAPI, LangGraph-style async pipeline, and a React (Vite) frontend using a neo-brutalist design system.

---

## Architecture

```
React (Vite)
    └── SSE connection → GET /audit?url=...
              ↓
         FastAPI (main.py)
              ↓
     Async Tool Pipeline (graph.py)
         ├── DNS Resolution
         ├── SSL / TLS Inspector
         ├── Security Header Analyzer
         ├── Exposed Endpoint Scanner
         ├── Robots & Sitemap Parser
         ├── Open Redirect Checker
         └── Cookie Security Auditor
              ↓
     Security Scorer (0–100)
              ↓
     LLM Synthesis (Claude)
              ↓
     Structured Report → SSE → React
```

Each tool runs sequentially, streaming log events back to the browser in real time via Server-Sent Events. The LLM synthesises all findings into a prioritised security brief at the end. If no API key is present, the pipeline still completes — LLM synthesis is an enhancement, not a dependency.

---

## Audit Modules

| Module | Checks |
|---|---|
| DNS Resolution | A record, reverse DNS (PTR), multi-IP / CDN detection |
| SSL / TLS | Certificate validity, expiry, TLS version, issuer, SAN coverage |
| Security Headers | HSTS, CSP, X-Frame-Options, X-Content-Type-Options, Referrer-Policy, Permissions-Policy, info-leaking headers |
| Exposed Endpoints | `/.env`, `/.git/config`, `/backup.sql`, `/admin`, `/phpmyadmin`, Spring Actuator, Swagger UI, and 10+ more — with baseline comparison to eliminate false positives on catch-all servers |
| Robots & Sitemap | Sensitive path disclosure in `robots.txt`, sitemap presence |
| Open Redirects | Unvalidated redirect parameters (`?url=`, `?next=`, `?redirect=`) and common redirect paths |
| Cookie Security | `Secure`, `HttpOnly`, `SameSite` flags on all cookies |

---

## Scoring

Each finding reduces from a base score of 100:

| Severity | Deduction |
|---|---|
| CRITICAL | -25 |
| HIGH | -15 |
| MEDIUM | -8 |
| LOW | -3 |

Score drives the gauge colour: green (≥70), amber (40–69), red (<40).

---

## Stack

**Backend**
- Python 3.12
- FastAPI + Uvicorn (SSE streaming)
- httpx (async HTTP)
- anthropic SDK (LLM synthesis)
- python-dotenv

**Frontend**
- React 18 + Vite
- lucide-react (icons)
- Space Grotesk + JetBrains Mono
- Vanilla CSS (neo-brutalist design system)

---

## Setup

### Prerequisites

- Python 3.12+
- Node.js 18+
- An Anthropic API key (optional — pipeline works without it)

### Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Copy and fill in your API key
cp ../.env.example ../.env
# ANTHROPIC_API_KEY=sk-ant-...

uvicorn main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Open [http://localhost:5173](http://localhost:5173).

The Vite dev server proxies `/audit` and `/health` to `localhost:8000` — no CORS config needed in development.

---

## Usage

1. Enter any URL in the input bar (with or without `https://`)
2. Click **Run Security Audit**
3. Watch findings stream live in the agent log panel
4. Review the structured report and LLM analysis when the scan completes

The scanner probes the root domain for headers and endpoint checks, regardless of the path provided. Deep paths (e.g. `/join/abc123`) are handled gracefully — the tool extracts the hostname automatically.

---

## False Positive Handling

The endpoint scanner establishes a baseline response (HTTP status + content size) for a known-nonexistent path before probing sensitive paths. This eliminates false positives from:

- Catch-all servers returning 403 for everything
- Path-based routing platforms (GitHub Pages, Vercel) that return 200 for non-existent routes
- Soft-404 servers that return 200 with a standard error page

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | No | Enables LLM synthesis of findings into a plain-English security brief |

---

## Project Structure

```
know-your-website/
├── backend/
│   ├── main.py                        # FastAPI app, /audit SSE endpoint
│   ├── requirements.txt
│   ├── agent/
│   │   ├── graph.py                   # Async pipeline orchestrator
│   │   ├── state.py                   # AuditState TypedDict
│   │   ├── prompts.py                 # LLM system prompt + synthesis prompt
│   │   └── tools/
│   │       ├── dns_resolver.py
│   │       ├── ssl_inspector.py
│   │       ├── header_analyzer.py     # Probes root domain, falls back to full URL
│   │       ├── endpoint_scanner.py    # Baseline-aware false positive filtering
│   │       ├── robots_parser.py
│   │       ├── redirect_checker.py
│   │       ├── cookie_auditor.py
│   │       └── security_scorer.py
│   └── models/
│       ├── report.py                  # Pydantic: Finding, ModuleResult, AuditReport
│       └── request.py
└── frontend/
    ├── index.html
    ├── vite.config.js
    ├── package.json
    └── src/
        ├── App.jsx
        ├── main.jsx
        ├── hooks/
        │   └── useAuditStream.js      # SSE connection manager
        ├── components/
        │   ├── LeftPanel.jsx          # URL input, score gauge, module checklist
        │   ├── AgentLog.jsx           # Live terminal stream
        │   ├── ReportCard.jsx         # Findings grid + LLM summary
        │   ├── ScoreGauge.jsx         # SVG ring with animated score
        │   └── FindingBadge.jsx       # Severity chip (CRITICAL/HIGH/MEDIUM/LOW/PASS)
        └── styles/
            └── theme.css              # Design tokens, neo-brutalist variables
```

---

## Design

Neo-brutalist aesthetic: hard 3px borders, 5px box shadows with no blur, flat fills, cream base (`#F5F0E8`) with ink black (`#0A0A0A`). The agent log renders raw terminal output inside the structured UI — that contrast is intentional.

Type: Space Grotesk (UI) + JetBrains Mono (data, terminal, badges).

Colour is used semantically only: red = critical, amber = warning, green = pass, cyan = agent/system.

---

## Known Limitations

- GitHub Pages and similar CDN-hosted static sites may show headers as missing if the scanning environment has network restrictions on `.github.io` domains
- The SSL issuer field reflects the scanning environment's egress proxy rather than the real CA in sandboxed environments (TLS version and expiry are always accurate)
- Open redirect detection uses a test payload (`https://evil-example.com`) — some WAFs may block this probe

---

## License

MIT

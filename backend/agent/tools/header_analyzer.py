import httpx
from urllib.parse import urlparse
from backend.models.report import ModuleResult, Finding, SeverityLevel


REQUIRED_HEADERS = {
    "strict-transport-security": ("HSTS", SeverityLevel.HIGH),
    "content-security-policy": ("CSP", SeverityLevel.HIGH),
    "x-frame-options": ("X-Frame-Options", SeverityLevel.MEDIUM),
    "x-content-type-options": ("X-Content-Type-Options", SeverityLevel.MEDIUM),
    "referrer-policy": ("Referrer-Policy", SeverityLevel.LOW),
    "permissions-policy": ("Permissions-Policy", SeverityLevel.LOW),
    "x-xss-protection": ("X-XSS-Protection", SeverityLevel.LOW),
}

INFO_HEADERS = ["server", "x-powered-by", "x-aspnet-version", "x-aspnetmvc-version"]


def get_root_url(url: str) -> str:
    """Strip path/query from URL to get root origin."""
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}"


async def run_header_analyzer(url: str, log) -> ModuleResult:
    await log("Fetching HTTP response headers...")
    findings = []

    # Always probe the root domain for headers — deep paths may 403/404
    root_url = get_root_url(url)
    probe_url = root_url
    used_fallback = False

    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=10) as client:
            resp = await client.get(probe_url)

            # If root returns a non-useful status, try the original full URL
            if resp.status_code in (403, 404, 410) and probe_url != url:
                await log(f"Root returned HTTP {resp.status_code}, retrying full URL...")
                resp = await client.get(url)
                used_fallback = True

        target_label = url if used_fallback else root_url
        headers_lower = {k.lower(): v for k, v in resp.headers.items()}
        await log(f"HTTP {resp.status_code} from {target_label} — {len(headers_lower)} headers received")

        for header_key, (label, severity) in REQUIRED_HEADERS.items():
            if header_key in headers_lower:
                val = headers_lower[header_key]
                await log(f"{label}: present — {val[:60]}", level="ok")
                findings.append(Finding(key=label, detail=f"Present: {val[:80]}", severity=SeverityLevel.PASS))
            else:
                await log(f"{label}: missing", level="warn")
                findings.append(Finding(key=label, detail=f"Header not set — {label} missing", severity=severity))

        for h in INFO_HEADERS:
            if h in headers_lower:
                val = headers_lower[h]
                await log(f"Info-leaking header detected: {h}: {val}", level="warn")
                findings.append(Finding(
                    key=h.title(),
                    detail=f"Leaks server info: {val}",
                    severity=SeverityLevel.LOW
                ))

        severities = [f.severity for f in findings if f.severity != SeverityLevel.PASS]
        if SeverityLevel.HIGH in severities:
            status = SeverityLevel.HIGH
        elif SeverityLevel.MEDIUM in severities:
            status = SeverityLevel.MEDIUM
        elif SeverityLevel.LOW in severities:
            status = SeverityLevel.LOW
        else:
            status = SeverityLevel.PASS

    except Exception as e:
        await log(f"Header fetch failed: {e}", level="err")
        findings.append(Finding(key="Fetch Error", detail=str(e), severity=SeverityLevel.HIGH))
        status = SeverityLevel.HIGH

    return ModuleResult(module="Security Headers", status=status, findings=findings)

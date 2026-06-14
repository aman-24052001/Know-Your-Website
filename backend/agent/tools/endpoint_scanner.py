import httpx
from backend.models.report import ModuleResult, Finding, SeverityLevel


# (path, label, severity if accessible)
PROBE_PATHS = [
    ("/.env", "Environment File", SeverityLevel.CRITICAL),
    ("/.env.local", "Local Env File", SeverityLevel.CRITICAL),
    ("/backup.sql", "SQL Backup", SeverityLevel.CRITICAL),
    ("/dump.sql", "SQL Dump", SeverityLevel.CRITICAL),
    ("/db.sqlite", "SQLite DB", SeverityLevel.CRITICAL),
    ("/.git/config", "Git Config", SeverityLevel.CRITICAL),
    ("/wp-config.php.bak", "WP Config Backup", SeverityLevel.CRITICAL),
    ("/admin", "Admin Panel", SeverityLevel.HIGH),
    ("/admin/login", "Admin Login", SeverityLevel.HIGH),
    ("/phpmyadmin", "phpMyAdmin", SeverityLevel.HIGH),
    ("/server-status", "Apache Server Status", SeverityLevel.HIGH),
    ("/actuator", "Spring Actuator", SeverityLevel.HIGH),
    ("/actuator/env", "Spring Env Endpoint", SeverityLevel.CRITICAL),
    ("/api/v1/users", "User API Endpoint", SeverityLevel.MEDIUM),
    ("/swagger-ui.html", "Swagger UI", SeverityLevel.MEDIUM),
    ("/openapi.json", "OpenAPI Spec", SeverityLevel.MEDIUM),
    ("/robots.txt", "Robots.txt", SeverityLevel.PASS),
    ("/sitemap.xml", "Sitemap", SeverityLevel.PASS),
]

BAD_CODES = {200}


async def probe_path(client: httpx.AsyncClient, base_url: str, path: str):
    try:
        resp = await client.get(base_url.rstrip("/") + path, follow_redirects=False)
        return resp.status_code, len(resp.content)
    except Exception:
        return None, None


async def run_endpoint_scanner(url: str, log) -> ModuleResult:
    await log("Scanning for exposed endpoints and sensitive files...")
    findings = []
    base = url.rstrip("/")

    async with httpx.AsyncClient(timeout=8, follow_redirects=False) as client:
        # Baseline: what does the server return for a known-nonexistent path?
        baseline_code, baseline_size = await probe_path(client, base, "/___kyw_nonexistent_probe___")
        await log(f"Baseline probe: HTTP {baseline_code} ({baseline_size} bytes)")

        for path, label, sev in PROBE_PATHS:
            code, size = await probe_path(client, base, path)
            if code is None:
                continue

            # Benign paths — just log presence
            if sev == SeverityLevel.PASS:
                if code == 200:
                    await log(f"{path} — HTTP {code} (expected)", level="ok")
                continue

            # Skip if response is same code AND similar size as baseline (catch-all server)
            if code == baseline_code:
                size_similar = baseline_size is not None and abs(size - baseline_size) < 200
                if size_similar or code != 200:
                    await log(f"{path} — HTTP {code} (matches baseline, skipping)", level="ok")
                    continue

            if code in BAD_CODES:
                # Additional check: if baseline is also 200, only flag if size differs significantly
                if baseline_code == 200:
                    if baseline_size is not None and abs(size - baseline_size) < 500:
                        await log(f"{path} — HTTP {code} but same size as baseline (soft-404 server)", level="ok")
                        continue
                await log(f"{path} returned HTTP {code} — {label} exposed", level="err")
                findings.append(Finding(
                    key=label,
                    detail=f"{path} returned HTTP {code} — accessible without authentication",
                    severity=sev
                ))
            elif code == 403 and baseline_code != 403:
                await log(f"{path} — HTTP 403, resource exists but access denied", level="warn")
                findings.append(Finding(
                    key=label,
                    detail=f"{path} exists (HTTP 403) — blocked but discoverable",
                    severity=SeverityLevel.LOW
                ))
            else:
                await log(f"{path} — HTTP {code}, not accessible", level="ok")
                findings.append(Finding(
                    key=label,
                    detail=f"{path} returned HTTP {code} — not accessible",
                    severity=SeverityLevel.PASS
                ))

    if not findings:
        await log("No sensitive endpoints exposed", level="ok")
        findings.append(Finding(
            key="Endpoint Scan",
            detail="No sensitive files or panels accessible",
            severity=SeverityLevel.PASS
        ))
        status = SeverityLevel.PASS
    else:
        severities = [f.severity for f in findings if f.severity != SeverityLevel.PASS]
        if not severities:
            status = SeverityLevel.PASS
        elif SeverityLevel.CRITICAL in severities:
            status = SeverityLevel.CRITICAL
        elif SeverityLevel.HIGH in severities:
            status = SeverityLevel.HIGH
        elif SeverityLevel.MEDIUM in severities:
            status = SeverityLevel.MEDIUM
        else:
            status = SeverityLevel.LOW

    return ModuleResult(module="Exposed Endpoints", status=status, findings=findings)

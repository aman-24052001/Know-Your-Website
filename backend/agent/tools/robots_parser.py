import httpx
from backend.models.report import ModuleResult, Finding, SeverityLevel


SENSITIVE_PATTERNS = [
    "admin", "backup", "config", "secret", "private", "internal",
    "staging", "dev", "test", "debug", "password", "passwd", ".env",
    "database", "db", "api/internal", "console"
]


async def run_robots_parser(url: str, log) -> ModuleResult:
    await log("Parsing robots.txt and sitemap.xml...")
    findings = []
    base = url.rstrip("/")

    async with httpx.AsyncClient(timeout=8, follow_redirects=True) as client:
        # robots.txt
        try:
            r = await client.get(f"{base}/robots.txt")
            if r.status_code == 200:
                content = r.text
                lines = content.splitlines()
                disallowed = [l.split(":", 1)[1].strip() for l in lines if l.lower().startswith("disallow:")]
                await log(f"robots.txt found — {len(disallowed)} Disallow rule(s)")

                flagged = []
                for path in disallowed:
                    if any(s in path.lower() for s in SENSITIVE_PATTERNS):
                        flagged.append(path)

                if flagged:
                    await log(f"Sensitive paths disclosed in robots.txt: {', '.join(flagged[:3])}", level="warn")
                    findings.append(Finding(
                        key="robots.txt Disclosure",
                        detail=f"Sensitive paths revealed: {', '.join(flagged[:5])}",
                        severity=SeverityLevel.MEDIUM
                    ))
                else:
                    await log("robots.txt found — no sensitive paths disclosed", level="ok")
                    findings.append(Finding(
                        key="robots.txt",
                        detail=f"Present, {len(disallowed)} disallow rules, no sensitive path disclosure",
                        severity=SeverityLevel.PASS
                    ))
            else:
                await log(f"robots.txt not found (HTTP {r.status_code})")
                findings.append(Finding(key="robots.txt", detail="Not present", severity=SeverityLevel.LOW))
        except Exception as e:
            await log(f"robots.txt fetch error: {e}", level="warn")

        # sitemap.xml
        try:
            r = await client.get(f"{base}/sitemap.xml")
            if r.status_code == 200:
                size = len(r.text)
                await log(f"sitemap.xml found — {size} bytes", level="ok")
                findings.append(Finding(
                    key="sitemap.xml",
                    detail=f"Present, {size} bytes",
                    severity=SeverityLevel.PASS
                ))
            else:
                await log("sitemap.xml not found")
                findings.append(Finding(key="sitemap.xml", detail="Not present", severity=SeverityLevel.LOW))
        except Exception as e:
            await log(f"sitemap.xml fetch error: {e}", level="warn")

    severities = [f.severity for f in findings if f.severity != SeverityLevel.PASS]
    if SeverityLevel.MEDIUM in severities:
        status = SeverityLevel.MEDIUM
    elif SeverityLevel.LOW in severities:
        status = SeverityLevel.LOW
    else:
        status = SeverityLevel.PASS

    return ModuleResult(module="Robots & Sitemap", status=status, findings=findings)

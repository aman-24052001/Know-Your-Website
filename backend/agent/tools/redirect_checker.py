import httpx
from backend.models.report import ModuleResult, Finding, SeverityLevel


REDIRECT_PARAMS = ["url", "redirect", "next", "return", "goto", "redir", "destination", "target", "link", "to"]
TEST_PAYLOAD = "https://evil-example.com"


async def run_redirect_checker(url: str, log) -> ModuleResult:
    await log("Checking for open redirect vulnerabilities...")
    findings = []
    base = url.rstrip("/")
    flagged = []

    async with httpx.AsyncClient(timeout=8, follow_redirects=False) as client:
        for param in REDIRECT_PARAMS:
            test_url = f"{base}/?{param}={TEST_PAYLOAD}"
            try:
                resp = await client.get(test_url)
                location = resp.headers.get("location", "")
                if resp.status_code in (301, 302, 303, 307, 308) and "evil-example.com" in location:
                    await log(f"Open redirect confirmed at ?{param}= — redirects to {location}", level="err")
                    flagged.append(param)
                    findings.append(Finding(
                        key=f"Open Redirect (?{param}=)",
                        detail=f"Redirects to attacker-controlled URL via ?{param}= parameter",
                        severity=SeverityLevel.HIGH
                    ))
            except Exception:
                pass

        # Also test common redirect paths
        for path in ["/go", "/redirect", "/out", "/r", "/link", "/forward"]:
            test_url = f"{base}{path}?url={TEST_PAYLOAD}"
            try:
                resp = await client.get(test_url)
                location = resp.headers.get("location", "")
                if resp.status_code in (301, 302, 303, 307, 308) and "evil-example.com" in location:
                    await log(f"Open redirect at {path}?url= confirmed", level="err")
                    findings.append(Finding(
                        key=f"Open Redirect ({path})",
                        detail=f"Path {path} performs unvalidated external redirect",
                        severity=SeverityLevel.HIGH
                    ))
            except Exception:
                pass

    if not findings:
        await log("No open redirect vulnerabilities detected", level="ok")
        findings.append(Finding(
            key="Open Redirects",
            detail="No unvalidated redirect parameters found",
            severity=SeverityLevel.PASS
        ))
        status = SeverityLevel.PASS
    else:
        await log(f"{len(findings)} open redirect(s) found", level="warn")
        status = SeverityLevel.HIGH

    return ModuleResult(module="Open Redirects", status=status, findings=findings)

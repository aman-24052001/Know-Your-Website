import httpx
from backend.models.report import ModuleResult, Finding, SeverityLevel


async def run_cookie_auditor(url: str, log) -> ModuleResult:
    await log("Auditing cookie security flags...")
    findings = []

    try:
        async with httpx.AsyncClient(timeout=8, follow_redirects=True) as client:
            resp = await client.get(url)

        cookies_raw = resp.headers.get_list("set-cookie") if hasattr(resp.headers, "get_list") else []

        # httpx stores cookies differently — parse from headers
        raw_headers = [(k, v) for k, v in resp.headers.raw if k.lower() == b"set-cookie"]
        cookie_headers = [v.decode("utf-8", errors="replace") for _, v in raw_headers]

        if not cookie_headers:
            await log("No Set-Cookie headers found")
            findings.append(Finding(
                key="Cookies",
                detail="No cookies set on initial response",
                severity=SeverityLevel.PASS
            ))
            return ModuleResult(module="Cookie Security", status=SeverityLevel.PASS, findings=findings)

        await log(f"Found {len(cookie_headers)} cookie(s) to audit")

        for raw in cookie_headers:
            parts = [p.strip() for p in raw.split(";")]
            name = parts[0].split("=")[0].strip()
            attrs_lower = [p.lower() for p in parts[1:]]

            await log(f"Auditing cookie: {name}")
            cookie_issues = []

            if "secure" not in attrs_lower:
                await log(f"  {name}: missing Secure flag", level="err")
                cookie_issues.append(Finding(
                    key=f"{name} — Secure",
                    detail=f"Cookie '{name}' missing Secure flag — transmittable over HTTP",
                    severity=SeverityLevel.CRITICAL
                ))

            if "httponly" not in attrs_lower:
                await log(f"  {name}: missing HttpOnly flag", level="err")
                cookie_issues.append(Finding(
                    key=f"{name} — HttpOnly",
                    detail=f"Cookie '{name}' missing HttpOnly — accessible via JavaScript (XSS risk)",
                    severity=SeverityLevel.HIGH
                ))

            samesite_set = any("samesite" in a for a in attrs_lower)
            if not samesite_set:
                await log(f"  {name}: no SameSite attribute", level="warn")
                cookie_issues.append(Finding(
                    key=f"{name} — SameSite",
                    detail=f"Cookie '{name}' has no SameSite attribute — CSRF risk",
                    severity=SeverityLevel.MEDIUM
                ))

            if not cookie_issues:
                await log(f"  {name}: all security flags present", level="ok")
                findings.append(Finding(
                    key=name,
                    detail="Secure, HttpOnly, and SameSite all set",
                    severity=SeverityLevel.PASS
                ))
            else:
                findings.extend(cookie_issues)

        severities = [f.severity for f in findings if f.severity != SeverityLevel.PASS]
        if SeverityLevel.CRITICAL in severities:
            status = SeverityLevel.CRITICAL
        elif SeverityLevel.HIGH in severities:
            status = SeverityLevel.HIGH
        elif SeverityLevel.MEDIUM in severities:
            status = SeverityLevel.MEDIUM
        else:
            status = SeverityLevel.PASS

    except Exception as e:
        await log(f"Cookie audit error: {e}", level="err")
        findings.append(Finding(key="Error", detail=str(e), severity=SeverityLevel.MEDIUM))
        status = SeverityLevel.MEDIUM

    return ModuleResult(module="Cookie Security", status=status, findings=findings)

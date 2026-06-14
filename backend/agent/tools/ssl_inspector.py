import ssl
import socket
import asyncio
from datetime import datetime
from backend.models.report import ModuleResult, Finding, SeverityLevel


async def run_ssl_inspector(url: str, log) -> ModuleResult:
    await log("Inspecting SSL/TLS certificate...")
    findings = []
    hostname = url.replace("https://", "").replace("http://", "").split("/")[0]

    try:
        loop = asyncio.get_event_loop()

        def get_cert():
            ctx = ssl.create_default_context()
            with ctx.wrap_socket(socket.socket(), server_hostname=hostname) as s:
                s.settimeout(8)
                s.connect((hostname, 443))
                return s.getpeercert(), s.version()

        cert, tls_version = await loop.run_in_executor(None, get_cert)

        # TLS version
        await log(f"TLS version: {tls_version}")
        if tls_version in ("TLSv1.2", "TLSv1.3"):
            findings.append(Finding(key="TLS Version", detail=f"{tls_version} — acceptable", severity=SeverityLevel.PASS))
        else:
            findings.append(Finding(key="TLS Version", detail=f"{tls_version} — outdated, upgrade required", severity=SeverityLevel.HIGH))

        # Expiry
        expiry_str = cert.get("notAfter", "")
        if expiry_str:
            expiry = datetime.strptime(expiry_str, "%b %d %H:%M:%S %Y %Z")
            days_left = (expiry - datetime.utcnow()).days
            await log(f"Certificate expires: {expiry.strftime('%Y-%m-%d')} ({days_left} days remaining)")
            if days_left < 0:
                findings.append(Finding(key="Cert Expiry", detail="Certificate has EXPIRED", severity=SeverityLevel.CRITICAL))
            elif days_left < 14:
                findings.append(Finding(key="Cert Expiry", detail=f"Expires in {days_left} days — renew immediately", severity=SeverityLevel.HIGH))
            elif days_left < 30:
                findings.append(Finding(key="Cert Expiry", detail=f"Expires in {days_left} days — renew soon", severity=SeverityLevel.MEDIUM))
            else:
                findings.append(Finding(key="Cert Expiry", detail=f"Valid for {days_left} more days", severity=SeverityLevel.PASS))

        # Issuer
        issuer = dict(x[0] for x in cert.get("issuer", []))
        org = issuer.get("organizationName", "Unknown")
        await log(f"Certificate issuer: {org}")
        findings.append(Finding(key="Issuer", detail=org, severity=SeverityLevel.PASS))

        # SAN
        sans = [v for t, v in cert.get("subjectAltName", []) if t == "DNS"]
        await log(f"Subject Alternative Names: {', '.join(sans[:3])}")
        findings.append(Finding(key="SAN Count", detail=f"{len(sans)} domain(s) covered", severity=SeverityLevel.PASS))

        severities = [f.severity for f in findings]
        if SeverityLevel.CRITICAL in severities:
            status = SeverityLevel.CRITICAL
        elif SeverityLevel.HIGH in severities:
            status = SeverityLevel.HIGH
        elif SeverityLevel.MEDIUM in severities:
            status = SeverityLevel.MEDIUM
        else:
            status = SeverityLevel.PASS

    except ssl.SSLError as e:
        await log(f"SSL error: {e}", level="err")
        findings.append(Finding(key="SSL Error", detail=str(e), severity=SeverityLevel.CRITICAL))
        status = SeverityLevel.CRITICAL
    except Exception as e:
        await log(f"Could not connect on port 443: {e}", level="warn")
        findings.append(Finding(key="No HTTPS", detail="Port 443 unreachable — site may not support HTTPS", severity=SeverityLevel.CRITICAL))
        status = SeverityLevel.CRITICAL

    return ModuleResult(module="SSL / TLS", status=status, findings=findings)

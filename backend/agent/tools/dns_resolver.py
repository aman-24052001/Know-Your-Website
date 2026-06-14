import socket
import asyncio
from backend.models.report import ModuleResult, Finding, SeverityLevel


async def run_dns_resolver(url: str, log) -> ModuleResult:
    await log(f"Resolving DNS records for {url}...")
    findings = []

    try:
        hostname = url.replace("https://", "").replace("http://", "").split("/")[0]
        loop = asyncio.get_event_loop()

        # A record
        ip = await loop.run_in_executor(None, socket.gethostbyname, hostname)
        await log(f"A record resolved — IP: {ip}")
        findings.append(Finding(key="A Record", detail=f"Resolved to {ip}", severity=SeverityLevel.PASS))

        # Reverse DNS
        try:
            reverse = await loop.run_in_executor(None, socket.gethostbyaddr, ip)
            await log(f"Reverse DNS: {reverse[0]}")
            findings.append(Finding(key="Reverse DNS", detail=reverse[0], severity=SeverityLevel.PASS))
        except Exception:
            await log("Reverse DNS lookup failed — no PTR record")
            findings.append(Finding(key="Reverse DNS", detail="No PTR record found", severity=SeverityLevel.LOW))

        # MX records via socket (basic)
        try:
            mx = await loop.run_in_executor(None, socket.getaddrinfo, hostname, None)
            addrs = list({r[4][0] for r in mx})
            await log(f"Resolved {len(addrs)} address(es) for {hostname}")
            if len(addrs) > 1:
                findings.append(Finding(key="Multiple IPs", detail=f"{len(addrs)} IPs — possible CDN/load balancer", severity=SeverityLevel.PASS))
        except Exception:
            pass

        status = SeverityLevel.PASS
        if any(f.severity == SeverityLevel.LOW for f in findings):
            status = SeverityLevel.LOW

    except Exception as e:
        await log(f"DNS resolution failed: {e}", level="err")
        findings.append(Finding(key="DNS Error", detail=str(e), severity=SeverityLevel.HIGH))
        status = SeverityLevel.HIGH

    return ModuleResult(module="DNS Resolution", status=status, findings=findings)

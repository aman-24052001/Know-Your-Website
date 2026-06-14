import asyncio
import os
from datetime import datetime, timezone
from typing import AsyncGenerator
import json
from urllib.parse import urlparse

# Load .env from project root
from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../../../.env"))

import anthropic

from backend.agent.tools.dns_resolver import run_dns_resolver
from backend.agent.tools.ssl_inspector import run_ssl_inspector
from backend.agent.tools.header_analyzer import run_header_analyzer
from backend.agent.tools.endpoint_scanner import run_endpoint_scanner
from backend.agent.tools.robots_parser import run_robots_parser
from backend.agent.tools.redirect_checker import run_redirect_checker
from backend.agent.tools.cookie_auditor import run_cookie_auditor
from backend.agent.tools.security_scorer import compute_score
from backend.agent.prompts import SYSTEM_PROMPT, build_synthesis_prompt
from backend.models.report import AuditReport


TOOL_PIPELINE = [
    run_dns_resolver,
    run_ssl_inspector,
    run_header_analyzer,
    run_endpoint_scanner,
    run_robots_parser,
    run_redirect_checker,
    run_cookie_auditor,
]


def normalise_url(url: str) -> str:
    if not url.startswith("http"):
        url = "https://" + url
    return url


def get_root_url(url: str) -> str:
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}"


async def run_audit(url: str) -> AsyncGenerator[str, None]:
    """
    Streams SSE-formatted events:
      data: {"type": "log", "ts": "...", "text": "...", "level": "info|ok|warn|err"}
      data: {"type": "report", "payload": {...}}
    """
    queue: asyncio.Queue[str | None] = asyncio.Queue()

    url = normalise_url(url)
    root_url = get_root_url(url)

    async def log(text: str, level: str = "info"):
        ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
        event = json.dumps({"type": "log", "ts": ts, "text": text, "level": level})
        await queue.put(f"data: {event}\n\n")

    async def run_pipeline():
        module_results = []

        await log(f"Initialising audit agent for {url}", level="info")
        if root_url != url:
            await log(f"Root domain: {root_url}", level="info")
        await asyncio.sleep(0.15)

        for tool_fn in TOOL_PIPELINE:
            try:
                result = await tool_fn(url, log)
                module_results.append(result)
            except Exception as e:
                await log(f"Tool error in {tool_fn.__name__}: {e}", level="err")

        score = compute_score(module_results)
        await log(f"All modules complete — score: {score}/100", level="ok")

        # LLM synthesis
        api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
        if not api_key:
            await log("ANTHROPIC_API_KEY not set — skipping LLM synthesis", level="warn")
            summary = f"Score: {score}/100. {sum(1 for m in module_results for f in m.findings if f.severity.value in ('CRITICAL','HIGH'))} high-severity issues found. Add ANTHROPIC_API_KEY to .env for LLM analysis."
        else:
            await log("Synthesising report with LLM...", level="info")
            try:
                client = anthropic.AsyncAnthropic(api_key=api_key)
                modules_dict = [m.model_dump() for m in module_results]
                prompt = build_synthesis_prompt(url, modules_dict, score)

                summary_parts = []
                async with client.messages.stream(
                    model="claude-sonnet-4-6",
                    max_tokens=300,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": prompt}],
                ) as stream:
                    async for text in stream.text_stream:
                        summary_parts.append(text)

                summary = "".join(summary_parts).strip()
                await log("LLM analysis complete", level="ok")

            except Exception as e:
                await log(f"LLM synthesis failed: {e}", level="warn")
                summary = f"Score: {score}/100. LLM synthesis unavailable. Review module findings for details."

        report = AuditReport(
            url=url,
            score=score,
            modules=module_results,
            summary=summary,
            scanned_at=datetime.now(timezone.utc).isoformat(),
            checks_performed=sum(len(m.findings) for m in module_results),
        )

        report_event = json.dumps({"type": "report", "payload": report.model_dump()})
        await queue.put(f"data: {report_event}\n\n")
        await queue.put(None)

    asyncio.create_task(run_pipeline())

    while True:
        item = await queue.get()
        if item is None:
            break
        yield item

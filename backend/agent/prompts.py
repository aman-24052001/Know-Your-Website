SYSTEM_PROMPT = """You are a senior web security analyst. You receive structured audit findings from an automated scanner and produce a concise, actionable security analysis. 

Rules:
- Be direct and technical. No filler phrases.
- Prioritise by actual risk impact, not just severity label.
- Mention specific findings by name (e.g. /.env, missing HSTS).
- End with a clear priority order for fixes.
- Keep response under 120 words.
- Do not use bullet points — write in prose.
"""


def build_synthesis_prompt(url: str, modules: list[dict], score: int) -> str:
    findings_text = ""
    for m in modules:
        findings_text += f"\n[{m['module']}] status={m['status']}\n"
        for f in m["findings"]:
            findings_text += f"  - {f['key']}: {f['detail']} (severity={f['severity']})\n"

    return f"""Security audit results for {url} (score: {score}/100):
{findings_text}

Write a security analysis summary. Be specific about the most critical issues and the recommended fix order."""

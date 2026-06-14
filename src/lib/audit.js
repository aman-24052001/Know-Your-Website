// Severity levels
export const SEV = {
  CRITICAL: 'CRITICAL',
  HIGH: 'HIGH',
  MEDIUM: 'MEDIUM',
  LOW: 'LOW',
  PASS: 'PASS',
}

const DEDUCTIONS = { CRITICAL: 25, HIGH: 15, MEDIUM: 8, LOW: 3, PASS: 0 }

export function computeScore(modules) {
  let score = 100
  for (const mod of modules) {
    for (const f of mod.findings) {
      score -= DEDUCTIONS[f.severity] || 0
    }
  }
  return Math.max(0, Math.min(100, score))
}

function worstSeverity(findings) {
  const order = [SEV.CRITICAL, SEV.HIGH, SEV.MEDIUM, SEV.LOW, SEV.PASS]
  for (const sev of order) {
    if (findings.some(f => f.severity === sev)) return sev
  }
  return SEV.PASS
}

// ── DNS Resolution ──────────────────────────────────────────────
// Browser can't do raw DNS — we use DoH (DNS over HTTPS via Cloudflare)
export async function runDNS(url, log) {
  await log('Resolving DNS records via DNS-over-HTTPS...', 'info')
  const findings = []
  try {
    const hostname = new URL(url).hostname

    // A records
    const r4 = await fetch(`https://cloudflare-dns.com/dns-query?name=${hostname}&type=A`, {
      headers: { Accept: 'application/dns-json' }
    })
    const d4 = await r4.json()
    const aRecords = (d4.Answer || []).filter(a => a.type === 1).map(a => a.data)

    if (aRecords.length > 0) {
      await log(`A records: ${aRecords.join(', ')}`, 'ok')
      findings.push({ key: 'A Record', detail: `${aRecords.length} IP(s): ${aRecords.join(', ')}`, severity: SEV.PASS })
      if (aRecords.length > 1) {
        findings.push({ key: 'Multiple IPs', detail: `${aRecords.length} addresses — likely CDN or load balancer`, severity: SEV.PASS })
      }
    } else {
      await log('No A records found', 'warn')
      findings.push({ key: 'A Record', detail: 'No A records resolved', severity: SEV.HIGH })
    }

    // AAAA
    const r6 = await fetch(`https://cloudflare-dns.com/dns-query?name=${hostname}&type=AAAA`, {
      headers: { Accept: 'application/dns-json' }
    })
    const d6 = await r6.json()
    const aaaaRecords = (d6.Answer || []).filter(a => a.type === 28)
    if (aaaaRecords.length > 0) {
      await log(`IPv6 support: ${aaaaRecords.length} AAAA record(s)`, 'ok')
      findings.push({ key: 'IPv6', detail: `${aaaaRecords.length} AAAA record(s) — dual-stack`, severity: SEV.PASS })
    } else {
      await log('No IPv6 records', 'info')
      findings.push({ key: 'IPv6', detail: 'No AAAA records — IPv4 only', severity: SEV.LOW })
    }

    // MX
    const rmx = await fetch(`https://cloudflare-dns.com/dns-query?name=${hostname}&type=MX`, {
      headers: { Accept: 'application/dns-json' }
    })
    const dmx = await rmx.json()
    const mxRecords = (dmx.Answer || []).filter(a => a.type === 15)
    if (mxRecords.length > 0) {
      await log(`MX records found: ${mxRecords.length}`, 'ok')
      findings.push({ key: 'MX Records', detail: `${mxRecords.length} mail exchanger(s) configured`, severity: SEV.PASS })
    }

  } catch (e) {
    await log(`DNS lookup failed: ${e.message}`, 'err')
    findings.push({ key: 'DNS Error', detail: e.message, severity: SEV.HIGH })
  }

  return { module: 'DNS Resolution', status: worstSeverity(findings), findings }
}

// ── SSL / TLS ────────────────────────────────────────────────────
// Browser can't inspect TLS directly — use SSL Labs API (free, no key needed)
export async function runSSL(url, log) {
  await log('Checking SSL/TLS via Qualys SSL Labs API...', 'info')
  const findings = []
  try {
    const hostname = new URL(url).hostname

    // Start analysis
    const startR = await fetch(
      `https://api.ssllabs.com/api/v3/analyze?host=${hostname}&startNew=on&all=done`,
      { headers: { Accept: 'application/json' } }
    )
    const startD = await startR.json()

    if (startD.status === 'ERROR') {
      throw new Error(startD.statusMessage || 'SSL Labs error')
    }

    // Poll until ready (max 30s for static analysis)
    let data = startD
    let attempts = 0
    while (data.status !== 'READY' && data.status !== 'ERROR' && attempts < 6) {
      await new Promise(r => setTimeout(r, 5000))
      const pollR = await fetch(
        `https://api.ssllabs.com/api/v3/analyze?host=${hostname}&all=done`,
        { headers: { Accept: 'application/json' } }
      )
      data = await pollR.json()
      attempts++
      await log(`SSL analysis: ${data.status}...`, 'info')
    }

    if (data.status === 'READY' && data.endpoints?.length > 0) {
      const ep = data.endpoints[0]
      const grade = ep.grade || 'Unknown'
      await log(`SSL grade: ${grade}`, grade.startsWith('A') ? 'ok' : 'warn')

      const gradeSev = grade.startsWith('A') ? SEV.PASS : grade.startsWith('B') ? SEV.MEDIUM : SEV.HIGH
      findings.push({ key: 'SSL Grade', detail: `Qualys grade: ${grade}`, severity: gradeSev })

      if (ep.details) {
        const proto = ep.details.protocols?.map(p => p.name + ' ' + p.version).join(', ')
        if (proto) {
          await log(`Protocols: ${proto}`, 'ok')
          const hasTLS13 = ep.details.protocols?.some(p => p.version === '1.3')
          findings.push({
            key: 'TLS Protocols',
            detail: proto,
            severity: hasTLS13 ? SEV.PASS : SEV.MEDIUM
          })
        }
        if (ep.details.cert) {
          const expiry = new Date(ep.details.cert.notAfter)
          const daysLeft = Math.floor((expiry - Date.now()) / 86400000)
          await log(`Certificate expires: ${expiry.toISOString().split('T')[0]} (${daysLeft} days)`, daysLeft < 14 ? 'warn' : 'ok')
          findings.push({
            key: 'Cert Expiry',
            detail: `${expiry.toISOString().split('T')[0]} — ${daysLeft} days remaining`,
            severity: daysLeft < 0 ? SEV.CRITICAL : daysLeft < 14 ? SEV.HIGH : daysLeft < 30 ? SEV.MEDIUM : SEV.PASS
          })
        }
      }
    } else {
      // SSL Labs unavailable or cached miss — fall back to basic check
      await log('SSL Labs result unavailable — performing basic HTTPS check', 'warn')
      try {
        const r = await fetch(url, { method: 'HEAD', signal: AbortSignal.timeout(8000) })
        findings.push({ key: 'HTTPS', detail: `Site reachable over HTTPS (HTTP ${r.status})`, severity: SEV.PASS })
        await log('HTTPS connection successful', 'ok')
      } catch {
        findings.push({ key: 'HTTPS', detail: 'Could not connect over HTTPS', severity: SEV.CRITICAL })
        await log('HTTPS connection failed', 'err')
      }
    }

  } catch (e) {
    await log(`SSL check error: ${e.message} — trying basic HTTPS`, 'warn')
    try {
      const r = await fetch(url, { method: 'HEAD', signal: AbortSignal.timeout(8000) })
      findings.push({ key: 'HTTPS', detail: `HTTPS reachable (HTTP ${r.status})`, severity: SEV.PASS })
      await log('HTTPS reachable', 'ok')
    } catch {
      findings.push({ key: 'HTTPS', detail: 'HTTPS unreachable', severity: SEV.CRITICAL })
    }
  }

  if (findings.length === 0) {
    findings.push({ key: 'SSL', detail: 'Could not determine SSL status', severity: SEV.MEDIUM })
  }

  return { module: 'SSL / TLS', status: worstSeverity(findings), findings }
}

// ── Security Headers ─────────────────────────────────────────────
// Use a CORS-friendly headers API
const REQUIRED_HEADERS = [
  { key: 'strict-transport-security', label: 'HSTS', severity: SEV.HIGH },
  { key: 'content-security-policy', label: 'CSP', severity: SEV.HIGH },
  { key: 'x-frame-options', label: 'X-Frame-Options', severity: SEV.MEDIUM },
  { key: 'x-content-type-options', label: 'X-Content-Type-Options', severity: SEV.MEDIUM },
  { key: 'referrer-policy', label: 'Referrer-Policy', severity: SEV.LOW },
  { key: 'permissions-policy', label: 'Permissions-Policy', severity: SEV.LOW },
]

const INFO_HEADERS = ['server', 'x-powered-by', 'x-aspnet-version']

export async function runHeaders(url, log) {
  await log('Fetching security headers via securityheaders.com API...', 'info')
  const findings = []

  try {
    const hostname = new URL(url).hostname
    const rootUrl = `https://${hostname}`

    // Use securityheaders.com API
    const apiUrl = `https://securityheaders.com/?q=${encodeURIComponent(rootUrl)}&followRedirects=on&hide=on`
    const r = await fetch(apiUrl, { signal: AbortSignal.timeout(12000) })
    const html = await r.text()

    // Parse grade from response
    const gradeMatch = html.match(/class="score-([A-Fa-f+])"/)
    if (gradeMatch) {
      const grade = gradeMatch[1].replace('plus', '+')
      await log(`Security headers grade: ${grade}`, grade === 'A' || grade === 'A+' ? 'ok' : 'warn')
    }

    // Check which headers are present/missing from HTML
    for (const { key, label, severity } of REQUIRED_HEADERS) {
      const present = html.toLowerCase().includes(key.toLowerCase())
      if (present) {
        await log(`${label}: present`, 'ok')
        findings.push({ key: label, detail: 'Header present', severity: SEV.PASS })
      } else {
        await log(`${label}: missing`, 'warn')
        findings.push({ key: label, detail: `${label} not set`, severity })
      }
    }

  } catch (e) {
    // Fall back: try fetch with no-cors and analyse what we can
    await log(`Headers API unavailable: ${e.message} — trying direct fetch`, 'warn')
    try {
      const hostname = new URL(url).hostname
      // Use allorigins proxy to get headers
      const proxyUrl = `https://api.allorigins.win/get?url=${encodeURIComponent(`https://${hostname}`)}`
      const r = await fetch(proxyUrl, { signal: AbortSignal.timeout(10000) })
      const data = await r.json()
      const headers = data.headers || {}

      for (const { key, label, severity } of REQUIRED_HEADERS) {
        const val = headers[key] || headers[key.toLowerCase()]
        if (val) {
          await log(`${label}: present — ${String(val).slice(0, 50)}`, 'ok')
          findings.push({ key: label, detail: `Present: ${String(val).slice(0, 80)}`, severity: SEV.PASS })
        } else {
          await log(`${label}: missing`, 'warn')
          findings.push({ key: label, detail: `${label} not set`, severity })
        }
      }

      for (const h of INFO_HEADERS) {
        const val = headers[h]
        if (val) {
          await log(`Info-leaking header: ${h}: ${val}`, 'warn')
          findings.push({ key: h, detail: `Leaks server info: ${val}`, severity: SEV.LOW })
        }
      }
    } catch (e2) {
      await log(`Could not fetch headers: ${e2.message}`, 'err')
      findings.push({ key: 'Fetch Error', detail: 'Could not retrieve headers — CORS or network restriction', severity: SEV.MEDIUM })
    }
  }

  return { module: 'Security Headers', status: worstSeverity(findings), findings }
}

// ── Exposed Endpoints ────────────────────────────────────────────
const PROBE_PATHS = [
  { path: '/.env', label: 'Environment File', severity: SEV.CRITICAL },
  { path: '/.env.local', label: 'Local Env File', severity: SEV.CRITICAL },
  { path: '/.git/config', label: 'Git Config', severity: SEV.CRITICAL },
  { path: '/backup.sql', label: 'SQL Backup', severity: SEV.CRITICAL },
  { path: '/dump.sql', label: 'SQL Dump', severity: SEV.CRITICAL },
  { path: '/db.sqlite', label: 'SQLite DB', severity: SEV.CRITICAL },
  { path: '/admin', label: 'Admin Panel', severity: SEV.HIGH },
  { path: '/phpmyadmin', label: 'phpMyAdmin', severity: SEV.HIGH },
  { path: '/server-status', label: 'Server Status', severity: SEV.HIGH },
  { path: '/actuator/env', label: 'Spring Actuator Env', severity: SEV.CRITICAL },
  { path: '/swagger-ui.html', label: 'Swagger UI', severity: SEV.MEDIUM },
  { path: '/openapi.json', label: 'OpenAPI Spec', severity: SEV.MEDIUM },
  { path: '/wp-config.php.bak', label: 'WP Config Backup', severity: SEV.CRITICAL },
]

async function probe(url) {
  try {
    const r = await fetch(url, {
      method: 'HEAD',
      redirect: 'manual',
      signal: AbortSignal.timeout(6000),
    })
    return { status: r.status, size: null }
  } catch {
    return { status: null, size: null }
  }
}

export async function runEndpoints(url, log) {
  await log('Scanning for exposed endpoints and sensitive files...', 'info')
  const findings = []
  const base = `https://${new URL(url).hostname}`

  // Baseline
  const { status: baseStatus } = await probe(`${base}/___kyw_nonexistent_probe___`)
  await log(`Baseline probe: HTTP ${baseStatus ?? 'timeout'} for nonexistent path`, 'info')

  for (const { path, label, severity } of PROBE_PATHS) {
    const { status } = await probe(`${base}${path}`)
    if (status === null) continue

    if (status === baseStatus && status !== 200) {
      await log(`${path} — HTTP ${status} (matches baseline, skipping)`, 'ok')
      continue
    }

    if (status === 200) {
      await log(`${path} returned HTTP 200 — ${label} may be exposed`, 'err')
      findings.push({ key: label, detail: `${path} returned HTTP 200 — verify accessibility`, severity })
    } else if (status === 403 && baseStatus !== 403) {
      await log(`${path} — HTTP 403, resource exists but blocked`, 'warn')
      findings.push({ key: label, detail: `${path} exists (HTTP 403) — blocked but discoverable`, severity: SEV.LOW })
    } else {
      await log(`${path} — HTTP ${status}, not accessible`, 'ok')
    }
  }

  if (findings.length === 0) {
    await log('No sensitive endpoints exposed', 'ok')
    findings.push({ key: 'Endpoint Scan', detail: 'No sensitive files or panels accessible', severity: SEV.PASS })
  }

  return { module: 'Exposed Endpoints', status: worstSeverity(findings), findings }
}

// ── Robots & Sitemap ─────────────────────────────────────────────
const SENSITIVE_PATTERNS = ['admin', 'backup', 'config', 'secret', 'private', 'internal', 'staging', 'dev', 'debug', '.env', 'database']

export async function runRobots(url, log) {
  await log('Parsing robots.txt and sitemap.xml...', 'info')
  const findings = []
  const base = `https://${new URL(url).hostname}`

  // robots.txt
  try {
    const r = await fetch(`${base}/robots.txt`, { signal: AbortSignal.timeout(8000) })
    if (r.ok) {
      const text = await r.text()
      const disallowed = text.split('\n')
        .filter(l => l.toLowerCase().startsWith('disallow:'))
        .map(l => l.split(':')[1]?.trim())
        .filter(Boolean)
      await log(`robots.txt found — ${disallowed.length} Disallow rule(s)`, 'ok')

      const flagged = disallowed.filter(p => SENSITIVE_PATTERNS.some(s => p.toLowerCase().includes(s)))
      if (flagged.length > 0) {
        await log(`Sensitive paths disclosed in robots.txt: ${flagged.slice(0, 3).join(', ')}`, 'warn')
        findings.push({ key: 'robots.txt Disclosure', detail: `Sensitive paths revealed: ${flagged.slice(0, 5).join(', ')}`, severity: SEV.MEDIUM })
      } else {
        findings.push({ key: 'robots.txt', detail: `Present — ${disallowed.length} disallow rules, no sensitive disclosure`, severity: SEV.PASS })
      }
    } else {
      await log(`robots.txt not found (HTTP ${r.status})`, 'info')
      findings.push({ key: 'robots.txt', detail: 'Not present', severity: SEV.LOW })
    }
  } catch (e) {
    await log(`robots.txt fetch error: ${e.message}`, 'warn')
    findings.push({ key: 'robots.txt', detail: `Fetch error: ${e.message}`, severity: SEV.LOW })
  }

  // sitemap.xml
  try {
    const r = await fetch(`${base}/sitemap.xml`, { signal: AbortSignal.timeout(8000) })
    if (r.ok) {
      const text = await r.text()
      await log(`sitemap.xml found — ${text.length} bytes`, 'ok')
      findings.push({ key: 'sitemap.xml', detail: `Present — ${text.length} bytes`, severity: SEV.PASS })
    } else {
      await log('sitemap.xml not found', 'info')
      findings.push({ key: 'sitemap.xml', detail: 'Not present', severity: SEV.LOW })
    }
  } catch (e) {
    findings.push({ key: 'sitemap.xml', detail: `Fetch error: ${e.message}`, severity: SEV.LOW })
  }

  return { module: 'Robots & Sitemap', status: worstSeverity(findings), findings }
}

// ── Open Redirects ───────────────────────────────────────────────
const REDIRECT_PARAMS = ['url', 'redirect', 'next', 'return', 'goto', 'redir', 'destination', 'target']
const TEST_PAYLOAD = 'https://evil-example-kyw.com'

export async function runRedirects(url, log) {
  await log('Checking for open redirect vulnerabilities...', 'info')
  const findings = []
  const base = `https://${new URL(url).hostname}`
  const flagged = []

  for (const param of REDIRECT_PARAMS) {
    try {
      const testUrl = `${base}/?${param}=${encodeURIComponent(TEST_PAYLOAD)}`
      const r = await fetch(testUrl, { redirect: 'manual', signal: AbortSignal.timeout(5000) })
      if ([301, 302, 303, 307, 308].includes(r.status)) {
        const loc = r.headers.get('location') || ''
        if (loc.includes('evil-example-kyw.com')) {
          await log(`Open redirect confirmed at ?${param}=`, 'err')
          flagged.push(param)
          findings.push({ key: `Open Redirect (?${param}=)`, detail: `Redirects to attacker URL via ?${param}= parameter`, severity: SEV.HIGH })
        }
      }
    } catch { /* timeout or network — skip */ }
  }

  // Common redirect paths
  for (const path of ['/go', '/redirect', '/out', '/r', '/forward']) {
    try {
      const testUrl = `${base}${path}?url=${encodeURIComponent(TEST_PAYLOAD)}`
      const r = await fetch(testUrl, { redirect: 'manual', signal: AbortSignal.timeout(5000) })
      const loc = r.headers.get('location') || ''
      if ([301, 302, 303, 307, 308].includes(r.status) && loc.includes('evil-example-kyw.com')) {
        await log(`Open redirect at ${path}`, 'err')
        findings.push({ key: `Open Redirect (${path})`, detail: `${path} performs unvalidated external redirect`, severity: SEV.HIGH })
      }
    } catch { /* skip */ }
  }

  if (findings.length === 0) {
    await log('No open redirect vulnerabilities detected', 'ok')
    findings.push({ key: 'Open Redirects', detail: 'No unvalidated redirect parameters found', severity: SEV.PASS })
  }

  return { module: 'Open Redirects', status: worstSeverity(findings), findings }
}

// ── Cookie Security ──────────────────────────────────────────────
export async function runCookies(url, log) {
  await log('Auditing cookie security via proxy...', 'info')
  const findings = []

  try {
    // Use allorigins to get headers including Set-Cookie
    const proxyUrl = `https://api.allorigins.win/get?url=${encodeURIComponent(url)}`
    const r = await fetch(proxyUrl, { signal: AbortSignal.timeout(10000) })
    const data = await r.json()
    const headers = data.headers || {}

    const cookieHeaders = Object.entries(headers)
      .filter(([k]) => k.toLowerCase() === 'set-cookie')
      .map(([, v]) => v)

    // allorigins may join cookies with comma
    const rawCookies = cookieHeaders.flatMap(c => c.split(/,(?=[^ ])/))

    if (rawCookies.length === 0) {
      await log('No Set-Cookie headers found', 'info')
      findings.push({ key: 'Cookies', detail: 'No cookies set on initial response', severity: SEV.PASS })
      return { module: 'Cookie Security', status: SEV.PASS, findings }
    }

    await log(`Found ${rawCookies.length} cookie(s) to audit`, 'info')

    for (const raw of rawCookies) {
      const parts = raw.split(';').map(p => p.trim())
      const name = parts[0].split('=')[0].trim()
      const attrsLower = parts.slice(1).map(p => p.toLowerCase())

      await log(`Auditing cookie: ${name}`, 'info')

      if (!attrsLower.includes('secure')) {
        await log(`  ${name}: missing Secure flag`, 'err')
        findings.push({ key: `${name} — Secure`, detail: `Cookie '${name}' missing Secure flag — transmittable over HTTP`, severity: SEV.CRITICAL })
      }
      if (!attrsLower.includes('httponly')) {
        await log(`  ${name}: missing HttpOnly flag`, 'err')
        findings.push({ key: `${name} — HttpOnly`, detail: `Cookie '${name}' missing HttpOnly — accessible via JavaScript`, severity: SEV.HIGH })
      }
      if (!attrsLower.some(a => a.startsWith('samesite'))) {
        await log(`  ${name}: no SameSite attribute`, 'warn')
        findings.push({ key: `${name} — SameSite`, detail: `Cookie '${name}' has no SameSite attribute — CSRF risk`, severity: SEV.MEDIUM })
      }
      if (attrsLower.includes('secure') && attrsLower.includes('httponly') && attrsLower.some(a => a.startsWith('samesite'))) {
        await log(`  ${name}: all security flags present`, 'ok')
        findings.push({ key: name, detail: 'Secure, HttpOnly, and SameSite all set', severity: SEV.PASS })
      }
    }
  } catch (e) {
    await log(`Cookie audit error: ${e.message}`, 'warn')
    findings.push({ key: 'Cookie Check', detail: `Could not retrieve cookies: ${e.message}`, severity: SEV.LOW })
  }

  return { module: 'Cookie Security', status: worstSeverity(findings), findings }
}

// ── LLM Synthesis ────────────────────────────────────────────────
export async function synthesiseWithLLM(url, modules, score, apiKey) {
  const findingsText = modules.map(m => {
    const lines = m.findings.map(f => `  - ${f.key}: ${f.detail} (${f.severity})`).join('\n')
    return `[${m.module}] status=${m.status}\n${lines}`
  }).join('\n\n')

  const prompt = `Security audit results for ${url} (score: ${score}/100):\n\n${findingsText}\n\nWrite a security analysis summary. Be direct and technical. Mention specific findings by name. Prioritise by actual risk. End with a clear fix order. Under 120 words. Prose only, no bullets.`

  const response = await fetch('https://api.anthropic.com/v1/messages', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'x-api-key': apiKey,
      'anthropic-version': '2023-06-01',
      'anthropic-dangerous-direct-browser-access': 'true',
    },
    body: JSON.stringify({
      model: 'claude-sonnet-4-6',
      max_tokens: 300,
      system: 'You are a senior web security analyst. Be direct and technical. No filler phrases. No bullet points. Write in prose.',
      messages: [{ role: 'user', content: prompt }],
    }),
  })

  if (!response.ok) {
    const err = await response.json()
    throw new Error(err.error?.message || `API error ${response.status}`)
  }

  const data = await response.json()
  return data.content?.[0]?.text || ''
}

// ── Main pipeline ────────────────────────────────────────────────
export const PIPELINE = [
  runDNS,
  runSSL,
  runHeaders,
  runEndpoints,
  runRobots,
  runRedirects,
  runCookies,
]

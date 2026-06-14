import { Shield, CheckCircle, XCircle, Sparkles, RefreshCw, Globe, Lock, FileText, ArrowUpRight, Cookie, AlertTriangle } from 'lucide-react'
import { FindingBadge } from './FindingBadge.jsx'

const DOT_COLOR = { CRITICAL: '#FF3B00', HIGH: '#FF7A00', MEDIUM: '#c8960a', LOW: '#888', PASS: '#00C853' }
const MODULE_ICON = {
  'DNS Resolution': Globe, 'SSL / TLS': Lock, 'Security Headers': FileText,
  'Exposed Endpoints': Shield, 'Robots & Sitemap': FileText,
  'Open Redirects': ArrowUpRight, 'Cookie Security': Cookie,
}

function FindingCard({ mod, isMobile }) {
  const Icon = MODULE_ICON[mod.module] || Shield
  const nonPass = mod.findings.filter(f => f.severity !== 'PASS')
  const pass = mod.findings.filter(f => f.severity === 'PASS')

  return (
    <div style={{ border: 'var(--border)', boxShadow: 'var(--shadow)', background: '#fff', overflow: 'hidden' }}>
      <div style={{
        padding: '12px 16px', borderBottom: 'var(--border)',
        display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 8,
      }}>
        <div style={{
          display: 'flex', alignItems: 'center', gap: 8,
          fontSize: 12, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.04em',
          minWidth: 0,
        }}>
          <Icon size={14} style={{ flexShrink: 0 }} />
          <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{mod.module}</span>
        </div>
        <FindingBadge severity={mod.status} />
      </div>
      <div style={{ padding: '12px 16px' }}>
        {nonPass.slice(0, 3).map((f, i) => (
          <div key={i} style={{ display: 'flex', alignItems: 'flex-start', gap: 8, marginBottom: 7, fontSize: 12, lineHeight: 1.5 }}>
            <div style={{
              width: 8, height: 8, flexShrink: 0, marginTop: 4,
              border: '2px solid', borderColor: DOT_COLOR[f.severity], background: DOT_COLOR[f.severity],
            }} />
            <div style={{ wordBreak: 'break-word' }}>
              <span style={{ fontWeight: 600 }}>{f.key}</span>
              {' '}<span style={{ color: '#555' }}>{f.detail}</span>
            </div>
          </div>
        ))}
        {nonPass.length === 0 && pass.length > 0 && (
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 12, color: '#00C853', fontWeight: 600 }}>
            <CheckCircle size={13} />All {pass.length} check(s) passed
          </div>
        )}
        {nonPass.length > 3 && (
          <div style={{ fontSize: 11, color: '#888', fontFamily: 'var(--font-mono)', marginTop: 4 }}>
            +{nonPass.length - 3} more
          </div>
        )}
      </div>
    </div>
  )
}

export function ReportCard({ report, onReset, isMobile }) {
  if (!report) return null

  const allFindings = report.modules.flatMap(m => m.findings)
  const counts = { CRITICAL: 0, HIGH: 0, MEDIUM: 0, PASS: 0 }
  for (const f of allFindings) { if (f.severity in counts) counts[f.severity]++ }
  const hostname = report.url.replace(/https?:\/\//, '').split('/')[0]

  return (
    <div>
      {/* Stat strip */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', borderBottom: 'var(--border)' }}>
        {[
          { num: counts.CRITICAL, label: 'Critical', color: '#FF3B00' },
          { num: counts.HIGH,     label: 'High',     color: '#FF7A00' },
          { num: counts.MEDIUM,   label: 'Medium',   color: '#c8960a' },
          { num: counts.PASS,     label: 'Passed',   color: '#00C853' },
        ].map(({ num, label, color }, i) => (
          <div key={i} style={{
            padding: isMobile ? '10px 12px' : '12px 20px',
            borderRight: i < 3 ? 'var(--border)' : 'none',
          }}>
            <div style={{
              fontFamily: 'var(--font-mono)',
              fontSize: isMobile ? 18 : 22,
              fontWeight: 700, color, marginBottom: 2,
            }}>{num}</div>
            <div style={{ fontSize: 10, color: '#777', fontWeight: 500, textTransform: 'uppercase', letterSpacing: '0.05em' }}>{label}</div>
          </div>
        ))}
      </div>

      {/* Body */}
      <div style={{ padding: isMobile ? 16 : 24 }}>

        {/* Header */}
        <div style={{ marginBottom: 20 }}>
          <div style={{ fontSize: isMobile ? 16 : 19, fontWeight: 800, marginBottom: 3 }}>
            Audit Report — {hostname}
          </div>
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: '#666' }}>
            {new Date(report.scannedAt).toLocaleTimeString()} &nbsp;|&nbsp; {report.modules.length} modules &nbsp;|&nbsp; {report.checksPerformed} checks
          </div>
        </div>

        {/* Module badges */}
        <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', marginBottom: 20 }}>
          {report.modules.map((m, i) => {
            const isPass = m.status === 'PASS'
            const isCrit = ['CRITICAL', 'HIGH'].includes(m.status)
            return (
              <span key={i} style={{
                border: '2px solid',
                borderColor: isPass ? '#00C853' : isCrit ? '#FF3B00' : '#c8960a',
                background: isPass ? '#00C853' : isCrit ? '#FF3B00' : '#FFE500',
                color: isPass ? '#fff' : isCrit ? '#fff' : '#0A0A0A',
                padding: '3px 8px',
                fontFamily: 'var(--font-mono)', fontSize: 10, fontWeight: 700,
                textTransform: 'uppercase', letterSpacing: '0.05em', whiteSpace: 'nowrap',
              }}>
                {m.module.split(' ')[0]}
              </span>
            )
          })}
        </div>

        {/* Findings label */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 14 }}>
          <span style={{ fontFamily: 'var(--font-mono)', fontSize: 10, fontWeight: 700, letterSpacing: '0.1em', textTransform: 'uppercase', color: '#888', whiteSpace: 'nowrap' }}>
            Findings
          </span>
          <div style={{ flex: 1, height: 1.5, background: '#ccc' }} />
        </div>

        {/* Findings grid — single col on mobile */}
        <div style={{
          display: 'grid',
          gridTemplateColumns: isMobile ? '1fr' : '1fr 1fr',
          gap: 14,
          marginBottom: 20,
        }}>
          {report.modules.map((mod, i) => <FindingCard key={i} mod={mod} isMobile={isMobile} />)}
        </div>

        {/* LLM Summary */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 14 }}>
          <span style={{ fontFamily: 'var(--font-mono)', fontSize: 10, fontWeight: 700, letterSpacing: '0.1em', textTransform: 'uppercase', color: '#888', whiteSpace: 'nowrap' }}>
            Agent Analysis
          </span>
          <div style={{ flex: 1, height: 1.5, background: '#ccc' }} />
        </div>

        <div style={{
          border: 'var(--border)', boxShadow: 'var(--shadow-lg)',
          background: 'var(--ink)', color: 'var(--cream)',
          padding: isMobile ? '16px 20px' : '20px 24px',
          position: 'relative', overflow: 'hidden',
        }}>
          <div style={{ position: 'absolute', top: 0, left: 0, width: 4, height: '100%', background: '#00C8E0' }} />
          <div style={{
            display: 'flex', alignItems: 'center', gap: 8,
            fontFamily: 'var(--font-mono)', fontSize: 10, fontWeight: 700,
            color: '#00C8E0', letterSpacing: '0.12em', textTransform: 'uppercase', marginBottom: 12,
          }}>
            <Sparkles size={13} color="#00C8E0" />
            LLM Security Analysis
          </div>
          <div style={{ fontSize: 13, lineHeight: 1.8, color: '#c8c5be', wordBreak: 'break-word' }}>
            {report.summary}
          </div>
          <div style={{ marginTop: 16 }}>
            <button onClick={onReset} style={{
              height: 36, padding: '0 16px',
              background: 'transparent', border: '2px solid #333', color: 'var(--cream)',
              fontFamily: 'var(--font-mono)', fontSize: 11, fontWeight: 700,
              letterSpacing: '0.06em', textTransform: 'uppercase',
              display: 'flex', alignItems: 'center', gap: 7,
            }}>
              <RefreshCw size={12} />New Audit
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}

import { Shield, AlertTriangle, CheckCircle, XCircle, Sparkles, Download, RefreshCw } from 'lucide-react'
import { FindingBadge } from './FindingBadge.jsx'

const DOT_COLOR = { CRITICAL: '#FF3B00', HIGH: '#FF7A00', MEDIUM: '#c8960a', LOW: '#888', PASS: '#00C853' }

const MODULE_ICON = {
  'DNS Resolution': Shield,
  'SSL / TLS': CheckCircle,
  'Security Headers': AlertTriangle,
  'Exposed Endpoints': XCircle,
  'Robots & Sitemap': Shield,
  'Open Redirects': AlertTriangle,
  'Cookie Security': XCircle,
}

function FindingCard({ mod }) {
  const Icon = MODULE_ICON[mod.module] || Shield
  const nonPass = mod.findings.filter(f => f.severity !== 'PASS')
  const pass = mod.findings.filter(f => f.severity === 'PASS')

  return (
    <div style={{
      border: 'var(--border)', boxShadow: 'var(--shadow)', background: '#fff', overflow: 'hidden',
    }}>
      <div style={{
        padding: '12px 16px', borderBottom: 'var(--border)',
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 12, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.04em' }}>
          <Icon size={14} />
          {mod.module}
        </div>
        <FindingBadge severity={mod.status} />
      </div>
      <div style={{ padding: '12px 16px' }}>
        {nonPass.slice(0, 4).map((f, i) => (
          <div key={i} style={{ display: 'flex', alignItems: 'flex-start', gap: 8, marginBottom: 7, fontSize: 12, lineHeight: 1.5 }}>
            <div style={{
              width: 8, height: 8, border: '2px solid',
              borderColor: DOT_COLOR[f.severity], background: DOT_COLOR[f.severity],
              flexShrink: 0, marginTop: 4
            }} />
            <div>
              <span style={{ fontWeight: 600 }}>{f.key}</span>
              {' '}
              <span style={{ color: '#555' }}>{f.detail}</span>
            </div>
          </div>
        ))}
        {pass.length > 0 && nonPass.length === 0 && (
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 12, color: '#00C853', fontWeight: 600 }}>
            <CheckCircle size={13} />
            All {pass.length} check(s) passed
          </div>
        )}
        {nonPass.length > 4 && (
          <div style={{ fontSize: 11, color: '#888', fontFamily: 'var(--font-mono)', marginTop: 4 }}>
            +{nonPass.length - 4} more findings
          </div>
        )}
      </div>
    </div>
  )
}

export function ReportCard({ report, onReset }) {
  if (!report) {
    return (
      <div style={{ padding: 32, color: '#aaa', fontFamily: 'var(--font-mono)', fontSize: 12 }}>
        Run an audit to see results here.
      </div>
    )
  }

  const criticalCount = report.modules.flatMap(m => m.findings).filter(f => f.severity === 'CRITICAL').length
  const highCount     = report.modules.flatMap(m => m.findings).filter(f => f.severity === 'HIGH').length
  const medCount      = report.modules.flatMap(m => m.findings).filter(f => f.severity === 'MEDIUM').length
  const passCount     = report.modules.flatMap(m => m.findings).filter(f => f.severity === 'PASS').length

  const hostname = report.url.replace(/https?:\/\//, '').split('/')[0]

  return (
    <div>
      {/* Stat strip */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', borderBottom: 'var(--border)' }}>
        {[
          { num: criticalCount, label: 'Critical', color: '#FF3B00' },
          { num: highCount,     label: 'High',     color: '#FF7A00' },
          { num: medCount,      label: 'Medium',   color: '#c8960a' },
          { num: passCount,     label: 'Passed',   color: '#00C853' },
        ].map(({ num, label, color }, i) => (
          <div key={i} style={{ padding: '12px 20px', borderRight: i < 3 ? 'var(--border)' : 'none' }}>
            <div style={{ fontFamily: 'var(--font-mono)', fontSize: 22, fontWeight: 700, color, marginBottom: 2 }}>{num}</div>
            <div style={{ fontSize: 11, color: '#777', fontWeight: 500, textTransform: 'uppercase', letterSpacing: '0.05em' }}>{label}</div>
          </div>
        ))}
      </div>

      {/* Report body */}
      <div style={{ padding: 24, overflowY: 'auto', maxHeight: 'calc(100vh - 58px - 260px - 57px)' }}>

        {/* Header row */}
        <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', marginBottom: 20, gap: 20 }}>
          <div>
            <div style={{ fontSize: 19, fontWeight: 800, marginBottom: 3 }}>
              Audit Report — {hostname}
            </div>
            <div style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: '#666' }}>
              {new Date(report.scanned_at).toLocaleTimeString()} &nbsp;|&nbsp; {report.modules.length} modules &nbsp;|&nbsp; {report.checks_performed} checks
            </div>
          </div>
          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', justifyContent: 'flex-end' }}>
            {report.modules.map((m, i) => {
              const isPass = m.status === 'PASS'
              const isCrit = m.status === 'CRITICAL' || m.status === 'HIGH'
              return (
                <span key={i} style={{
                  border: '2px solid',
                  borderColor: isPass ? '#00C853' : isCrit ? '#FF3B00' : '#c8960a',
                  background: isPass ? '#00C853' : isCrit ? '#FF3B00' : '#FFE500',
                  color: isPass ? '#fff' : isCrit ? '#fff' : '#0A0A0A',
                  padding: '4px 10px',
                  fontFamily: 'var(--font-mono)', fontSize: 10, fontWeight: 700,
                  textTransform: 'uppercase', letterSpacing: '0.05em',
                  whiteSpace: 'nowrap',
                }}>
                  {m.module.split(' ')[0]}
                </span>
              )
            })}
          </div>
        </div>

        {/* Section label */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 14 }}>
          <span style={{ fontFamily: 'var(--font-mono)', fontSize: 10, fontWeight: 700, letterSpacing: '0.1em', textTransform: 'uppercase', color: '#888' }}>
            Findings
          </span>
          <div style={{ flex: 1, height: 1.5, background: '#ccc' }} />
        </div>

        {/* Finding cards grid */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 20 }}>
          {report.modules.map((mod, i) => (
            <FindingCard key={i} mod={mod} />
          ))}
        </div>

        {/* LLM Summary */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 14 }}>
          <span style={{ fontFamily: 'var(--font-mono)', fontSize: 10, fontWeight: 700, letterSpacing: '0.1em', textTransform: 'uppercase', color: '#888' }}>
            Agent Analysis
          </span>
          <div style={{ flex: 1, height: 1.5, background: '#ccc' }} />
        </div>

        <div style={{
          border: 'var(--border)', boxShadow: 'var(--shadow-lg)',
          background: 'var(--ink)', color: 'var(--cream)',
          padding: '20px 24px', position: 'relative', overflow: 'hidden',
        }}>
          <div style={{ position: 'absolute', top: 0, left: 0, width: 4, height: '100%', background: '#00C8E0' }} />
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontFamily: 'var(--font-mono)', fontSize: 10, fontWeight: 700, color: '#00C8E0', letterSpacing: '0.12em', textTransform: 'uppercase', marginBottom: 12 }}>
            <Sparkles size={13} color="#00C8E0" />
            LLM Security Analysis
          </div>
          <div style={{ fontSize: 13, lineHeight: 1.8, color: '#c8c5be', fontWeight: 400 }}>
            {report.summary}
          </div>
          <div style={{ marginTop: 16, display: 'flex', gap: 10 }}>
            <button onClick={onReset} style={{
              height: 36, padding: '0 16px',
              background: 'transparent', border: '2px solid #333', color: 'var(--cream)',
              fontFamily: 'var(--font-mono)', fontSize: 11, fontWeight: 700,
              letterSpacing: '0.06em', textTransform: 'uppercase',
              display: 'flex', alignItems: 'center', gap: 7,
              transition: 'border-color 0.15s',
            }}>
              <RefreshCw size={12} />
              New Audit
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}

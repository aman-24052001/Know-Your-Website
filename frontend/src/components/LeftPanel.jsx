import { useState } from 'react'
import { Search, Shield, Lock, FileText, Globe, Cookie, ArrowUpRight, CheckCircle, AlertTriangle, XCircle } from 'lucide-react'
import { ScoreGauge } from './ScoreGauge.jsx'
import { FindingBadge } from './FindingBadge.jsx'

const MODULE_ICONS = {
  'DNS Resolution':    Globe,
  'SSL / TLS':         Lock,
  'Security Headers':  FileText,
  'Exposed Endpoints': Shield,
  'Robots & Sitemap':  FileText,
  'Open Redirects':    ArrowUpRight,
  'Cookie Security':   Cookie,
}

const IDLE_MODULES = [
  'DNS Resolution', 'SSL / TLS', 'Security Headers',
  'Exposed Endpoints', 'Robots & Sitemap', 'Open Redirects', 'Cookie Security',
]

function CheckIcon({ status }) {
  if (!status || status === 'PASS') return <div style={{ width: 28, height: 28, border: '2px solid var(--ink)', background: status === 'PASS' ? '#00C853' : 'transparent', display: 'flex', alignItems: 'center', justifyContent: 'center' }}><CheckCircle size={13} color={status === 'PASS' ? '#fff' : '#ccc'} /></div>
  if (status === 'MEDIUM' || status === 'LOW') return <div style={{ width: 28, height: 28, border: '2px solid var(--ink)', background: '#FFE500', display: 'flex', alignItems: 'center', justifyContent: 'center' }}><AlertTriangle size={13} color="#0A0A0A" /></div>
  return <div style={{ width: 28, height: 28, border: '2px solid var(--ink)', background: '#FF3B00', display: 'flex', alignItems: 'center', justifyContent: 'center' }}><XCircle size={13} color="#fff" /></div>
}

export function LeftPanel({ report, status, onStart }) {
  const [url, setUrl] = useState('')

  const handleSubmit = () => {
    const cleaned = url.trim()
    if (!cleaned) return
    onStart(cleaned)
  }

  const modules = report
    ? report.modules.map(m => ({ name: m.module, status: m.status }))
    : IDLE_MODULES.map(name => ({ name, status: null }))

  return (
    <div style={{ borderRight: 'var(--border)', display: 'flex', flexDirection: 'column', overflowY: 'auto' }}>

      {/* URL Input */}
      <div style={{ padding: 20, borderBottom: 'var(--border)' }}>
        <div style={{ fontFamily: 'var(--font-mono)', fontSize: 10, fontWeight: 700, letterSpacing: '0.12em', textTransform: 'uppercase', color: '#666', marginBottom: 10 }}>
          Target URL
        </div>
        <div style={{
          border: 'var(--border)', background: '#fff', boxShadow: 'var(--shadow)',
          display: 'flex', alignItems: 'center', overflow: 'hidden', marginBottom: 10,
        }}>
          <span style={{
            padding: '0 12px', fontFamily: 'var(--font-mono)', fontSize: 12, fontWeight: 700,
            color: '#888', borderRight: '2px solid var(--ink)', background: '#f0ede6',
            height: 46, display: 'flex', alignItems: 'center', whiteSpace: 'nowrap',
          }}>
            https://
          </span>
          <input
            value={url}
            onChange={e => setUrl(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && handleSubmit()}
            placeholder="example.com"
            style={{
              flex: 1, border: 'none', outline: 'none',
              padding: '0 12px', height: 46,
              fontFamily: 'var(--font-mono)', fontSize: 13,
              background: '#fff', color: 'var(--ink)',
            }}
          />
        </div>
        <button
          onClick={handleSubmit}
          disabled={status === 'running'}
          style={{
            width: '100%', height: 50,
            background: status === 'running' ? '#333' : 'var(--ink)',
            color: 'var(--cream)',
            border: 'var(--border)', boxShadow: 'var(--shadow)',
            fontFamily: 'var(--font-ui)', fontSize: 13, fontWeight: 700,
            letterSpacing: '0.06em', textTransform: 'uppercase',
            display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 10,
            transition: 'transform 0.08s, box-shadow 0.08s',
            cursor: status === 'running' ? 'not-allowed' : 'pointer',
          }}
          onMouseEnter={e => { if (status !== 'running') { e.currentTarget.style.transform = 'translate(-2px,-2px)'; e.currentTarget.style.boxShadow = '7px 7px 0px var(--ink)' } }}
          onMouseLeave={e => { e.currentTarget.style.transform = ''; e.currentTarget.style.boxShadow = 'var(--shadow)' }}
        >
          <Search size={16} />
          {status === 'running' ? 'Scanning...' : 'Run Security Audit'}
        </button>
      </div>

      {/* Score */}
      <div style={{ padding: 20, borderBottom: 'var(--border)' }}>
        <div style={{ fontFamily: 'var(--font-mono)', fontSize: 10, fontWeight: 700, letterSpacing: '0.12em', textTransform: 'uppercase', color: '#666', marginBottom: 12 }}>
          Security Score
        </div>
        <div style={{ border: 'var(--border)', boxShadow: 'var(--shadow)', background: '#fff', padding: 16 }}>
          {report ? (
            <ScoreGauge score={report.score} />
          ) : (
            <div style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: '#aaa' }}>
              Run an audit to see your score
            </div>
          )}
        </div>
      </div>

      {/* Module checklist */}
      <div style={{ padding: 20 }}>
        <div style={{ fontFamily: 'var(--font-mono)', fontSize: 10, fontWeight: 700, letterSpacing: '0.12em', textTransform: 'uppercase', color: '#666', marginBottom: 12 }}>
          Audit Modules
        </div>
        {modules.map(({ name, status: modStatus }, i) => {
          const Icon = MODULE_ICONS[name] || Shield
          return (
            <div key={i} style={{
              display: 'flex', alignItems: 'center', justifyContent: 'space-between',
              padding: '9px 0',
              borderBottom: i < modules.length - 1 ? '1.5px solid #e0ddd6' : 'none',
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 10, fontSize: 12.5, fontWeight: 500 }}>
                <CheckIcon status={modStatus} />
                {name}
              </div>
              {modStatus && <FindingBadge severity={modStatus} />}
            </div>
          )
        })}
      </div>
    </div>
  )
}

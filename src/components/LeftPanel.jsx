import { useState } from 'react'
import { Search, Globe, Lock, FileText, Shield, ArrowUpRight, Cookie, CheckCircle, AlertTriangle, XCircle, Settings } from 'lucide-react'
import { ScoreGauge } from './ScoreGauge.jsx'
import { FindingBadge } from './FindingBadge.jsx'

const MODULE_ICONS = {
  'DNS Resolution': Globe,
  'SSL / TLS': Lock,
  'Security Headers': FileText,
  'Exposed Endpoints': Shield,
  'Robots & Sitemap': FileText,
  'Open Redirects': ArrowUpRight,
  'Cookie Security': Cookie,
}

const IDLE_MODULES = [
  'DNS Resolution', 'SSL / TLS', 'Security Headers',
  'Exposed Endpoints', 'Robots & Sitemap', 'Open Redirects', 'Cookie Security',
]

function CheckIcon({ status }) {
  const base = {
    width: 28, height: 28, border: '2px solid var(--ink)',
    display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0,
  }
  if (status === 'PASS') return <div style={{ ...base, background: '#00C853' }}><CheckCircle size={13} color="#fff" /></div>
  if (status === 'MEDIUM' || status === 'LOW') return <div style={{ ...base, background: '#FFE500' }}><AlertTriangle size={13} color="#0A0A0A" /></div>
  if (status === 'HIGH' || status === 'CRITICAL') return <div style={{ ...base, background: '#FF3B00' }}><XCircle size={13} color="#fff" /></div>
  return <div style={{ ...base }}><CheckCircle size={13} color="#ccc" /></div>
}

export function LeftPanel({ report, status, onStart, onSettingsOpen, hasApiKey, isMobile }) {
  const [url, setUrl] = useState('')

  const handleSubmit = () => {
    const cleaned = url.trim()
    if (!cleaned || status === 'running') return
    onStart(cleaned)
  }

  const modules = report
    ? report.modules.map(m => ({ name: m.module, status: m.status }))
    : IDLE_MODULES.map(name => ({ name, status: null }))

  return (
    <div style={{
      borderRight: isMobile ? 'none' : 'var(--border)',
      borderBottom: isMobile ? 'var(--border)' : 'none',
      display: 'flex',
      flexDirection: 'column',
      // Desktop: scrollable within the column; Mobile: natural height
      overflowY: isMobile ? 'visible' : 'auto',
    }}>

      {/* URL Input */}
      <div style={{ padding: isMobile ? 16 : 20, borderBottom: 'var(--border)' }}>
        <div style={{
          fontFamily: 'var(--font-mono)', fontSize: 10, fontWeight: 700,
          letterSpacing: '0.12em', textTransform: 'uppercase', color: '#666', marginBottom: 10,
        }}>
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
            color: 'var(--cream)', border: 'var(--border)', boxShadow: 'var(--shadow)',
            fontFamily: 'var(--font-ui)', fontSize: 13, fontWeight: 700,
            letterSpacing: '0.06em', textTransform: 'uppercase',
            display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 10,
            cursor: status === 'running' ? 'not-allowed' : 'pointer',
          }}
        >
          <Search size={16} />
          {status === 'running' ? 'Scanning...' : 'Run Security Audit'}
        </button>
      </div>

      {/* API Key status */}
      <div style={{
        padding: isMobile ? '10px 16px' : '12px 20px',
        borderBottom: 'var(--border)',
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <div style={{
            width: 7, height: 7, borderRadius: '50%',
            background: hasApiKey ? '#00C853' : '#ccc',
            flexShrink: 0,
          }} />
          <span style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: '#777', letterSpacing: '0.04em' }}>
            {hasApiKey ? 'LLM ANALYSIS ENABLED' : 'NO API KEY — LLM DISABLED'}
          </span>
        </div>
        <button onClick={onSettingsOpen} style={{
          background: 'none', border: '2px solid var(--ink)', padding: '4px 10px',
          display: 'flex', alignItems: 'center', gap: 5,
          fontFamily: 'var(--font-mono)', fontSize: 10, fontWeight: 700,
          textTransform: 'uppercase', letterSpacing: '0.05em', flexShrink: 0,
        }}>
          <Settings size={11} />
          Key
        </button>
      </div>

      {/* Score */}
      <div style={{ padding: isMobile ? 16 : 20, borderBottom: 'var(--border)' }}>
        <div style={{
          fontFamily: 'var(--font-mono)', fontSize: 10, fontWeight: 700,
          letterSpacing: '0.12em', textTransform: 'uppercase', color: '#666', marginBottom: 12,
        }}>
          Security Score
        </div>
        <div style={{ border: 'var(--border)', boxShadow: 'var(--shadow)', background: '#fff', padding: 16 }}>
          {report
            ? <ScoreGauge score={report.score} />
            : <div style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: '#aaa' }}>Run an audit to see your score</div>
          }
        </div>
      </div>

      {/* Module checklist */}
      <div style={{ padding: isMobile ? 16 : 20 }}>
        <div style={{
          fontFamily: 'var(--font-mono)', fontSize: 10, fontWeight: 700,
          letterSpacing: '0.12em', textTransform: 'uppercase', color: '#666', marginBottom: 12,
        }}>
          Audit Modules
        </div>
        {modules.map(({ name, status: modStatus }, i) => (
          <div key={i} style={{
            display: 'flex', alignItems: 'center', justifyContent: 'space-between',
            padding: '9px 0',
            borderBottom: i < modules.length - 1 ? '1.5px solid #e0ddd6' : 'none',
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10, fontSize: 13, fontWeight: 500 }}>
              <CheckIcon status={modStatus} />
              {name}
            </div>
            {modStatus && <FindingBadge severity={modStatus} />}
          </div>
        ))}
      </div>
    </div>
  )
}

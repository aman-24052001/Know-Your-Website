import { useState, useEffect } from 'react'
import { Shield } from 'lucide-react'
import { useAudit } from './hooks/useAudit.js'
import { LeftPanel } from './components/LeftPanel.jsx'
import { AgentLog } from './components/AgentLog.jsx'
import { ReportCard } from './components/ReportCard.jsx'
import { SettingsPanel } from './components/SettingsPanel.jsx'

export default function App() {
  const { logs, report, status, startAudit, reset } = useAudit()
  const [showSettings, setShowSettings] = useState(false)
  const [hasApiKey, setHasApiKey] = useState(false)

  useEffect(() => {
    const check = () => setHasApiKey(!!localStorage.getItem('kyw_api_key'))
    check()
    window.addEventListener('storage', check)
    return () => window.removeEventListener('storage', check)
  }, [])

  const handleSettingsClose = () => {
    setShowSettings(false)
    setHasApiKey(!!localStorage.getItem('kyw_api_key'))
  }

  const handleStart = (url) => {
    const apiKey = localStorage.getItem('kyw_api_key') || ''
    startAudit(url, apiKey)
  }

  return (
    <div style={{ height: '100vh', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
      {/* Nav */}
      <nav style={{ background: 'var(--ink)', borderBottom: 'var(--border)', padding: '0 28px', height: 58, display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexShrink: 0 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, color: 'var(--cream)', fontSize: 15, fontWeight: 700, letterSpacing: '0.04em', textTransform: 'uppercase' }}>
          <div style={{ width: 28, height: 28, background: '#00C8E0', border: '2px solid var(--cream)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <Shield size={15} color="var(--ink)" />
          </div>
          Know Your Website
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
          <div style={{ background: 'var(--cream)', border: '2px solid var(--cream)', color: 'var(--ink)', padding: '3px 10px', fontFamily: 'var(--font-mono)', fontSize: 10, fontWeight: 700, letterSpacing: '0.08em', textTransform: 'uppercase' }}>
            Static · v2.0
          </div>
          <div style={{ width: 1, height: 20, background: '#333' }} />
          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <div style={{ width: 7, height: 7, borderRadius: '50%', background: status === 'running' ? '#FFE500' : '#00C853' }} />
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: '#666', letterSpacing: '0.06em' }}>
              {status === 'running' ? 'SCANNING' : 'READY'}
            </span>
          </div>
        </div>
      </nav>

      {/* Layout */}
      <div style={{ display: 'grid', gridTemplateColumns: '360px 1fr', flex: 1, overflow: 'hidden' }}>
        <LeftPanel
          report={report}
          status={status}
          onStart={handleStart}
          onSettingsOpen={() => setShowSettings(true)}
          hasApiKey={hasApiKey}
        />
        <div style={{ display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
          <AgentLog logs={logs} status={status} />
          <div style={{ flex: 1, overflowY: 'auto' }}>
            {status === 'done' && report
              ? <ReportCard report={report} onReset={reset} />
              : (
                <div style={{ padding: 32 }}>
                  {status === 'idle' && (
                    <div style={{ color: '#aaa', fontFamily: 'var(--font-mono)', fontSize: 12, lineHeight: 2 }}>
                      <div>Enter a URL on the left and run an audit.</div>
                      <div style={{ marginTop: 8, color: '#ccc', fontSize: 11 }}>
                        7 modules — DNS · SSL · Headers · Endpoints · Robots · Redirects · Cookies
                      </div>
                      <div style={{ marginTop: 4, color: '#ccc', fontSize: 11 }}>
                        Runs entirely in your browser. No backend required.
                      </div>
                    </div>
                  )}
                  {status === 'running' && (
                    <div style={{ color: '#888', fontFamily: 'var(--font-mono)', fontSize: 12 }}>
                      Scan in progress — report will appear here when complete.
                    </div>
                  )}
                  {status === 'error' && (
                    <div style={{ color: 'var(--red)', fontFamily: 'var(--font-mono)', fontSize: 12 }}>
                      Scan error — check the agent log for details.
                    </div>
                  )}
                </div>
              )
            }
          </div>
        </div>
      </div>

      {showSettings && <SettingsPanel onClose={handleSettingsClose} />}
    </div>
  )
}

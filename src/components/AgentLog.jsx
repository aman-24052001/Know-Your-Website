import { useEffect, useRef } from 'react'
import { Terminal } from 'lucide-react'

const LEVEL_COLOR = { ok: '#00C853', warn: '#FFE500', err: '#FF3B00', info: '#00C8E0' }

export function AgentLog({ logs, status, isMobile }) {
  const bottomRef = useRef(null)
  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: 'smooth' }) }, [logs])

  return (
    <div style={{
      borderBottom: 'var(--border)',
      background: '#0A0A0A',
      display: 'flex',
      flexDirection: 'column',
      // Desktop: fixed height in the column; Mobile: natural height, min to show content
      flex: isMobile ? 'none' : '0 0 260px',
      minHeight: isMobile ? 200 : 'auto',
      maxHeight: isMobile ? 320 : 'none',
    }}>
      {/* Header */}
      <div style={{
        padding: '10px 20px', borderBottom: '2px solid #1e1e1e',
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        flexShrink: 0,
      }}>
        <div style={{
          display: 'flex', alignItems: 'center', gap: 8,
          fontFamily: 'var(--font-mono)', fontSize: 11, fontWeight: 700,
          color: '#00C8E0', letterSpacing: '0.1em', textTransform: 'uppercase',
        }}>
          <Terminal size={13} color="#00C8E0" />
          Agent Log
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          {status === 'running' && (
            <div style={{
              width: 7, height: 7, borderRadius: '50%',
              background: '#00C853', animation: 'pulse 1.4s infinite',
            }} />
          )}
          <span style={{
            fontFamily: 'var(--font-mono)', fontSize: 10, color: '#555',
            letterSpacing: '0.08em', textTransform: 'uppercase',
          }}>
            {status === 'running' ? 'SCANNING' : status === 'done' ? 'COMPLETE' : 'READY'}
          </span>
        </div>
      </div>

      {/* Log body */}
      <div style={{
        flex: 1, overflowY: 'auto', padding: '14px 20px',
        fontFamily: 'var(--font-mono)', fontSize: 11.5, lineHeight: 1.9,
      }}>
        {logs.length === 0 && <div style={{ color: '#333' }}>Waiting for audit to start...</div>}
        {logs.map(log => (
          <div key={log.id} style={{ display: 'flex', gap: 12 }}>
            <span style={{ color: '#383838', flexShrink: 0 }}>{log.ts}</span>
            <span style={{ color: LEVEL_COLOR[log.level] || '#9a9a9a', wordBreak: 'break-word' }}>{log.text}</span>
          </div>
        ))}
        {status === 'running' && (
          <div style={{ display: 'flex', gap: 12 }}>
            <span style={{ color: '#383838' }}>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>
            <span style={{
              display: 'inline-block', width: 8, height: 14,
              background: '#00C8E0', animation: 'blink 1s step-end infinite',
            }} />
          </div>
        )}
        <div ref={bottomRef} />
      </div>
    </div>
  )
}

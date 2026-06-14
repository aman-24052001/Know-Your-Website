import { useState, useRef, useCallback } from 'react'

export function useAuditStream() {
  const [logs, setLogs] = useState([])
  const [report, setReport] = useState(null)
  const [status, setStatus] = useState('idle') // idle | running | done | error
  const esRef = useRef(null)

  const startAudit = useCallback((url) => {
    if (esRef.current) {
      esRef.current.close()
    }

    setLogs([])
    setReport(null)
    setStatus('running')

    const encoded = encodeURIComponent(url)
    const es = new EventSource(`/audit?url=${encoded}`)
    esRef.current = es

    es.onmessage = (e) => {
      try {
        const msg = JSON.parse(e.data)

        if (msg.type === 'log') {
          setLogs(prev => [...prev, {
            ts: msg.ts,
            text: msg.text,
            level: msg.level || 'info',
            id: Date.now() + Math.random()
          }])
        }

        if (msg.type === 'report') {
          setReport(msg.payload)
          setStatus('done')
          es.close()
        }
      } catch (err) {
        console.error('SSE parse error', err)
      }
    }

    es.onerror = () => {
      setStatus('error')
      es.close()
    }
  }, [])

  const reset = useCallback(() => {
    if (esRef.current) esRef.current.close()
    setLogs([])
    setReport(null)
    setStatus('idle')
  }, [])

  return { logs, report, status, startAudit, reset }
}

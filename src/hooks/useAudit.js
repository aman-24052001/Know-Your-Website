import { useState, useCallback, useRef } from 'react'
import { PIPELINE, computeScore, synthesiseWithLLM } from '../lib/audit.js'

export function useAudit() {
  const [logs, setLogs] = useState([])
  const [report, setReport] = useState(null)
  const [status, setStatus] = useState('idle') // idle | running | done | error
  const abortRef = useRef(false)

  const startAudit = useCallback(async (rawUrl, apiKey) => {
    abortRef.current = false
    setLogs([])
    setReport(null)
    setStatus('running')

    let url = rawUrl.trim()
    if (!url.startsWith('http')) url = 'https://' + url

    const addLog = (text, level = 'info') => {
      const ts = new Date().toTimeString().slice(0, 8)
      setLogs(prev => [...prev, { id: Date.now() + Math.random(), ts, text, level }])
    }

    addLog(`Initialising audit for ${url}`, 'info')

    const moduleResults = []

    for (const tool of PIPELINE) {
      if (abortRef.current) break
      try {
        const result = await tool(url, addLog)
        moduleResults.push(result)
      } catch (e) {
        addLog(`Tool error: ${e.message}`, 'err')
      }
    }

    const score = computeScore(moduleResults)
    addLog(`All modules complete — score: ${score}/100`, 'ok')

    let summary = `Score: ${score}/100. ${
      moduleResults.flatMap(m => m.findings).filter(f => ['CRITICAL','HIGH'].includes(f.severity)).length
    } high-severity issues found.`

    if (apiKey) {
      addLog('Synthesising report with LLM...', 'info')
      try {
        summary = await synthesiseWithLLM(url, moduleResults, score, apiKey)
        addLog('LLM analysis complete', 'ok')
      } catch (e) {
        addLog(`LLM synthesis failed: ${e.message}`, 'warn')
      }
    } else {
      addLog('No API key — skipping LLM synthesis', 'warn')
    }

    setReport({
      url,
      score,
      modules: moduleResults,
      summary,
      scannedAt: new Date().toISOString(),
      checksPerformed: moduleResults.reduce((acc, m) => acc + m.findings.length, 0),
    })
    setStatus('done')
  }, [])

  const reset = useCallback(() => {
    abortRef.current = true
    setLogs([])
    setReport(null)
    setStatus('idle')
  }, [])

  return { logs, report, status, startAudit, reset }
}

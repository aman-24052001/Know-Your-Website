import { useState, useEffect } from 'react'
import { Key, Eye, EyeOff, Check, X } from 'lucide-react'

export function SettingsPanel({ onClose }) {
  const [key, setKey] = useState('')
  const [show, setShow] = useState(false)
  const [saved, setSaved] = useState(false)

  useEffect(() => {
    const stored = localStorage.getItem('kyw_api_key') || ''
    setKey(stored)
  }, [])

  const handleSave = () => {
    if (key.trim()) {
      localStorage.setItem('kyw_api_key', key.trim())
    } else {
      localStorage.removeItem('kyw_api_key')
    }
    setSaved(true)
    setTimeout(() => { setSaved(false); onClose() }, 800)
  }

  const handleClear = () => {
    setKey('')
    localStorage.removeItem('kyw_api_key')
  }

  return (
    <div style={{
      position: 'fixed', inset: 0, background: 'rgba(10,10,10,0.6)',
      display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 100,
    }} onClick={onClose}>
      <div style={{
        background: 'var(--cream)', border: 'var(--border)', boxShadow: 'var(--shadow-lg)',
        width: 480, padding: 28,
      }} onClick={e => e.stopPropagation()}>

        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 20 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, fontWeight: 700, fontSize: 15, textTransform: 'uppercase', letterSpacing: '0.04em' }}>
            <Key size={16} />
            Anthropic API Key
          </div>
          <button onClick={onClose} style={{ background: 'none', border: 'none', padding: 4 }}>
            <X size={18} />
          </button>
        </div>

        <div style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: '#666', lineHeight: 1.7, marginBottom: 16 }}>
          Your key is stored only in your browser (localStorage).<br />
          It is sent directly to the Anthropic API — never to any server.<br />
          Without a key, all 7 audit modules still run — only the LLM summary is skipped.
        </div>

        <div style={{
          border: 'var(--border)', background: '#fff', boxShadow: 'var(--shadow)',
          display: 'flex', alignItems: 'center', overflow: 'hidden', marginBottom: 12,
        }}>
          <input
            type={show ? 'text' : 'password'}
            value={key}
            onChange={e => setKey(e.target.value)}
            placeholder="sk-ant-..."
            style={{
              flex: 1, border: 'none', outline: 'none',
              padding: '0 14px', height: 46,
              fontFamily: 'var(--font-mono)', fontSize: 13,
              background: '#fff', color: 'var(--ink)',
            }}
          />
          <button onClick={() => setShow(s => !s)} style={{
            background: 'none', border: 'none', borderLeft: '2px solid var(--ink)',
            padding: '0 14px', height: 46, display: 'flex', alignItems: 'center',
          }}>
            {show ? <EyeOff size={15} /> : <Eye size={15} />}
          </button>
        </div>

        <div style={{ display: 'flex', gap: 10 }}>
          <button onClick={handleSave} style={{
            flex: 1, height: 44,
            background: saved ? '#00C853' : 'var(--ink)',
            color: 'var(--cream)', border: 'var(--border)', boxShadow: 'var(--shadow)',
            fontFamily: 'var(--font-ui)', fontWeight: 700, fontSize: 13,
            letterSpacing: '0.05em', textTransform: 'uppercase',
            display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8,
            transition: 'background 0.2s',
          }}>
            {saved ? <><Check size={15} /> Saved</> : 'Save Key'}
          </button>
          {key && (
            <button onClick={handleClear} style={{
              height: 44, padding: '0 16px',
              background: 'transparent', border: 'var(--border)',
              fontFamily: 'var(--font-mono)', fontSize: 11, fontWeight: 700,
              textTransform: 'uppercase', letterSpacing: '0.05em',
            }}>
              Clear
            </button>
          )}
        </div>

        <div style={{ marginTop: 14, fontFamily: 'var(--font-mono)', fontSize: 10, color: '#aaa' }}>
          Get a key at console.anthropic.com
        </div>
      </div>
    </div>
  )
}

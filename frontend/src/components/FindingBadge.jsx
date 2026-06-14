const COLORS = {
  CRITICAL: { bg: '#FF3B00', color: '#fff' },
  HIGH:     { bg: '#FF7A00', color: '#fff' },
  MEDIUM:   { bg: '#FFE500', color: '#0A0A0A' },
  LOW:      { bg: '#ccc',    color: '#0A0A0A' },
  PASS:     { bg: '#00C853', color: '#fff' },
}

export function FindingBadge({ severity }) {
  const c = COLORS[severity] || COLORS.LOW
  return (
    <span style={{
      fontFamily: 'var(--font-mono)',
      fontSize: 10,
      fontWeight: 700,
      padding: '3px 8px',
      border: '2px solid var(--ink)',
      textTransform: 'uppercase',
      letterSpacing: '0.06em',
      background: c.bg,
      color: c.color,
      whiteSpace: 'nowrap',
    }}>
      {severity}
    </span>
  )
}

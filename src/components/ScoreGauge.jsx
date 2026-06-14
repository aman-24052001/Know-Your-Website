export function ScoreGauge({ score }) {
  const radius = 34
  const circ = 2 * Math.PI * radius
  const offset = circ - (score / 100) * circ
  const color = score >= 70 ? '#00C853' : score >= 40 ? '#FFE500' : '#FF3B00'
  const label = score >= 70 ? 'Low Risk' : score >= 40 ? 'Moderate Risk' : 'High Risk'

  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 20 }}>
      <div style={{ position: 'relative', width: 80, height: 80, flexShrink: 0 }}>
        <svg width="80" height="80" viewBox="0 0 80 80" style={{ transform: 'rotate(-90deg)' }}>
          <circle cx="40" cy="40" r={radius} fill="none" stroke="#e0ddd6" strokeWidth="8" />
          <circle cx="40" cy="40" r={radius} fill="none" stroke={color} strokeWidth="8"
            strokeDasharray={circ} strokeDashoffset={offset}
            style={{ transition: 'stroke-dashoffset 1s ease, stroke 0.5s ease' }} />
        </svg>
        <div style={{
          position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center',
          fontFamily: 'var(--font-mono)', fontSize: 20, fontWeight: 700,
        }}>{score}</div>
      </div>
      <div>
        <div style={{ fontSize: 13, fontWeight: 700, marginBottom: 4 }}>{label}</div>
        <div style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: '#777', lineHeight: 1.7 }}>
          Score: {score} / 100
        </div>
      </div>
    </div>
  )
}

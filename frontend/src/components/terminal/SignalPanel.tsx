import { useMemo } from 'react'
import { useNavigate } from 'react-router-dom'
import { useSignals } from '../../api/hooks'
import type { Signal } from '../../types'

function scoreTone(score: number): string {
  if (score >= 70) return 'var(--ra-bullish)'
  if (score >= 50) return '#eab308'
  return 'var(--ra-bearish)'
}

function tierScore(tier: Signal['confidence_tier']): number {
  if (tier === 'HIGH') return 90
  if (tier === 'MEDIUM') return 65
  return 40
}

function phaseScore(phase: string): number {
  const p = phase.toLowerCase()
  if (p.includes('trend') || p.includes('expansion')) return 80
  if (p.includes('compression') || p.includes('range')) return 55
  return 35
}

function isGreenSignal(signal: Signal): boolean {
  const phase = signal.market_phase.toLowerCase()
  const phaseBullish = phase.includes('trend') || phase.includes('expansion') || phase.includes('bull')
  return phaseBullish
}

function ScoreBar({ label, score }: { label: string; score: number }) {
  const width = Math.max(0, Math.min(100, Math.round(score)))
  const color = scoreTone(score)

  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between">
        <span className="text-[10px] uppercase tracking-wide text-gray-500">{label}</span>
        <span className="font-mono text-[11px]" style={{ color }}>{Math.round(score)}</span>
      </div>
      <div className="h-1.5 rounded bg-gray-800">
        <div className="h-full rounded" style={{ width: `${width}%`, backgroundColor: color }} />
      </div>
    </div>
  )
}

export default function SignalPanel() {
  const navigate = useNavigate()
  const { data: signals = [] } = useSignals(0)

  const topSignals = useMemo(
    () => [...signals].filter(isGreenSignal).sort((a, b) => b.probability - a.probability).slice(0, 10),
    [signals],
  )

  // No active signals — hide the panel entirely (no empty cards)
  if (topSignals.length === 0) return null

  return (
    <div className="max-h-[30vh] border-t border-gray-800 p-2 overflow-y-auto">
    <section className="rounded border border-gray-800 bg-gray-900/60 p-2">
      <div className="mb-2 px-1">
        <h2 className="text-[11px] font-semibold uppercase tracking-wide text-gray-400">Signal Panel</h2>
      </div>

      <div className="grid gap-2">
        {topSignals.map(signal => (
          <button
            key={`${signal.symbol}-${signal.date}`}
            onClick={() => navigate(`/chart?symbol=${signal.symbol}`)}
            className="space-y-2 rounded border border-gray-800 bg-gray-950/70 p-2 text-left transition-colors hover:border-gray-700"
          >
            <div className="flex items-center justify-between">
              <span className="font-mono text-sm font-semibold text-gray-100">{signal.symbol}</span>
              <span className="font-mono text-[11px] text-gray-400">{Math.round(signal.probability * 100)}</span>
            </div>
            <ScoreBar label="Probability" score={signal.probability * 100} />
            <ScoreBar label="Confidence" score={tierScore(signal.confidence_tier)} />
            <ScoreBar label="Market Phase" score={phaseScore(signal.market_phase)} />
          </button>
        ))}
      </div>
    </section>
    </div>
  )
}

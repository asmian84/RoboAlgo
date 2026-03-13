import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import type { CommandCenterSignal } from '../../types'

function cellColor(score: number | null): string {
  if (score == null || score < 0) return '#1f2937'
  if (score >= 80) return '#166534'   // bright green
  if (score >= 60) return '#15803d'   // green
  if (score >= 40) return '#92400e'   // amber
  return '#7f1d1d'                    // red
}

function textColor(score: number | null): string {
  if (score == null || score < 0) return '#4b5563'
  if (score >= 60) return '#bbf7d0'
  if (score >= 40) return '#fde68a'
  return '#fca5a5'
}

interface Props {
  signals: CommandCenterSignal[]
}

export default function OpportunityHeatmap({ signals }: Props) {
  const navigate = useNavigate()
  const [tooltip, setTooltip] = useState<CommandCenterSignal | null>(null)
  const [tooltipPos, setTooltipPos] = useState({ x: 0, y: 0 })

  // Sort by setup_quality_score, take top 40
  const cells = [...signals]
    .sort((a, b) => (b.setup_quality_score ?? 0) - (a.setup_quality_score ?? 0))
    .slice(0, 40)

  return (
    <div className="bg-gray-900 border border-gray-800 rounded-xl p-4">
      <div className="mb-3">
        <h2 className="text-sm font-semibold text-gray-100">Opportunity Heatmap</h2>
        <p className="text-xs text-gray-500 mt-0.5">Color = setup quality · darker green = higher score</p>
      </div>

      {/* Legend */}
      <div className="flex items-center gap-3 mb-3 text-[10px] text-gray-500">
        {[
          { label: '80-100', color: '#166534' },
          { label: '60-79', color: '#15803d' },
          { label: '40-59', color: '#92400e' },
          { label: '0-39', color: '#7f1d1d' },
        ].map(({ label, color }) => (
          <div key={label} className="flex items-center gap-1">
            <div className="w-3 h-3 rounded" style={{ backgroundColor: color }} />
            <span>{label}</span>
          </div>
        ))}
      </div>

      {/* Grid */}
      <div
        className="grid gap-1"
        style={{ gridTemplateColumns: 'repeat(10, minmax(0, 1fr))' }}
        onMouseLeave={() => setTooltip(null)}
      >
        {cells.map(sig => {
          const score = sig.setup_quality_score
          const bg    = cellColor(score)
          const tc    = textColor(score)
          return (
            <button
              key={sig.symbol}
              className="rounded p-1 flex flex-col items-center justify-center transition-transform hover:scale-105 cursor-pointer"
              style={{ backgroundColor: bg, minHeight: '48px' }}
              onClick={() => navigate(`/chart?symbol=${sig.symbol}`)}
              onMouseEnter={e => {
                setTooltip(sig)
                setTooltipPos({ x: e.clientX, y: e.clientY })
              }}
              onMouseMove={e => setTooltipPos({ x: e.clientX, y: e.clientY })}
            >
              <span className="text-[9px] font-mono font-bold" style={{ color: tc }}>
                {sig.symbol.length > 4 ? sig.symbol.slice(0, 4) : sig.symbol}
              </span>
              {score != null && (
                <span className="text-[8px] font-mono" style={{ color: tc, opacity: 0.8 }}>
                  {score.toFixed(0)}
                </span>
              )}
            </button>
          )
        })}
      </div>

      {/* Tooltip */}
      {tooltip && (
        <div
          className="fixed z-50 bg-gray-800 border border-gray-700 rounded-lg p-3 text-xs shadow-xl pointer-events-none"
          style={{ left: tooltipPos.x + 10, top: tooltipPos.y - 80 }}
        >
          <p className="font-mono font-bold text-gray-100 mb-1">{tooltip.symbol}</p>
          <p className="text-gray-400">Setup Quality: <span className="text-emerald-400 font-mono">{tooltip.setup_quality_score?.toFixed(0) ?? '—'}</span></p>
          <p className="text-gray-400">Regime: <span className="text-gray-200">{tooltip.market_state ?? '—'}</span></p>
          <p className="text-gray-400">Liq Align: <span className="text-blue-400 font-mono">{tooltip.liquidity_alignment?.toFixed(0) ?? '—'}</span></p>
          <p className="text-gray-400">Tier: <span className="text-amber-400">{tooltip.signal_tier}</span></p>
        </div>
      )}
    </div>
  )
}

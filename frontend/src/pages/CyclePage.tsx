import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useCycleHeatmap, useRecommendation, useMTF } from '../api/hooks'
import type { CycleEntry } from '../types'

const PHASES = [
  { label: 'Markdown',     color: '#ef4444', range: [0.75, 1.00] },
  { label: 'Distribution', color: '#eab308', range: [0.50, 0.75] },
  { label: 'Accumulation', color: '#3b82f6', range: [0.25, 0.50] },
  { label: 'Markup',       color: '#22c55e', range: [0.00, 0.25] },
]

function phaseInfo(phase: number | null): { color: string; label: string } {
  if (phase == null) return { color: '#374151', label: '—' }
  if (phase >= 0.75) return { color: '#ef4444', label: 'Markdown' }
  if (phase >= 0.50) return { color: '#eab308', label: 'Distribution' }
  if (phase >= 0.25) return { color: '#3b82f6', label: 'Accumulation' }
  return { color: '#22c55e', label: 'Markup' }
}

function fmt(v: number | null | undefined, decimals = 2) {
  return v == null ? '—' : v.toFixed(decimals)
}

function DrillPanel({ entry, onClose }: { entry: CycleEntry; onClose: () => void }) {
  const navigate = useNavigate()
  const { data: rec, isLoading: recLoading } = useRecommendation(entry.symbol)
  const { data: mtf } = useMTF(entry.symbol)
  const { color, label } = phaseInfo(entry.cycle_phase)

  const signalColor = (s: string) => {
    if (s === 'bullish') return 'text-emerald-400'
    if (s === 'bearish') return 'text-red-400'
    return 'text-gray-400'
  }

  return (
    <div className="fixed inset-0 z-50 flex items-start justify-end" onClick={onClose}>
      {/* Backdrop */}
      <div className="absolute inset-0 bg-black/60" />
      {/* Panel */}
      <div
        className="relative z-10 w-full max-w-sm h-full bg-gray-950 border-l border-gray-800 overflow-y-auto shadow-2xl"
        onClick={e => e.stopPropagation()}
      >
        {/* Header */}
        <div className="sticky top-0 bg-gray-950 border-b border-gray-800 px-4 py-3 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-sm flex-shrink-0" style={{ backgroundColor: color }} />
            <span className="font-bold text-white text-lg">{entry.symbol}</span>
            <span className="text-xs px-2 py-0.5 rounded-full font-medium text-white" style={{ backgroundColor: color + 'cc' }}>
              {label}
            </span>
          </div>
          <button onClick={onClose} className="text-gray-500 hover:text-white transition-colors text-xl leading-none">✕</button>
        </div>

        <div className="p-4 space-y-4">
          {/* Cycle metrics */}
          <div className="bg-gray-900 rounded-lg border border-gray-800 p-3">
            <p className="text-[10px] text-gray-500 uppercase font-semibold mb-2 tracking-wider">Cycle Metrics</p>
            <div className="grid grid-cols-3 gap-3 text-center">
              <div>
                <p className="text-lg font-bold text-white">{fmt(entry.cycle_length, 0)}<span className="text-xs text-gray-500 ml-0.5">d</span></p>
                <p className="text-[10px] text-gray-500">Length</p>
              </div>
              <div>
                <p className="text-lg font-bold" style={{ color }}>{fmt(entry.cycle_phase)}</p>
                <p className="text-[10px] text-gray-500">Phase</p>
              </div>
              <div>
                <p className="text-lg font-bold text-white">{fmt(entry.cycle_strength)}</p>
                <p className="text-[10px] text-gray-500">Strength</p>
              </div>
            </div>
            {/* Phase bar */}
            <div className="mt-3">
              <div className="flex justify-between text-[9px] text-gray-600 mb-1">
                <span>Markup</span><span>Accum</span><span>Distrib</span><span>Markdown</span>
              </div>
              <div className="h-2 rounded-full bg-gray-800 relative overflow-hidden">
                <div className="absolute inset-0 flex">
                  <div className="flex-1 bg-emerald-600/40" />
                  <div className="flex-1 bg-blue-600/40" />
                  <div className="flex-1 bg-yellow-600/40" />
                  <div className="flex-1 bg-red-600/40" />
                </div>
                {entry.cycle_phase != null && (
                  <div
                    className="absolute top-0 w-2 h-2 rounded-full bg-white shadow"
                    style={{ left: `calc(${entry.cycle_phase * 100}% - 4px)` }}
                  />
                )}
              </div>
            </div>
          </div>

          {/* Recommendation summary */}
          {recLoading ? (
            <div className="text-xs text-gray-500 text-center py-4">Loading analysis...</div>
          ) : rec ? (
            <>
              {/* Score + conviction */}
              <div className="bg-gray-900 rounded-lg border border-gray-800 p-3">
                <p className="text-[10px] text-gray-500 uppercase font-semibold mb-2 tracking-wider">Signal</p>
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-2xl font-black" style={{ color: rec.recommendation_color }}>
                      {Math.round(rec.overall_score)}%
                    </p>
                    <p className="text-xs text-gray-400">{rec.recommendation}</p>
                  </div>
                  <div className="text-right">
                    <span className={`text-sm font-bold px-2 py-1 rounded ${
                      rec.conviction === 'HIGH' ? 'bg-emerald-900/50 text-emerald-300' :
                      rec.conviction === 'MEDIUM' ? 'bg-yellow-900/50 text-yellow-300' :
                      'bg-red-900/50 text-red-300'
                    }`}>{rec.conviction}</span>
                    <p className="text-[10px] text-gray-600 mt-1">conviction</p>
                  </div>
                </div>
              </div>

              {/* Trade plan */}
              {rec.trade_plan && (
                <div className="bg-gray-900 rounded-lg border border-gray-800 p-3">
                  <p className="text-[10px] text-gray-500 uppercase font-semibold mb-2 tracking-wider">Trade Plan</p>
                  <div className="grid grid-cols-2 gap-2 text-xs">
                    {[
                      { label: 'Buy', value: rec.trade_plan.buy_price, cls: 'text-emerald-400' },
                      { label: 'Accumulate', value: rec.trade_plan.accumulate_price, cls: 'text-blue-400' },
                      { label: 'Scale', value: rec.trade_plan.scale_price, cls: 'text-yellow-400' },
                      { label: 'Sell', value: rec.trade_plan.sell_price, cls: 'text-red-400' },
                    ].map(({ label, value, cls }) => (
                      <div key={label} className="bg-gray-800/60 rounded px-2 py-1.5">
                        <p className="text-[9px] text-gray-500 uppercase">{label}</p>
                        <p className={`font-mono font-bold ${cls}`}>{value != null ? `$${value.toFixed(2)}` : '—'}</p>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* MTF timeframes */}
              {(mtf?.timeframes || rec.mtf_timeframes) && (
                <div className="bg-gray-900 rounded-lg border border-gray-800 p-3">
                  <p className="text-[10px] text-gray-500 uppercase font-semibold mb-2 tracking-wider">Multi-Timeframe</p>
                  <div className="space-y-1">
                    {(mtf?.timeframes || rec.mtf_timeframes || []).map(tf => (
                      <div key={tf.timeframe} className="flex items-center justify-between text-xs">
                        <span className="text-gray-400 w-8">{tf.timeframe}</span>
                        <span className={`font-medium ${signalColor(tf.signal)}`}>
                          {tf.signal === 'bullish' ? '▲' : tf.signal === 'bearish' ? '▼' : '—'} {tf.signal}
                        </span>
                        <span className="text-gray-500 font-mono text-[10px]">
                          {tf.confidence != null ? `${(tf.confidence * 100).toFixed(0)}%` : ''}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Recent patterns */}
              {rec.pattern_context.recent.length > 0 && (
                <div className="bg-gray-900 rounded-lg border border-gray-800 p-3">
                  <p className="text-[10px] text-gray-500 uppercase font-semibold mb-2 tracking-wider">Recent Patterns</p>
                  <div className="space-y-1">
                    {rec.pattern_context.recent.slice(0, 4).map((p, i) => (
                      <div key={i} className="flex items-center justify-between text-xs">
                        <span className={signalColor(p.direction)}>{p.direction === 'bullish' ? '▲' : '▼'}</span>
                        <span className="text-gray-300 flex-1 mx-2 truncate">{p.name}</span>
                        <span className="text-gray-600 text-[10px]">{p.date}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </>
          ) : null}

          {/* View chart button */}
          <button
            onClick={() => navigate(`/chart?symbol=${entry.symbol}`)}
            className="w-full bg-emerald-700 hover:bg-emerald-600 text-white rounded-lg py-2.5 text-sm font-semibold transition-colors"
          >
            View Chart →
          </button>
        </div>
      </div>
    </div>
  )
}

export default function CyclePage() {
  const { data: entries, isLoading } = useCycleHeatmap()
  const [selected, setSelected] = useState<CycleEntry | null>(null)

  if (isLoading) return <div className="text-gray-500 py-20 text-center">Loading cycle data...</div>

  // Group entries by phase, sorted by strength descending within each group
  const grouped = PHASES.map(phase => ({
    ...phase,
    entries: (entries || [])
      .filter(e => {
        const p = e.cycle_phase ?? 0
        return p >= phase.range[0] && p < phase.range[1]
      })
      .sort((a, b) => (b.cycle_strength ?? 0) - (a.cycle_strength ?? 0)),
  }))

  return (
    <div>
      <div className="flex items-center justify-between mb-2">
        <h2 className="text-2xl font-bold">Cycle Heatmap</h2>
        <span className="text-xs text-gray-600">opacity = strength · click cell for details</span>
      </div>

      {(!entries || entries.length === 0) && (
        <p className="text-gray-500 text-center py-10">
          No cycle data. Run: python scripts/run_roboalgo.py --step cycles
        </p>
      )}

      <div className="space-y-5">
        {grouped.map(group => group.entries.length === 0 ? null : (
          <div key={group.label}>
            {/* Group header */}
            <div
              className="flex items-center gap-2 mb-2 px-3 py-1.5 rounded-lg"
              style={{ backgroundColor: group.color + '22', borderLeft: `3px solid ${group.color}` }}
            >
              <div className="w-2.5 h-2.5 rounded-sm" style={{ backgroundColor: group.color }} />
              <span className="text-sm font-bold text-white">{group.label}</span>
              <span className="text-xs text-gray-400 ml-1">{group.entries.length} instruments</span>
            </div>

            {/* Cells grid */}
            <div className="grid grid-cols-6 sm:grid-cols-8 md:grid-cols-10 lg:grid-cols-12 gap-2">
              {group.entries.map(entry => {
                const strength = entry.cycle_strength ?? 0.5
                const opacity = Math.min(0.65 + strength * 0.35, 1)
                const isActive = selected?.symbol === entry.symbol
                return (
                  <div
                    key={entry.symbol}
                    className={`rounded-lg p-2 text-center transition-all cursor-pointer hover:scale-105 hover:ring-2 hover:ring-white/30 ${
                      isActive ? 'ring-2 ring-white scale-105' : ''
                    }`}
                    style={{ backgroundColor: group.color, opacity }}
                    title={`${entry.symbol} · ${group.label} · strength=${strength.toFixed(2)} · ${entry.cycle_length?.toFixed(0)}d cycle`}
                    onClick={() => setSelected(isActive ? null : entry)}
                  >
                    <p className="text-xs font-bold text-white drop-shadow">{entry.symbol}</p>
                    <p className="text-[10px] text-white/80 font-mono">{entry.cycle_length?.toFixed(0)}d</p>
                    <p className="text-[9px] text-white/60">{(strength * 100).toFixed(0)}%</p>
                  </div>
                )
              })}
            </div>
          </div>
        ))}
      </div>

      {/* Drill panel */}
      {selected && (
        <DrillPanel entry={selected} onClose={() => setSelected(null)} />
      )}
    </div>
  )
}

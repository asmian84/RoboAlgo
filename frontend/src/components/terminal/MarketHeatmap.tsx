import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import type { CommandCenterInstrument } from '../../types'

// ── Color map ─────────────────────────────────────────────────────────────────

const STATE_BG: Record<string, string> = {
  EXPANSION:   '#431407',
  TREND:       '#052e16',
  COMPRESSION: '#172554',
  CHAOS:       '#450a0a',
}

const STATE_BORDER: Record<string, string> = {
  EXPANSION:   '#c2410c',
  TREND:       '#16a34a',
  COMPRESSION: '#2563eb',
  CHAOS:       '#dc2626',
}

const STATE_TEXT: Record<string, string> = {
  EXPANSION:   '#fb923c',
  TREND:       '#4ade80',
  COMPRESSION: '#93c5fd',
  CHAOS:       '#f87171',
}

// ── Cell ─────────────────────────────────────────────────────────────────────

function HeatCell({
  inst,
  onClick,
}: {
  inst:    CommandCenterInstrument
  onClick: () => void
}) {
  const bg     = STATE_BG[inst.state]     ?? '#1f2937'
  const border = STATE_BORDER[inst.state] ?? '#374151'
  const tc     = STATE_TEXT[inst.state]   ?? '#9ca3af'
  const vol    = (inst.volatility_percentile ?? 0) * 100

  return (
    <button
      className="flex flex-col items-center justify-center rounded p-1.5 transition-transform hover:scale-105 border"
      style={{ backgroundColor: bg, borderColor: border + '60', minHeight: '52px' }}
      onClick={onClick}
      title={`${inst.symbol} · ${inst.state} · Vol ${vol.toFixed(0)}%`}
    >
      <span className="text-[10px] font-mono font-bold" style={{ color: tc }}>
        {inst.symbol.length > 5 ? inst.symbol.slice(0, 5) : inst.symbol}
      </span>
      {/* State label (abbreviated) */}
      <span className="text-[8px] font-bold uppercase" style={{ color: tc, opacity: 0.7 }}>
        {inst.state === 'COMPRESSION' ? 'COMP' : inst.state.slice(0, 3)}
      </span>
      {/* Volatility mini-bar */}
      <div className="w-full h-0.5 bg-black/20 rounded-full overflow-hidden mt-1">
        <div className="h-full" style={{ width: `${vol}%`, backgroundColor: tc, opacity: 0.6 }} />
      </div>
    </button>
  )
}

// ── Panel ─────────────────────────────────────────────────────────────────────

interface Props {
  instruments: CommandCenterInstrument[]
}

export default function MarketHeatmap({ instruments }: Props) {
  const navigate   = useNavigate()
  const [filter, setFilter] = useState<string>('all')

  const STATE_ORDER = ['EXPANSION', 'TREND', 'COMPRESSION', 'CHAOS']

  const filtered = filter === 'all'
    ? instruments
    : instruments.filter(i => i.state === filter)

  // Count by state
  const counts: Record<string, number> = {}
  instruments.forEach(i => { counts[i.state] = (counts[i.state] ?? 0) + 1 })

  return (
    <div className="bg-gray-900 border border-gray-800 rounded-xl flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-gray-800">
        <div>
          <h2 className="text-sm font-semibold text-gray-100">Market Heatmap</h2>
          <p className="text-xs text-gray-500 mt-0.5">Instruments colored by market state · click to open chart</p>
        </div>
        <div className="flex items-center gap-1 flex-wrap">
          <button
            onClick={() => setFilter('all')}
            className={`px-2 py-0.5 rounded text-[10px] font-medium transition-colors ${filter === 'all' ? 'bg-gray-600 text-white' : 'text-gray-500 hover:text-gray-300'}`}
          >
            All ({instruments.length})
          </button>
          {STATE_ORDER.map(s => {
            const n = counts[s] ?? 0
            if (!n) return null
            const tc = STATE_TEXT[s] ?? '#9ca3af'
            const bg = STATE_BG[s]   ?? '#1f2937'
            return (
              <button
                key={s}
                onClick={() => setFilter(filter === s ? 'all' : s)}
                className={`px-2 py-0.5 rounded text-[10px] font-bold transition-colors border ${filter === s ? 'opacity-100' : 'opacity-60 hover:opacity-100'}`}
                style={{ color: tc, backgroundColor: bg, borderColor: STATE_BORDER[s] + '60' }}
              >
                {s === 'COMPRESSION' ? 'COMP' : s} {n}
              </button>
            )
          })}
        </div>
      </div>

      {/* Grid */}
      {filtered.length === 0 ? (
        <div className="flex items-center justify-center h-32 text-gray-600 text-xs">
          No instruments — run market state pipeline first
        </div>
      ) : (
        <div
          className="p-4 grid gap-1.5"
          style={{ gridTemplateColumns: 'repeat(auto-fill, minmax(64px, 1fr))' }}
        >
          {filtered.map(inst => (
            <HeatCell
              key={inst.symbol}
              inst={inst}
              onClick={() => navigate(`/chart?symbol=${inst.symbol}`)}
            />
          ))}
        </div>
      )}

      {/* Legend */}
      <div className="flex items-center gap-4 px-4 py-2 border-t border-gray-800 text-[10px]">
        {STATE_ORDER.map(s => (
          <div key={s} className="flex items-center gap-1.5">
            <div
              className="w-3 h-3 rounded border"
              style={{ backgroundColor: STATE_BG[s], borderColor: STATE_BORDER[s] }}
            />
            <span style={{ color: STATE_TEXT[s] }}>{s === 'COMPRESSION' ? 'COMP' : s}</span>
          </div>
        ))}
      </div>
    </div>
  )
}

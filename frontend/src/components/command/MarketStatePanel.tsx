/**
 * Market State panel — shows per-instrument state (EXPANSION / TREND / COMPRESSION / CHAOS)
 * with volatility, trend strength, MA alignment, and size multiplier.
 *
 * Extracted as a standalone component so it can be used on any page.
 */
import { useNavigate } from 'react-router-dom'
import type { CommandCenterInstrument } from '../../types'

// ── Color constants ──────────────────────────────────────────────────────────

const STATE_COLORS: Record<string, string> = {
  EXPANSION:   '#f97316',
  TREND:       '#22c55e',
  COMPRESSION: '#60a5fa',
  CHAOS:       '#ef4444',
}

const ALIGN_COLORS: Record<string, string> = {
  bullish: '#22c55e',
  bearish: '#ef4444',
  neutral: '#9ca3af',
}

// ── Helper sub-components ────────────────────────────────────────────────────

function StateBadge({ state }: { state: string }) {
  const color = STATE_COLORS[state] ?? '#6b7280'
  return (
    <span
      className="px-1.5 py-0.5 rounded text-xs font-bold uppercase"
      style={{ color, backgroundColor: color + '20' }}
    >
      {state}
    </span>
  )
}

function ScoreBar({ score, max = 100 }: { score: number; max?: number }) {
  const pct = Math.max(0, Math.min(100, (score / max) * 100))
  const color = score >= 80 ? '#22c55e' : score >= 65 ? '#4ade80' : score >= 50 ? '#eab308' : score >= 30 ? '#f97316' : '#ef4444'
  return (
    <div className="flex items-center gap-2">
      <div className="flex-1 h-1.5 bg-gray-800 rounded-full overflow-hidden">
        <div className="h-full rounded-full transition-all" style={{ width: `${pct}%`, backgroundColor: color }} />
      </div>
      <span className="text-xs font-mono w-8 text-right" style={{ color }}>{score.toFixed(0)}</span>
    </div>
  )
}

function PctBar({ value, max = 100, color = '#22c55e' }: { value: number; max?: number; color?: string }) {
  const pct = Math.max(0, Math.min(100, (value / max) * 100))
  return (
    <div className="flex items-center gap-1.5">
      <div className="flex-1 h-1 bg-gray-800 rounded-full overflow-hidden">
        <div className="h-full rounded-full" style={{ width: `${pct}%`, backgroundColor: color }} />
      </div>
      <span className="text-xs font-mono w-10 text-right" style={{ color }}>{(value * 100).toFixed(1)}%</span>
    </div>
  )
}

// ── Instrument row ───────────────────────────────────────────────────────────

function InstrumentRow({ inst, onClick }: { inst: CommandCenterInstrument; onClick: () => void }) {
  const volPct = (inst.volatility_percentile ?? 0) * 100
  const trendStr = inst.trend_strength ?? 0

  return (
    <tr
      className="border-b border-gray-800/40 hover:bg-gray-800/30 cursor-pointer transition-colors"
      onClick={onClick}
    >
      <td className="px-4 py-2 font-mono font-bold text-gray-100">{inst.symbol}</td>
      <td className="px-4 py-2">
        <StateBadge state={inst.state} />
      </td>
      <td className="px-4 py-2 w-28">
        <PctBar value={volPct / 100} max={1} color={STATE_COLORS[inst.state] ?? '#9ca3af'} />
      </td>
      <td className="px-4 py-2 w-28">
        <ScoreBar score={trendStr} />
      </td>
      <td className="px-4 py-2">
        <span
          className="text-xs font-semibold capitalize"
          style={{ color: ALIGN_COLORS[inst.ma_alignment ?? 'neutral'] ?? '#9ca3af' }}
        >
          {inst.ma_alignment ?? '—'}
        </span>
      </td>
      <td className="px-4 py-2 text-right text-gray-400">
        {inst.size_multiplier != null ? `${inst.size_multiplier}×` : '—'}
      </td>
    </tr>
  )
}

// ── Main panel ───────────────────────────────────────────────────────────────

interface Props {
  data: {
    counts: Record<string, number>
    instruments: CommandCenterInstrument[]
    error?: string
  }
}

const STATE_ORDER = ['EXPANSION', 'TREND', 'COMPRESSION', 'CHAOS']

export default function MarketStatePanel({ data }: Props) {
  const navigate = useNavigate()
  const { counts = {}, instruments = [] } = data

  return (
    <div className="bg-gray-900 border border-gray-800 rounded-xl flex flex-col">
      <div className="flex items-center justify-between px-4 py-3 border-b border-gray-800">
        <div>
          <h2 className="text-sm font-semibold text-gray-100">Market State</h2>
          <p className="text-xs text-gray-500 mt-0.5">{instruments.length} instruments tracked</p>
        </div>
        <div className="flex gap-1.5 flex-wrap">
          {STATE_ORDER.map(s => {
            const n = counts[s] ?? 0
            if (!n && !instruments.length) return null
            return (
              <span
                key={s}
                className="px-2 py-0.5 rounded-full text-xs font-bold"
                style={{ color: STATE_COLORS[s], backgroundColor: STATE_COLORS[s] + '20' }}
              >
                {s.slice(0, 3)} {n}
              </span>
            )
          })}
        </div>
      </div>
      <div className="flex-1 overflow-auto">
        {instruments.length === 0 ? (
          <div className="flex items-center justify-center h-24 text-gray-600 text-sm">
            No market state data — run pipeline first
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-gray-800 text-gray-500 uppercase">
                  <th className="text-left px-4 py-2 font-medium">Symbol</th>
                  <th className="text-left px-4 py-2 font-medium">State</th>
                  <th className="px-4 py-2 font-medium">Volatility</th>
                  <th className="px-4 py-2 font-medium">Trend</th>
                  <th className="text-left px-4 py-2 font-medium">Alignment</th>
                  <th className="text-right px-4 py-2 font-medium">Size</th>
                </tr>
              </thead>
              <tbody>
                {instruments.map(inst => (
                  <InstrumentRow key={inst.symbol} inst={inst} onClick={() => navigate(`/chart?symbol=${inst.symbol}`)} />
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  )
}

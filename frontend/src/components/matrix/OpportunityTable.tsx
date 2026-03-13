import { useState, useMemo } from 'react'
import { useNavigate } from 'react-router-dom'
import type { CommandCenterSignal } from '../../types'

const STATE_COLORS: Record<string, string> = {
  EXPANSION:   '#f97316',
  TREND:       '#22c55e',
  COMPRESSION: '#60a5fa',
  CHAOS:       '#ef4444',
}

const TIER_COLORS: Record<string, string> = {
  HIGH:   '#ef4444',
  MEDIUM: '#f97316',
  WATCH:  '#eab308',
  NONE:   '#6b7280',
}

function scoreColor(s: number | null): string {
  if (s == null)  return '#6b7280'
  if (s >= 70)    return '#22c55e'
  if (s >= 50)    return '#eab308'
  return '#ef4444'
}

function ScoreBar({ score }: { score: number | null }) {
  if (score == null) return <span className="text-gray-600 text-xs">—</span>
  const color = scoreColor(score)
  return (
    <div className="flex items-center gap-1.5">
      <div className="w-16 h-1.5 bg-gray-800 rounded-full overflow-hidden">
        <div className="h-full rounded-full" style={{ width: `${Math.min(score, 100)}%`, backgroundColor: color }} />
      </div>
      <span className="text-[10px] font-mono w-6" style={{ color }}>{score.toFixed(0)}</span>
    </div>
  )
}

type FilterLevel = 'all' | 'gt50' | 'gt70' | 'gt80'

interface Props {
  signals: CommandCenterSignal[]
}

export default function OpportunityTable({ signals }: Props) {
  const navigate = useNavigate()
  const [filter, setFilter] = useState<FilterLevel>('all')
  const [sortKey, setSortKey] = useState<keyof CommandCenterSignal>('setup_quality_score')

  const filtered = useMemo(() => {
    let base = [...signals]
    if (filter === 'gt80') base = base.filter(s => (s.setup_quality_score ?? 0) >= 80)
    else if (filter === 'gt70') base = base.filter(s => (s.setup_quality_score ?? 0) >= 70)
    else if (filter === 'gt50') base = base.filter(s => (s.setup_quality_score ?? 0) >= 50)
    return base.sort((a, b) => ((b[sortKey] as number) ?? 0) - ((a[sortKey] as number) ?? 0))
  }, [signals, filter, sortKey])

  const FILTERS: { id: FilterLevel; label: string }[] = [
    { id: 'all',  label: `All (${signals.length})` },
    { id: 'gt50', label: 'Score > 50' },
    { id: 'gt70', label: 'Score > 70' },
    { id: 'gt80', label: 'Score > 80' },
  ]

  type SortCol = { key: keyof CommandCenterSignal; label: string }
  const SORT_COLS: SortCol[] = [
    { key: 'setup_quality_score',    label: 'Setup Q' },
    { key: 'breakout_quality_score', label: 'Breakout' },
    { key: 'liquidity_alignment',    label: 'Liq Align' },
    { key: 'confluence_score',       label: 'Confluence' },
  ]

  return (
    <div className="bg-gray-900 border border-gray-800 rounded-xl flex flex-col">
      {/* Header + filters */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-gray-800 flex-wrap gap-2">
        <h2 className="text-sm font-semibold text-gray-100">Opportunity Table</h2>
        <div className="flex items-center gap-2 flex-wrap">
          {/* Sort */}
          <div className="flex items-center gap-1">
            <span className="text-[10px] text-gray-600">Sort:</span>
            {SORT_COLS.map(c => (
              <button
                key={c.key as string}
                onClick={() => setSortKey(c.key)}
                className={`px-2 py-0.5 rounded text-[10px] font-medium transition-colors ${
                  sortKey === c.key ? 'bg-emerald-700 text-white' : 'bg-gray-800 text-gray-500 hover:text-gray-300'
                }`}
              >
                {c.label}
              </button>
            ))}
          </div>
          {/* Filter */}
          <div className="flex items-center gap-1">
            {FILTERS.map(f => (
              <button
                key={f.id}
                onClick={() => setFilter(f.id)}
                className={`px-2 py-0.5 rounded text-[10px] font-medium transition-colors ${
                  filter === f.id ? 'bg-blue-700 text-white' : 'bg-gray-800 text-gray-500 hover:text-gray-300'
                }`}
              >
                {f.label}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Table */}
      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr className="border-b border-gray-800 text-gray-500 uppercase">
              <th className="text-left px-4 py-2 font-medium">Symbol</th>
              <th className="px-4 py-2 font-medium">Setup Q</th>
              <th className="px-4 py-2 font-medium">Breakout</th>
              <th className="px-4 py-2 font-medium">Liq Shelf</th>
              <th className="px-4 py-2 font-medium">Liq Align</th>
              <th className="text-left px-4 py-2 font-medium">State</th>
              <th className="text-left px-4 py-2 font-medium">Vol Regime</th>
              <th className="text-left px-4 py-2 font-medium">Signal</th>
            </tr>
          </thead>
          <tbody>
            {filtered.length === 0 ? (
              <tr>
                <td colSpan={8} className="px-4 py-8 text-center text-gray-600">
                  No signals match the current filter
                </td>
              </tr>
            ) : (
              filtered.map(sig => {
                const stateColor = STATE_COLORS[sig.market_state] ?? '#6b7280'
                const tierColor  = TIER_COLORS[sig.signal_tier]   ?? '#6b7280'
                return (
                  <tr
                    key={sig.symbol}
                    className="border-b border-gray-800/40 hover:bg-gray-800/30 cursor-pointer transition-colors"
                    onClick={() => navigate(`/chart?symbol=${sig.symbol}`)}
                  >
                    <td className="px-4 py-2 font-mono font-bold text-gray-100">{sig.symbol}</td>
                    <td className="px-4 py-2"><ScoreBar score={sig.setup_quality_score} /></td>
                    <td className="px-4 py-2"><ScoreBar score={sig.breakout_quality_score} /></td>
                    <td className="px-4 py-2"><ScoreBar score={sig.liquidity_shelf_score} /></td>
                    <td className="px-4 py-2"><ScoreBar score={sig.liquidity_alignment} /></td>
                    <td className="px-4 py-2">
                      <span
                        className="text-[10px] font-bold px-1.5 py-0.5 rounded uppercase"
                        style={{ color: stateColor, backgroundColor: stateColor + '20' }}
                      >
                        {sig.market_state === 'COMPRESSION' ? 'COMP' : (sig.market_state ?? '?').slice(0, 4)}
                      </span>
                    </td>
                    <td className="px-4 py-2 text-gray-400 text-[10px]">
                      {sig.volatility_regime || '—'}
                    </td>
                    <td className="px-4 py-2">
                      <span
                        className="text-[10px] font-bold px-1.5 py-0.5 rounded"
                        style={{ color: tierColor, backgroundColor: tierColor + '20' }}
                      >
                        {sig.signal_tier}
                      </span>
                    </td>
                  </tr>
                )
              })
            )}
          </tbody>
        </table>
      </div>
    </div>
  )
}

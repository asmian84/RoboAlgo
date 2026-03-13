import { useNavigate } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import api from '../../api/client'

// ── Sniper live feed ──────────────────────────────────────────────────────────

function useSniperFeed() {
  return useQuery<any[]>({
    queryKey: ['sniper', 'sidebar'],
    queryFn: async () => {
      const r = await api.get('/sniper-entry/scan', { params: { top_n: 500 } })
      const results = r.data?.results ?? { negative_gamma: [], technical: [] }
      // Combine both buckets, sort by entry_quality descending
      const combined = [
        ...(results.negative_gamma ?? []),
        ...(results.technical ?? []),
      ]
      // Dedupe by symbol, keep highest quality
      const seen = new Map<string, any>()
      for (const item of combined) {
        const prev = seen.get(item.symbol)
        if (!prev || (item.entry_quality ?? 0) > (prev.entry_quality ?? 0)) {
          seen.set(item.symbol, item)
        }
      }
      return [...seen.values()].sort((a, b) => (b.entry_quality ?? 0) - (a.entry_quality ?? 0))
    },
    staleTime: 5 * 60_000,
    refetchInterval: 5 * 60_000,
  })
}

// ── Quality bar ───────────────────────────────────────────────────────────────

function QualityBar({ score }: { score: number }) {
  const pct = Math.min(100, Math.max(0, score))
  const color = pct >= 70 ? '#22c55e' : pct >= 50 ? '#f59e0b' : '#6b7280'
  return (
    <div className="h-1 rounded-full bg-gray-800 mt-1">
      <div className="h-full rounded-full transition-all" style={{ width: `${pct}%`, backgroundColor: color }} />
    </div>
  )
}

// ── Panel ─────────────────────────────────────────────────────────────────────

interface Props {
  signals?: any[]           // kept for TerminalLayout compat, unused
  selectedSymbol?: string
}

export default function WatchlistPanel({ selectedSymbol }: Props) {
  const navigate = useNavigate()
  const { data: picks = [], isLoading } = useSniperFeed()

  return (
    <aside className="h-full min-h-0 overflow-y-auto bg-gray-950 p-2">
      {/* Header */}
      <div className="flex items-center justify-between mb-2 px-1">
        <h2 className="text-[11px] font-semibold uppercase tracking-wide text-gray-400">
          🎯 Sniper Picks
        </h2>
        {picks.length > 0 && (
          <span className="text-[10px] text-gray-600">{picks.length}</span>
        )}
      </div>

      {/* Loading */}
      {isLoading && (
        <div className="flex items-center gap-2 px-1 py-3">
          <div className="w-3 h-3 border border-emerald-500 border-t-transparent rounded-full animate-spin" />
          <span className="text-[10px] text-gray-600">Scanning…</span>
        </div>
      )}

      {/* Picks list */}
      {!isLoading && picks.length === 0 && (
        <div className="rounded border border-gray-800 bg-gray-900/50 px-3 py-4 text-xs text-gray-500">
          No sniper setups found.
        </div>
      )}

      <div className="grid gap-1">
        {picks.map((item, i) => {
          const active  = item.symbol === selectedSymbol
          const quality = item.entry_quality ?? 0
          const rsi     = item.rsi ?? null
          const fromLow = item.distance_from_all_low_pct ?? item.distance_from_60_low_pct ?? null

          return (
            <button
              key={item.symbol}
              onClick={() => navigate(`/chart?symbol=${item.symbol}`)}
              className={`rounded border p-2 text-left transition-colors w-full ${
                active
                  ? 'border-emerald-500/50 bg-emerald-500/10'
                  : 'border-gray-800 bg-gray-900/50 hover:border-gray-700 hover:bg-gray-900'
              }`}
            >
              {/* Symbol + rank + price */}
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-1.5">
                  <span className="text-[9px] text-gray-600 font-mono w-4 shrink-0">#{i + 1}</span>
                  <span className="font-mono text-sm font-bold text-gray-100">{item.symbol}</span>
                </div>
                <span className="text-[10px] font-mono text-gray-400">${item.current_price?.toFixed(2)}</span>
              </div>

              {/* Quality bar */}
              <QualityBar score={quality} />

              {/* Metrics row */}
              <div className="flex items-center justify-between mt-1.5">
                <span className="text-[10px] font-bold" style={{
                  color: quality >= 70 ? '#22c55e' : quality >= 50 ? '#f59e0b' : '#9ca3af'
                }}>
                  {quality.toFixed(0)} <span className="font-normal text-gray-600">qual</span>
                </span>
                <div className="flex gap-2 text-[9px] text-gray-500">
                  {rsi != null && (
                    <span style={{ color: rsi < 25 ? '#f87171' : rsi < 35 ? '#fb923c' : '#9ca3af' }}>
                      RSI {rsi.toFixed(0)}
                    </span>
                  )}
                  {fromLow != null && (
                    <span>{fromLow.toFixed(1)}% low</span>
                  )}
                </div>
              </div>
            </button>
          )
        })}
      </div>
    </aside>
  )
}

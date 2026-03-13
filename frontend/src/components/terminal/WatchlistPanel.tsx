import { useMemo } from 'react'
import { useNavigate } from 'react-router-dom'
import { useCategories, useSignals } from '../../api/hooks'
import type { CommandCenterSignal, Signal } from '../../types'

interface Props {
  signals: CommandCenterSignal[]
  selectedSymbol?: string
}

const STATE_COLORS: Record<string, string> = {
  TREND: 'var(--ra-bullish)',
  EXPANSION: 'var(--ra-expansion)',
  COMPRESSION: 'var(--ra-compression)',
  CHAOS: 'var(--ra-bearish)',
}

function normalizeState(state: string | null | undefined): string {
  if (!state) return 'WATCH'
  const s = state.toUpperCase()
  if (s.includes('EXPANSION')) return 'EXPANSION'
  if (s.includes('COMPRESSION') || s.includes('RANGE')) return 'COMPRESSION'
  if (s.includes('TREND') || s.includes('BULL')) return 'TREND'
  if (s.includes('BEAR') || s.includes('CHAOS') || s.includes('RISK')) return 'CHAOS'
  return s
}

function scoreBlocks(score: number | null): string {
  if (score == null) return '----------'
  const filled = Math.max(0, Math.min(10, Math.round(score / 10)))
  return `${'█'.repeat(filled)}${'░'.repeat(10 - filled)}`
}

export default function WatchlistPanel({ signals, selectedSymbol }: Props) {
  const navigate = useNavigate()
  const { data: latestSignals = [] } = useSignals(0)
  const { data: categories } = useCategories()

  const latestBySymbol = useMemo(() => {
    const map = new Map<string, Signal>()
    for (const sig of [...latestSignals].sort((a, b) => b.probability - a.probability)) {
      if (!map.has(sig.symbol)) map.set(sig.symbol, sig)
    }
    return map
  }, [latestSignals])

  const commandBySymbol = useMemo(() => new Map(signals.map(s => [s.symbol, s])), [signals])

  const hotSymbols = useMemo(() => {
    const fromLatest = [...latestSignals].sort((a, b) => b.probability - a.probability).map(s => s.symbol)
    const fromCommand = [...signals].sort((a, b) => (b.setup_quality_score ?? 0) - (a.setup_quality_score ?? 0)).map(s => s.symbol)
    const leaders = categories?.underlying_leaders ?? []
    return [...new Set([...fromLatest, ...fromCommand, ...leaders])].slice(0, 24)
  }, [categories?.underlying_leaders, latestSignals, signals])

  const watchlist = useMemo(() => {
    return hotSymbols.map(symbol => {
      const cmd = commandBySymbol.get(symbol)
      const latest = latestBySymbol.get(symbol)
      const score = cmd?.setup_quality_score ?? (latest ? latest.probability * 100 : null)
      const state = normalizeState(cmd?.market_state ?? latest?.market_phase)
      return { symbol, score, state }
    })
  }, [commandBySymbol, hotSymbols, latestBySymbol])

  return (
    <aside className="h-full min-h-0 overflow-y-auto border-r border-gray-800 bg-gray-950 p-2">
      <div className="mb-2 px-1">
        <h2 className="text-[11px] font-semibold uppercase tracking-wide text-gray-400">Hot Watchlist</h2>
      </div>

      <div className="grid gap-1">
        {watchlist.map(item => {
          const active = item.symbol === selectedSymbol
          const stateColor = STATE_COLORS[item.state] ?? '#9ca3af'
          return (
            <button
              key={item.symbol}
              onClick={() => navigate(`/chart?symbol=${item.symbol}`)}
              className={`rounded border p-2 text-left transition-colors ${
                active
                  ? 'border-emerald-500/50 bg-emerald-500/10'
                  : 'border-gray-800 bg-gray-900/50 hover:border-gray-700 hover:bg-gray-900'
              }`}
            >
              <p className="font-mono text-sm font-semibold text-gray-100">{item.symbol}</p>
              <p className="font-mono text-[11px] text-gray-300">
                {scoreBlocks(item.score)}{' '}
                <span className="font-semibold">{Math.round(item.score ?? 0)}</span>
              </p>
              <p className="text-[10px] font-semibold uppercase tracking-wide" style={{ color: stateColor }}>
                {item.state}
              </p>
            </button>
          )
        })}

        {watchlist.length === 0 && (
          <div className="rounded border border-gray-800 bg-gray-900/50 px-3 py-4 text-xs text-gray-500">
            No symbols available.
          </div>
        )}
      </div>
    </aside>
  )
}

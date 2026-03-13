import { useMemo } from 'react'
import { useNavigate } from 'react-router-dom'
import { useCommandCenter } from '../api/hooks'

function scoreColor(score: number): string {
  if (score >= 80) return 'rgba(34,197,94,0.28)'
  if (score >= 65) return 'rgba(168,85,247,0.25)'
  if (score >= 50) return 'rgba(249,115,22,0.25)'
  return 'rgba(239,68,68,0.22)'
}

function stateColor(state: string): string {
  if (state === 'EXPANSION') return 'var(--ra-expansion)'
  if (state === 'COMPRESSION') return 'var(--ra-compression)'
  if (state === 'TREND') return 'var(--ra-bullish)'
  return 'var(--ra-bearish)'
}

export default function OpportunityMatrixPage() {
  const navigate = useNavigate()
  const { data, isLoading, refetch, isFetching } = useCommandCenter()

  const tiles = useMemo(
    () => [...(data?.opportunity_map?.signals ?? [])].sort((a, b) => (b.setup_quality_score ?? 0) - (a.setup_quality_score ?? 0)),
    [data?.opportunity_map?.signals],
  )

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between gap-2">
        <div>
          <h1 className="text-lg font-semibold text-gray-100">Opportunity Matrix</h1>
          <p className="text-xs text-gray-500">Tile scan for fastest symbol triage.</p>
        </div>
        <button
          onClick={() => refetch()}
          disabled={isFetching}
          className="rounded border border-gray-700 bg-gray-900 px-2 py-1 text-xs text-gray-300 transition-colors hover:bg-gray-800 disabled:opacity-50"
        >
          {isFetching ? 'Refreshing...' : 'Refresh'}
        </button>
      </div>

      {isLoading && !data && (
        <div className="grid grid-cols-2 gap-2 md:grid-cols-4 xl:grid-cols-6">
          {Array.from({ length: 12 }).map((_, index) => (
            <div key={index} className="h-20 animate-pulse rounded border border-gray-800 bg-gray-900" />
          ))}
        </div>
      )}

      <div className="grid grid-cols-2 gap-2 md:grid-cols-4 xl:grid-cols-6">
        {tiles.map(tile => {
          const setupScore = Math.round(tile.setup_quality_score ?? 0)
          return (
            <button
              key={tile.symbol}
              onClick={() => navigate(`/chart?symbol=${tile.symbol}`)}
              className="rounded border border-gray-800 p-2 text-left transition-transform hover:scale-[1.02]"
              style={{ backgroundColor: scoreColor(setupScore) }}
            >
              <p className="font-mono text-sm font-semibold text-gray-100">{tile.symbol}</p>
              <p className="font-mono text-base text-gray-200">{setupScore}</p>
              <p className="text-[10px] font-semibold uppercase tracking-wide" style={{ color: stateColor(tile.market_state) }}>
                {tile.market_state}
              </p>
            </button>
          )
        })}
      </div>
    </div>
  )
}

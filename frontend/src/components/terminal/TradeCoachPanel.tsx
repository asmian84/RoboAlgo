import { useMemo } from 'react'
import { useSimilarSetups, useTradeCoach } from '../../api/hooks'
import type { CommandCenterSignal } from '../../types'

interface Props {
  symbol?: string
  signals: CommandCenterSignal[]
}

function toPct(value: number | null | undefined): string {
  if (value == null) return 'N/A'
  return `${(value * 100).toFixed(1)}%`
}

export default function TradeCoachPanel({ symbol, signals }: Props) {
  const fallback = useMemo(() => [...signals].sort((a, b) => (b.confluence_score ?? 0) - (a.confluence_score ?? 0))[0]?.symbol, [signals])
  const activeSymbol = symbol ?? fallback ?? ''

  const { data: explanation, isLoading: loadingExplanation } = useTradeCoach(activeSymbol)
  const { data: similar, isLoading: loadingSimilar } = useSimilarSetups(activeSymbol)

  const loading = loadingExplanation || loadingSimilar

  // No data and not loading — hide the panel entirely (no empty cards)
  if (!loading && !explanation && !similar) return null

  const hasSetupType  = !!explanation?.setup_type
  const hasEvidence   = (explanation?.evidence?.length ?? 0) > 0
  const hasRiskItems  = (explanation?.risk_factors?.length ?? 0) > 0
  const hasSimilar    = !!similar && (similar.sample_size ?? 0) > 0

  // Nothing meaningful to show yet (still loading or all sections empty)
  if (!loading && !hasSetupType && !hasEvidence && !hasRiskItems && !hasSimilar) return null

  return (
    <div className="max-h-[34vh] border-t border-gray-800 p-2 overflow-y-auto">
    <section className="rounded border border-gray-800 bg-gray-900/60 p-2">
      <div className="mb-2 px-1">
        <h2 className="text-[11px] font-semibold uppercase tracking-wide text-gray-400">
          Trade Coach {activeSymbol ? <span className="font-mono text-gray-200">· {activeSymbol}</span> : null}
        </h2>
      </div>

      <div className="space-y-2 text-xs">
        {/* Signal explanation — only when we have a setup type */}
        {(loading || hasSetupType) && (
          <article className="rounded border border-gray-800 bg-gray-950/70 p-2">
            <h3 className="mb-1 text-[10px] font-semibold uppercase tracking-wide text-gray-500">Signal explanation</h3>
            <p className="text-gray-300">
              {loading ? 'Analysing signal context...' : `Setup: ${explanation!.setup_type}`}
            </p>
          </article>
        )}

        {/* Evidence — only when populated */}
        {hasEvidence && (
          <article className="rounded border border-gray-800 bg-gray-950/70 p-2">
            <h3 className="mb-1 text-[10px] font-semibold uppercase tracking-wide text-gray-500">Evidence</h3>
            <ul className="space-y-1 text-gray-300">
              {explanation!.evidence!.slice(0, 4).map((item, idx) => (
                <li key={`${item}-${idx}`}>• {item}</li>
              ))}
            </ul>
          </article>
        )}

        {/* Risk warnings — only when populated */}
        {hasRiskItems && (
          <article className="rounded border border-gray-800 bg-gray-950/70 p-2">
            <h3 className="mb-1 text-[10px] font-semibold uppercase tracking-wide text-gray-500">Risk warnings</h3>
            <ul className="space-y-1 text-gray-300">
              {explanation!.risk_factors!.slice(0, 3).map((item, idx) => (
                <li key={`${item}-${idx}`}>• {item}</li>
              ))}
            </ul>
          </article>
        )}

        {/* Historical performance — only when we have sample data */}
        {hasSimilar && (
          <article className="rounded border border-gray-800 bg-gray-950/70 p-2">
            <h3 className="mb-1 text-[10px] font-semibold uppercase tracking-wide text-gray-500">Historical performance</h3>
            <div className="grid grid-cols-2 gap-2 text-[11px]">
              <div className="rounded bg-gray-900/80 p-2">
                <p className="text-gray-500">Sample</p>
                <p className="font-mono text-gray-200">{similar!.sample_size}</p>
              </div>
              <div className="rounded bg-gray-900/80 p-2">
                <p className="text-gray-500">Win Rate</p>
                <p className="font-mono text-gray-200">{toPct(similar!.win_rate)}</p>
              </div>
              <div className="rounded bg-gray-900/80 p-2">
                <p className="text-gray-500">Avg Return</p>
                <p className="font-mono text-gray-200">{toPct(similar!.avg_return)}</p>
              </div>
              <div className="rounded bg-gray-900/80 p-2">
                <p className="text-gray-500">Profit Factor</p>
                <p className="font-mono text-gray-200">{similar!.profit_factor?.toFixed(2) ?? 'N/A'}</p>
              </div>
            </div>
          </article>
        )}
      </div>
    </section>
    </div>
  )
}

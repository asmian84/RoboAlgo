import { useQuery } from '@tanstack/react-query'
import api from '../../api/client'

// ── Types ─────────────────────────────────────────────────────────────────────

interface Scenario {
  label:        string
  probability:  number
  price_target: number
  direction:    'bullish' | 'bearish' | 'neutral'
  description:  string
}

interface SignalExplanation {
  symbol:             string
  market_state:       string
  setup_type:         string
  setup_quality_score:number | null
  evidence:           string[]
  risk_factors:       string[]
  scenario_map:       Scenario[]
  error?:             string
  computed_at:        string
}

interface SimilarSetups {
  found:        boolean
  setup_type:   string
  sample_size:  number
  win_rate:     number
  avg_return:   number
  max_drawdown: number
  profit_factor:number
  message?:     string
  error?:       string
}

// ── Color helpers ─────────────────────────────────────────────────────────────

const DIR_COLORS: Record<string, string> = {
  bullish: '#22c55e',
  bearish: '#ef4444',
  neutral: '#eab308',
}

function qualColor(s: number | null): string {
  if (s == null)  return '#6b7280'
  if (s >= 70)    return '#22c55e'
  if (s >= 50)    return '#eab308'
  return '#ef4444'
}

// ── Sub-components ─────────────────────────────────────────────────────────

function ScenarioMap({ scenarios }: { scenarios: Scenario[] }) {
  const total = scenarios.reduce((a, s) => a + s.probability, 0)

  return (
    <div>
      <p className="text-xs font-bold text-gray-400 uppercase tracking-wider mb-2">Scenario Map</p>
      <div className="space-y-2">
        {scenarios.map(s => {
          const pct   = total > 0 ? s.probability / total : 0
          const color = DIR_COLORS[s.direction] ?? '#6b7280'
          return (
            <div key={s.label} className="space-y-1">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <span
                    className="text-[10px] font-bold px-1.5 py-0.5 rounded"
                    style={{ color, backgroundColor: color + '20' }}
                  >
                    {s.direction === 'bullish' ? '▲' : s.direction === 'bearish' ? '▼' : '◆'}
                    {' '}{s.label}
                  </span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-xs font-mono font-bold" style={{ color }}>
                    ${s.price_target.toFixed(2)}
                  </span>
                  <span className="text-xs font-bold text-gray-300">
                    {(pct * 100).toFixed(0)}%
                  </span>
                </div>
              </div>
              <div className="h-1.5 bg-gray-800 rounded-full overflow-hidden">
                <div
                  className="h-full rounded-full transition-all"
                  style={{ width: `${pct * 100}%`, backgroundColor: color }}
                />
              </div>
              <p className="text-[10px] text-gray-600">{s.description}</p>
            </div>
          )
        })}
      </div>
    </div>
  )
}

function HistoricalStats({ data }: { data: SimilarSetups }) {
  if (!data.found) {
    return (
      <p className="text-xs text-gray-600 italic">
        {data.message ?? 'No historical data for this setup type.'}
      </p>
    )
  }

  const metrics = [
    { label: 'Sample Size',   value: String(data.sample_size),              color: '#9ca3af' },
    { label: 'Win Rate',      value: `${(data.win_rate * 100).toFixed(0)}%`, color: data.win_rate >= 0.5 ? '#22c55e' : '#ef4444' },
    { label: 'Avg Return',    value: `${(data.avg_return * 100).toFixed(2)}%`, color: data.avg_return >= 0 ? '#22c55e' : '#ef4444' },
    { label: 'Profit Factor', value: data.profit_factor.toFixed(2),          color: data.profit_factor >= 1.5 ? '#22c55e' : data.profit_factor >= 1 ? '#eab308' : '#ef4444' },
    { label: 'Max Drawdown',  value: `${(data.max_drawdown * 100).toFixed(1)}%`, color: '#ef4444' },
  ]

  return (
    <div>
      <p className="text-xs font-bold text-gray-400 uppercase tracking-wider mb-2">
        Historical Performance · {data.setup_type}
      </p>
      <div className="grid grid-cols-3 gap-2">
        {metrics.map(m => (
          <div key={m.label} className="bg-gray-800/40 rounded-lg p-2.5 text-center">
            <p className="text-[10px] text-gray-500">{m.label}</p>
            <p className="text-sm font-bold font-mono mt-0.5" style={{ color: m.color }}>{m.value}</p>
          </div>
        ))}
      </div>
    </div>
  )
}

// ── Main component ─────────────────────────────────────────────────────────────

interface Props {
  symbol: string
}

export default function TradeCoachPanel({ symbol }: Props) {
  const { data: explanation, isLoading: expLoading } = useQuery<SignalExplanation>({
    queryKey: ['trade-coach', 'signal', symbol],
    queryFn:  () => api.get(`/trade-coach/signal/${symbol}`).then(r => r.data),
    enabled:  !!symbol,
    staleTime: 5 * 60_000,
  })

  const { data: similar, isLoading: simLoading } = useQuery<SimilarSetups>({
    queryKey: ['trade-coach', 'similar', symbol],
    queryFn:  () => api.get(`/trade-coach/similar/${symbol}`).then(r => r.data),
    enabled:  !!symbol,
    staleTime: 5 * 60_000,
  })

  const isLoading = expLoading || simLoading
  const qColor    = qualColor(explanation?.setup_quality_score ?? null)

  return (
    <div className="bg-gray-900 border border-gray-800 rounded-xl flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-gray-800">
        <div>
          <h2 className="text-sm font-semibold text-gray-100 flex items-center gap-2">
            🤖 Trade Coach
            <span className="text-gray-500 font-normal text-xs">·</span>
            <span className="font-mono text-emerald-400">{symbol}</span>
          </h2>
          <p className="text-xs text-gray-500 mt-0.5">Signal explanation · Scenarios · Historical performance</p>
        </div>
        {explanation?.setup_quality_score != null && (
          <div
            className="flex flex-col items-center px-3 py-1 rounded-lg border"
            style={{ borderColor: qColor + '40', backgroundColor: qColor + '10' }}
          >
            <span className="text-lg font-bold font-mono" style={{ color: qColor }}>
              {explanation.setup_quality_score.toFixed(0)}
            </span>
            <span className="text-[10px]" style={{ color: qColor }}>Quality</span>
          </div>
        )}
      </div>

      {isLoading && (
        <div className="flex items-center justify-center h-32 text-xs text-gray-600 animate-pulse">
          Analysing {symbol}…
        </div>
      )}

      {!isLoading && explanation && (
        <div className="p-4 space-y-5">
          {/* Market state + setup type */}
          <div className="flex items-center gap-3 flex-wrap">
            <div>
              <p className="text-[10px] text-gray-500 uppercase">Market State</p>
              <p className="text-sm font-bold text-gray-200">{explanation.market_state}</p>
            </div>
            <div>
              <p className="text-[10px] text-gray-500 uppercase">Setup Type</p>
              <p className="text-sm font-bold text-gray-200">{explanation.setup_type || '—'}</p>
            </div>
          </div>

          {/* Evidence */}
          {explanation.evidence.length > 0 && (
            <div>
              <p className="text-xs font-bold text-gray-400 uppercase tracking-wider mb-2">Evidence</p>
              <ul className="space-y-1.5">
                {explanation.evidence.map((e, i) => (
                  <li key={i} className="flex items-start gap-2 text-xs">
                    <span className="text-emerald-500 mt-0.5 flex-shrink-0">✓</span>
                    <span className="text-gray-300">{e}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* Risk warnings */}
          {explanation.risk_factors.length > 0 && (
            <div>
              <p className="text-xs font-bold text-gray-400 uppercase tracking-wider mb-2">Risk Warnings</p>
              <div className="space-y-1.5">
                {explanation.risk_factors.map((r, i) => {
                  const isHigh = r.toLowerCase().includes('critical') || r.toLowerCase().includes('safe_mode')
                  const color  = isHigh ? '#ef4444' : '#eab308'
                  return (
                    <div
                      key={i}
                      className="flex items-start gap-2 px-3 py-2 rounded-lg text-xs border-l-2"
                      style={{ backgroundColor: color + '10', borderColor: color }}
                    >
                      <span style={{ color }}>⚠</span>
                      <span className="text-gray-300">{r}</span>
                    </div>
                  )
                })}
              </div>
            </div>
          )}

          {/* Scenario map */}
          {explanation.scenario_map.length > 0 && (
            <ScenarioMap scenarios={explanation.scenario_map} />
          )}

          {/* Historical stats */}
          {similar && <HistoricalStats data={similar} />}
        </div>
      )}

      {!isLoading && explanation?.error && (
        <div className="p-4 text-xs text-red-400">
          {explanation.error}
        </div>
      )}
    </div>
  )
}

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import api from '../api/client'
import TradeReplayPlayer from '../components/chart/TradeReplayPlayer'

// ── Types ─────────────────────────────────────────────────────────────────────

interface StrategyStats {
  setup_type:    string
  fitness_score: number
  status:        'strong' | 'acceptable' | 'weak' | 'disabled'
  trade_count:   number
  return_count:  number
  win_rate:      number | null
  avg_return:    number | null
  profit_factor: number | null
  max_drawdown:  number | null
  sharpe_ratio:  number | null
}

interface EvolutionSuggestion {
  setup_type:    string
  fitness_score: number
  status:        string
  suggestions:   string[]
}

interface EvolutionReport {
  system_fitness:         number | null
  total_strategies:       number
  underperforming_count:  number
  strategies:             StrategyStats[]
  suggestions:            EvolutionSuggestion[]
  safety_note:            string
  generated_at:           string
}

// ── Color helpers ─────────────────────────────────────────────────────────────

const STATUS_COLORS: Record<string, string> = {
  strong:     '#22c55e',
  acceptable: '#eab308',
  weak:       '#f97316',
  disabled:   '#ef4444',
}

function fitnessColor(f: number): string {
  if (f >= 80) return '#22c55e'
  if (f >= 60) return '#eab308'
  if (f >= 40) return '#f97316'
  return '#ef4444'
}

function pctFmt(v: number | null, dp = 1): string {
  if (v == null) return '—'
  return `${(v * 100).toFixed(dp)}%`
}

function numFmt(v: number | null, dp = 2): string {
  if (v == null) return '—'
  return v.toFixed(dp)
}

// ── Strategy fitness table ────────────────────────────────────────────────────

function FitnessBar({ score }: { score: number }) {
  const color = fitnessColor(score)
  return (
    <div className="flex items-center gap-2">
      <div className="flex-1 h-2 bg-gray-800 rounded-full overflow-hidden">
        <div className="h-full rounded-full" style={{ width: `${score}%`, backgroundColor: color }} />
      </div>
      <span className="text-xs font-mono w-8 text-right" style={{ color }}>{score.toFixed(0)}</span>
    </div>
  )
}

function StrategyRow({ s }: { s: StrategyStats }) {
  const statusColor = STATUS_COLORS[s.status] ?? '#6b7280'

  return (
    <tr className="border-b border-gray-800/40 hover:bg-gray-800/20">
      <td className="px-4 py-3">
        <div>
          <p className="text-sm font-semibold text-gray-200">{s.setup_type?.replace(/_/g, ' ')}</p>
          <span
            className="text-[10px] font-bold px-1.5 py-0.5 rounded uppercase"
            style={{ color: statusColor, backgroundColor: statusColor + '20' }}
          >
            {s.status}
          </span>
        </div>
      </td>
      <td className="px-4 py-3 w-40">
        <FitnessBar score={s.fitness_score} />
      </td>
      <td className="px-4 py-3 text-right font-mono text-gray-300">
        {s.trade_count ?? 0}
      </td>
      <td className="px-4 py-3 text-right font-mono">
        {s.win_rate != null ? (
          <span className={s.win_rate >= 0.5 ? 'text-emerald-400' : 'text-red-400'}>
            {(s.win_rate * 100).toFixed(0)}%
          </span>
        ) : '—'}
      </td>
      <td className="px-4 py-3 text-right font-mono">
        {s.avg_return != null ? (
          <span className={s.avg_return >= 0 ? 'text-emerald-400' : 'text-red-400'}>
            {s.avg_return >= 0 ? '+' : ''}{(s.avg_return * 100).toFixed(2)}%
          </span>
        ) : '—'}
      </td>
      <td className="px-4 py-3 text-right font-mono text-gray-400">
        {numFmt(s.profit_factor, 2)}
      </td>
      <td className="px-4 py-3 text-right font-mono text-red-400">
        {pctFmt(s.max_drawdown)}
      </td>
      <td className="px-4 py-3 text-right font-mono text-gray-400">
        {numFmt(s.sharpe_ratio, 2)}
      </td>
    </tr>
  )
}

// ── Suggestions panel ─────────────────────────────────────────────────────────

function SuggestionsPanel({ suggestions }: { suggestions: EvolutionSuggestion[] }) {
  return (
    <div className="bg-gray-900 border border-gray-800 rounded-xl">
      <div className="px-4 py-3 border-b border-gray-800">
        <h2 className="text-sm font-semibold text-gray-100">Optimization Suggestions</h2>
        <p className="text-xs text-gray-500 mt-0.5">Deterministic analysis — no automatic changes</p>
      </div>
      <div className="divide-y divide-gray-800/50">
        {suggestions.length === 0 ? (
          <div className="px-4 py-8 text-center text-xs text-gray-600">
            No suggestions — run more trades to generate analysis
          </div>
        ) : (
          suggestions.map(s => {
            const color = STATUS_COLORS[s.status] ?? '#6b7280'
            return (
              <div key={s.setup_type} className="px-4 py-3">
                <div className="flex items-center gap-2 mb-2">
                  <span className="text-xs font-semibold text-gray-200">
                    {s.setup_type.replace(/_/g, ' ')}
                  </span>
                  <span
                    className="text-[10px] font-bold px-1.5 py-0.5 rounded"
                    style={{ color, backgroundColor: color + '20' }}
                  >
                    {s.fitness_score?.toFixed(0) ?? '—'} · {s.status}
                  </span>
                </div>
                <ul className="space-y-1">
                  {s.suggestions.map((tip, i) => (
                    <li key={i} className="flex items-start gap-2 text-xs">
                      <span className="text-amber-500 flex-shrink-0 mt-0.5">→</span>
                      <span className="text-gray-400">{tip}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )
          })
        )}
      </div>
    </div>
  )
}

// ── Page ─────────────────────────────────────────────────────────────────────

export default function EvolutionPage() {
  const queryClient = useQueryClient()

  const { data, isLoading, isError, refetch, isFetching } = useQuery<EvolutionReport>({
    queryKey: ['evolution', 'report'],
    queryFn:  () => api.get('/evolution/report').then(r => r.data),
    staleTime: 5 * 60_000,
  })

  const sysColor   = data?.system_fitness != null ? fitnessColor(data.system_fitness) : '#6b7280'
  const sysLabel   = data?.system_fitness != null
    ? data.system_fitness >= 80 ? 'Strong'
    : data.system_fitness >= 60 ? 'Acceptable'
    : data.system_fitness >= 40 ? 'Weak' : 'At Risk'
    : '—'

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-start justify-between gap-4">
        <div>
          <h1 className="text-xl font-bold text-gray-100 flex items-center gap-2">
            ⚙ Strategy Evolution
          </h1>
          <p className="text-xs text-gray-500 mt-0.5">
            Deterministic performance analysis · Statistical optimization suggestions
          </p>
        </div>
        <button
          onClick={() => refetch()}
          disabled={isFetching}
          className="px-3 py-1.5 text-xs font-medium bg-gray-800 hover:bg-gray-700 text-gray-300 rounded-lg border border-gray-700 disabled:opacity-50 transition-colors"
        >
          {isFetching ? '↻ Loading…' : '↻ Refresh'}
        </button>
      </div>

      {isError && (
        <div className="bg-red-900/20 border border-red-900 rounded-xl p-4 text-red-400 text-sm">
          Failed to load evolution report. Ensure the API is running and trades exist.
        </div>
      )}

      {data && (
        <>
          {/* System summary */}
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
            <div className="bg-gray-900 border border-gray-800 rounded-xl p-4 flex flex-col items-center">
              <p className="text-xs text-gray-500 mb-1">System Fitness</p>
              <p className="text-3xl font-bold font-mono" style={{ color: sysColor }}>
                {data.system_fitness?.toFixed(0) ?? '—'}
              </p>
              <p className="text-xs font-semibold mt-1" style={{ color: sysColor }}>{sysLabel}</p>
            </div>
            <div className="bg-gray-900 border border-gray-800 rounded-xl p-4 text-center">
              <p className="text-xs text-gray-500 mb-1">Total Strategies</p>
              <p className="text-2xl font-bold font-mono text-gray-200">{data.total_strategies}</p>
            </div>
            <div className="bg-gray-900 border border-gray-800 rounded-xl p-4 text-center">
              <p className="text-xs text-gray-500 mb-1">Underperforming</p>
              <p className="text-2xl font-bold font-mono" style={{ color: data.underperforming_count > 0 ? '#f97316' : '#22c55e' }}>
                {data.underperforming_count}
              </p>
            </div>
            <div className="bg-gray-900 border border-gray-800 rounded-xl p-4 flex items-center justify-center">
              <div className="text-center">
                <p className="text-xs text-gray-600 italic max-w-[160px] leading-relaxed">{data.safety_note}</p>
              </div>
            </div>
          </div>

          {/* Strategy fitness table */}
          <div className="bg-gray-900 border border-gray-800 rounded-xl overflow-hidden">
            <div className="px-4 py-3 border-b border-gray-800">
              <h2 className="text-sm font-semibold text-gray-100">Strategy Fitness Table</h2>
            </div>
            {data.strategies.length === 0 ? (
              <div className="flex items-center justify-center h-24 text-xs text-gray-600">
                No closed trades yet — trade more to generate evolution data
              </div>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full text-xs">
                  <thead>
                    <tr className="border-b border-gray-800 text-gray-500 uppercase">
                      <th className="text-left px-4 py-2 font-medium">Strategy</th>
                      <th className="px-4 py-2 font-medium">Fitness</th>
                      <th className="text-right px-4 py-2 font-medium">Trades</th>
                      <th className="text-right px-4 py-2 font-medium">Win%</th>
                      <th className="text-right px-4 py-2 font-medium">Avg Return</th>
                      <th className="text-right px-4 py-2 font-medium">Prof Factor</th>
                      <th className="text-right px-4 py-2 font-medium">Max DD</th>
                      <th className="text-right px-4 py-2 font-medium">Sharpe</th>
                    </tr>
                  </thead>
                  <tbody>
                    {data.strategies.map(s => <StrategyRow key={s.setup_type} s={s} />)}
                  </tbody>
                </table>
              </div>
            )}
          </div>

          {/* Optimization suggestions */}
          <SuggestionsPanel suggestions={data.suggestions} />

          {/* Trade Replay Player */}
          <TradeReplayPlayer className="h-[520px]" />
        </>
      )}

      {isLoading && !data && (
        <div className="space-y-3">
          {[...Array(3)].map((_, i) => (
            <div key={i} className="bg-gray-900 border border-gray-800 rounded-xl h-32 animate-pulse" />
          ))}
        </div>
      )}
    </div>
  )
}

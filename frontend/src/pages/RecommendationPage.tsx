import { useState } from 'react'
import { useRecommendation, useInstruments } from '../api/hooks'
import type { MTFEntry } from '../types'

const TF_ORDER = ['15m', '30m', '1h', '2h', '4h', '1d', '1w', '1M']

function ConfidenceBar({ value, signal }: { value: number | null; signal: string }) {
  if (value === null) return <span className="text-gray-600 text-xs">N/A</span>
  const color =
    signal === 'bullish' ? '#22c55e' :
    signal === 'bearish' ? '#ef4444' : '#eab308'
  return (
    <div className="flex items-center gap-2">
      <div className="w-28 h-2 bg-gray-700 rounded-full overflow-hidden">
        <div className="h-full rounded-full transition-all" style={{ width: `${value}%`, backgroundColor: color }} />
      </div>
      <span className="text-xs font-mono" style={{ color }}>{value.toFixed(1)}%</span>
    </div>
  )
}

function SignalBadge({ signal }: { signal: string }) {
  const cfg = {
    bullish: { label: '▲ BULLISH', color: '#22c55e', bg: '#14532d' },
    bearish: { label: '▼ BEARISH', color: '#ef4444', bg: '#450a0a' },
    neutral: { label: '● NEUTRAL', color: '#eab308', bg: '#422006' },
    no_data: { label: '— N/A',    color: '#6b7280', bg: '#1f2937' },
  }[signal] ?? { label: signal, color: '#6b7280', bg: '#1f2937' }
  return (
    <span className="text-xs font-bold px-2 py-0.5 rounded" style={{ color: cfg.color, backgroundColor: cfg.bg }}>
      {cfg.label}
    </span>
  )
}

function MTFTable({ timeframes }: { timeframes: MTFEntry[] }) {
  const ordered = TF_ORDER.map(tf => timeframes.find(t => t.timeframe === tf)).filter(Boolean) as MTFEntry[]
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="text-gray-500 text-xs border-b border-gray-800">
            <th className="text-left py-2 pr-4 font-medium">Timeframe</th>
            <th className="text-left py-2 pr-4 font-medium">Signal</th>
            <th className="text-left py-2 pr-8 font-medium">Confidence</th>
            <th className="text-right py-2 pr-4 font-mono font-medium">RSI</th>
            <th className="text-right py-2 pr-4 font-mono font-medium">MACD Hist</th>
            <th className="text-right py-2 pr-4 font-mono font-medium">BB Pos</th>
            <th className="text-right py-2 font-mono font-medium">MA %</th>
          </tr>
        </thead>
        <tbody>
          {ordered.map(tf => (
            <tr key={tf.timeframe} className="border-b border-gray-800/50 hover:bg-gray-800/30">
              <td className="py-2 pr-4 font-mono font-bold text-gray-300">{tf.timeframe}</td>
              <td className="py-2 pr-4"><SignalBadge signal={tf.signal} /></td>
              <td className="py-2 pr-8"><ConfidenceBar value={tf.confidence} signal={tf.signal} /></td>
              <td className="py-2 pr-4 text-right font-mono text-xs text-gray-400">
                {tf.details?.rsi != null ? (
                  <span style={{ color: tf.details.rsi < 30 ? '#22c55e' : tf.details.rsi > 70 ? '#ef4444' : '#9ca3af' }}>
                    {tf.details.rsi.toFixed(1)}
                  </span>
                ) : '—'}
              </td>
              <td className="py-2 pr-4 text-right font-mono text-xs">
                {tf.details?.macd_hist != null ? (
                  <span style={{ color: tf.details.macd_hist > 0 ? '#22c55e' : '#ef4444' }}>
                    {tf.details.macd_hist > 0 ? '+' : ''}{tf.details.macd_hist.toFixed(4)}
                  </span>
                ) : '—'}
              </td>
              <td className="py-2 pr-4 text-right font-mono text-xs text-gray-400">
                {tf.details?.bb_position != null ? tf.details.bb_position.toFixed(2) : '—'}
              </td>
              <td className="py-2 text-right font-mono text-xs text-gray-400">
                {tf.details?.above_ma != null ? `${(tf.details.above_ma * 100).toFixed(0)}%` : '—'}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function TradePlanCard({ plan }: { plan: NonNullable<import('../types').Recommendation['trade_plan']> }) {
  const tierColor = plan.confidence_tier === 'HIGH' ? '#22c55e' : plan.confidence_tier === 'MEDIUM' ? '#eab308' : '#f97316'
  return (
    <div className="bg-gray-800/60 rounded-lg p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-semibold text-gray-300">3-Tier Trade Plan</h3>
        <div className="flex items-center gap-2">
          <span className="text-xs text-gray-500">{plan.date}</span>
          <span className="text-xs font-bold px-2 py-0.5 rounded" style={{ color: tierColor, backgroundColor: tierColor + '20' }}>
            {plan.confidence_tier}
          </span>
        </div>
      </div>
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        {[
          { label: 'BUY (Entry)', value: plan.buy_price, color: '#60a5fa' },
          { label: 'ACCUMULATE', value: plan.accumulate_price, color: '#a78bfa' },
          { label: 'SCALE (T2)', value: plan.scale_price, color: '#34d399' },
          { label: 'TARGET (T3)', value: plan.sell_price, color: '#fbbf24' },
        ].map(({ label, value, color }) => (
          <div key={label} className="bg-gray-900/60 rounded p-3 text-center">
            <div className="text-xs text-gray-500 mb-1">{label}</div>
            <div className="font-mono font-bold text-base" style={{ color }}>
              {value != null ? `$${value.toFixed(2)}` : '—'}
            </div>
          </div>
        ))}
      </div>
      <p className="text-xs text-gray-600 mt-2">Phase: {plan.market_phase}</p>
    </div>
  )
}

// ── Behavioral Signal Panel ───────────────────────────────────────────────────
function BehavioralPanel({ behavioral }: { behavioral: import('../types').Recommendation['behavioral'] }) {
  const SIGNAL_ICONS: Record<string, string> = {
    FEAR_CAPITULATION: '🔥',
    MILD_FEAR:         '⚠',
    NEUTRAL:           '○',
    MILD_GREED:        '△',
    GREED_FOMO:        '🚨',
  }

  const strengthPct = behavioral.strength

  return (
    <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
      <h3 className="text-sm font-semibold text-gray-300 mb-4">Behavioral Sentiment</h3>
      <div className="flex items-start gap-4 flex-wrap">
        {/* Signal pill */}
        <div
          className="px-4 py-2 rounded-full text-sm font-black tracking-wide border"
          style={{ color: behavioral.color, borderColor: behavioral.color + '66', backgroundColor: behavioral.color + '18' }}
        >
          {SIGNAL_ICONS[behavioral.signal] ?? '●'} {behavioral.label}
        </div>

        {/* Strength bar */}
        <div className="flex-1 min-w-[160px]">
          <div className="flex justify-between text-[10px] text-gray-500 mb-1">
            <span>Strength</span>
            <span style={{ color: behavioral.color }}>{strengthPct}%</span>
          </div>
          <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
            <div className="h-full rounded-full" style={{ width: `${strengthPct}%`, backgroundColor: behavioral.color }} />
          </div>
        </div>
      </div>

      <p className="text-xs text-gray-400 mt-3 leading-relaxed">{behavioral.description}</p>

      <div
        className="mt-3 inline-block px-3 py-1.5 rounded text-xs font-bold tracking-wide"
        style={{ color: behavioral.color, backgroundColor: behavioral.color + '18', border: `1px solid ${behavioral.color}44` }}
      >
        ACTION: {behavioral.action}
      </div>
    </div>
  )
}

// ── News Sentiment + Earnings Risk ────────────────────────────────────────────
function NewsSentimentCard({ sentiment }: { sentiment: import('../types').Recommendation['news_sentiment'] }) {
  const noData = sentiment.score == null
  return (
    <div className="bg-gray-900 border border-gray-800 rounded-xl p-4">
      <div className="text-xs text-gray-500 uppercase tracking-wider mb-2">News Sentiment</div>
      <div className="flex items-center gap-3">
        <div className="text-2xl font-black" style={{ color: noData ? '#6b7280' : sentiment.color }}>
          {noData ? 'N/A' : sentiment.score!.toFixed(3)}
        </div>
        <div>
          <div className="text-xs font-bold" style={{ color: sentiment.color }}>{sentiment.label}</div>
          {!noData && (
            <div className="text-[10px] text-gray-600">{sentiment.article_count} articles</div>
          )}
        </div>
      </div>
      {!noData && (
        <div className="mt-2 h-1.5 bg-gray-800 rounded-full overflow-hidden">
          {/* -1 to +1 normalized to 0-100% */}
          <div className="h-full rounded-full" style={{
            width: `${((sentiment.score! + 1) / 2) * 100}%`,
            backgroundColor: sentiment.color,
          }} />
        </div>
      )}
    </div>
  )
}

function IndexCorrelationCard({ corr }: { corr: import('../types').Recommendation['index_correlation'] }) {
  const noData = corr.correlation == null
  return (
    <div className="bg-gray-900 border border-gray-800 rounded-xl p-4">
      <div className="text-xs text-gray-500 uppercase tracking-wider mb-2">Index Correlation</div>
      <div className="flex items-center gap-3">
        <div className="text-2xl font-black" style={{ color: noData ? '#6b7280' : corr.color }}>
          {noData ? 'N/A' : corr.correlation!.toFixed(2)}
        </div>
        <div>
          <div className="text-xs font-bold" style={{ color: corr.color }}>{corr.label}</div>
          {corr.benchmark && (
            <div className="text-[10px] text-gray-600">vs {corr.benchmark} (20d)</div>
          )}
        </div>
      </div>
      {corr.description && (
        <p className="text-[10px] text-gray-600 mt-2 leading-relaxed">{corr.description}</p>
      )}
    </div>
  )
}

function EarningsRiskCard({ earnings }: { earnings: import('../types').Recommendation['earnings_risk'] }) {
  const hasDate = earnings.earnings_date != null
  const isRisky = earnings.has_risk

  return (
    <div className={`bg-gray-900 border rounded-xl p-4 ${isRisky ? 'border-yellow-700' : 'border-gray-800'}`}>
      <div className="text-xs text-gray-500 uppercase tracking-wider mb-2">Earnings Risk</div>
      {!hasDate ? (
        <div className="text-gray-600 text-sm">No upcoming earnings</div>
      ) : (
        <>
          <div className="text-2xl font-black" style={{ color: isRisky ? '#eab308' : '#9ca3af' }}>
            {earnings.days_until}d
          </div>
          <div className="text-xs mt-1" style={{ color: isRisky ? '#eab308' : '#6b7280' }}>
            {isRisky ? '⚠ Earnings within 5 days' : 'Earnings upcoming'}
          </div>
          <div className="text-[10px] text-gray-600 mt-0.5">{earnings.earnings_date}</div>
        </>
      )}
    </div>
  )
}

// ── Symbol search input ───────────────────────────────────────────────────────
function SymbolSearch({ symbol, onChange }: { symbol: string; onChange: (s: string) => void }) {
  const { data: instruments } = useInstruments()
  const [search, setSearch] = useState('')
  const [focused, setFocused] = useState(false)

  const matches = (instruments ?? []).filter(i =>
    i.symbol.toUpperCase().includes(search.toUpperCase())
  )

  return (
    <div className="relative w-48">
      <input
        type="text"
        placeholder={symbol}
        value={search}
        onChange={e => setSearch(e.target.value)}
        onFocus={() => setFocused(true)}
        onBlur={() => setTimeout(() => setFocused(false), 150)}
        className="w-full bg-gray-800 border border-gray-700 rounded px-3 py-2 text-sm text-white placeholder-gray-400 focus:outline-none focus:border-emerald-600"
      />
      {focused && search && matches.length > 0 && (
        <div className="absolute z-20 w-full mt-1 bg-gray-800 border border-gray-700 rounded shadow-xl max-h-56 overflow-y-auto">
          {matches.slice(0, 30).map(i => (
            <button
              key={i.symbol}
              onMouseDown={() => { onChange(i.symbol); setSearch('') }}
              className={`w-full text-left px-3 py-2 text-sm hover:bg-gray-700 ${i.symbol === symbol ? 'text-emerald-400' : 'text-gray-300'}`}
            >
              <span className="font-bold font-mono">{i.symbol}</span>
              {i.name && <span className="text-gray-500 ml-2 text-xs truncate">{i.name}</span>}
            </button>
          ))}
        </div>
      )}
    </div>
  )
}

// ── Main page ─────────────────────────────────────────────────────────────────
export default function RecommendationPage() {
  const [symbol, setSymbol] = useState('TQQQ')
  const { data, isLoading, error } = useRecommendation(symbol)

  const bullishCount = data?.mtf_timeframes.filter(t => t.signal === 'bullish').length ?? 0
  const totalValid = data?.mtf_timeframes.filter(t => t.signal !== 'no_data').length ?? 0

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between flex-wrap gap-3">
        <div>
          <h2 className="text-xl font-bold text-white">Trade Recommendation</h2>
          <p className="text-sm text-gray-500 mt-0.5">Multi-timeframe confluence + XGBoost + patterns + news</p>
        </div>
        <SymbolSearch symbol={symbol} onChange={setSymbol} />
      </div>

      {isLoading && (
        <div className="text-center py-16 text-gray-500">
          <div className="text-2xl mb-2">Analyzing {symbol}...</div>
          <div className="text-sm">Downloading 8 timeframes from market data</div>
        </div>
      )}

      {error && (
        <div className="bg-red-900/30 border border-red-700 rounded p-4 text-red-400 text-sm">
          Failed to load recommendation for {symbol}
        </div>
      )}

      {data && !isLoading && (
        <>
          {/* Earnings risk warning banner */}
          {data.earnings_risk?.has_risk && (
            <div className="bg-yellow-900/30 border border-yellow-700 rounded-lg px-4 py-3 flex items-center gap-3">
              <span className="text-yellow-400 text-lg">⚠</span>
              <div>
                <span className="text-yellow-300 font-bold text-sm">Earnings Risk</span>
                <span className="text-yellow-600 text-sm ml-2">
                  {data.symbol} reports in {data.earnings_risk.days_until} days ({data.earnings_risk.earnings_date})
                  — consider smaller position or wait until after earnings.
                </span>
              </div>
            </div>
          )}

          {/* Overall recommendation */}
          <div className="bg-gray-900 border border-gray-800 rounded-xl p-6 flex flex-col sm:flex-row items-center gap-6">
            <div className="text-center sm:text-left">
              <div className="text-4xl font-black tracking-wide" style={{ color: data.recommendation_color }}>
                {data.recommendation}
              </div>
              <div className="text-gray-400 text-sm mt-1">
                {data.symbol} <span className="text-gray-600">/ underlying: {data.underlying}</span>
              </div>
            </div>
            <div className="flex-1 grid grid-cols-2 gap-3 sm:grid-cols-4">
              <div className="bg-gray-800 rounded-lg p-3 text-center">
                <div className="text-2xl font-bold text-white">{data.overall_score.toFixed(1)}</div>
                <div className="text-xs text-gray-500 mt-0.5">Overall Score</div>
              </div>
              <div className="bg-gray-800 rounded-lg p-3 text-center">
                <div className="text-2xl font-bold" style={{ color: data.components.mtf_weighted_avg >= 60 ? '#22c55e' : data.components.mtf_weighted_avg <= 40 ? '#ef4444' : '#eab308' }}>
                  {data.components.mtf_weighted_avg.toFixed(1)}
                </div>
                <div className="text-xs text-gray-500 mt-0.5">MTF Avg</div>
              </div>
              <div className="bg-gray-800 rounded-lg p-3 text-center">
                <div className="text-2xl font-bold text-blue-400">{data.components.xgb_probability.toFixed(1)}%</div>
                <div className="text-xs text-gray-500 mt-0.5">XGBoost Prob</div>
              </div>
              <div className="bg-gray-800 rounded-lg p-3 text-center">
                <div className="text-2xl font-bold text-purple-400">{bullishCount}/{totalValid}</div>
                <div className="text-xs text-gray-500 mt-0.5">TF Alignment</div>
              </div>
            </div>
          </div>

          {/* Behavioral + News + Earnings + Correlation row */}
          <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-4">
            <div className="md:col-span-2 xl:col-span-1">
              <BehavioralPanel behavioral={data.behavioral} />
            </div>
            <NewsSentimentCard sentiment={data.news_sentiment} />
            <EarningsRiskCard earnings={data.earnings_risk} />
            <IndexCorrelationCard corr={data.index_correlation} />
          </div>

          {/* MTF Table */}
          <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
            <h3 className="text-sm font-semibold text-gray-300 mb-4">
              Multi-Timeframe Analysis
              <span className="text-gray-600 font-normal ml-2">· computed on {data.underlying}</span>
            </h3>
            <MTFTable timeframes={data.mtf_timeframes} />
          </div>

          {/* Trade Plan */}
          {data.trade_plan && <TradePlanCard plan={data.trade_plan} />}

          {/* Pattern context */}
          {data.pattern_context.total > 0 && (
            <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-sm font-semibold text-gray-300">Pattern Context (last 14 days)</h3>
                <div className="flex gap-3 text-xs">
                  <span className="text-green-400">▲ {data.pattern_context.bullish} bullish</span>
                  <span className="text-red-400">▼ {data.pattern_context.bearish} bearish</span>
                </div>
              </div>
              <div className="space-y-1">
                {data.pattern_context.recent.map((p, i) => (
                  <div key={i} className="flex items-center gap-3 text-xs py-1 border-b border-gray-800/50">
                    <span className="text-gray-600 w-20 shrink-0">{p.date}</span>
                    <span className="font-medium" style={{ color: p.direction === 'bullish' ? '#22c55e' : p.direction === 'bearish' ? '#ef4444' : '#eab308' }}>
                      {p.direction === 'bullish' ? '▲' : p.direction === 'bearish' ? '▼' : '●'} {p.name}
                    </span>
                    <div className="flex-1 h-1 bg-gray-800 rounded-full overflow-hidden">
                      <div className="h-full rounded-full" style={{
                        width: `${p.strength * 100}%`,
                        backgroundColor: p.direction === 'bullish' ? '#22c55e' : '#ef4444',
                      }} />
                    </div>
                    <span className="text-gray-600">{(p.strength * 100).toFixed(0)}%</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </>
      )}
    </div>
  )
}

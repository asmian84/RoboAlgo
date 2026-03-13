/**
 * RoboAlgo — Market Regime Timeline Page
 *
 * Visualises the full market-state history + trade events for a symbol:
 *   • Regime bands (coloured ReferenceArea behind price / P&L)
 *   • Cumulative P&L line (green above zero, red below)
 *   • Trade entry/exit markers on the P&L line
 *   • Brush component for zoom
 *   • Regime stats summary cards
 *   • Trade table below the chart
 */

import { useState, useMemo } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  ComposedChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceArea,
  ReferenceLine,
  Brush,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
} from 'recharts'
import { useInstruments, useRegimeTimeline } from '../api/hooks'
import type { RegimeTimelineBar, RegimeStatePeriod, RegimeTradeEntry } from '../types'

// ── Colour palette ────────────────────────────────────────────────────────────

const STATE_COLORS: Record<string, string> = {
  EXPANSION:   '#f97316',
  TREND:       '#22c55e',
  COMPRESSION: '#60a5fa',
  CHAOS:       '#ef4444',
  UNKNOWN:     '#6b7280',
}

const STATE_BG: Record<string, string> = {
  EXPANSION:   '#f9731614',
  TREND:       '#22c55e14',
  COMPRESSION: '#60a5fa14',
  CHAOS:       '#ef444414',
  UNKNOWN:     '#6b728014',
}

const TRADE_EVENT_COLORS: Record<string, string> = {
  ENTRY: '#22c55e',
  EXIT:  '#f97316',
  SETUP: '#94a3b8',
  TRIGGER: '#eab308',
}

// ── Small helper components ───────────────────────────────────────────────────

function StateBadge({ state }: { state: string }) {
  const color = STATE_COLORS[state] ?? '#6b7280'
  return (
    <span
      className="inline-block px-2 py-0.5 rounded text-xs font-semibold"
      style={{ color, background: color + '22', border: `1px solid ${color}44` }}
    >
      {state}
    </span>
  )
}

function PnlChip({ value }: { value: number | null }) {
  if (value === null || value === undefined) return <span className="text-gray-600">—</span>
  const pos = value >= 0
  return (
    <span className={pos ? 'text-emerald-400' : 'text-red-400'}>
      {pos ? '+' : ''}${value.toFixed(2)}
    </span>
  )
}

function WinRateBar({ rate }: { rate: number }) {
  const pct = Math.round(rate * 100)
  return (
    <div className="flex items-center gap-2">
      <div className="flex-1 h-1.5 bg-gray-700 rounded-full overflow-hidden">
        <div
          className="h-full rounded-full"
          style={{
            width: `${pct}%`,
            background: pct >= 60 ? '#22c55e' : pct >= 45 ? '#eab308' : '#ef4444',
          }}
        />
      </div>
      <span className="text-xs text-gray-300 w-8 text-right">{pct}%</span>
    </div>
  )
}

// ── Custom Tooltip ────────────────────────────────────────────────────────────

function TimelineTooltip({ active, payload, label }: any) {
  if (!active || !payload?.length) return null
  const d = payload[0]?.payload as RegimeTimelineBar
  if (!d) return null

  return (
    <div className="bg-gray-900 border border-gray-700 rounded p-3 text-xs shadow-xl max-w-xs">
      <p className="text-gray-400 mb-1 font-mono">{d.date}</p>
      <StateBadge state={d.market_state} />
      {d.close_price != null && (
        <p className="mt-1 text-gray-300">Price: <span className="text-white font-mono">${d.close_price.toFixed(2)}</span></p>
      )}
      {d.trend_strength != null && (
        <p className="text-gray-300">Trend str: <span className="text-white">{d.trend_strength}</span></p>
      )}
      <div className="border-t border-gray-700 mt-2 pt-2">
        <p className="text-gray-300">
          Daily P&L: <PnlChip value={d.daily_pnl} />
        </p>
        <p className="text-gray-300">
          Cumulative P&L: <PnlChip value={d.cumulative_pnl} />
        </p>
      </div>
      {d.trade_event && (
        <div className="border-t border-gray-700 mt-2 pt-2">
          <span
            className="inline-block px-2 py-0.5 rounded text-xs font-bold"
            style={{
              color: TRADE_EVENT_COLORS[d.trade_event] ?? '#fff',
              background: (TRADE_EVENT_COLORS[d.trade_event] ?? '#fff') + '22',
            }}
          >
            ● {d.trade_event} #{d.trade_id}
          </span>
          {d.trade_pnl != null && (
            <p className="mt-1 text-gray-300">P&L: <PnlChip value={d.trade_pnl} /></p>
          )}
        </div>
      )}
    </div>
  )
}

// ── Regime stats card ─────────────────────────────────────────────────────────

function RegimeStatsCard({
  state,
  stats,
}: {
  state: string
  stats: { trades: number; wins: number; total_pnl: number; win_rate: number; avg_pnl: number }
}) {
  const color = STATE_COLORS[state] ?? '#6b7280'
  return (
    <div
      className="rounded-lg p-3 border"
      style={{ borderColor: color + '44', background: color + '0d' }}
    >
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs font-bold" style={{ color }}>{state}</span>
        <span className="text-xs text-gray-500">{stats.trades} trades</span>
      </div>
      <WinRateBar rate={stats.win_rate} />
      <div className="mt-2 grid grid-cols-2 gap-1 text-xs">
        <div>
          <p className="text-gray-500">Total P&L</p>
          <p className={stats.total_pnl >= 0 ? 'text-emerald-400' : 'text-red-400'}>
            {stats.total_pnl >= 0 ? '+' : ''}${stats.total_pnl.toFixed(0)}
          </p>
        </div>
        <div>
          <p className="text-gray-500">Avg/trade</p>
          <p className={stats.avg_pnl >= 0 ? 'text-emerald-400' : 'text-red-400'}>
            {stats.avg_pnl >= 0 ? '+' : ''}${stats.avg_pnl.toFixed(0)}
          </p>
        </div>
      </div>
    </div>
  )
}

// ── Preset date ranges ────────────────────────────────────────────────────────

function isoOffset(days: number): string {
  const d = new Date()
  d.setDate(d.getDate() - days)
  return d.toISOString().slice(0, 10)
}

const DATE_PRESETS = [
  { label: '1M',   days: 30  },
  { label: '3M',   days: 90  },
  { label: '6M',   days: 180 },
  { label: '1Y',   days: 365 },
  { label: '2Y',   days: 730 },
]

// ── Main Page ─────────────────────────────────────────────────────────────────

export default function RegimeTimelinePage() {
  const navigate  = useNavigate()
  const { data: instruments } = useInstruments()

  const [symbol,    setSymbol]    = useState('SOXL')
  const [startDate, setStartDate] = useState<string | undefined>(isoOffset(365))
  const [endDate,   setEndDate]   = useState<string | undefined>(undefined)
  const [activePreset, setActivePreset] = useState<string>('1Y')
  const [selectedTrade, setSelectedTrade] = useState<number | null>(null)

  const { data, isLoading, isError, refetch, isFetching } = useRegimeTimeline(
    symbol,
    startDate,
    endDate,
  )

  // Unique symbols for selector
  const symbols = useMemo(
    () => instruments?.map(i => i.symbol).sort() ?? [],
    [instruments],
  )

  // Determine Y-axis domain for P&L chart
  const pnlExtent = useMemo(() => {
    if (!data?.timeline?.length) return [-100, 100]
    const vals = data.timeline.map(d => d.cumulative_pnl)
    const min   = Math.min(...vals)
    const max   = Math.max(...vals)
    const pad   = Math.max(Math.abs(max - min) * 0.1, 50)
    return [Math.floor(min - pad), Math.ceil(max + pad)]
  }, [data])

  function applyPreset(days: number, label: string) {
    setActivePreset(label)
    setStartDate(isoOffset(days))
    setEndDate(undefined)
  }

  function handleSymbolChange(sym: string) {
    setSymbol(sym)
    setSelectedTrade(null)
  }

  // ── Chart data — mark trade events for custom dots ────────────────────────
  const chartData = useMemo(() => data?.timeline ?? [], [data])

  // Entries / Exits for scatter overlay
  const entryPoints = useMemo(
    () => chartData.filter(d => d.trade_event === 'ENTRY'),
    [chartData],
  )
  const exitPoints = useMemo(
    () => chartData.filter(d => d.trade_event === 'EXIT'),
    [chartData],
  )

  // ── Render ────────────────────────────────────────────────────────────────

  return (
    <div className="flex flex-col gap-4 p-4 min-h-0">
      {/* ── Header ── */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
        <div>
          <h1 className="text-xl font-bold text-white flex items-center gap-2">
            <span className="text-emerald-400">◈</span> Market Regime Timeline
          </h1>
          <p className="text-xs text-gray-500 mt-0.5">
            Regime bands · Trade events · Cumulative P&amp;L
          </p>
        </div>

        <div className="flex items-center gap-2 flex-wrap">
          {/* Symbol selector */}
          <select
            className="bg-gray-800 border border-gray-700 text-gray-200 text-sm rounded px-2 py-1.5 focus:outline-none focus:ring-1 focus:ring-emerald-500"
            value={symbol}
            onChange={e => handleSymbolChange(e.target.value)}
          >
            {symbols.length ? (
              symbols.map(s => <option key={s}>{s}</option>)
            ) : (
              <option>{symbol}</option>
            )}
          </select>

          {/* Date presets */}
          <div className="flex gap-1">
            {DATE_PRESETS.map(p => (
              <button
                key={p.label}
                onClick={() => applyPreset(p.days, p.label)}
                className={`text-xs px-2 py-1 rounded transition-colors ${
                  activePreset === p.label
                    ? 'bg-emerald-600 text-white'
                    : 'bg-gray-800 text-gray-400 hover:text-gray-200 hover:bg-gray-700'
                }`}
              >
                {p.label}
              </button>
            ))}
          </div>

          {/* Refresh */}
          <button
            onClick={() => refetch()}
            disabled={isFetching}
            className="text-xs px-3 py-1.5 bg-gray-800 border border-gray-700 text-gray-300 rounded hover:bg-gray-700 disabled:opacity-50 transition-colors"
          >
            {isFetching ? '↻' : '⟳'} Refresh
          </button>
        </div>
      </div>

      {/* ── Error / Loading states ── */}
      {isError && (
        <div className="rounded-lg bg-red-900/20 border border-red-800 p-4 text-sm text-red-300">
          Failed to load regime timeline. Check that the API is running.
        </div>
      )}
      {data?.error && (
        <div className="rounded-lg bg-yellow-900/20 border border-yellow-800 p-4 text-sm text-yellow-300">
          ⚠ {data.error}
        </div>
      )}

      {/* ── Regime stats row ── */}
      {data?.regime_stats && Object.keys(data.regime_stats).length > 0 && (
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
          {Object.entries(data.regime_stats).map(([state, stats]) => (
            <RegimeStatsCard key={state} state={state} stats={stats} />
          ))}
        </div>
      )}

      {/* ── Main chart ── */}
      <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-4 text-xs text-gray-500">
            <span>
              {data?.start_date} → {data?.end_date ?? 'today'}
            </span>
            <span>{chartData.length} days</span>
            {data?.trades.length ? (
              <span>{data.trades.length} completed trades</span>
            ) : null}
          </div>
          {/* Regime legend */}
          <div className="hidden sm:flex items-center gap-3 text-xs">
            {Object.entries(STATE_COLORS).filter(([k]) => k !== 'UNKNOWN').map(([state, color]) => (
              <span key={state} className="flex items-center gap-1">
                <span className="inline-block w-3 h-3 rounded-sm" style={{ background: color + '55', border: `1px solid ${color}` }} />
                <span style={{ color }}>{state}</span>
              </span>
            ))}
          </div>
        </div>

        {isLoading ? (
          <div className="h-64 flex items-center justify-center text-gray-600">
            Loading timeline…
          </div>
        ) : chartData.length === 0 ? (
          <div className="h-64 flex items-center justify-center text-gray-600 text-sm">
            No market state data for {symbol} in the selected window.
          </div>
        ) : (
          <ResponsiveContainer width="100%" height={340}>
            <ComposedChart data={chartData} margin={{ top: 8, right: 16, bottom: 0, left: 8 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />

              <XAxis
                dataKey="date"
                tick={{ fill: '#6b7280', fontSize: 10 }}
                tickLine={false}
                tickFormatter={d => d.slice(5)} /* MM-DD */
                interval="preserveStartEnd"
              />

              <YAxis
                domain={pnlExtent}
                tick={{ fill: '#6b7280', fontSize: 10 }}
                tickLine={false}
                tickFormatter={v => `$${v >= 0 ? '+' : ''}${v}`}
                width={64}
              />

              <Tooltip content={<TimelineTooltip />} />

              {/* ── Regime background bands ── */}
              {data?.state_periods?.map((period, i) => (
                <ReferenceArea
                  key={i}
                  x1={period.start_date}
                  x2={period.end_date}
                  fill={STATE_COLORS[period.state] ?? '#6b7280'}
                  fillOpacity={0.08}
                  strokeOpacity={0}
                />
              ))}

              {/* ── Zero line ── */}
              <ReferenceLine y={0} stroke="#374151" strokeDasharray="4 2" />

              {/* ── Cumulative P&L line ── */}
              <Line
                type="monotone"
                dataKey="cumulative_pnl"
                stroke="#22c55e"
                strokeWidth={2}
                dot={false}
                activeDot={{ r: 4, fill: '#22c55e' }}
                name="Cumulative P&L"
              />

              {/* ── Trade entry markers ── */}
              {entryPoints.map((d, i) => (
                <ReferenceLine
                  key={`entry-${i}`}
                  x={d.date}
                  stroke="#22c55e"
                  strokeWidth={1.5}
                  strokeDasharray="2 2"
                  label={{
                    value: '▲',
                    position: 'top',
                    fill: '#22c55e',
                    fontSize: 10,
                  }}
                />
              ))}

              {/* ── Trade exit markers ── */}
              {exitPoints.map((d, i) => (
                <ReferenceLine
                  key={`exit-${i}`}
                  x={d.date}
                  stroke={d.trade_pnl != null && d.trade_pnl >= 0 ? '#f97316' : '#ef4444'}
                  strokeWidth={1.5}
                  strokeDasharray="2 2"
                  label={{
                    value: '✕',
                    position: 'top',
                    fill: d.trade_pnl != null && d.trade_pnl >= 0 ? '#f97316' : '#ef4444',
                    fontSize: 10,
                  }}
                />
              ))}

              {/* ── Brush / zoom ── */}
              <Brush
                dataKey="date"
                height={24}
                stroke="#374151"
                fill="#111827"
                travellerWidth={6}
                tickFormatter={d => d.slice(5)}
              />
            </ComposedChart>
          </ResponsiveContainer>
        )}
      </div>

      {/* ── Bottom two-col layout ── */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* ── State Periods list ── */}
        <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
          <h2 className="text-sm font-semibold text-gray-300 mb-3">Regime Periods</h2>
          {!data?.state_periods?.length ? (
            <p className="text-gray-600 text-xs">No period data</p>
          ) : (
            <div className="space-y-1 max-h-72 overflow-y-auto pr-1">
              {data.state_periods.map((p, i) => (
                <div
                  key={i}
                  className="flex items-center justify-between text-xs py-1 border-b border-gray-800"
                >
                  <StateBadge state={p.state} />
                  <span className="text-gray-500 font-mono ml-2">
                    {p.start_date.slice(5)} → {p.end_date.slice(5)}
                  </span>
                  <span className="text-gray-600 ml-1">{p.days}d</span>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* ── Trade table ── */}
        <div className="lg:col-span-2 bg-gray-900 border border-gray-800 rounded-lg p-4">
          <h2 className="text-sm font-semibold text-gray-300 mb-3">
            Completed Trades
            {data?.trades.length ? (
              <span className="ml-2 text-xs text-gray-600">({data.trades.length})</span>
            ) : null}
          </h2>

          {!data?.trades?.length ? (
            <p className="text-gray-600 text-xs">No completed trades in this window.</p>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr className="text-gray-600 border-b border-gray-800">
                    <th className="text-left py-1 pr-3">Entry</th>
                    <th className="text-left py-1 pr-3">Exit</th>
                    <th className="text-left py-1 pr-3">Regime</th>
                    <th className="text-left py-1 pr-3">Setup</th>
                    <th className="text-right py-1 pr-3">Entry $</th>
                    <th className="text-right py-1 pr-3">Exit $</th>
                    <th className="text-right py-1 pr-3">P&L</th>
                    <th className="text-right py-1">Ret%</th>
                  </tr>
                </thead>
                <tbody>
                  {data.trades.map(t => {
                    const isSelected = selectedTrade === t.id
                    return (
                      <tr
                        key={t.id}
                        className={`border-b border-gray-800/50 cursor-pointer transition-colors ${
                          isSelected ? 'bg-gray-800' : 'hover:bg-gray-800/40'
                        }`}
                        onClick={() => setSelectedTrade(isSelected ? null : t.id)}
                      >
                        <td className="py-1 pr-3 font-mono text-gray-400">
                          {t.entry_date?.slice(5) ?? '—'}
                        </td>
                        <td className="py-1 pr-3 font-mono text-gray-400">
                          {t.exit_date?.slice(5) ?? '—'}
                        </td>
                        <td className="py-1 pr-3">
                          {t.market_state ? <StateBadge state={t.market_state} /> : <span className="text-gray-600">—</span>}
                        </td>
                        <td className="py-1 pr-3 text-gray-400 truncate max-w-[120px]">
                          {t.setup_type ?? '—'}
                        </td>
                        <td className="py-1 pr-3 text-right font-mono text-gray-300">
                          {t.entry_price != null ? `$${t.entry_price.toFixed(2)}` : '—'}
                        </td>
                        <td className="py-1 pr-3 text-right font-mono text-gray-300">
                          {t.exit_price != null ? `$${t.exit_price.toFixed(2)}` : '—'}
                        </td>
                        <td className="py-1 pr-3 text-right">
                          <PnlChip value={t.pnl} />
                        </td>
                        <td className="py-1 text-right">
                          {t.return_pct != null ? (
                            <span className={t.return_pct >= 0 ? 'text-emerald-400' : 'text-red-400'}>
                              {t.return_pct >= 0 ? '+' : ''}{t.return_pct.toFixed(1)}%
                            </span>
                          ) : (
                            <span className="text-gray-600">—</span>
                          )}
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
                {/* Totals row */}
                {data.trades.length > 0 && (() => {
                  const totalPnl = data.trades.reduce((s, t) => s + (t.pnl ?? 0), 0)
                  const wins     = data.trades.filter(t => (t.pnl ?? 0) > 0).length
                  return (
                    <tfoot>
                      <tr className="border-t border-gray-700 text-gray-400">
                        <td colSpan={6} className="py-1.5 text-gray-500">
                          {wins}/{data.trades.length} wins &nbsp;|&nbsp;
                          {Math.round((wins / data.trades.length) * 100)}% win rate
                        </td>
                        <td className="py-1.5 text-right font-semibold">
                          <PnlChip value={totalPnl} />
                        </td>
                        <td />
                      </tr>
                    </tfoot>
                  )
                })()}
              </table>
            </div>
          )}
        </div>
      </div>

      {/* ── Footer: legend ── */}
      <div className="text-xs text-gray-600 flex flex-wrap gap-4 pb-2">
        <span>▲ <span className="text-emerald-400">ENTRY</span></span>
        <span>✕ <span className="text-orange-400">EXIT (profit)</span></span>
        <span>✕ <span className="text-red-400">EXIT (loss)</span></span>
        <span className="text-gray-500">Drag chart bottom bar to zoom · Click row to highlight trade</span>
      </div>
    </div>
  )
}

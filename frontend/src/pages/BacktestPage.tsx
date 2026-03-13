import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useBacktestStats, useBacktestTicker, useBacktestDrill, useInstruments } from '../api/hooks'
import type { BacktestBucket, BacktestSignalRow, BacktestDrillRow } from '../types'

// ── Shared sub-components ─────────────────────────────────────────────────────
function WinBar({ rate, color }: { rate: number; color: string }) {
  return (
    <div className="flex items-center gap-2">
      <div className="w-24 h-2 bg-gray-700 rounded-full overflow-hidden">
        <div className="h-full rounded-full transition-all" style={{ width: `${rate}%`, backgroundColor: color }} />
      </div>
      <span className="text-sm font-black font-mono" style={{ color }}>{rate.toFixed(1)}%</span>
    </div>
  )
}

function StatCard({ label, value, sub, color }: { label: string; value: string; sub?: string; color?: string }) {
  return (
    <div className="bg-gray-900 border border-gray-800 rounded-xl p-4">
      <div className="text-xs text-gray-500 uppercase tracking-wider mb-1">{label}</div>
      <div className="text-2xl font-black" style={{ color: color || '#f3f4f6' }}>{value}</div>
      {sub && <div className="text-xs text-gray-600 mt-1">{sub}</div>}
    </div>
  )
}

function BucketRow({ label, b, highlight, expanded, onToggle }: {
  label: string; b: BacktestBucket; highlight?: boolean
  expanded: boolean; onToggle: () => void
}) {
  const winColor  = b.win_rate >= 60 ? '#22c55e' : b.win_rate >= 45 ? '#eab308' : '#ef4444'
  const stopColor = b.stop_rate <= 20 ? '#22c55e' : b.stop_rate <= 35 ? '#eab308' : '#ef4444'
  return (
    <tr
      className={`border-b border-gray-800/40 cursor-pointer transition-colors ${
        expanded ? 'bg-gray-800/40' : highlight ? 'bg-emerald-900/10 hover:bg-emerald-900/20' : 'hover:bg-gray-800/20'
      }`}
      onClick={onToggle}
    >
      <td className={`py-3 px-4 font-bold text-sm ${highlight ? 'text-emerald-400' : 'text-gray-300'}`}>
        <span className="flex items-center gap-2">
          <span className="text-gray-600 text-xs">{expanded ? '▼' : '▶'}</span>
          {label}
        </span>
      </td>
      <td className="py-3 px-4 text-right text-gray-400 text-sm font-mono">{b.total.toLocaleString()}</td>
      <td className="py-3 px-4 text-right text-gray-500 text-xs font-mono">{b.open.toLocaleString()}</td>
      <td className="py-3 px-4"><WinBar rate={b.win_rate} color={winColor} /></td>
      <td className="py-3 px-4">
        <span className="text-sm font-bold font-mono" style={{ color: '#34d399' }}>
          {b.target_rate.toFixed(1)}%
        </span>
      </td>
      <td className="py-3 px-4"><WinBar rate={b.stop_rate} color={stopColor} /></td>
      <td className="py-3 px-4 text-right font-mono text-xs text-gray-400">
        {b.avg_t1_return != null ? `+${b.avg_t1_return.toFixed(1)}%` : '—'}
      </td>
      <td className="py-3 px-4 text-right font-mono text-xs text-gray-400">
        {b.avg_tgt_return != null ? `+${b.avg_tgt_return.toFixed(1)}%` : '—'}
      </td>
      <td className="py-3 px-4 text-right font-mono text-xs text-red-400">
        {b.avg_stop_return != null ? `${b.avg_stop_return.toFixed(1)}%` : '—'}
      </td>
    </tr>
  )
}

// ── Drill-down panel (signals in a probability bucket) ────────────────────────
function DrillPanel({ bucket }: { bucket: string }) {
  const navigate = useNavigate()
  const { data, isLoading } = useBacktestDrill(bucket, '', true)

  if (isLoading) {
    return (
      <tr><td colSpan={9} className="py-4 px-6 bg-gray-800/30">
        <div className="text-gray-500 text-xs text-center">Loading {bucket} signals…</div>
      </td></tr>
    )
  }
  if (!data || data.signals.length === 0) {
    return (
      <tr><td colSpan={9} className="py-3 px-6 bg-gray-800/30">
        <div className="text-gray-600 text-xs text-center">No signals found for {bucket}.</div>
      </td></tr>
    )
  }

  return (
    <tr>
      <td colSpan={9} className="px-4 py-3 bg-gray-800/30 border-b border-gray-700">
        <div className="text-xs text-gray-500 mb-2 font-medium">
          Latest signal per symbol in <span className="text-white font-bold">{bucket}</span> · {data.signals.length} instruments
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="text-[10px] text-gray-600 uppercase tracking-wider border-b border-gray-700/50">
                <th className="text-left pb-1.5 pr-4">Symbol</th>
                <th className="text-right pb-1.5 pr-4">Prob</th>
                <th className="text-left pb-1.5 pr-4">Phase</th>
                <th className="text-right pb-1.5 pr-4">Entry</th>
                <th className="text-center pb-1.5 pr-4">Outcome</th>
                <th className="text-right pb-1.5">Return</th>
              </tr>
            </thead>
            <tbody>
              {data.signals.map((sig: BacktestDrillRow) => {
                const ret = sig.outcome === 'target' ? sig.tgt_return :
                            sig.outcome === 't1'     ? sig.t1_return  :
                            sig.outcome === 'stop'   ? sig.stop_return : null
                const retColor = ret == null ? '#6b7280' : ret >= 0 ? '#22c55e' : '#ef4444'
                const probColor = sig.probability >= 95 ? '#22c55e' : sig.probability >= 90 ? '#86efac' : '#eab308'
                const outcomeColor = sig.outcome === 'target' ? '#22c55e' : sig.outcome === 't1' ? '#34d399' : sig.outcome === 'stop' ? '#ef4444' : '#6b7280'
                return (
                  <tr
                    key={sig.symbol}
                    className="border-b border-gray-700/30 hover:bg-gray-700/30 cursor-pointer"
                    onClick={e => { e.stopPropagation(); navigate(`/chart?symbol=${sig.symbol}`) }}
                  >
                    <td className="py-1.5 pr-4 font-black text-white">{sig.symbol}</td>
                    <td className="py-1.5 pr-4 text-right font-mono font-bold" style={{ color: probColor }}>{sig.probability.toFixed(1)}%</td>
                    <td className="py-1.5 pr-4 text-gray-400">{sig.market_phase}</td>
                    <td className="py-1.5 pr-4 text-right font-mono text-gray-300">${sig.buy_price.toFixed(2)}</td>
                    <td className="py-1.5 pr-4 text-center font-bold text-[10px]" style={{ color: outcomeColor }}>
                      {sig.outcome.toUpperCase()}
                    </td>
                    <td className="py-1.5 text-right font-mono font-bold" style={{ color: retColor }}>
                      {ret != null ? `${ret >= 0 ? '+' : ''}${ret.toFixed(1)}%` : '—'}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </td>
    </tr>
  )
}

function PhaseRow({ phase, b }: { phase: string; b: BacktestBucket }) {
  const winColor = b.win_rate >= 60 ? '#22c55e' : b.win_rate >= 45 ? '#eab308' : '#ef4444'
  return (
    <tr className="border-b border-gray-800/40 hover:bg-gray-800/20">
      <td className="py-2.5 px-4 text-sm text-gray-300 font-medium">{phase}</td>
      <td className="py-2.5 px-4 text-right text-gray-500 text-xs font-mono">{b.total.toLocaleString()}</td>
      <td className="py-2.5 px-4"><WinBar rate={b.win_rate} color={winColor} /></td>
      <td className="py-2.5 px-4 text-right font-mono text-sm font-bold text-red-400">{b.stop_rate.toFixed(1)}%</td>
      <td className="py-2.5 px-4 text-right font-mono text-xs text-gray-400">
        {b.avg_t1_return != null ? `+${b.avg_t1_return.toFixed(1)}%` : '—'}
      </td>
    </tr>
  )
}

// ── Outcome badge ─────────────────────────────────────────────────────────────
function OutcomeBadge({ outcome }: { outcome: BacktestSignalRow['outcome'] }) {
  const cfg = {
    target: { label: 'TARGET ✓', color: '#22c55e', bg: '#14532d22' },
    t1:     { label: 'T1 ✓',     color: '#34d399', bg: '#06473522' },
    stop:   { label: 'STOP ✗',   color: '#ef4444', bg: '#45090a22' },
    open:   { label: 'OPEN',     color: '#6b7280', bg: '#1f293722' },
  }[outcome]
  return (
    <span
      className="text-[10px] font-bold px-1.5 py-0.5 rounded font-mono"
      style={{ color: cfg.color, backgroundColor: cfg.bg }}
    >
      {cfg.label}
    </span>
  )
}

// ── Ticker stats header cards ─────────────────────────────────────────────────
function TickerStats({ stats, forward_days }: { stats: BacktestBucket; forward_days: number }) {
  return (
    <div className="grid grid-cols-2 md:grid-cols-3 xl:grid-cols-6 gap-3 mb-4">
      <StatCard label="Signals" value={stats.total.toLocaleString()} sub={`${stats.closed} closed · ${stats.open} open`} />
      <StatCard
        label="T1 Win Rate" value={`${stats.win_rate.toFixed(1)}%`}
        sub={`hit +2 ATR in ${forward_days}d`}
        color={stats.win_rate >= 55 ? '#22c55e' : stats.win_rate >= 40 ? '#eab308' : '#ef4444'}
      />
      <StatCard
        label="Full Target" value={`${stats.target_rate.toFixed(1)}%`}
        sub="hit +4 ATR" color="#34d399"
      />
      <StatCard
        label="Stop Rate" value={`${stats.stop_rate.toFixed(1)}%`}
        sub="stopped out"
        color={stats.stop_rate <= 25 ? '#22c55e' : stats.stop_rate <= 40 ? '#eab308' : '#ef4444'}
      />
      <StatCard
        label="Avg T1 Ret"
        value={stats.avg_t1_return != null ? `+${stats.avg_t1_return.toFixed(1)}%` : '—'}
        color="#22c55e"
      />
      <StatCard
        label="Avg Stop"
        value={stats.avg_stop_return != null ? `${stats.avg_stop_return.toFixed(1)}%` : '—'}
        color="#ef4444"
      />
    </div>
  )
}

// ── Ticker signal history table ───────────────────────────────────────────────
function TickerSignalTable({ signals }: { signals: BacktestSignalRow[] }) {
  const [filter, setFilter] = useState<'all' | 'stop' | 't1' | 'target' | 'open'>('all')

  const visible = filter === 'all' ? signals : signals.filter(s => s.outcome === filter)

  return (
    <div>
      {/* Outcome filter */}
      <div className="flex gap-1 mb-3 flex-wrap">
        {(['all', 'target', 't1', 'stop', 'open'] as const).map(f => (
          <button
            key={f}
            onClick={() => setFilter(f)}
            className={`px-3 py-1 rounded text-xs font-bold uppercase transition-all ${
              filter === f ? 'bg-gray-700 text-white' : 'bg-gray-900 text-gray-500 hover:text-gray-300'
            }`}
          >
            {f === 'all' ? `All (${signals.length})` :
             f === 'target' ? `Target (${signals.filter(s => s.outcome === 'target').length})` :
             f === 't1'     ? `T1 (${signals.filter(s => s.outcome === 't1').length})` :
             f === 'stop'   ? `Stop (${signals.filter(s => s.outcome === 'stop').length})` :
                              `Open (${signals.filter(s => s.outcome === 'open').length})`}
          </button>
        ))}
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr className="border-b border-gray-800 text-gray-600 text-[10px] uppercase tracking-wider">
              <th className="text-left py-2 px-3">Date</th>
              <th className="text-right py-2 px-3">Prob</th>
              <th className="text-center py-2 px-3">Tier</th>
              <th className="text-left py-2 px-3">Phase</th>
              <th className="text-right py-2 px-3">Entry</th>
              <th className="text-right py-2 px-3">Stop</th>
              <th className="text-right py-2 px-3">T1</th>
              <th className="text-right py-2 px-3">Target</th>
              <th className="text-center py-2 px-3">Outcome</th>
              <th className="text-right py-2 px-3">Return</th>
            </tr>
          </thead>
          <tbody>
            {visible.map((sig, i) => {
              const ret = sig.outcome === 'target' ? sig.tgt_return :
                          sig.outcome === 't1'     ? sig.t1_return  :
                          sig.outcome === 'stop'   ? sig.stop_return : null
              const retColor = ret == null ? '#6b7280' : ret >= 0 ? '#22c55e' : '#ef4444'
              const tierColor = sig.confidence_tier === 'HIGH' ? '#22c55e' : sig.confidence_tier === 'MEDIUM' ? '#eab308' : '#f97316'
              const probColor = sig.probability >= 90 ? '#22c55e' : sig.probability >= 80 ? '#eab308' : '#9ca3af'

              return (
                <tr key={i} className="border-b border-gray-800/30 hover:bg-gray-800/20">
                  <td className="py-2 px-3 font-mono text-gray-400">{sig.date}</td>
                  <td className="py-2 px-3 text-right font-mono font-bold" style={{ color: probColor }}>
                    {sig.probability.toFixed(1)}%
                  </td>
                  <td className="py-2 px-3 text-center">
                    <span className="font-bold text-[10px]" style={{ color: tierColor }}>{sig.confidence_tier}</span>
                  </td>
                  <td className="py-2 px-3 text-gray-400">{sig.market_phase}</td>
                  <td className="py-2 px-3 text-right font-mono text-gray-300">${sig.buy_price.toFixed(2)}</td>
                  <td className="py-2 px-3 text-right font-mono text-red-400">
                    {sig.accumulate_price != null ? `$${sig.accumulate_price.toFixed(2)}` : '—'}
                  </td>
                  <td className="py-2 px-3 text-right font-mono text-emerald-400">
                    {sig.scale_price != null ? `$${sig.scale_price.toFixed(2)}` : '—'}
                  </td>
                  <td className="py-2 px-3 text-right font-mono text-yellow-400">
                    {sig.sell_price != null ? `$${sig.sell_price.toFixed(2)}` : '—'}
                  </td>
                  <td className="py-2 px-3 text-center"><OutcomeBadge outcome={sig.outcome} /></td>
                  <td className="py-2 px-3 text-right font-mono font-bold" style={{ color: retColor }}>
                    {ret != null ? `${ret >= 0 ? '+' : ''}${ret.toFixed(1)}%` : '—'}
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
        {visible.length === 0 && (
          <p className="text-center text-gray-600 py-8 text-sm">No signals match this filter.</p>
        )}
      </div>
    </div>
  )
}

// ── Ticker backtest panel ─────────────────────────────────────────────────────
function TickerPanel() {
  const { data: instruments } = useInstruments()
  const [search, setSearch] = useState('')
  const [symbol, setSymbol] = useState('TQQQ')
  const [inputFocused, setInputFocused] = useState(false)

  const filtered = (instruments ?? []).filter(i =>
    i.symbol.toUpperCase().includes(search.toUpperCase())
  )

  const { data, isLoading } = useBacktestTicker(symbol)

  return (
    <div className="space-y-4">
      {/* Symbol search */}
      <div className="relative w-64">
        <input
          type="text"
          placeholder="Search symbol…"
          value={search}
          onChange={e => setSearch(e.target.value)}
          onFocus={() => setInputFocused(true)}
          onBlur={() => setTimeout(() => setInputFocused(false), 150)}
          className="w-full bg-gray-800 border border-gray-700 rounded px-3 py-2 text-sm text-white placeholder-gray-600 focus:outline-none focus:border-emerald-600"
        />
        {inputFocused && search && filtered.length > 0 && (
          <div className="absolute z-10 w-full mt-1 bg-gray-800 border border-gray-700 rounded shadow-xl max-h-60 overflow-y-auto">
            {filtered.slice(0, 40).map(i => (
              <button
                key={i.symbol}
                onMouseDown={() => { setSymbol(i.symbol); setSearch(i.symbol) }}
                className={`w-full text-left px-3 py-2 text-sm hover:bg-gray-700 ${i.symbol === symbol ? 'text-emerald-400' : 'text-gray-300'}`}
              >
                <span className="font-bold font-mono">{i.symbol}</span>
                {i.name && <span className="text-gray-500 ml-2 text-xs">{i.name}</span>}
              </button>
            ))}
          </div>
        )}
        {!search && (
          <div className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-600 text-xs font-mono">{symbol}</div>
        )}
      </div>

      {isLoading && (
        <div className="text-gray-500 py-10 text-center text-sm">Loading backtest for {symbol}…</div>
      )}

      {data && !isLoading && (
        <>
          {data.stats ? (
            <TickerStats stats={data.stats} forward_days={data.forward_days} />
          ) : (
            <div className="text-gray-500 text-sm">No signals found for {symbol}.</div>
          )}

          {data.signals.length > 0 && (
            <div className="bg-gray-900 border border-gray-800 rounded-xl overflow-hidden">
              <div className="px-4 py-3 border-b border-gray-800">
                <h3 className="text-sm font-bold text-gray-400 uppercase tracking-wider">
                  {symbol} — Signal History
                </h3>
                <p className="text-xs text-gray-600 mt-0.5">
                  {data.forward_days}-day forward window · Stop = −1 ATR · T1 = +2 ATR · Target = +4 ATR
                </p>
              </div>
              <div className="p-4">
                <TickerSignalTable signals={data.signals} />
              </div>
            </div>
          )}
        </>
      )}
    </div>
  )
}

// ── All-instruments backtest ───────────────────────────────────────────────────
const PROB_ORDER  = ['95-100%', '90-95%', '80-90%', '70-80%', '<70%']
const PHASE_ORDER = [
  'Accumulation', 'Early Bull', 'Recovery', 'Momentum Bull',
  'Late Bull', 'Distribution', 'Early Bear', 'Late Bear', 'Markdown', 'Capitulation',
]

function AllPanel() {
  const { data, isLoading } = useBacktestStats()
  const [expandedBucket, setExpandedBucket] = useState<string | null>(null)

  if (isLoading) {
    return (
      <div className="flex flex-col items-center justify-center h-64 gap-4">
        <div className="text-gray-400 text-lg font-bold">Computing backtest results…</div>
        <div className="text-gray-600 text-sm">Analyzing all historical signals against forward price data.</div>
        <div className="text-gray-700 text-xs">This may take 10–30 seconds on first load (cached 1h after).</div>
      </div>
    )
  }
  if (!data) return <div className="text-gray-500 text-center mt-20">No backtest data available.</div>

  const { overall, by_probability, by_phase, forward_days } = data

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-2 md:grid-cols-4 xl:grid-cols-6 gap-3">
        <StatCard label="Signals Analyzed" value={overall.total.toLocaleString()} sub={`${overall.closed.toLocaleString()} closed · ${overall.open.toLocaleString()} still open`} />
        <StatCard label="T1 Win Rate" value={`${overall.win_rate.toFixed(1)}%`} sub="hit +2 ATR before stop" color={overall.win_rate >= 55 ? '#22c55e' : overall.win_rate >= 40 ? '#eab308' : '#ef4444'} />
        <StatCard label="Full Target Rate" value={`${overall.target_rate.toFixed(1)}%`} sub="hit +4 ATR before stop" color="#34d399" />
        <StatCard label="Stop Rate" value={`${overall.stop_rate.toFixed(1)}%`} sub="hit −1 ATR (stopped out)" color={overall.stop_rate <= 25 ? '#22c55e' : overall.stop_rate <= 40 ? '#eab308' : '#ef4444'} />
        <StatCard label="Avg T1 Return" value={overall.avg_t1_return != null ? `+${overall.avg_t1_return.toFixed(1)}%` : '—'} sub="when T1 reached" color="#22c55e" />
        <StatCard label="Avg Stop Loss" value={overall.avg_stop_return != null ? `${overall.avg_stop_return.toFixed(1)}%` : '—'} sub="when stopped out" color="#ef4444" />
      </div>

      <div className="bg-gray-900 border border-gray-800 rounded-xl overflow-hidden">
        <div className="px-4 py-3 border-b border-gray-800">
          <h3 className="text-sm font-bold text-gray-400 uppercase tracking-wider">Win Rate by Probability Range</h3>
          <p className="text-xs text-gray-600 mt-0.5">Higher XGBoost probability → higher historical success · {forward_days}-day window</p>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-800 text-gray-600 text-[10px] uppercase tracking-wider">
                <th className="text-left py-2.5 px-4">Probability</th>
                <th className="text-right py-2.5 px-4">Signals</th>
                <th className="text-right py-2.5 px-4">Open</th>
                <th className="text-left py-2.5 px-4">T1 Win Rate</th>
                <th className="text-left py-2.5 px-4">Full Target</th>
                <th className="text-left py-2.5 px-4">Stop Rate</th>
                <th className="text-right py-2.5 px-4">Avg T1 Ret</th>
                <th className="text-right py-2.5 px-4">Avg Tgt Ret</th>
                <th className="text-right py-2.5 px-4">Avg Stop</th>
              </tr>
            </thead>
            <tbody>
              {PROB_ORDER.map(label => {
                const b = by_probability[label]
                if (!b) return null
                const isExpanded = expandedBucket === label
                return (
                  <>
                    <BucketRow
                      key={label} label={label} b={b}
                      highlight={label === '90-95%' || label === '95-100%'}
                      expanded={isExpanded}
                      onToggle={() => setExpandedBucket(isExpanded ? null : label)}
                    />
                    {isExpanded && <DrillPanel key={`drill-${label}`} bucket={label} />}
                  </>
                )
              })}
            </tbody>
          </table>
        </div>
      </div>

      <div className="bg-gray-900 border border-gray-800 rounded-xl overflow-hidden">
        <div className="px-4 py-3 border-b border-gray-800">
          <h3 className="text-sm font-bold text-gray-400 uppercase tracking-wider">Win Rate by Market Phase</h3>
          <p className="text-xs text-gray-600 mt-0.5">Phase quality matters as much as probability</p>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-800 text-gray-600 text-[10px] uppercase tracking-wider">
                <th className="text-left py-2 px-4">Phase</th>
                <th className="text-right py-2 px-4">Signals</th>
                <th className="text-left py-2 px-4">T1 Win Rate</th>
                <th className="text-right py-2 px-4">Stop Rate</th>
                <th className="text-right py-2 px-4">Avg T1 Ret</th>
              </tr>
            </thead>
            <tbody>
              {PHASE_ORDER.map(phase => {
                const b = by_phase[phase]
                if (!b) return null
                return <PhaseRow key={phase} phase={phase} b={b} />
              })}
            </tbody>
          </table>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-xs">
        <div className="bg-gray-900 border border-emerald-900 rounded-xl p-4">
          <div className="font-bold text-emerald-400 mb-2">HIGH CONVICTION SETUP</div>
          <div className="text-gray-400 leading-relaxed">Probability 90%+ in Accumulation or Early Bull phase. Enter at ENTRY, scale in at ADD level, sell ⅓ at T1.</div>
        </div>
        <div className="bg-gray-900 border border-yellow-900 rounded-xl p-4">
          <div className="font-bold text-yellow-400 mb-2">RISK / REWARD</div>
          <div className="text-gray-400 leading-relaxed">T1 = +2 ATR, Stop = −1 ATR → 2:1 risk/reward minimum. Positive expectancy even at 50% win rate.</div>
        </div>
        <div className="bg-gray-900 border border-blue-900 rounded-xl p-4">
          <div className="font-bold text-blue-400 mb-2">3-TIER EXIT STRATEGY</div>
          <div className="text-gray-400 leading-relaxed">Sell ⅓ at T1 (guarantee profit), ⅓ at Target, keep ⅓ as house money. Move stop to break-even after T1.</div>
        </div>
      </div>
    </div>
  )
}

// ── Main page ─────────────────────────────────────────────────────────────────
type Tab = 'all' | 'ticker'

export default function BacktestPage() {
  const [tab, setTab] = useState<Tab>('all')

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between flex-wrap gap-3">
        <div>
          <h2 className="text-2xl font-black text-white tracking-tight">BACKTEST RESULTS</h2>
          <p className="text-xs text-gray-500 mt-1">Historical signal performance tested against forward price data.</p>
        </div>
        <div className="flex gap-1 bg-gray-900 border border-gray-800 rounded-xl p-1">
          {([['all', 'All Instruments'], ['ticker', 'Single Ticker']] as const).map(([id, label]) => (
            <button
              key={id}
              onClick={() => setTab(id)}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                tab === id ? 'bg-emerald-700 text-white' : 'text-gray-500 hover:text-gray-300 hover:bg-gray-800/50'
              }`}
            >
              {label}
            </button>
          ))}
        </div>
      </div>

      {tab === 'all'    && <AllPanel />}
      {tab === 'ticker' && <TickerPanel />}
    </div>
  )
}

import { useNavigate } from 'react-router-dom'
import { useCommandCenter, useMarketBreadth } from '../api/hooks'
import type { MarketBreadthData } from '../api/hooks'
import type {
  CommandCenterInstrument,
  CommandCenterSignal,
  CommandCenterTrade,
  CommandCenterData,
  ActiveStrategyData,
} from '../types'
import SignalReliabilityPanel from '../components/command/SignalReliabilityPanel'
import MarketSafetyPanel      from '../components/command/MarketSafetyPanel'
import MarketHeatmap          from '../components/terminal/MarketHeatmap'

// ── Color constants ─────────────────────────────────────────────────────────────

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

const LIFECYCLE_COLORS: Record<string, string> = {
  SETUP:   '#6b7280',
  TRIGGER: '#eab308',
  ENTRY:   '#60a5fa',
  ACTIVE:  '#22c55e',
}

const ALIGN_COLORS: Record<string, string> = {
  bullish: '#22c55e',
  bearish: '#ef4444',
  neutral: '#9ca3af',
}

// ── Active Strategy Banner ─────────────────────────────────────────────────────

const STRATEGY_ICONS: Record<string, string> = {
  breakout:   '⚡',
  pullback:   '↩',
  watchlist:  '👁',
  defensive:  '🛡',
}

const RISK_MODE_COLORS: Record<string, string> = {
  aggressive: '#f97316',
  normal:     '#22c55e',
  reduced:    '#60a5fa',
  defensive:  '#ef4444',
}

// ── Market Breadth Panel (McClellan, VIX, Fear/Greed) ─────────────────────────

function MarketBreadthPanel({ breadth }: { breadth: MarketBreadthData }) {
  const vix   = breadth.vix
  const mco   = breadth.mco
  const fg    = breadth.fear_greed
  const label = breadth.fear_greed_label ?? 'N/A'

  const vixColor = !vix ? '#6b7280' :
    vix >= 30 ? '#ef4444' : vix >= 20 ? '#f97316' : vix >= 15 ? '#9ca3af' : '#22c55e'

  const mcoColor = !mco || mco === 0 ? '#9ca3af' : mco > 0 ? '#22c55e' : '#ef4444'

  const fgColor = !fg ? '#6b7280' :
    fg < 20 ? '#ef4444' : fg < 40 ? '#f97316' : fg < 60 ? '#9ca3af' : fg < 80 ? '#22c55e' : '#10b981'

  // Semi-circle gauge for Fear/Greed
  const fgPct   = fg != null ? Math.max(0, Math.min(100, fg)) : 50
  const fgAngle = (fgPct / 100) * 180 - 90 // -90° (left=fear) to +90° (right=greed)
  const needleX = 50 + 38 * Math.cos((fgAngle * Math.PI) / 180)
  const needleY = 50 + 38 * Math.sin((fgAngle * Math.PI) / 180)

  return (
    <div className="bg-gray-900 border border-gray-800 rounded-xl p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-widest">Market Internals</h3>
        <span className="text-[10px] text-gray-600 font-mono">
          {breadth.adv_latest != null && breadth.dec_latest != null
            ? `A/D ${Math.round(breadth.adv_latest).toLocaleString()} / ${Math.round(breadth.dec_latest).toLocaleString()}`
            : 'NYSE Breadth'}
        </span>
      </div>

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">

        {/* VIX */}
        <div className="bg-gray-950 border border-gray-800 rounded-lg p-3">
          <p className="text-[10px] text-gray-600 uppercase tracking-wide mb-1">VIX · Volatility</p>
          <p className="text-2xl font-mono font-bold" style={{ color: vixColor }}>
            {vix != null ? vix.toFixed(1) : '—'}
          </p>
          <p className="text-[10px] mt-0.5" style={{ color: vixColor }}>
            {vix == null ? '' : vix >= 30 ? 'Extreme Fear' : vix >= 20 ? 'Fear / Elevated' : vix >= 15 ? 'Normal Range' : 'Complacency'}
          </p>
          {breadth.vix_change != null && (
            <p className="text-[10px] font-mono text-gray-600 mt-0.5">
              {breadth.vix_change >= 0 ? '+' : ''}{breadth.vix_change.toFixed(2)} today
            </p>
          )}
        </div>

        {/* McClellan Oscillator */}
        <div className="bg-gray-950 border border-gray-800 rounded-lg p-3">
          <p className="text-[10px] text-gray-600 uppercase tracking-wide mb-1">McClellan Osc.</p>
          <p className="text-2xl font-mono font-bold" style={{ color: mcoColor }}>
            {mco != null ? (mco > 0 ? '+' : '') + mco.toFixed(1) : '—'}
          </p>
          <p className="text-[10px] mt-0.5" style={{ color: mcoColor }}>
            {mco == null ? 'Fetching…' : mco > 50 ? 'Strong Breadth ↑' : mco > 0 ? 'Mild Breadth ↑' : mco > -50 ? 'Mild Breadth ↓' : 'Weak Breadth ↓'}
          </p>
          {breadth.mco_sum != null && (
            <p className="text-[10px] font-mono text-gray-600 mt-0.5">
              Sum: {breadth.mco_sum > 0 ? '+' : ''}{breadth.mco_sum.toFixed(0)}
            </p>
          )}
        </div>

        {/* Fear / Greed gauge */}
        <div className="bg-gray-950 border border-gray-800 rounded-lg p-3 flex flex-col items-center">
          <p className="text-[10px] text-gray-600 uppercase tracking-wide mb-1 self-start">Fear / Greed</p>
          {/* Mini semi-circle gauge */}
          <svg viewBox="0 0 100 60" className="w-20 h-12">
            {/* Background arc (fear=red → neutral=gray → greed=green) */}
            <defs>
              <linearGradient id="fgGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%"   stopColor="#ef4444" stopOpacity="0.5" />
                <stop offset="50%"  stopColor="#9ca3af" stopOpacity="0.3" />
                <stop offset="100%" stopColor="#22c55e" stopOpacity="0.5" />
              </linearGradient>
            </defs>
            <path d="M5 50 A45 45 0 0 1 95 50" fill="none" stroke="url(#fgGrad)" strokeWidth="8" />
            {/* Needle */}
            {fg != null && (
              <line x1="50" y1="50" x2={needleX.toFixed(1)} y2={needleY.toFixed(1)}
                stroke={fgColor} strokeWidth="2" strokeLinecap="round" />
            )}
            <circle cx="50" cy="50" r="3" fill={fgColor} />
          </svg>
          <p className="text-xl font-mono font-bold -mt-1" style={{ color: fgColor }}>
            {fg != null ? fg : '—'}
          </p>
          <p className="text-[10px] text-center mt-0.5" style={{ color: fgColor }}>{label}</p>
        </div>

        {/* SPY momentum */}
        <div className="bg-gray-950 border border-gray-800 rounded-lg p-3">
          <p className="text-[10px] text-gray-600 uppercase tracking-wide mb-1">SPY Momentum</p>
          {breadth.spy_momentum != null ? (
            <>
              <p className="text-2xl font-mono font-bold"
                style={{ color: breadth.spy_momentum > 0 ? '#22c55e' : '#ef4444' }}>
                {breadth.spy_momentum > 0 ? '+' : ''}{breadth.spy_momentum.toFixed(1)}%
              </p>
              <p className="text-[10px] mt-0.5"
                style={{ color: breadth.spy_above_ma ? '#22c55e' : '#ef4444' }}>
                {breadth.spy_above_ma ? '↑ Above 125-day MA' : '↓ Below 125-day MA'}
              </p>
              <p className="text-[10px] font-mono text-gray-600 mt-0.5">vs 125-day avg</p>
            </>
          ) : (
            <p className="text-sm text-gray-600 mt-2">Loading…</p>
          )}
        </div>
      </div>

      {/* McClellan direction bar */}
      {mco != null && (
        <div className="mt-3">
          <div className="flex items-center justify-between text-[10px] text-gray-600 mb-1">
            <span>Bearish Breadth</span>
            <span className="font-medium" style={{ color: mcoColor }}>
              MCO {mco > 0 ? '+' : ''}{mco.toFixed(1)} · {breadth.mco_direction?.toUpperCase()}
            </span>
            <span>Bullish Breadth</span>
          </div>
          <div className="h-1.5 rounded-full bg-gray-800 overflow-hidden">
            <div
              className="h-full rounded-full transition-all"
              style={{
                width: `${Math.max(2, Math.min(100, 50 + mco / 2))}%`,
                backgroundColor: mcoColor,
              }}
            />
          </div>
        </div>
      )}
    </div>
  )
}

function ActiveStrategyBanner({ data }: { data: ActiveStrategyData }) {
  const color    = data.color ?? '#6b7280'
  const icon     = STRATEGY_ICONS[data.strategy_key] ?? '◉'
  const modeColor = RISK_MODE_COLORS[data.risk_mode] ?? '#6b7280'

  return (
    <div
      className="rounded-xl border px-4 py-3 flex flex-col sm:flex-row sm:items-center gap-3"
      style={{ borderColor: color + '55', background: color + '0d' }}
    >
      {/* Left: Regime + Strategy */}
      <div className="flex items-center gap-3 flex-1">
        <span className="text-2xl" role="img">{icon}</span>
        <div>
          <div className="flex items-center gap-2 flex-wrap">
            <span className="text-xs text-gray-500 uppercase tracking-wider">Market Regime</span>
            <span
              className="px-2 py-0.5 rounded text-sm font-bold"
              style={{ color, background: color + '22' }}
            >
              {data.dominant_regime}
            </span>
            <span className="text-gray-700">·</span>
            <span className="text-xs text-gray-500 uppercase tracking-wider">Strategy Mode</span>
            <span className="text-sm font-bold text-white">{data.strategy_type}</span>
          </div>
          <p className="text-xs text-gray-500 mt-0.5 max-w-xl">{data.entry_description}</p>
        </div>
      </div>

      {/* Right: Risk metrics */}
      <div className="flex items-center gap-4 text-xs shrink-0">
        <div className="text-center">
          <p className="text-gray-600">Risk/Trade</p>
          <p className="font-mono font-bold text-gray-200">
            {(data.risk_per_trade * 100).toFixed(1)}%
          </p>
        </div>
        <div className="text-center">
          <p className="text-gray-600">Max Positions</p>
          <p className="font-mono font-bold text-gray-200">{data.max_positions}</p>
        </div>
        <div className="text-center">
          <p className="text-gray-600">Size</p>
          <p className="font-mono font-bold text-gray-200">{data.position_multiplier}×</p>
        </div>
        <div className="text-center">
          <p className="text-gray-600">Risk Mode</p>
          <span
            className="px-1.5 py-0.5 rounded font-bold uppercase"
            style={{ color: modeColor, background: modeColor + '22' }}
          >
            {data.risk_mode}
          </span>
        </div>
      </div>

      {/* Regime distribution pills */}
      {Object.keys(data.regime_counts).length > 0 && (
        <div className="flex gap-1 flex-wrap shrink-0">
          {Object.entries(data.regime_counts)
            .sort(([, a], [, b]) => b - a)
            .map(([state, count]) => (
              <span
                key={state}
                className="text-xs px-1.5 py-0.5 rounded"
                style={{
                  color: STATE_COLORS[state] ?? '#6b7280',
                  background: (STATE_COLORS[state] ?? '#6b7280') + '22',
                }}
              >
                {state} {count}
              </span>
            ))}
        </div>
      )}
    </div>
  )
}

// ── Helpers ────────────────────────────────────────────────────────────────────

function StateBadge({ state }: { state: string }) {
  const color = STATE_COLORS[state] ?? '#6b7280'
  return (
    <span
      className="px-1.5 py-0.5 rounded text-xs font-bold uppercase"
      style={{ color, backgroundColor: color + '20' }}
    >
      {state}
    </span>
  )
}

function TierBadge({ tier }: { tier: string }) {
  const color = TIER_COLORS[tier] ?? '#6b7280'
  return (
    <span
      className="px-1.5 py-0.5 rounded text-xs font-bold uppercase"
      style={{ color, backgroundColor: color + '20' }}
    >
      {tier}
    </span>
  )
}

function LifecycleBadge({ state }: { state: string }) {
  const color = LIFECYCLE_COLORS[state] ?? '#6b7280'
  return (
    <span
      className="px-1.5 py-0.5 rounded text-xs font-bold"
      style={{ color, backgroundColor: color + '20' }}
    >
      {state}
    </span>
  )
}

function ScoreBar({ score, max = 100 }: { score: number; max?: number }) {
  const pct = Math.max(0, Math.min(100, (score / max) * 100))
  // Green = high score (good), red = low score (bad)
  const color = score >= 80 ? '#22c55e' : score >= 65 ? '#4ade80' : score >= 50 ? '#eab308' : score >= 30 ? '#f97316' : '#ef4444'
  return (
    <div className="flex items-center gap-2">
      <div className="flex-1 h-1.5 bg-gray-800 rounded-full overflow-hidden">
        <div className="h-full rounded-full transition-all" style={{ width: `${pct}%`, backgroundColor: color }} />
      </div>
      <span className="text-xs font-mono w-8 text-right" style={{ color }}>{score.toFixed(0)}</span>
    </div>
  )
}

function PctBar({ value, max = 100, color = '#22c55e' }: { value: number; max?: number; color?: string }) {
  const pct = Math.max(0, Math.min(100, (value / max) * 100))
  return (
    <div className="flex items-center gap-1.5">
      <div className="flex-1 h-1 bg-gray-800 rounded-full overflow-hidden">
        <div className="h-full rounded-full" style={{ width: `${pct}%`, backgroundColor: color }} />
      </div>
      <span className="text-xs font-mono w-10 text-right" style={{ color }}>{(value * 100).toFixed(1)}%</span>
    </div>
  )
}

function PanelCard({ title, subtitle, badge, children, className = '' }: {
  title: string; subtitle?: string; badge?: React.ReactNode; children: React.ReactNode; className?: string
}) {
  return (
    <div className={`bg-gray-900 border border-gray-800 rounded-xl flex flex-col ${className}`}>
      <div className="flex items-center justify-between px-4 py-3 border-b border-gray-800">
        <div>
          <h2 className="text-sm font-semibold text-gray-100">{title}</h2>
          {subtitle && <p className="text-xs text-gray-500 mt-0.5">{subtitle}</p>}
        </div>
        {badge}
      </div>
      <div className="flex-1 overflow-auto">{children}</div>
    </div>
  )
}

function EmptyState({ message }: { message: string }) {
  return (
    <div className="flex items-center justify-center h-24 text-gray-600 text-sm">
      {message}
    </div>
  )
}

function fmt(v: number | null | undefined, decimals = 2, prefix = '') {
  if (v == null) return '—'
  return `${prefix}${v.toFixed(decimals)}`
}

function fmtTime(iso: string | null) {
  if (!iso) return '—'
  try {
    return new Date(iso).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
  } catch {
    return iso
  }
}

function fmtDate(iso: string | null) {
  if (!iso) return '—'
  try {
    return new Date(iso).toLocaleDateString([], { month: 'short', day: 'numeric' })
  } catch {
    return iso
  }
}

// ── Panel 1: Market State ──────────────────────────────────────────────────────

function MarketStatePanel({ data }: { data: CommandCenterData['market_state_summary'] }) {
  const navigate = useNavigate()
  const { counts = {}, instruments = [] } = data

  const STATE_ORDER = ['EXPANSION', 'TREND', 'COMPRESSION', 'CHAOS']

  return (
    <PanelCard
      title="Market State"
      subtitle={`${instruments.length} instruments tracked`}
      badge={
        <div className="flex gap-1.5 flex-wrap">
          {STATE_ORDER.map(s => {
            const n = counts[s] ?? 0
            if (!n && !instruments.length) return null
            return (
              <span
                key={s}
                className="px-2 py-0.5 rounded-full text-xs font-bold"
                style={{ color: STATE_COLORS[s], backgroundColor: STATE_COLORS[s] + '20' }}
              >
                {s.slice(0, 3)} {n}
              </span>
            )
          })}
        </div>
      }
    >
      {instruments.length === 0 ? (
        <EmptyState message="No market state data — run pipeline first" />
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-gray-800 text-gray-500 uppercase">
                <th className="text-left px-4 py-2 font-medium">Symbol</th>
                <th className="text-left px-4 py-2 font-medium">State</th>
                <th className="px-4 py-2 font-medium">Volatility</th>
                <th className="px-4 py-2 font-medium">Trend</th>
                <th className="text-left px-4 py-2 font-medium">Alignment</th>
                <th className="text-right px-4 py-2 font-medium">Size</th>
              </tr>
            </thead>
            <tbody>
              {instruments.map(inst => (
                <InstrumentRow key={inst.symbol} inst={inst} onClick={() => navigate(`/chart?symbol=${inst.symbol}`)} />
              ))}
            </tbody>
          </table>
        </div>
      )}
    </PanelCard>
  )
}

function InstrumentRow({ inst, onClick }: { inst: CommandCenterInstrument; onClick: () => void }) {
  const volPct = (inst.volatility_percentile ?? 0) * 100
  const trendStr = inst.trend_strength ?? 0

  return (
    <tr
      className="border-b border-gray-800/40 hover:bg-gray-800/30 cursor-pointer transition-colors"
      onClick={onClick}
    >
      <td className="px-4 py-2 font-mono font-bold text-gray-100">{inst.symbol}</td>
      <td className="px-4 py-2">
        <StateBadge state={inst.state} />
      </td>
      <td className="px-4 py-2 w-28">
        <PctBar value={volPct / 100} max={1} color={STATE_COLORS[inst.state] ?? '#9ca3af'} />
      </td>
      <td className="px-4 py-2 w-28">
        <ScoreBar score={trendStr} />
      </td>
      <td className="px-4 py-2">
        <span
          className="text-xs font-semibold capitalize"
          style={{ color: ALIGN_COLORS[inst.ma_alignment ?? 'neutral'] ?? '#9ca3af' }}
        >
          {inst.ma_alignment ?? '—'}
        </span>
      </td>
      <td className="px-4 py-2 text-right text-gray-400">
        {inst.size_multiplier != null ? `${inst.size_multiplier}×` : '—'}
      </td>
    </tr>
  )
}

// ── Panel 2: Opportunity Map ───────────────────────────────────────────────────

function OpportunityMapPanel({ data }: { data: CommandCenterData['opportunity_map'] }) {
  const navigate = useNavigate()
  const { signals = [] } = data

  return (
    <PanelCard
      title="Opportunity Map"
      subtitle="Top confluence-scored setups"
      badge={
        <span className="text-xs text-gray-500">{signals.length} setups</span>
      }
    >
      {signals.length === 0 ? (
        <EmptyState message="No signals — run confluence engine first" />
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-gray-800 text-gray-500 uppercase">
                <th className="text-left px-4 py-2 font-medium">Symbol</th>
                <th className="px-4 py-2 font-medium">Confluence</th>
                <th className="px-4 py-2 font-medium">Quality</th>
                <th className="text-right px-4 py-2 font-medium">Move</th>
                <th className="text-right px-4 py-2 font-medium">Size</th>
                <th className="text-left px-4 py-2 font-medium">State</th>
                <th className="text-left px-4 py-2 font-medium">Tier</th>
                <th className="text-left px-4 py-2 font-medium">Flags</th>
              </tr>
            </thead>
            <tbody>
              {signals.map(sig => (
                <SignalRow key={sig.symbol} sig={sig} onClick={() => navigate(`/chart?symbol=${sig.symbol}`)} />
              ))}
            </tbody>
          </table>
        </div>
      )}
    </PanelCard>
  )
}

function QualityBar({ score }: { score: number | null }) {
  if (score == null) return <span className="text-gray-600 text-xs font-mono">—</span>
  const pct   = Math.max(0, Math.min(100, score))
  const color = score >= 70 ? '#22c55e' : score >= 50 ? '#eab308' : '#ef4444'
  return (
    <div className="flex items-center gap-1.5">
      <div className="w-14 h-1.5 bg-gray-800 rounded-full overflow-hidden">
        <div className="h-full rounded-full" style={{ width: `${pct}%`, backgroundColor: color }} />
      </div>
      <span className="text-[10px] font-mono" style={{ color }}>{score.toFixed(0)}</span>
    </div>
  )
}

function SignalRow({ sig, onClick }: { sig: CommandCenterSignal; onClick: () => void }) {
  return (
    <tr
      className="border-b border-gray-800/40 hover:bg-gray-800/30 cursor-pointer transition-colors"
      onClick={onClick}
    >
      <td className="px-4 py-2 font-mono font-bold text-gray-100">{sig.symbol}</td>
      <td className="px-4 py-2 w-32">
        <ScoreBar score={sig.confluence_score} />
      </td>
      <td className="px-4 py-2 w-28">
        <QualityBar score={sig.setup_quality_score} />
      </td>
      <td className="px-4 py-2 text-right">
        {sig.expected_move_pct != null ? (
          <span className="font-mono text-emerald-400">
            +{(sig.expected_move_pct * 100).toFixed(1)}%
          </span>
        ) : (
          <span className="text-gray-600">—</span>
        )}
      </td>
      <td className="px-4 py-2 text-right font-mono">
        {sig.position_size_multiplier != null ? (
          <span className={sig.size_approved ? 'text-emerald-400' : 'text-amber-400'}>
            {sig.position_size_multiplier.toFixed(1)}×
          </span>
        ) : (
          <span className="text-gray-600">—</span>
        )}
      </td>
      <td className="px-4 py-2">
        {sig.market_state ? <StateBadge state={sig.market_state} /> : <span className="text-gray-600">—</span>}
      </td>
      <td className="px-4 py-2">
        <TierBadge tier={sig.signal_tier} />
      </td>
      <td className="px-4 py-2">
        <span className="flex gap-1">
          {sig.is_compression && (
            <span className="px-1 py-0.5 rounded text-xs bg-blue-900/40 text-blue-400 font-bold">COMP</span>
          )}
          {sig.is_breakout && (
            <span className="px-1 py-0.5 rounded text-xs bg-orange-900/40 text-orange-400 font-bold">BKT</span>
          )}
        </span>
      </td>
    </tr>
  )
}

// ── Panel 3: Active Trades ─────────────────────────────────────────────────────

function ActiveTradesPanel({ data }: { data: CommandCenterData['active_trades'] }) {
  const navigate = useNavigate()
  const { trades = [], count } = data

  return (
    <PanelCard
      title="Active Trades"
      subtitle="Open lifecycle positions"
      badge={
        <span
          className="px-2 py-0.5 rounded-full text-xs font-bold"
          style={{ color: count > 0 ? '#22c55e' : '#6b7280', backgroundColor: (count > 0 ? '#22c55e' : '#6b7280') + '20' }}
        >
          {count} Open
        </span>
      }
    >
      {trades.length === 0 ? (
        <EmptyState message="No open trades" />
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-gray-800 text-gray-500 uppercase">
                <th className="text-left px-4 py-2 font-medium">Symbol</th>
                <th className="text-left px-4 py-2 font-medium">Status</th>
                <th className="text-left px-4 py-2 font-medium">Setup</th>
                <th className="text-right px-4 py-2 font-medium">Entry</th>
                <th className="text-right px-4 py-2 font-medium">Stop</th>
                <th className="text-right px-4 py-2 font-medium">T1</th>
                <th className="text-right px-4 py-2 font-medium">Since</th>
              </tr>
            </thead>
            <tbody>
              {trades.map(trade => (
                <TradeRow key={trade.id} trade={trade} onClick={() => navigate(`/chart?symbol=${trade.symbol}`)} />
              ))}
            </tbody>
          </table>
        </div>
      )}
    </PanelCard>
  )
}

function TradeRow({ trade, onClick }: { trade: CommandCenterTrade; onClick: () => void }) {
  return (
    <tr
      className="border-b border-gray-800/40 hover:bg-gray-800/30 cursor-pointer transition-colors"
      onClick={onClick}
    >
      <td className="px-4 py-2 font-mono font-bold text-gray-100">{trade.symbol}</td>
      <td className="px-4 py-2">
        <LifecycleBadge state={trade.state} />
      </td>
      <td className="px-4 py-2 text-gray-400 truncate max-w-[120px]">
        {trade.setup_type?.replace(/_/g, ' ') ?? '—'}
      </td>
      <td className="px-4 py-2 text-right font-mono text-gray-200">
        {fmt(trade.entry_price, 2, '$')}
      </td>
      <td className="px-4 py-2 text-right font-mono text-red-400">
        {fmt(trade.stop_price, 2, '$')}
      </td>
      <td className="px-4 py-2 text-right font-mono text-emerald-400">
        {fmt(trade.tier1_sell, 2, '$')}
      </td>
      <td className="px-4 py-2 text-right text-gray-500">
        {fmtDate(trade.setup_at)}
      </td>
    </tr>
  )
}

// ── Panel 4: Portfolio Risk ────────────────────────────────────────────────────

function PortfolioRiskPanel({ data }: { data: CommandCenterData['portfolio_risk'] }) {
  const pnlColor = (data.daily_pnl_pct ?? 0) >= 0 ? '#22c55e' : '#ef4444'
  const slotsAvail = data.slots_available ?? 0

  return (
    <PanelCard title="Portfolio Risk" subtitle="Exposure & position limits">
      <div className="p-4 space-y-4">
        {/* ── Top metric grid ── */}
        <div className="grid grid-cols-3 gap-3">
          <MetricTile
            label="Account Equity"
            value={data.account_equity != null ? `$${(data.account_equity / 1000).toFixed(0)}K` : '—'}
            color="#f3f4f6"
          />
          <MetricTile
            label="Positions"
            value={`${data.open_positions ?? 0} / ${data.max_positions ?? 5}`}
            color={slotsAvail > 0 ? '#22c55e' : '#ef4444'}
            sub={`${slotsAvail} slot${slotsAvail !== 1 ? 's' : ''} open`}
          />
          <MetricTile
            label="Daily P&L"
            value={data.daily_pnl_pct != null ? `${(data.daily_pnl_pct * 100).toFixed(2)}%` : '—'}
            color={pnlColor}
          />
          <MetricTile
            label="Risk Budget"
            value={data.risk_budget_remaining != null ? `${(data.risk_budget_remaining * 100).toFixed(1)}%` : '—'}
            color={data.risk_budget_remaining != null && data.risk_budget_remaining > 0.02 ? '#22c55e' : '#ef4444'}
            sub="remaining"
          />
          <MetricTile
            label="Daily Limit"
            value={data.daily_loss_limit != null ? `${(data.daily_loss_limit * 100).toFixed(0)}%` : '—'}
            color="#f97316"
            sub="max drawdown"
          />
          <MetricTile
            label="Risk/Trade"
            value="2.0%"
            color="#9ca3af"
            sub="base rule"
          />
        </div>

        {/* ── Sector exposure bars ── */}
        {data.sector_exposure && Object.keys(data.sector_exposure).length > 0 && (
          <div>
            <p className="text-xs text-gray-500 uppercase font-medium mb-2">Sector Exposure</p>
            <div className="space-y-1.5">
              {Object.entries(data.sector_exposure)
                .sort(([, a], [, b]) => b - a)
                .slice(0, 8)
                .map(([sector, pct]) => {
                  const over = pct > 0.40
                  const color = over ? '#ef4444' : pct > 0.25 ? '#f97316' : '#22c55e'
                  return (
                    <div key={sector} className="flex items-center gap-2">
                      <span className="text-xs text-gray-400 w-28 truncate">{sector}</span>
                      <div className="flex-1 h-1.5 bg-gray-800 rounded-full overflow-hidden">
                        <div
                          className="h-full rounded-full"
                          style={{ width: `${Math.min(pct, 1) * 100}%`, backgroundColor: color }}
                        />
                      </div>
                      <span className="text-xs font-mono w-10 text-right" style={{ color }}>
                        {(pct * 100).toFixed(0)}%
                      </span>
                    </div>
                  )
                })}
            </div>
          </div>
        )}
      </div>
    </PanelCard>
  )
}

function MetricTile({ label, value, color, sub }: { label: string; value: string; color?: string; sub?: string }) {
  return (
    <div className="bg-gray-800/40 rounded-lg p-3">
      <p className="text-xs text-gray-500 mb-1">{label}</p>
      <p className="text-base font-bold font-mono" style={{ color: color ?? '#f3f4f6' }}>{value}</p>
      {sub && <p className="text-xs text-gray-600 mt-0.5">{sub}</p>}
    </div>
  )
}

// ── Panel 5: System Health ─────────────────────────────────────────────────────

function SystemHealthPanel({ data }: { data: CommandCenterData['system_health'] }) {
  const score = data.data_quality_score ?? 0
  const scoreColor = score >= 90 ? '#22c55e' : score >= 75 ? '#eab308' : score >= 50 ? '#f97316' : '#ef4444'
  const scoreLabel = score >= 90 ? 'Excellent' : score >= 75 ? 'Good' : score >= 50 ? 'Fair' : 'Poor'

  const pipelineColors: Record<string, string> = { OK: '#22c55e', STALE: '#eab308', ERROR: '#ef4444' }
  const pipelineColor = pipelineColors[data.pipeline_status] ?? '#6b7280'

  return (
    <PanelCard title="System Health" subtitle="Data quality & pipeline status">
      <div className="p-4 space-y-4">
        {/* ── Quality score ── */}
        <div className="flex items-center gap-4">
          <div
            className="flex-none w-20 h-20 rounded-full border-4 flex flex-col items-center justify-center"
            style={{ borderColor: scoreColor }}
          >
            <span className="text-2xl font-bold font-mono" style={{ color: scoreColor }}>
              {score.toFixed(0)}
            </span>
            <span className="text-xs" style={{ color: scoreColor }}>{scoreLabel}</span>
          </div>
          <div className="flex-1 space-y-2">
            <StatusRow label="Pipeline" value={data.pipeline_status} color={pipelineColor} />
            <StatusRow
              label="Last Update"
              value={data.last_data_update
                ? `${fmtDate(data.last_data_update)} ${fmtTime(data.last_data_update)}`
                : '—'}
              color={pipelineColor}
            />
            <StatusRow
              label="Instruments"
              value={String(data.total_instruments ?? 0)}
              color="#9ca3af"
            />
          </div>
        </div>

        {/* ── Issue summary ── */}
        {data.data_issues && (
          <div className="grid grid-cols-2 gap-2">
            <div className="bg-gray-800/40 rounded-lg p-2.5 text-center">
              <p className="text-xs text-gray-500">Critical Issues</p>
              <p
                className="text-lg font-bold font-mono mt-0.5"
                style={{ color: (data.data_issues.total_critical ?? 0) > 0 ? '#ef4444' : '#22c55e' }}
              >
                {data.data_issues.total_critical ?? 0}
              </p>
            </div>
            <div className="bg-gray-800/40 rounded-lg p-2.5 text-center">
              <p className="text-xs text-gray-500">Below 80 Score</p>
              <p
                className="text-lg font-bold font-mono mt-0.5"
                style={{ color: (data.data_issues.below_80 ?? 0) > 0 ? '#f97316' : '#22c55e' }}
              >
                {data.data_issues.below_80 ?? 0}
              </p>
            </div>
          </div>
        )}

        {/* ── Worst symbols ── */}
        {data.worst_symbols && data.worst_symbols.length > 0 && (
          <div>
            <p className="text-xs text-gray-500 uppercase font-medium mb-1.5">Lowest Quality</p>
            <div className="flex flex-wrap gap-1.5">
              {data.worst_symbols.slice(0, 5).map(sym => (
                <span key={sym} className="px-2 py-0.5 rounded text-xs bg-red-900/20 text-red-400 font-mono">
                  {sym}
                </span>
              ))}
            </div>
          </div>
        )}
      </div>
    </PanelCard>
  )
}

function StatusRow({ label, value, color }: { label: string; value: string; color?: string }) {
  return (
    <div className="flex items-center justify-between">
      <span className="text-xs text-gray-500">{label}</span>
      <span className="text-xs font-semibold" style={{ color: color ?? '#f3f4f6' }}>{value}</span>
    </div>
  )
}

// ── Main Page ──────────────────────────────────────────────────────────────────

export default function CommandCenterPage() {
  const { data, isLoading, isError, dataUpdatedAt, refetch, isFetching } = useCommandCenter()
  const { data: breadth } = useMarketBreadth()

  const lastUpdated = dataUpdatedAt
    ? new Date(dataUpdatedAt).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
    : '—'

  const isStale = dataUpdatedAt ? Date.now() - dataUpdatedAt > 90_000 : true
  const dotColor = isFetching ? '#eab308' : isStale ? '#f97316' : '#22c55e'

  return (
    <div className="space-y-4">
      {/* ── Header ── */}
      <div className="flex items-start justify-between gap-4">
        <div>
          <h1 className="text-xl font-bold text-gray-100 flex items-center gap-2">
            <span>⌘</span> Command Center
          </h1>
          <p className="text-xs text-gray-500 mt-0.5">
            Unified operational dashboard · Auto-refreshes every 60s
          </p>
        </div>
        <div className="flex items-center gap-3 flex-shrink-0">
          {/* Live indicator */}
          <div className="flex items-center gap-1.5">
            <span
              className="inline-block w-2 h-2 rounded-full"
              style={{
                backgroundColor: dotColor,
                boxShadow: isFetching ? `0 0 0 2px ${dotColor}33` : undefined,
              }}
            />
            <span className="text-xs text-gray-500">
              {isFetching ? 'Refreshing…' : `Updated ${lastUpdated}`}
            </span>
          </div>
          {/* Refresh button */}
          <button
            onClick={() => refetch()}
            disabled={isFetching}
            className="px-3 py-1.5 text-xs font-medium bg-gray-800 hover:bg-gray-700 text-gray-300 rounded-lg border border-gray-700 disabled:opacity-50 transition-colors"
          >
            {isFetching ? '↻ Loading…' : '↻ Refresh'}
          </button>
        </div>
      </div>

      {/* ── Error state ── */}
      {isError && (
        <div className="bg-red-900/20 border border-red-900 rounded-xl p-4 text-red-400 text-sm">
          Failed to load Command Center data. Ensure the API is running and the pipeline has been executed at least once.
        </div>
      )}

      {/* ── Loading skeleton ── */}
      {isLoading && !data && (
        <div className="grid grid-cols-1 gap-4">
          {[...Array(3)].map((_, i) => (
            <div key={i} className="bg-gray-900 border border-gray-800 rounded-xl h-40 animate-pulse" />
          ))}
        </div>
      )}

      {/* ── Dashboard panels ── */}
      {data && (
        <div className="space-y-4">
          {/* Row 0: Active Strategy Banner */}
          {data.active_strategy && (
            <ActiveStrategyBanner data={data.active_strategy} />
          )}

          {/* Row 0.5: Market Breadth — VIX + McClellan + Fear/Greed */}
          {breadth && !breadth.error && <MarketBreadthPanel breadth={breadth} />}

          {/* Market Heatmap — right below Market Internals */}
          {data.market_state_summary?.instruments?.length > 0 && (
            <MarketHeatmap instruments={data.market_state_summary.instruments} />
          )}

          {/* Portfolio Risk (1/3) + System Health (2/3) */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
            <div className="lg:col-span-1">
              <PortfolioRiskPanel data={data.portfolio_risk} />
            </div>
            <div className="lg:col-span-2">
              <SystemHealthPanel data={data.system_health} />
            </div>
          </div>

          {/* Signal Reliability (2/3) + Market Safety (1/3) */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
            <div className="lg:col-span-2">
              {data.signal_reliability && (
                <SignalReliabilityPanel data={data.signal_reliability} />
              )}
            </div>
            <div className="lg:col-span-1">
              {data.market_safety && (
                <MarketSafetyPanel data={data.market_safety} />
              )}
            </div>
          </div>

          {/* ── Footer: Interactive guide ── */}
          <div className="bg-gray-900/50 border border-gray-800/50 rounded-xl px-4 py-3 text-xs text-gray-600">
            <span className="text-gray-500 font-medium">Interactive: </span>
            Click any symbol row to open the chart view with full decision trace, compression ranges, and breakout levels.
            <span className="ml-2 text-gray-700">· State colors: </span>
            {[['EXPANSION','#f97316'],['TREND','#22c55e'],['COMPRESSION','#60a5fa'],['CHAOS','#ef4444']].map(([s,c]) => (
              <span key={s} className="mr-2" style={{ color: c as string }}>●{s}</span>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

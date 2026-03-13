import { useNavigate } from 'react-router-dom'
import type { MarketSafetyData, HedgeSuggestion } from '../../types'

// ── Color maps ────────────────────────────────────────────────────────────────

const STATE_COLORS: Record<string, string> = {
  NORMAL:    '#22c55e',
  CAUTION:   '#eab308',
  SAFE_MODE: '#ef4444',
}

const STATE_BG: Record<string, string> = {
  NORMAL:    '#052e16',
  CAUTION:   '#422006',
  SAFE_MODE: '#450a0a',
}

const STATE_ICONS: Record<string, string> = {
  NORMAL:    '✓',
  CAUTION:   '⚠',
  SAFE_MODE: '⛔',
}

function componentColor(score: number): string {
  if (score >= 70) return '#22c55e'
  if (score >= 50) return '#eab308'
  return '#ef4444'
}

// ── Component bar row ─────────────────────────────────────────────────────────

function ComponentBar({ label, score }: { label: string; score: number }) {
  const color = componentColor(score)
  return (
    <div className="flex items-center gap-2">
      <span className="text-xs text-gray-400 w-24 capitalize">{label}</span>
      <div className="flex-1 h-1.5 bg-gray-800 rounded-full overflow-hidden">
        <div
          className="h-full rounded-full transition-all"
          style={{ width: `${Math.min(score, 100)}%`, backgroundColor: color }}
        />
      </div>
      <span className="text-xs font-mono w-8 text-right" style={{ color }}>{score.toFixed(0)}</span>
    </div>
  )
}

// ── Hedge category metadata ────────────────────────────────────────────────────

const HEDGE_CATEGORY_META: Record<HedgeSuggestion['category'], { label: string; color: string; icon: string }> = {
  volatility:      { label: 'Vol Surge',      color: '#a78bfa', icon: '⚡' },
  inverse_broad:   { label: 'Inverse Broad',  color: '#f97316', icon: '▼' },
  inverse_sector:  { label: 'Inverse Sector', color: '#fb923c', icon: '▼' },
  safe_haven:      { label: 'Safe Haven',     color: '#60a5fa', icon: '⛽' },
}

function HedgeRow({ hedge, onClick }: { hedge: HedgeSuggestion; onClick: () => void }) {
  const meta = HEDGE_CATEGORY_META[hedge.category]
  return (
    <button
      onClick={onClick}
      className="w-full flex items-start gap-2.5 px-3 py-2 rounded-lg hover:bg-gray-800/60 transition-colors text-left border border-gray-800/40 hover:border-gray-700/60"
    >
      <span className="text-sm flex-shrink-0 mt-0.5" style={{ color: meta.color }}>{meta.icon}</span>
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-1.5 flex-wrap">
          <span className="text-xs font-mono font-bold text-gray-100">{hedge.symbol}</span>
          <span
            className="text-[9px] font-bold px-1 py-0.5 rounded uppercase flex-shrink-0"
            style={{ color: meta.color, backgroundColor: meta.color + '20' }}
          >
            {meta.label}
          </span>
        </div>
        <p className="text-[10px] text-gray-500 mt-0.5 leading-tight">{hedge.description}</p>
      </div>
    </button>
  )
}

// ── Panel ─────────────────────────────────────────────────────────────────────

interface Props {
  data: MarketSafetyData
}

export default function MarketSafetyPanel({ data }: Props) {
  const navigate   = useNavigate()
  const stateColor = STATE_COLORS[data.safety_state] ?? '#6b7280'
  const stateBg    = STATE_BG[data.safety_state]   ?? '#111827'
  const stateIcon  = STATE_ICONS[data.safety_state] ?? '?'

  const components = data.components ?? { volatility: 50, gap: 100, portfolio: 100, data_quality: 50 }
  const hedges     = data.suggested_hedges ?? []

  return (
    <div className="bg-gray-900 border border-gray-800 rounded-xl flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-gray-800">
        <div>
          <h2 className="text-sm font-semibold text-gray-100">Market Safety</h2>
          <p className="text-xs text-gray-500 mt-0.5">Capital protection engine</p>
        </div>
        <span
          className="px-3 py-1 rounded-lg text-sm font-bold uppercase tracking-wide"
          style={{ color: stateColor, backgroundColor: stateColor + '20', border: `1px solid ${stateColor}40` }}
        >
          {stateIcon} {data.safety_state}
        </span>
      </div>

      <div className="p-4 space-y-4">
        {/* ── Safety score + action ── */}
        <div
          className="flex items-center gap-4 rounded-xl px-4 py-3 border"
          style={{ backgroundColor: stateBg, borderColor: stateColor + '40' }}
        >
          {/* Big score circle */}
          <div
            className="flex-none w-16 h-16 rounded-full border-4 flex flex-col items-center justify-center"
            style={{ borderColor: stateColor }}
          >
            <span className="text-xl font-bold font-mono" style={{ color: stateColor }}>
              {data.safety_score.toFixed(0)}
            </span>
          </div>

          {/* Action + multiplier */}
          <div className="flex-1 min-w-0">
            <p className="text-sm text-gray-200 font-medium leading-snug break-words">{data.system_action}</p>
            <div className="flex items-center gap-2 mt-2 text-xs flex-wrap">
              <span className="text-gray-500">Size</span>
              <span
                className="font-bold font-mono"
                style={{ color: data.size_multiplier >= 1 ? '#22c55e' : data.size_multiplier > 0 ? '#eab308' : '#ef4444' }}
              >
                {data.size_multiplier.toFixed(1)}×
              </span>
              <span className="text-gray-500">Trading</span>
              <span
                className="font-bold"
                style={{ color: data.trading_allowed ? '#22c55e' : '#ef4444' }}
              >
                {data.trading_allowed ? 'ALLOWED' : 'BLOCKED'}
              </span>
            </div>
          </div>
        </div>

        {/* ── Component scores ── */}
        <div className="space-y-2">
          <p className="text-xs text-gray-500 uppercase font-medium">Score Components</p>
          <ComponentBar label="Volatility (40%)"    score={components.volatility}   />
          <ComponentBar label="Gap Risk (25%)"       score={components.gap}          />
          <ComponentBar label="Portfolio (20%)"      score={components.portfolio}    />
          <ComponentBar label="Data Quality (15%)"   score={components.data_quality} />
        </div>

        {/* ── Active triggers ── */}
        {data.triggers && data.triggers.length > 0 && (
          <div>
            <p className="text-xs text-gray-500 uppercase font-medium mb-2">Active Triggers</p>
            <div className="space-y-1.5">
              {data.triggers.map((t, i) => (
                <div
                  key={i}
                  className="flex items-start gap-2 px-3 py-2 rounded-lg text-xs"
                  style={{ backgroundColor: stateColor + '10', borderLeft: `2px solid ${stateColor}` }}
                >
                  <span style={{ color: stateColor }}>⚡</span>
                  <span className="text-gray-300">{t}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* ── Suggested Hedges (volatility spike only) ── */}
        {hedges.length > 0 && (
          <div>
            <div className="flex items-center gap-2 mb-2">
              <p className="text-xs text-gray-500 uppercase font-medium">Suggested Hedges</p>
              <span className="text-[9px] font-bold px-1.5 py-0.5 rounded bg-red-900/40 text-red-400 uppercase tracking-wide">
                Vol Spike
              </span>
            </div>
            <div className="space-y-1.5">
              {hedges.map(h => (
                <HedgeRow
                  key={h.symbol}
                  hedge={h}
                  onClick={() => navigate(`/chart?symbol=${h.symbol}`)}
                />
              ))}
            </div>
          </div>
        )}

        {data.error && (
          <div className="px-3 py-2 rounded-lg bg-red-900/20 border border-red-800/40 text-xs text-red-400">
            {data.error}
          </div>
        )}
      </div>
    </div>
  )
}
